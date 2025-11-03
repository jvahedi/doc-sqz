"""
DOCX extraction utilities for docsqz.
Uses python-docx to pull text, structure, markdown, and metadata.
"""
import re
from pathlib import Path
from typing import Iterable, Any, Dict, List, Tuple

import pandas as pd
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table


def extract_docx_text(docx_path: str | Path) -> str:
    """
    Extract plain text from paragraphs in a DOCX.
    Returns concatenated string.
    """
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def _iter_block_items(doc_or_cell) -> Iterable[Any]:
    """Yield Paragraph and Table objects in document order (works for Document and Cell)."""
    if hasattr(doc_or_cell, "element") and hasattr(doc_or_cell.element, "body"):
        body = doc_or_cell.element.body   # document
        parent = doc_or_cell
    else:
        body = doc_or_cell._tc            # table cell
        parent = doc_or_cell
    for child in body.iterchildren():
        tag = child.tag.lower()
        if tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif tag.endswith("}tbl"):
            yield Table(child, parent)

def extract_docx_structure(docx_path: str | Path) -> pd.DataFrame:
    """
    Emit a single DataFrame in *true document order*:
      - paragraphs as rows (type='paragraph')
      - each table cell as a row (type='table_cell') with table_id, row, col
    """
    doc = Document(docx_path)
    rows: List[Dict[str, Any]] = []
    order = 0
    table_id = 0

    for blk in _iter_block_items(doc):
        if isinstance(blk, Paragraph):
            text = (blk.text or "").strip()
            rows.append({
                "order": order,
                "type": "paragraph",
                "text": text,
                "style": blk.style.name if blk.style else None,
                "table_id": None,
                "row": None,
                "col": None,
            })
            order += 1

        elif isinstance(blk, Table):
            # emit each cell
            for r_idx, r in enumerate(blk.rows):
                for c_idx, cell in enumerate(r.cells):
                    text = (cell.text or "").strip()
                    rows.append({
                        "order": order,
                        "type": "table_cell",
                        "text": text,
                        "style": None,
                        "table_id": table_id,
                        "row": r_idx,
                        "col": c_idx,
                    })
                    order += 1
            table_id += 1

    return pd.DataFrame(rows)

def extract_docx_markdown(docx_path: str | Path) -> str:
    """
    Approximate markdown export: paragraphs and tables with | separators.
    """
    lines = []
    doc = Document(docx_path)

    for p in doc.paragraphs:
        if p.style and "Heading" in p.style.name:
            level = "".join(ch for ch in p.style.name if ch.isdigit()) or "1"
            level = int(level)
            lines.append("#" * level + " " + p.text)
        else:
            lines.append(p.text)

    for tbl in doc.tables:
        for row in tbl.rows:
            cells = [cell.text.strip() for cell in row.cells]
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def extract_docx_metadata(docx_path: str | Path) -> dict:
    """
    Extract document core properties.
    """
    doc = Document(docx_path)
    props = doc.core_properties
    return {
        "author": props.author,
        "title": props.title,
        "subject": props.subject,
        "keywords": props.keywords,
        "last_modified_by": props.last_modified_by,
        "created": props.created,
        "modified": props.modified,
    }

# -----------------------
# Parsing: H1 → H2 → items (dict-config, simple)
# -----------------------

# Default parse configuration (tweak per project via `config=` in the call)
DEFAULT_PARSE_CONFIG: Dict[str, object] = {
    # Which styles mark top-level sections (H1) and sub-sections (H2)
    "heading_h1": ("Heading 1",),
    "heading_h2": ("Heading 2",),

    # Styles treated as list/body items under a heading
    "list_styles": ("List Paragraph", "Normal", "Normal (Web)"),

    # Strip numbering/letters from H2 text: "1. Handbooks" → "Handbooks", "A) Foo" → "Foo"
    "number_prefix_regex": r"^\s*([0-9]+|[A-Za-z]+)[\.\)]\s*",

    # Characters to trim off the front of list items (common bullets/dashes)
    "bullet_chars": "•-–—",

    # Optional cleanups/behaviors
    "strip_quotes": False,               # also strip wrapping quotes from items (“…”, "…")
    "accept_any_text_as_item": False,    # treat any non-empty text under H2 as item (even if style not in list_styles)
    "allow_items_without_h2": False,     # allow items directly under H1 (stored under "_items")

    # Optional filters (simple yes/no functions). None = keep all.
    # Example: lambda h1: "next steps" not in h1.lower()
    "h1_keep": None,                     # function(str)->bool to keep/drop H1 sections
    "h2_keep": None,                     # function(str)->bool to keep/drop H2 groups
}

_num_prefix_cache: Dict[str, re.Pattern] = {}

def _strip_number_prefix(s: str, pattern: str) -> str:
    rx = _num_prefix_cache.get(pattern)
    if rx is None:
        rx = re.compile(pattern)
        _num_prefix_cache[pattern] = rx
    return rx.sub("", s or "").strip()

def _clean_bullet(s: str, bullet_chars: str, strip_quotes: bool) -> str:
    s = (s or "").strip().lstrip(bullet_chars).strip()
    if strip_quotes:
        s = s.strip('"“”')
    return s

def parse_docx_sections(
    docx_path: str | Path,
    config: Dict[str, object] | None = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse a DOCX into nested sections by style:
      { <H1 text>: { <H2 text>: [items...] } }

    - H1 starts a new top-level section
    - H2 starts a new subgroup inside the current H1 (number/letter prefix stripped)
    - List/Normal lines become items under the current H1+H2
    - Behavior is adjustable via `config` (merged with DEFAULT_PARSE_CONFIG)
    """
    cfg = DEFAULT_PARSE_CONFIG.copy()
    if config:
        cfg.update(config)

    df = extract_docx_structure(docx_path)

    sections: Dict[str, Dict[str, List[str]]] = {}
    current_h1, current_h2 = None, None

    for _, row in df.iterrows():
        style = (row.get("style") or "").strip()
        text  = (row.get("text")  or "").strip()
        if not text:
            continue

        # H1 section
        if style in cfg["heading_h1"]:
            h1 = text
            keep_h1 = cfg["h1_keep"](h1) if cfg.get("h1_keep") else True
            if not keep_h1:
                current_h1, current_h2 = None, None
                continue
            current_h1, current_h2 = h1, None
            sections.setdefault(current_h1, {})
            continue

        # H2 group
        if style in cfg["heading_h2"]:
            if not current_h1:
                continue  # orphan H2: ignore
            h2 = _strip_number_prefix(text, cfg["number_prefix_regex"])
            keep_h2 = cfg["h2_keep"](h2) if cfg.get("h2_keep") else True
            if not keep_h2:
                current_h2 = None
                continue
            current_h2 = h2
            sections[current_h1].setdefault(current_h2, [])
            continue

        # Item lines
        is_listish = style in cfg["list_styles"] or cfg["accept_any_text_as_item"]
        if is_listish and current_h1:
            item = _clean_bullet(text, cfg["bullet_chars"], cfg["strip_quotes"])
            if not item:
                continue
            if current_h2:
                sections[current_h1][current_h2].append(item)
            elif cfg["allow_items_without_h2"]:
                bucket = sections[current_h1].setdefault("_items", [])
                bucket.append(item)

    return sections
