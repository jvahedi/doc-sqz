# text_clean.py
# Lightweight, reusable text cleaner for scraped PDFs/handbooks.

from __future__ import annotations
import re

import unicodedata
try:
    from ftfy import fix_text as _fix_text
except Exception:
    _fix_text = None  # optional dep

# ---------- Invisibles ----------
RE_BIDI = re.compile(r"[\u202A-\u202E\u2066-\u2069\u200E\u200F]")  # LRE/RLE/PDF/LRO/RLO + LRI/RLI/FSI/PDI + LRM/RLM
RE_ZERO = re.compile(r"[\u200B-\u200D\uFEFF]")                     # ZWSP/ZWNJ/ZWJ/BOM

# ---------- TOC heuristics ----------
RE_TOC_HEADER = re.compile(r"(?im)^\s*(table\s+of\s+contents|contents)\s*$")
RE_TOC_LEADER = re.compile(r"\s*\.{2,}\s*\d{1,4}")                 # "..... 37" anywhere in text
RE_TOC_SIMPLE = re.compile(r"^[^\n]{3,120}\s+\d{1,4}\s*$")         # "Title     42" at line end

def _repair_mojibake(s: str) -> str:
    # Only if ftfy is available; otherwise no-op
    if _fix_text is None:
        return s
    try:
        return _fix_text(s)
    except Exception:
        return s

def _decode_escapes(s: str) -> str:
    """
    Decode literal escape sequences that might be in source text.
    Only decodes specific patterns to avoid breaking legitimate backslashes.
    """
    # Decode Unicode escapes like \u2022 → •
    def replace_unicode(match):
        try:
            code = int(match.group(1), 16)
            return chr(code)
        except:
            return match.group(0)
    s = re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode, s)

    # Decode hex escapes like \x27 → ' (only valid hex codes)
    def replace_hex(match):
        try:
            code = int(match.group(1), 16)
            # Only decode printable characters or common ones
            if 0x20 <= code <= 0x7E or code in (0x09, 0x0A, 0x0D):
                return chr(code)
            return match.group(0)
        except:
            return match.group(0)
    s = re.sub(r'\\x([0-9a-fA-F]{2})', replace_hex, s)

    return s

def clean_text(
    text: str,
    *,
    toc_mode: str = "drop",          # "drop" | "keep" | "off"
    keep_word_boundaries: bool = True,
    normalize_blank_runs: bool = True,
    keep_toc_numbers: bool = True,   # keep page numbers when toc_mode="keep"
    force_ascii: bool = False        # convert Unicode punctuation to ASCII for Excel compatibility
) -> str:
    """
    Clean a block of text:
      1) Remove bidi/zero-width controls (replace with space if keep_word_boundaries=True).
      2) TOC handling:
         - "drop": remove TOC headered blocks and floating TOC-like lines
         - "keep": compress dot-leaders; strip page numbers unless keep_toc_numbers=True
         - "off":  leave as-is
      3) Normalize whitespace (collapse excessive blanks, trim line-edge spaces).
    """
    if not text:
        return text

    # --- NEW: repair common mojibake (Ã, Â, �, etc.) ---
    s = _repair_mojibake(str(text))

    # --- NEW: decode literal escape sequences (\u2022, \x27, etc.) ---
    s = _decode_escapes(s)

    # --- NEW: Unicode canonicalization (helps compatibility) ---
    s = unicodedata.normalize("NFKC", s)

    # Fix OCR artifact: lowercase 'o' used as bullet in lists (always applied)
    # Pattern 1: After punctuation: "; o word" or ". o word" → "; * word"
    s = re.sub(r'([;.])\s+o\s+([a-z])', r'\1 * \2', s)
    # Pattern 2: Start of line: "o word" → "* word" (multiline mode)
    s = re.sub(r'(?m)^o\s+([a-z])', r'* \1', s)
    # Pattern 3: After newline with spaces: "\n  o word" → "\n  * word"
    s = re.sub(r'(\n\s+)o\s+([a-z])', r'\1* \2', s)

    # Always convert bullets to asterisks (Excel fonts render them inconsistently)
    s = s.replace('\uf0b7', '* ')  # U+F0B7 private use area
    s = s.replace('•', '* ')       # U+2022 bullet (always converted)
    s = s.replace('●', '* ')       # U+25CF black circle (always converted)

    # Optional: Convert other Unicode punctuation to ASCII for Excel compatibility
    if force_ascii:
        # Dashes → hyphen
        s = s.replace('–', '-')       # U+2013 en-dash
        s = s.replace('—', '-')       # U+2014 em-dash

        # Fraction slash → regular slash
        s = s.replace('⁄', '/')       # U+2044 fraction slash

        # Smart quotes → straight quotes
        s = s.replace('"', '"').replace('"', '"')  # U+201C, U+201D
        s = s.replace(''', "'").replace(''', "'")  # U+2018, U+2019
        s = s.replace('…', '...')     # U+2026 ellipsis

    # 1) strip control chars (preserve word boundaries by using a space)
    repl = " " if keep_word_boundaries else ""
    s = RE_BIDI.sub(repl, s)
    s = RE_ZERO.sub(repl, s)
    # tidy line-edge spaces and internal multi-spaces (keep newlines)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    # 2) TOC handling
    if toc_mode == "drop":
        s = _drop_toc(s)
    elif toc_mode == "keep":
        s = _keep_toc_titles(s, keep_numbers=keep_toc_numbers)
    elif toc_mode != "off":
        raise ValueError('toc_mode must be "drop", "keep", or "off"')

    # 3) normalize blank runs
    if normalize_blank_runs:
        s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()

def _drop_toc(text: str) -> str:
    """Remove headered TOC blocks and floating leader+page lines."""
    lines = text.splitlines()
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        # Headered TOC block
        if RE_TOC_HEADER.match(line):
            i += 1
            while i < n and (lines[i].strip() == "" or RE_TOC_LEADER.search(lines[i]) or RE_TOC_SIMPLE.search(lines[i])):
                i += 1
            continue
        # Single line with multiple TOC entries (e.g., "OVERVIEW...5 STUDENT RECORDS...13")
        if len(RE_TOC_LEADER.findall(line)) >= 2:
            i += 1
            continue
        # Floating TOC run (leaders on multiple consecutive lines)
        if RE_TOC_LEADER.search(line) and any(RE_TOC_LEADER.search(l) for l in lines[i+1:i+6]):
            while i < n and (lines[i].strip() == "" or RE_TOC_LEADER.search(lines[i])):
                i += 1
            continue
        # Otherwise keep
        out.append(line)
        i += 1
    return "\n".join(out)

def _keep_toc_titles(text: str, keep_numbers: bool = False) -> str:
    """Keep lines but compress dot-leaders and optionally strip page numbers."""
    cleaned = []
    for line in text.splitlines():
        if keep_numbers:
            # Compress dots but keep numbers: "OVERVIEW.....5" -> "OVERVIEW... 5"
            L = re.sub(r'\.{3,}', '...', line)
            # Collapse multiple spaces, trim
            L = re.sub(r"[ \t]{2,}", " ", L).rstrip()
        else:
            # Strip dot-leaders with page numbers: "OVERVIEW.....5" -> "OVERVIEW..."
            L = re.sub(r'\s*\.{2,}\s*\d{1,4}(?=\s|$)', '...', line)
            # Also strip simple trailing page numbers without dots
            L = re.sub(r'\s+\d{1,4}(?=\s|$)', '', L)
            # Collapse multiple spaces, trim
            L = re.sub(r"[ \t]{2,}", " ", L).rstrip()
        cleaned.append(L)
    return "\n".join(cleaned)

# ---------- Debug helper ----------
_INVIS_MAP = {
    "\u202a":"⟦LRE⟧", "\u202b":"⟦RLE⟧", "\u202c":"⟦PDF⟧",
    "\u202d":"⟦LRO⟧", "\u202e":"⟦RLO⟧",
    "\u2066":"⟦LRI⟧", "\u2067":"⟦RLI⟧", "\u2068":"⟦FSI⟧", "\u2069":"⟦PDI⟧",
    "\u200e":"⟦LRM⟧", "\u200f":"⟦RLM⟧",
    "\u200b":"⟦ZWSP⟧", "\u200c":"⟦ZWNJ⟧", "\u200d":"⟦ZWJ⟧", "\ufeff":"⟦BOM⟧",
}
def visualize_invisibles(text: str) -> str:
    """Render hidden controls as bracketed tokens for inspection."""
    return "".join(_INVIS_MAP.get(ch, ch) for ch in text or "")
