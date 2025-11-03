# docsqz/docx/__init__.py

from .convert import (
    convert_docx,
    df_to_markdown_table
)
from .extract import (
    extract_docx_text,
    extract_docx_structure,
    extract_docx_markdown,
    extract_docx_metadata,
    parse_docx_sections,     # <-- new
    DEFAULT_PARSE_CONFIG,    # <-- optional: expose defaults so users can tweak
)

__all__ = [
    "convert_docx",
    "df_to_markdown_table",
    "extract_docx_text",
    "extract_docx_structure",
    "extract_docx_markdown",
    "extract_docx_metadata",
    "parse_docx_sections",
    "DEFAULT_PARSE_CONFIG",  # include only if you want notebooks to import it directly
]
