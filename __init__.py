# DOCSQZ/__init__.py

# import submodules
from . import config
from . import common
from . import docx
from . import pdf
from . import html

# Re-export high-level functions for flat API
from .docx.convert import (
    convert_docx,
    df_to_markdown_table
)
from .docx.extract import (
    extract_docx_text,
    extract_docx_structure,
    extract_docx_markdown,
    extract_docx_metadata,
    parse_docx_sections,   # <-- add this
    DEFAULT_PARSE_CONFIG,  # <-- optional: only if you want to expose it
)

from .pdf.flatten import flatten_acrobat_macos, flatten_aspose
from .pdf.extract import (
    extract_pdf_ocr,
    extract_pdf_layout,
    extract_pdf_adobe,
    extract_pdf_text,
    extract_pdf_auto,
    standardize_extraction_result,
    group_to_level,
)

from .html.extract import extract_html_structure

from .common.cleaner import clean_text

__all__ = [
    # docx
    "convert_docx",
    "df_to_markdown_table",
    "extract_docx_text",
    "extract_docx_structure",
    "extract_docx_markdown",
    "extract_docx_metadata",
    "parse_docx_sections",   # <-- add
    "DEFAULT_PARSE_CONFIG",  # <-- optional
    # pdf
    "flatten_acrobat_macos",
    "flatten_aspose",
    "extract_pdf_ocr",
    "extract_pdf_layout",
    "extract_pdf_adobe",
    "extract_pdf_text",
    "extract_pdf_auto",
    "standardize_extraction_result",
    "group_to_level",
    # html
    "extract_html_structure",
    # common
    "clean_text"
]
