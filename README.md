# docsqz

A document processing pipeline package that supports PDF flattening, OCR-based extraction, and DOCX conversion/extraction.

## Installation

```bash
pip install -e .
```

## Usage

```python
from docsqz.docx import convert_docx, extract_docx_text
from docsqz.pdf.flatten import flatten_acrobat_macos, flatten_aspose

# Flatten a PDF using Acrobat (macOS)
flatten_acrobat_macos("input.pdf")

# Flatten a PDF using Aspose
flatten_aspose("input.pdf")

# Convert a DOCX to another format
convert_docx("input.docx")

# Extract text from DOCX
extract_docx_text("input.docx")
```
