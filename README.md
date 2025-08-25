# 📄 docsqz

**docsqz** is a modular Python package for **document processing pipelines**, extracted from a notebook workflow.
It provides robust utilities for:

* **Flattening PDFs** (Acrobat on macOS, Aspose Cloud API)
* **Extracting structure from PDFs** (OCR, layout detection, Adobe Extract API)
* **Converting DOCX documents** (Pandoc route)
* **Extracting DOCX text/structure/metadata** (python-docx)
* **Visual overlays** for debugging layouts

---

## 🚀 Features

* **PDF Flattening**

  * macOS Acrobat “print to PDF” pipeline via `PRINTER` and `SPOOL_DIR`
  * Aspose API pipeline using `ASPOSE_CLIENT_ID/SECRET`

* **OCR & Layout Extraction**

  * PaddleOCR v4/v5 mobile/server engines
  * Tesseract OCR (via `pytesseract`)
  * Layout detection via PubLayNet/Detectron2
  * Adobe PDF Extract API → structured DataFrame

* **DOCX Handling**

  * Conversion with Pandoc
  * Extraction of plain text, hierarchical structure, Markdown, and metadata

* **Visualization**

  * Overlay bounding boxes on rendered PDF pages
  * Render document layouts for inspection

---

## 📦 Installation

```bash
git clone https://github.com/yourname/docsqz.git
cd docsqz
pip install -e .
```

### Dependencies

* Python 3.9+
* System:

  * **Tesseract OCR** (optional): `brew install tesseract` or `apt-get install tesseract-ocr`
  * **Acrobat (macOS)** for the printer-based flatten pipeline
* Python libraries:

  * `pymupdf`, `pillow`, `numpy`, `pandas`, `layoutparser`, `paddleocr`, `python-docx`, `pypandoc`

---

## ⚙️ Configuration

`docsqz` reads several environment variables:

| Variable               | Default       | Description                              |
| ---------------------- | ------------- | ---------------------------------------- |
| `PRINTER`              | `"PDFwriter"` | Name of Acrobat virtual printer on macOS |
| `SPOOL_DIR`            | `"/tmp"`      | Directory where spooler writes PDFs      |
| `PDF_TIMEOUT_SECS`     | `30`          | Timeout for Acrobat flattening           |
| `ASPOSE_CLIENT_ID`     | *required*    | Aspose Cloud API client ID               |
| `ASPOSE_CLIENT_SECRET` | *required*    | Aspose Cloud API client secret           |

Export these in your shell or `.env` file before use.

---

## 🏃 Quickstart

```python
from docsqz.pdf.flatten import flatten_acrobat_macos, flatten_aspose
from docsqz.pdf.extract import ocr_extract, adobe_extract
from docsqz.docx import convert_docx, extract_docx_text
from docsqz.common.visualize import overlay_bounds

# --- PDF flatten ---
flat_pdf = flatten_acrobat_macos("input.pdf")
# or
flat_pdf = flatten_aspose("input.pdf")

# --- OCR extraction ---
layout = ocr_extract(flat_pdf)

# --- DOCX convert & extract ---
out_docx = convert_docx("report.docx")
text = extract_docx_text(out_docx)

# --- Visualization ---
img = overlay_bounds(flat_pdf, [(50,50,200,200)], page_idx=0)
img.show()
```

---

## 📚 Module Reference

* `docsqz.config` – env vars & defaults
* `docsqz.common.visualize` – `overlay_bounds`, render helpers
* `docsqz.docx.convert` – `convert_docx()` (Pandoc)
* `docsqz.docx.extract` – `extract_docx_text()`, `extract_docx_structure()`, `extract_docx_markdown()`, `extract_docx_metadata()`
* `docsqz.pdf.flatten` – `flatten_acrobat_macos()`, `flatten_aspose()`
* `docsqz.pdf.extract` – `ocr_extract()`, `layout_extract()`, `adobe_extract()`

---

## 🛠 Troubleshooting

* **Acrobat flattening hangs** → check `PRINTER` name matches your system printer list, and `SPOOL_DIR` is writable.
* **Aspose flattening fails** → ensure `ASPOSE_CLIENT_ID/SECRET` are set and valid.
* **OCR results empty** → verify OCR engine installed; try increasing DPI.
* **Missing models** (PaddleOCR/Detectron2) → install model weights via `paddleocr` or `layoutparser`.

---

## 🤝 Contributing

* Keep notebook-to-module mapping verbatim (no heavy refactors).
* Add wrappers only at module bottoms.
* Follow PEP8, type annotate functions.

---

## 📜 License

MIT (or your chosen license).

⚠️ Note: This repo depends on third-party services (Adobe, Aspose) requiring API credentials. Do not commit secrets.
