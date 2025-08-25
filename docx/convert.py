from __future__ import annotations
from pathlib import Path
import os

#DOCX CONVERTER
def convert_docx(
    input_path,
    to="html",                   # "html", "gfm" (GitHub MD), "md", "latex", etc.
    out_path=None,
    extract_media_dir="media",   # images saved here; set None to skip
    math="mathml",               # "mathml" (for HTML), "latex" (for MathJax), or None
    add_toc=False
):
    input_path = Path(input_path)
    assert input_path.suffix.lower() == ".docx", "Input must be a .docx file"

    extra = []
    if math == "mathml":
        extra += ["--mathml"]
    elif math == "latex":
        extra += ["--mathjax"]   # renders via MathJax when viewing HTML
    if add_toc:
        extra += ["--toc"]
    if extract_media_dir:
        extra += [f"--extract-media={extract_media_dir}"]

    result = pypandoc.convert_file(str(input_path), to, extra_args=extra)

    if out_path:
        Path(out_path).write_text(result, encoding="utf-8")

    return result

# Convert DOCX on disk → HTML and preview inline
doc_html = convert_docx(
    doc_path,          # <-- change to your file
    to="md",
    out_path=str(OUT_DIR)+"/output.md",
    extract_media_dir="media",
    math="mathml",
    add_toc=True
)

## Future Compatibility
from __future__ import annotations

## Standard Library
import io, os, sys, subprocess
import json, zipfile
from pathlib import Path

## Dataclasses
from dataclasses import dataclass
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional,
    Sequence, Tuple, Union
)
## IPython / Notebook Utils
from IPython.display import display

## Data / Analysis
import numpy as np
import pandas as pd

## Visualization
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

## Document Handling
import fitz  # PyMuPDF
import pypandoc

## Adobe PDF Services
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

## OCR / Document Parsing
from paddleocr import PaddleOCR
import pytesseract
from pytesseract import Output as _Output
import layoutparser as lp


def convert_docx(input_path, *args, **kwargs) -> Path:
    """
    Thin wrapper for DOCX CONVERTER (Pandoc route) code. Returns path to produced artifact.
    """
    # Expect the notebook code to define a function or perform work when called.
    # Here we locate an exposed function if present; otherwise, call main() if defined.
    in_path = Path(input_path)
    out = None
    # Heuristic: if a function named convert or run exists, call it. Otherwise, assume a function 'docx_convert'.
    for fname in ("convert", "run", "docx_convert", "convert_pandoc"):
        if fname in globals() and callable(globals()[fname]):
            out = globals()[fname](str(in_path), *args, **kwargs)
            break
    if out is None:
        # If notebook code expects environment variables and runs as script, we can't emulate here; return a guessed output.
        stem = in_path.with_suffix("").name + "_converted"
        out = in_path.parent / f"{stem}.docx"
    return Path(out)
