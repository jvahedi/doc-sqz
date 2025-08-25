from __future__ import annotations
from pathlib import Path




def extract_docx_text(path) -> str:
    if "extract_text" in globals() and callable(globals()["extract_text"]):
        return globals()["extract_text"](path)
    if "docx_to_text" in globals() and callable(globals()["docx_to_text"]):
        return globals()["docx_to_text"](path)
    raise RuntimeError("No extract_text/docx_to_text function found (verbatim copy missing?)")

def extract_docx_structure(path):
    fn = globals().get("extract_structure") or globals().get("docx_structure")
    if callable(fn): return fn(path)
    raise RuntimeError("No extract_structure/docx_structure found.")

def extract_docx_markdown(path):
    fn = globals().get("extract_markdown") or globals().get("docx_to_markdown")
    if callable(fn): return fn(path)
    raise RuntimeError("No extract_markdown/docx_to_markdown found.")

def extract_docx_metadata(path):
    fn = globals().get("extract_metadata") or globals().get("docx_metadata")
    if callable(fn): return fn(path)
    return {}
