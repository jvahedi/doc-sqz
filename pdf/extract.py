"""
PDF extraction utilities for docsqz.

Includes:
- OCR extraction (Tesseract, PaddleOCR)
- Layout detection (PubLayNet via Detectron2, if installed)
- Adobe PDF Extract API (requires adobe-pdfservices-sdk)
"""

from __future__ import annotations
import io
import os
import json
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import layoutparser as lp

# ==========================================================
# --- OCR + Layout dependencies
# ==========================================================

import sys, subprocess

def install_ocr_requirements(kind: str = "paddle"):
    """Convenience installer for OCR backends."""
    pkgs = []
    kind = kind.lower()
    if kind == "tesseract":
        pkgs = ["pytesseract", "pillow", "pandas", "layoutparser", "pymupdf"]
    elif kind == "paddle":
        pkgs = ["paddleocr", "paddlepaddle", "pillow", "pandas", "layoutparser", "pymupdf"]
    elif kind == "publaynet":
        pkgs = ["layoutparser[detectron2]", "pillow", "pandas", "pymupdf"]
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])


# ==========================================================
# --- Core rendering
# ==========================================================

def render_pdf_page(pdf_path: str, page_idx: int = 0, dpi: int = 240):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix  = page.get_pixmap(dpi=dpi, alpha=False)
    img  = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img, np.array(img)


# ==========================================================
# --- OCR Engines
# ==========================================================

def make_tesseract_agent(tesseract_cmd: str | None = None, languages: str = "eng"):
    import pytesseract
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return {"engine": pytesseract, "languages": languages}


def ocr_tesseract_to_layout(img_np: np.ndarray, agent, level: str = "line"):
    """
    Convert Tesseract OCR output to layoutparser format.

    Args:
        img_np: Image as numpy array
        agent: Tesseract agent dict
        level: Grouping level - "line", "paragraph", or "block"

    Returns:
        (layout, dataframe) tuple
    """
    import pandas as _pd
    from pytesseract import Output as _Output
    pyt  = agent["engine"]
    lang = agent.get("languages", "eng")
    df = pyt.image_to_data(img_np, lang=lang, output_type=_Output.DATAFRAME)
    df = df.replace({np.nan: ""})
    if "conf" in df.columns:
        df["conf"] = _pd.to_numeric(df["conf"], errors="coerce").fillna(-1)
        df = df[df["conf"] >= 0]

    items = []
    level_lower = level.lower()

    # Determine grouping columns based on level
    if level_lower == "block":
        group_cols = [c for c in ("page_num", "block_num") if c in df.columns]
        block_type = "block"
    elif level_lower == "paragraph":
        group_cols = [c for c in ("page_num", "block_num", "par_num") if c in df.columns]
        block_type = "paragraph"
    elif level_lower == "line":
        group_cols = [c for c in ("page_num", "block_num", "par_num", "line_num") if c in df.columns]
        block_type = "line"
    else:
        # Default to word level
        group_cols = []
        block_type = "word"

    if group_cols and {"left","top","width","height"}.issubset(df.columns):
        # Group by specified level
        for _, sub in df.groupby(group_cols, dropna=False):
            toks = [t for t in sub.get("text","").astype(str) if t.strip()]
            if not toks: continue
            x1 = float(sub["left"].min()); y1 = float(sub["top"].min())
            x2 = float((sub["left"]+sub["width"]).max()); y2 = float((sub["top"]+sub["height"]).max())
            score = float(_pd.to_numeric(sub.get("conf", np.nan), errors="coerce").replace(0, np.nan).mean())
            items.append(lp.TextBlock(lp.Rectangle(x1,y1,x2,y2), text=" ".join(toks), type=block_type, score=score))
    else:
        # Fall back to word level
        need = {"left","top","width","height","text"}
        for _, r in df.iterrows():
            if not need.issubset(df.columns): break
            txt = str(r["text"]).strip()
            if not txt: continue
            x1 = float(r["left"]); y1 = float(r["top"])
            x2 = float(r["left"] + r["width"]); y2 = float(r["top"] + r["height"])
            score = float(r.get("conf", np.nan))
            items.append(lp.TextBlock(lp.Rectangle(x1,y1,x2,y2), text=txt, type="word", score=score))

    layout = lp.Layout(items)
    df_out = layout.to_dataframe()
    if "score" not in df_out.columns:
        df_out["score"] = np.nan
    return layout, df_out


def make_paddle_engine(variant: str = "mobile", version: str = "v4",
                       batch_size: int = 4, use_extras: bool = False, det_limit: int = 960):
    from paddleocr import PaddleOCR
    prefix = f"PP-OCR{version}_"
    det = prefix + f"{variant}_det"
    rec = prefix + f"{variant}_rec"
    return PaddleOCR(
        use_doc_orientation_classify=use_extras,
        use_doc_unwarping=use_extras,
        use_textline_orientation=use_extras,
        text_detection_model_name=det,
        text_recognition_model_name=rec,
        text_recognition_batch_size=batch_size,
        text_det_limit_side_len=det_limit,
        text_det_limit_type="max",
    )


def ocr_paddle_to_layout(img_np: np.ndarray, engine):
    res = engine.ocr(img_np)
    if not res or len(res) == 0:
        return lp.Layout([]), pd.DataFrame(columns=["x_1","y_1","x_2","y_2","text","score","type"])

    items, rows = [], []

    # PaddleOCR 3.2.0 returns OCRResult object (dict-like)
    page_result = res[0]

    # Access as dict keys
    polys = page_result.get('rec_polys', [])
    texts = page_result.get('rec_texts', [])
    scores = page_result.get('rec_scores', [])

    if len(polys) == 0 or len(texts) == 0:
        return lp.Layout([]), pd.DataFrame(columns=["x_1","y_1","x_2","y_2","text","score","type"])

    for i in range(len(texts)):
        txt = texts[i]
        score = scores[i] if i < len(scores) else 0.0
        poly = polys[i] if i < len(polys) else None

        if not txt or poly is None or len(poly) == 0:
            continue

        # poly is array of points [[x1,y1], [x2,y2], ...]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

        items.append(lp.TextBlock(lp.Rectangle(x1,y1,x2,y2), text=str(txt), type="ocr", score=float(score)))
        rows.append({"x_1":x1,"y_1":y1,"x_2":x2,"y_2":y2,"text":str(txt),"score":float(score),"type":"ocr"})

    return lp.Layout(items), pd.DataFrame(rows)


def make_publaynet_model(score_thresh: float = 0.5, device: str = "cpu"):
    import layoutparser as lp
    return lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
        device=device,
    )


# ==========================================================
# --- Unified interface
# ==========================================================

def make_ocr_engine(kind: str = "paddle_v4_mobile", **kwargs):
    kind = kind.lower()
    if kind == "tesseract":
        eng = make_tesseract_agent(
            tesseract_cmd=kwargs.get("tesseract_cmd"),
            languages=kwargs.get("languages", "eng"),
        )
    elif kind in ("paddle_v4_mobile", "paddle_v5_mobile", "paddle_v5_server"):
        variant = "mobile" if "mobile" in kind else "server"
        version = "v4" if "v4" in kind else "v5"
        eng = make_paddle_engine(
            variant=variant,
            version=version,
            batch_size=kwargs.get("batch_size", 4),
            use_extras=kwargs.get("use_extras", False),
            det_limit=kwargs.get("det_limit", 960),
        )
    elif kind == "publaynet":
        eng = make_publaynet_model(
            score_thresh=kwargs.get("score_thresh", 0.5),
            device=kwargs.get("device", "cpu"),
        )
    else:
        raise ValueError(f"Unknown kind '{kind}'")
    return {"kind": kind, "engine": eng}


def ocr_page(pdf_path: str, page_idx: int, dpi: int, engine_bundle):
    kind = engine_bundle["kind"]
    eng  = engine_bundle["engine"]
    img, img_np = render_pdf_page(pdf_path, page_idx=page_idx, dpi=dpi)
    if kind == "tesseract":
        layout, df = ocr_tesseract_to_layout(img_np, eng, level="line")
    elif kind.startswith("paddle_"):
        layout, df = ocr_paddle_to_layout(img_np, eng)
    elif kind == "publaynet":
        layout = eng.detect(img_np)
        df = layout.to_dataframe()
    else:
        raise ValueError(f"Unsupported kind '{kind}'")
    if not df.empty:
        if "page" not in df.columns:
            df.insert(0, "page", page_idx)
        else:
            df["page"] = page_idx
    return layout, df


def run_ocr(pdf_path: str, pages: list[int] | None = None,
            dpi: int = 240, engine_bundle=None):
    if engine_bundle is None:
        engine_bundle = make_ocr_engine("paddle_v4_mobile")
    doc = fitz.open(pdf_path)
    idxs = list(range(len(doc))) if pages is None else list(pages)
    layouts, per_dfs, rows = [], [], []
    for i in idxs:
        layout, df = ocr_page(pdf_path, page_idx=i, dpi=dpi, engine_bundle=engine_bundle)
        layouts.append(layout)
        per_dfs.append(df)
        if not df.empty:
            rows.append(df)
    agg_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return layouts, per_dfs, agg_df


# ==========================================================
# --- Adobe Extract API
# ==========================================================

from DOCSQZ import config
ADOBE_CLIENT_ID = config.ADOBE_CLIENT_ID
ADOBE_CLIENT_SECRET = config.ADOBE_CLIENT_SECRET

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

def adobe_make_client(client_id: str | None = None,
                      client_secret: str | None = None) -> PDFServices:
    cid = client_id or ADOBE_CLIENT_ID
    csec = client_secret or ADOBE_CLIENT_SECRET
    if not cid or not csec:
        raise ValueError("Missing ADOBE_CLIENT_ID / ADOBE_CLIENT_SECRET.")
    creds = ServicePrincipalCredentials(cid, csec)
    return PDFServices(creds)

def adobe_extract_zip_bytes(pdf_services: PDFServices,
                            input_pdf_path: str,
                            want_text: bool = True,
                            want_tables: bool = True,
                            renditions: list | None = None,
                            table_structure: TableStructureType = TableStructureType.XLSX) -> bytes:
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(input_pdf_path)
    with open(input_pdf_path, "rb") as f:
        asset = pdf_services.upload(io.BytesIO(f.read()), PDFServicesMediaType.PDF)
    elements = []
    if want_text: elements.append(ExtractElementType.TEXT)
    if want_tables: elements.append(ExtractElementType.TABLES)
    params = ExtractPDFParams(
        elements_to_extract=elements,
        elements_to_extract_renditions=list(renditions or []),
        table_structure_type=table_structure,
    )
    job = ExtractPDFJob(input_asset=asset, extract_pdf_params=params)
    loc = pdf_services.submit(job)
    resp = pdf_services.get_job_result(loc, ExtractPDFResult)
    res_asset = resp.get_result().get_resource()
    content = pdf_services.get_content(res_asset)
    return _adobe_to_bytes(content)

def adobe_load_structured_json(zip_bytes: bytes) -> dict:
    import zipfile
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        sd_name = next((n for n in zf.namelist() if n.endswith("structuredData.json")), None)
        if not sd_name:
            raise RuntimeError("structuredData.json not found in Adobe Extract output.")
        return json.loads(zf.read(sd_name).decode("utf-8"))

def adobe_elements_to_dataframe(structured_json: dict) -> pd.DataFrame:
    elements = structured_json.get("elements") or []
    def _scalarize(v):
        if isinstance(v, (str, int, float, bool)) or v is None: return v
        return json.dumps(v, ensure_ascii=False)
    def _coalesce_text(el):
        t = el.get("Text", "")
        if isinstance(t, list): return "".join(x.get("Text", "") for x in t)
        return t or ""
    def _get_page(el):
        if "Page" in el:
            pg = el["Page"]
            if isinstance(pg, dict):
                return pg.get("PageNumber") or pg.get("pageNumber")
            if isinstance(pg, int): return pg
        return el.get("PageNumber") or el.get("pageNumber")
    def _infer_type(el):
        t = el.get("Type") or el.get("type") or el.get("Label")
        if t: return str(t).lower()
        path = el.get("Path")
        if isinstance(path, str):
            parts = [p for p in path.split("/") if p]
            for c in ["Table","Text","Figure","Image","List","Header","Footer","Footnote","Annotation","FormField"]:
                if c in parts: return c.lower()
        attrs = el.get("attributes") or {}
        if any(k in attrs for k in ("NumRow","NumCol","RowIndex","ColIndex")): return "table"
        if "Text" in el: return "text"
        return None
    rows = []
    for el in elements:
        if not isinstance(el, dict):
            el = {"Type": type(el).__name__, "Value": el}
        flat = pd.json_normalize(el, sep=".")
        flat["page"] = _get_page(el)
        flat["type"] = _infer_type(el)
        flat["text"] = _coalesce_text(el)
        flat["raw_json"] = json.dumps(el, ensure_ascii=False)
        flat = flat.apply(lambda s: s.map(_scalarize))
        rows.append(flat.iloc[0].to_dict())
    df = pd.DataFrame(rows)
    if "Text" in df.columns: df = df.drop(columns=["Text"])
    return df

def _adobe_to_bytes(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    gis = getattr(obj, "get_input_stream", None)
    if callable(gis):
        s = gis(); return s if isinstance(s, (bytes, bytearray)) else s.read()
    r = getattr(obj, "read", None)
    if callable(r): return r()
    data = getattr(obj, "data", None)
    if isinstance(data, (bytes, bytearray)): return bytes(data)
    raise TypeError(f"Unsupported content type from Adobe SDK get_content: {type(obj)}")


# ==========================================================
# --- Native text extraction (PyMuPDF)
# ==========================================================

def extract_pdf_text(pdf_path: str, pages: list[int] | None = None) -> pd.DataFrame:
    """
    Extract embedded text from PDF using PyMuPDF (no OCR).
    Returns DataFrame with text blocks per page.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page indices to extract. If None, extracts all pages.

    Returns:
        DataFrame with columns: page, text, x_1, y_1, x_2, y_2, type
    """
    doc = fitz.open(pdf_path)
    idxs = list(range(len(doc))) if pages is None else list(pages)

    rows = []
    for page_idx in idxs:
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        blocks = text_dict.get("blocks", [])

        for block in blocks:
            if block.get("type") == 0:  # Text block
                # Extract all text from lines/spans
                lines = block.get("lines", [])
                text_parts = []
                for line in lines:
                    for span in line.get("spans", []):
                        text_parts.append(span.get("text", ""))

                text = " ".join(text_parts).strip()
                if not text:
                    continue

                bbox = block.get("bbox", [0, 0, 0, 0])
                rows.append({
                    "page": page_idx,
                    "text": text,
                    "x_1": float(bbox[0]),
                    "y_1": float(bbox[1]),
                    "x_2": float(bbox[2]),
                    "y_2": float(bbox[3]),
                    "type": "text_block"
                })

    return pd.DataFrame(rows)


# ==========================================================
# --- Grouping and normalization utilities
# ==========================================================

def group_to_level(df: pd.DataFrame, method: str, target_level: str = "paragraph") -> pd.DataFrame:
    """
    Group extracted text to target granularity level.

    Args:
        df: DataFrame from any extraction method
        method: "ocr", "native_text", "adobe"
        target_level: "line", "paragraph", "block" (ignored for native_text and adobe)

    Returns:
        DataFrame grouped to target level
    """
    if df.empty:
        return df

    if method == "ocr":
        # OCR DataFrames have block_num, par_num, line_num columns from Tesseract
        # or just bounding boxes from PaddleOCR
        # For now, return as-is (already at line/detection level)
        # Future: could group by spatial proximity or block_num/par_num if available
        return df

    elif method == "native_text":
        # PyMuPDF blocks are already paragraph-like, just return as-is
        return df

    elif method == "adobe":
        # Adobe already returns semantic elements
        # Optionally filter to text-only elements
        if "type" in df.columns:
            # Keep text, headings, paragraphs; exclude tables, figures
            text_types = ["text", "p", "h1", "h2", "h3", "h4", "h5", "h6", "paragraph"]
            df_filtered = df[df["type"].str.lower().isin(text_types)] if not df.empty else df
            return df_filtered
        return df

    else:
        # Unknown method, return as-is
        return df


def standardize_extraction_result(result, method: str, level: str = "paragraph") -> dict:
    """
    Normalize outputs from different extraction methods to a standard format.

    Args:
        result: Raw output from extraction method
        method: "ocr", "native_text", "adobe"
        level: Target granularity level (default: "paragraph")

    Returns:
        dict with keys:
            - df: Standardized DataFrame with columns (page, text, x_1, y_1, x_2, y_2, type)
            - method: Extraction method used
            - raw: Original output (for advanced users)
    """
    if method == "ocr":
        # OCR methods return: (layouts, per_dfs, agg_df)
        layouts, per_dfs, agg_df = result
        normalized_df = group_to_level(agg_df, method, target_level=level)
        return {
            "df": normalized_df,
            "method": method,
            "raw": {"layouts": layouts, "per_page_dfs": per_dfs}
        }

    elif method == "native_text":
        # Native text returns: DataFrame
        df = result
        normalized_df = group_to_level(df, method, target_level=level)
        return {
            "df": normalized_df,
            "method": method,
            "raw": None
        }

    elif method == "adobe":
        # Adobe returns: (df, structured_json)
        df, structured = result
        normalized_df = group_to_level(df, method, target_level=level)
        return {
            "df": normalized_df,
            "method": method,
            "raw": {"structured_json": structured}
        }

    else:
        raise ValueError(f"Unknown extraction method: {method}")


# ==========================================================
# --- Public wrappers
# ==========================================================

def extract_pdf_ocr(pdf_path: str, kind: str = "paddle_v4_mobile", **kwargs):
    engine = make_ocr_engine(kind, **kwargs)
    return run_ocr(pdf_path, engine_bundle=engine)

def extract_pdf_layout(pdf_path: str, kind: str = "publaynet", **kwargs):
    engine = make_ocr_engine(kind, **kwargs)
    return run_ocr(pdf_path, engine_bundle=engine)

def extract_pdf_adobe(pdf_path: str,
                      client_id: str | None = None,
                      client_secret: str | None = None,
                      save_zip_to: str | None = None):
    svc = adobe_make_client(client_id, client_secret)
    zip_bytes = adobe_extract_zip_bytes(svc, input_pdf_path=str(pdf_path))
    if save_zip_to:
        with open(save_zip_to, "wb") as f: f.write(zip_bytes)
    structured = adobe_load_structured_json(zip_bytes)
    df = adobe_elements_to_dataframe(structured)
    return df, structured

def is_page_blank(page, text_char_count: int = 0, blank_threshold: int = 50) -> bool:
    """
    Detect if a PDF page is blank/mostly empty.

    Args:
        page: PyMuPDF page object
        text_char_count: Number of characters extracted by native text extraction
        blank_threshold: If text < this AND page is visually blank, consider it blank

    Returns:
        True if page appears blank, False otherwise
    """
    # If there's meaningful text, definitely not blank
    if text_char_count >= blank_threshold:
        return False

    # Check for images - if page has images, it's not blank (might be scanned)
    image_list = page.get_images(full=False)
    if len(image_list) > 0:
        return False  # Has images, might need OCR

    # Render page at low resolution to check pixel variance
    pix = page.get_pixmap(dpi=72, alpha=False)  # Low res for speed
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Check if image is mostly uniform (blank pages have low variance)
    # Calculate standard deviation of pixel values
    std_dev = np.std(img_array)

    # If very low variance (< 5) and little text, it's blank
    if std_dev < 5 and text_char_count < blank_threshold:
        return True

    return False


def extract_pdf_auto(pdf_path: str,
                     prefer_native: bool = True,
                     ocr_threshold: int = 50,
                     ocr_fallback_kind: str = "paddle_v4_mobile",
                     ocr_engine = None,
                     standardize: bool = True,
                     skip_blank_pages: bool = True,
                     **kwargs) -> dict:
    """
    Smart PDF extraction: tries native text first per-page, falls back to OCR if needed.

    Args:
        pdf_path: Path to PDF file
        prefer_native: Try native text extraction first (default: True)
        ocr_threshold: Minimum character count per page to consider native extraction successful
        ocr_fallback_kind: OCR engine to use if fallback needed (default: "paddle_v4_mobile")
        ocr_engine: Pre-created OCR engine to reuse (optional, creates new if None)
        standardize: Return standardized format (default: True)
        skip_blank_pages: Skip OCR on pages that appear blank (default: True)
        **kwargs: Additional arguments passed to OCR engine (only if ocr_engine is None)

    Returns:
        dict with keys:
            - df: Combined DataFrame with standardized columns
            - method: Primary extraction method used
            - raw: Metadata including page_methods dict
    """
    doc = fitz.open(pdf_path)
    all_rows = []
    page_methods = {}

    # Standard columns to keep
    standard_cols = ["page", "text", "x_1", "y_1", "x_2", "y_2", "type"]

    # Create OCR engine once if not provided (reuse for all pages that need OCR)
    engine_created_here = False

    if prefer_native:
        # Process each page individually
        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # Try native text extraction for this page
            native_df = extract_pdf_text(pdf_path, pages=[page_idx])

            # Check if native extraction was successful for this page
            page_chars = native_df["text"].str.len().sum() if not native_df.empty else 0

            if page_chars >= ocr_threshold:
                # Native extraction successful for this page
                # Standardize to common columns
                page_df = native_df[standard_cols].copy()
                all_rows.append(page_df)
                page_methods[page_idx] = "native_text"
            else:
                # Check if page is blank before running expensive OCR
                if skip_blank_pages and is_page_blank(page, text_char_count=page_chars):
                    # Page is blank, skip OCR
                    page_methods[page_idx] = "skipped_blank"
                    continue

                # Fall back to OCR for this page (likely scanned)
                # Create engine only if not provided and not created yet
                if ocr_engine is None:
                    ocr_engine = make_ocr_engine(ocr_fallback_kind, **kwargs)
                    engine_created_here = True
                layout, ocr_df = ocr_page(pdf_path, page_idx=page_idx, dpi=240, engine_bundle=ocr_engine)

                # Standardize to common columns
                ocr_cols = [c for c in standard_cols if c in ocr_df.columns]
                page_df = ocr_df[ocr_cols].copy()

                # Add missing columns with defaults
                for col in standard_cols:
                    if col not in page_df.columns:
                        if col in ("x_1", "y_1", "x_2", "y_2"):
                            page_df[col] = 0.0
                        else:
                            page_df[col] = ""

                page_df = page_df[standard_cols]
                all_rows.append(page_df)
                page_methods[page_idx] = "ocr"

        # Combine all pages
        combined_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=standard_cols)

        # Determine primary method (most common)
        method_counts = pd.Series(list(page_methods.values())).value_counts()
        primary_method = method_counts.index[0] if not method_counts.empty else "native_text"

        # Check if mixed methods were used
        is_mixed = len(set(page_methods.values())) > 1

        if standardize:
            return {
                "df": combined_df,
                "method": primary_method,
                "raw": {
                    "page_methods": page_methods,
                    "mixed": is_mixed
                }
            }
        else:
            return combined_df
    else:
        # Use OCR directly for all pages
        ocr_result = extract_pdf_ocr(pdf_path, kind=ocr_fallback_kind, **kwargs)
        if standardize:
            return standardize_extraction_result(ocr_result, method="ocr")
        else:
            layouts, per_dfs, agg_df = ocr_result
            return agg_df
