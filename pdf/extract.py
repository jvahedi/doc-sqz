from __future__ import annotations
from pathlib import Path

# PDF = "your_xfa_flattened.pdf"

# # Paddle v4 mobile (fast baseline)
# eng = make_ocr_engine("paddle_v4_mobile")
# layouts, dfs, agg = run_ocr(PDF, pages=[0,1], dpi=240, engine_bundle=eng)
# agg.head()

# # Paddle v5 server (heavier, higher quality)
# eng = make_ocr_engine("paddle_v5_server", batch_size=4)
# layouts, dfs, agg = run_ocr(PDF, pages=[0,1], dpi=240, engine_bundle=eng)
# len(agg)

# # Tesseract (render at ~300 DPI for best results)
# eng = make_ocr_engine("tesseract", languages="eng")
# layouts, dfs, agg = run_ocr(PDF, pages=[0,1], dpi=300, engine_bundle=eng)
# dfs[0].head()

# # PubLayNet: layout detection only (no text OCR)
# eng = make_ocr_engine("publaynet", score_thresh=0.2, device="cpu")
# layouts, dfs, agg = run_ocr(PDF, pages=[0], dpi=300, engine_bundle=eng)
# layouts[0][:5]  # first few blocks

import pytesseract

def make_tesseract_agent(tesseract_cmd: str | None = None, languages: str = "eng"):
    import pytesseract
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return {"engine": pytesseract, "languages": languages}

def ocr_tesseract_to_layout(img_np: np.ndarray, agent, level: str = "line"):
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
    if level.lower() == "line" and {"line_num","left","top","width","height"}.issubset(df.columns):
        group_cols = [c for c in ("page_num","block_num","par_num","line_num") if c in df.columns]
        for _, sub in df.groupby(group_cols, dropna=False):
            toks = [t for t in sub.get("text","").astype(str) if t.strip()]
            if not toks: continue
            x1 = float(sub["left"].min());                 y1 = float(sub["top"].min())
            x2 = float((sub["left"]+sub["width"]).max());  y2 = float((sub["top"]+sub["height"]).max())
            score = float(_pd.to_numeric(sub.get("conf", np.nan), errors="coerce").replace(0, np.nan).mean())
            items.append(lp.TextBlock(lp.Rectangle(x1,y1,x2,y2), text=" ".join(toks), type="line", score=score))
    else:
        # word level
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
    try:
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
    except Exception:
        # fallback to v4 mobile if the chosen combo fails
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="PP-OCRv4_mobile_rec",
            text_recognition_batch_size=min(batch_size, 4),
            text_det_limit_side_len=det_limit,
            text_det_limit_type="max",
        )

def ocr_paddle_to_layout(img_np: np.ndarray, engine):
    res = engine.ocr(img_np)
    if not res or not res[0]:
        return lp.Layout([]), pd.DataFrame(columns=["x_1","y_1","x_2","y_2","text","score","type"])
    items, rows = [], []
    for quad, (txt, conf) in res[0]:
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
        items.append(lp.TextBlock(lp.Rectangle(x1,y1,x2,y2), text=str(txt), type="ocr", score=float(conf)))
        rows.append({"x_1":x1,"y_1":y1,"x_2":x2,"y_2":y2,"text":str(txt),"score":float(conf),"type":"ocr"})
    return lp.Layout(items), pd.DataFrame(rows)

def make_publaynet_model(score_thresh: float = 0.5, device: str = "cpu"):
    return lp.Detectron2LayoutModel(
        'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
        label_map={0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"},
        device=device,
    )

"""
adobe_extract_utils.py
Adobe PDF Extract → tidy pandas DataFrame (page/type/text/raw_json) with helpers.
Import-only: no CLI, no argparse. Credentials are in-module constants.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

# --- Your Adobe credentials (module-level constants) ---
# NOTE: Hard-coding secrets is risky if this repo is ever shared.
ADOBE_CLIENT_ID = "c2d7bbc5bea04542b50cc6c54bb44b45"
ADOBE_CLIENT_SECRET = "p8e-LZ1gF1r8CNl7MHgqQ1r6CiqY1B0em1uO"

# --- Adobe SDK (v4) ---
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

# __all__ = [
#     "make_client",
#     "extract_zip_bytes",
#     "load_structured_json",
#     "elements_to_dataframe",
#     "extract_pdf_to_dataframe",
# ]

# =========================
# Public API
# =========================

def make_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
) -> PDFServices:
    """
    Create a PDFServices client. Defaults to in-module constants.
    You can still override with explicit args if needed.
    """
    client_id = client_id or ADOBE_CLIENT_ID
    client_secret = client_secret or ADOBE_CLIENT_SECRET
    if not client_id or not client_secret:
        raise ValueError("Missing ADOBE_CLIENT_ID / ADOBE_CLIENT_SECRET.")
    creds = ServicePrincipalCredentials(client_id, client_secret)
    return PDFServices(creds)


def extract_zip_bytes(
    pdf_services: PDFServices,
    input_pdf_path: str,
    want_text: bool = True,
    want_tables: bool = True,
    renditions: Optional[Iterable[ExtractRenditionsElementType]] = None,
    table_structure: TableStructureType = TableStructureType.XLSX,
) -> bytes:
    """
    Upload a local PDF, run Extract API, and return the result ZIP as raw bytes.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(input_pdf_path)

    # 1) Upload
    with open(input_pdf_path, "rb") as f:
        asset = pdf_services.upload(io.BytesIO(f.read()), PDFServicesMediaType.PDF)

    # 2) Build params
    elements_to_extract: List[ExtractElementType] = []
    if want_text:
        elements_to_extract.append(ExtractElementType.TEXT)
    if want_tables:
        elements_to_extract.append(ExtractElementType.TABLES)

    params = ExtractPDFParams(
        elements_to_extract=elements_to_extract,
        elements_to_extract_renditions=list(renditions or []),
        table_structure_type=table_structure,
    )

    # 3) Run job
    job = ExtractPDFJob(input_asset=asset, extract_pdf_params=params)
    loc = pdf_services.submit(job)
    resp = pdf_services.get_job_result(loc, ExtractPDFResult)

    # 4) Get ZIP content (handle multiple return shapes)
    res_asset = resp.get_result().get_resource()
    content = pdf_services.get_content(res_asset)
    return _to_bytes(content)


def load_structured_json(zip_bytes: bytes) -> dict:
    """
    From Extract API ZIP bytes, load and return structuredData.json as dict.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        sd_name = next((n for n in zf.namelist() if n.endswith("structuredData.json")), None)
        if not sd_name:
            raise RuntimeError("structuredData.json not found in Extract output.")
        return json.loads(zf.read(sd_name).decode("utf-8"))


def elements_to_dataframe(structured_json: dict) -> pd.DataFrame:
    """
    Flatten structured JSON 'elements' to a single DataFrame with:
        - page: page number (best-effort across schema variants)
        - type: inferred element type (text/table/figure/…)
        - text: coalesced text (handles list-of-runs)
        - raw_json: full JSON per element (string)
      plus flattened element keys as columns.
    """
    elements = structured_json.get("elements") or []

    def _scalarize(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return json.dumps(v, ensure_ascii=False)

    rows = []
    for el in elements:
        if not isinstance(el, dict):
            el = {"Type": type(el).__name__, "Value": el}
        flat = pd.json_normalize(el, sep=".")
        flat["page"] = _get_page(el)
        flat["type"] = _infer_type(el)
        flat["text"] = _coalesce_text(el)
        flat["raw_json"] = json.dumps(el, ensure_ascii=False)
        flat = flat.apply(lambda s: s.map(_scalarize))  # ensure scalar-friendly values
        rows.append(flat.iloc[0].to_dict())

    df = pd.DataFrame(rows)

    # Move key columns to front if present
    front = [c for c in ("page", "type", "text") if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]

    # Drop any capital "Text" column from source to avoid confusion with our 'text'
    if "Text" in df.columns:
        df = df.drop(columns=["Text"])
    return df


def extract_pdf_to_dataframe(
    input_pdf_path: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    save_zip_to: Optional[str] = None,
    want_text: bool = True,
    want_tables: bool = True,
    renditions: Optional[Iterable[ExtractRenditionsElementType]] = None,
    table_structure: TableStructureType = TableStructureType.XLSX,
) -> Tuple[pd.DataFrame, dict]:
    """
    High-level convenience: run Extract on a local PDF path and return (DataFrame, structured_json).
    Optionally write the raw result ZIP to disk.
    """
    svc = make_client(client_id, client_secret)
    zip_bytes = extract_zip_bytes(
        svc,
        input_pdf_path=input_pdf_path,
        want_text=want_text,
        want_tables=want_tables,
        renditions=renditions,
        table_structure=table_structure,
    )
    if save_zip_to:
        with open(save_zip_to, "wb") as f:
            f.write(zip_bytes)

    structured = load_structured_json(zip_bytes)
    df = elements_to_dataframe(structured)
    return df, structured


# =========================
# Internals
# =========================

def _to_bytes(obj: Union[bytes, bytearray, io.IOBase, object]) -> bytes:
    """Robustly turn Adobe SDK get_content(...) return into bytes."""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    gis = getattr(obj, "get_input_stream", None)
    if callable(gis):
        s = gis()
        return s if isinstance(s, (bytes, bytearray)) else s.read()

    r = getattr(obj, "read", None)
    if callable(r):
        return r()

    data = getattr(obj, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)

    raise TypeError(f"Unsupported content type from get_content: {type(obj)}")


def _coalesce_text(el: dict) -> str:
    t = el.get("Text", "")
    if isinstance(t, list):
        return "".join(x.get("Text", "") for x in t)
    return t or ""


def _get_page(el: dict) -> Optional[int]:
    # Handles {"Page":{"PageNumber": n}}, {"PageNumber": n}, or int
    if "Page" in el:
        pg = el["Page"]
        if isinstance(pg, dict):
            return pg.get("PageNumber") or pg.get("pageNumber")
        if isinstance(pg, int):
            return pg
    return el.get("PageNumber") or el.get("pageNumber")


def _infer_type(el: dict) -> Optional[str]:
    # 1) direct keys (some responses use other casings/labels)
    t = el.get("Type") or el.get("type") or el.get("Label")
    if t:
        return str(t).lower()

    # 2) look at Path like "/Root/Document/Pages/0/Text/..."
    path = el.get("Path")
    if isinstance(path, str):
        parts = [p for p in path.split("/") if p]
        cand = ["Table", "Text", "Figure", "Image", "List", "Header", "Footer", "Footnote", "Annotation", "FormField"]
        for c in cand:
            if c in parts:
                return c.lower()

    # 3) table-ish attributes imply 'table'
    attrs = el.get("attributes") or {}
    if any(k in attrs for k in ("NumRow", "NumCol", "RowIndex", "ColIndex")):
        return "table"

    # 4) has Text => call it 'text'
    if "Text" in el:
        return "text"

    return None

def _show_image_inline(img: Image.Image, title: str | None = None, **kwargs):
    """Display PIL image inline in notebooks using matplotlib (no return)."""
    import matplotlib.pyplot as plt  # lazy import
    figsize = kwargs.pop("figsize", None)  # e.g., (10, 14)
    dpi = kwargs.pop("dpi", None)
    if figsize or dpi:
        plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

PDF_PATH = "Documents/Educational_FY2023_NSGP_S_VA_Highland School Educational Foundation_IJ_flattened.pdf"

df, structured = extract_pdf_to_dataframe(PDF_PATH)
print(df.shape)
df.head(10)

# 1) Using a column like "Bounds" in your elements_df
pg = 1
img = overlay_from_dataframe(
    df,
    pdf_path="Documents/Educational_FY2023_NSGP_S_VA_Highland School Educational Foundation_IJ_flattened.pdf",
    page=pg,
    bounds_col="Bounds",     # auto-detects common alternatives if missing
    dpi=240,
    normalized=False,        # True if coords are 0..1
    adobe_top_left=True,     # Flip Y if your source uses top-left origin (Adobe Extract)
    color=(0, 255, 0),
    width=3,
    show=True,
)
#img.show()
#img.save("page_overlay_pg"+str(pg)+".png")

# Quick peek
try:
    display(df.head(10))
    display(df.tail(5))
except Exception as e:
    print("Load data into `df` first.", e)

# Structure & summary
try:
    info = df.info()
    display(df.describe(include='all'))
except Exception as e:
    print("Load data into `df` first.", e)

# Basic distributions (edit columns as needed)
try:
    # Example for a numeric column:
    # df['some_numeric_column'].hist()
    # plt.show()
    pass
except Exception as e:
    print("Load data into `df` first.", e)

# Prototype: write quick, task-focused code here
# Example:
# result = (
#     df.pipe(lambda d: d.assign(example_col=1))
# )
# result.head()

# Cleaning steps template
# df = df.dropna(subset=['col'])           # example
# df['col'] = df['col'].astype('Int64')    # types
# df = df.drop_duplicates()                # dedupe
# df = df.rename(columns={'old':'new'})    # rename
# TODO: add your canonical cleaning ops here

# Transformations template
# df['feature'] = df['a'] / (df['b'] + 1e-9)
# df['date'] = pd.to_datetime(df['date'])
# from sklearn.decomposition import PCA
# TODO: add domain-specific transforms

# Analysis / modeling template
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# X = df[['x1','x2']]; y = df['y']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression().fit(X_train, y_train)
# print("R^2:", model.score(X_test, y_test))

# Plotting template (add real columns)
# df['some_numeric_column'].hist()
# plt.title('Example')  # optional
# plt.show()

# Extract reusable utilities here.
# Keep functions pure and documented.
# Example:
# def clean_names(df):
#     df = df.copy()
#     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
#     return df

# TODO: move stable code from Build/Clean/Transform here

# Save outputs (edit targets as needed)
# df.to_csv(OUT_DIR / "data_clean.csv", index=False)
# with open(OUT_DIR / "notes.txt", "w") as f:
#     f.write("Summary of results...")

print("Ready to deliver.")


def ocr_extract(path, *args, **kwargs):
    for fname in ("ocr_extract", "extract_ocr", "run_ocr"):
        if fname in globals() and callable(globals()[fname]):
            return globals()[fname](path, *args, **kwargs)
    raise RuntimeError("No ocr function found in copied code.")

def layout_extract(path, *args, **kwargs):
    for fname in ("layout_extract", "extract_layout", "detect_layout"):
        if fname in globals() and callable(globals()[fname]):
            return globals()[fname](path, *args, **kwargs)
    raise RuntimeError("No layout function found in copied code.")

def adobe_extract(path, *args, **kwargs):
    for fname in ("adobe_extract", "extract_adobe", "extract_elements"):
        if fname in globals() and callable(globals()[fname]):
            return globals()[fname](path, *args, **kwargs)
    raise RuntimeError("No adobe extract function found in copied code.")
