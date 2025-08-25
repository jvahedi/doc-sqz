from __future__ import annotations
import io
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import numpy as np

"""
docproc_utils.py
Modular pipeline:
  1) Aspose: XFA -> AcroForm -> Flatten (in place) -> Download
  2) OCR/Layout: render PDF pages, run OCR, get Layout + DataFrames
  3) Overlays: draw boxes on page images and (optionally) show inline (no matplotlib)

Deps (typical):
  pip install requests pymupdf pillow numpy pandas layoutparser paddleocr
  # Optional: tesseract stack if you choose to use it instead of Paddle
"""

from __future__ import annotations

# -----------------------
# Common imports
# -----------------------
import io
import os
import os.path as op
import json
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# ========== 1) Aspose flattening ==========
# -----------------------

ASPOSE_BASE = "https://api.aspose.cloud"
# If you want hard-coded creds, set these; env vars still override.
ASPOSE_CLIENT_ID     = os.getenv("ASPOSE_CLIENT_ID",     "42a849c0-13d4-4ab3-ad82-b7d9099cbc22")
ASPOSE_CLIENT_SECRET = os.getenv("ASPOSE_CLIENT_SECRET", "12cefab4412f32e62bb528a1b7ced607")
ASPOSE_STORAGE_NAME  = os.getenv("ASPOSE_STORAGE_NAME",  "Data")

def aspose_get_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 120,
) -> str:
    """Fetch OAuth token from Aspose Cloud."""
    import requests
    cid = client_id or ASPOSE_CLIENT_ID
    csec = client_secret or ASPOSE_CLIENT_SECRET
    r = requests.post(
        f"{base}/connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": csec,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def aspose_upload_file(
    local_path: str,
    remote_path: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 300,
) -> None:
    """Upload a local file to Aspose storage at remote_path."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    with open(local_path, "rb") as f:
        r = requests.put(
            f"{base}/v3.0/pdf/storage/file/{remote_path}",
            headers={"Authorization": f"Bearer {tok}"},
            params={"storageName": storage},
            data=f,
            timeout=timeout,
        )
    r.raise_for_status()

def aspose_convert_xfa_to_acroform(
    input_remote_path: str,
    out_remote_path: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 600,
) -> None:
    """Convert XFA PDF to AcroForm; writes to out_remote_path on Aspose storage."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    name_in  = op.basename(input_remote_path)
    folder_in = op.dirname(input_remote_path)
    r = requests.put(
        f"{base}/v3.0/pdf/{name_in}/convert/xfatoacroform",
        headers={"Authorization": f"Bearer {tok}"},
        params={"folder": folder_in, "outPath": out_remote_path, "storageName": storage},
        timeout=timeout,
    )
    r.raise_for_status()

def aspose_flatten_in_place(
    remote_path: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 600,
) -> None:
    """Flatten form fields in place on the given remote PDF path."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    name_conv   = op.basename(remote_path)
    folder_conv = op.dirname(remote_path)
    r = requests.put(
        f"{base}/v3.0/pdf/{name_conv}/fields/flatten",
        headers={"Authorization": f"Bearer {tok}"},
        params={"folder": folder_conv, "storageName": storage},
        timeout=timeout,
    )
    r.raise_for_status()

def aspose_download_file(
    remote_path: str,
    local_out: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 600,
) -> str:
    """Download a file from Aspose storage to local_out. Returns local_out."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    r = requests.get(
        f"{base}/v3.0/pdf/storage/file/{remote_path}",
        headers={"Authorization": f"Bearer {tok}"},
        params={"storageName": storage},
        stream=True, timeout=timeout,
    )
    r.raise_for_status()
    os.makedirs(op.dirname(local_out) or ".", exist_ok=True)
    with open(local_out, "wb") as out:
        for chunk in r.iter_content(1 << 20):
            out.write(chunk)
    return local_out

def flatten_xfa_pdf_with_aspose(
    local_in: str,
    in_remote: str = "incoming/input.pdf",
    out_remote: str = "converted/xfa_to_acroform.pdf",
    local_out: str = "your_xfa_flattened.pdf",
    storage_name: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> str:
    """
    End-to-end: upload -> XFA->AcroForm -> flatten in place -> download.
    Returns local_out path.
    """
    tok = aspose_get_token(client_id=client_id, client_secret=client_secret)
    aspose_upload_file(local_in, in_remote, storage_name=storage_name, token=tok)
    aspose_convert_xfa_to_acroform(in_remote, out_remote, storage_name=storage_name, token=tok)
    aspose_flatten_in_place(out_remote, storage_name=storage_name, token=tok)
    return aspose_download_file(out_remote, local_out, storage_name=storage_name, token=tok)
yout_boxes(image: Image.Image, layout: lp.Layout, box_width: int = 2, color=(0, 128, 255)) -> Image.Image:
    """Return a PIL.Image with LayoutParser boxes/labels drawn."""
    img2 = image.copy()
    draw = ImageDraw.Draw(img2)
    font = ImageFont.load_default()
    for i, ele in enumerate(layout):
        x1, y1, x2, y2 = ele.coordinates
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        label = f"{i}: ocr"
        lb, tb, rb, bb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([lb, tb, rb, bb], fill=color)
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
    return img2

"""
docproc_utils.py
Modular pipeline:
  1) Aspose: XFA -> AcroForm -> Flatten (in place) -> Download
  2) OCR/Layout: render PDF pages, run OCR, get Layout + DataFrames
  3) Overlays: draw boxes on page images and (optionally) show inline (no matplotlib)

Deps (typical):
  pip install requests pymupdf pillow numpy pandas layoutparser paddleocr
  # Optional: tesseract stack if you choose to use it instead of Paddle
"""

from __future__ import annotations

# -----------------------
# Common imports
# -----------------------
import io
import os
import os.path as op
import json
from typing import Iterable, Optional, Sequence, Tuple, List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# ========== 1) Aspose flattening ==========
# -----------------------

ASPOSE_BASE = "https://api.aspose.cloud"
# If you want hard-coded creds, set these; env vars still override.
ASPOSE_CLIENT_ID     = os.getenv("ASPOSE_CLIENT_ID",     "42a849c0-13d4-4ab3-ad82-b7d9099cbc22")
ASPOSE_CLIENT_SECRET = os.getenv("ASPOSE_CLIENT_SECRET", "12cefab4412f32e62bb528a1b7ced607")
ASPOSE_STORAGE_NAME  = os.getenv("ASPOSE_STORAGE_NAME",  "Data")

def aspose_get_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 120,
) -> str:
    """Fetch OAuth token from Aspose Cloud."""
    import requests
    cid = client_id or ASPOSE_CLIENT_ID
    csec = client_secret or ASPOSE_CLIENT_SECRET
    r = requests.post(
        f"{base}/connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": csec,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def aspose_upload_file(
    local_path: str,
    remote_path: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 300,
) -> None:
    """Upload a local file to Aspose storage at remote_path."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    with open(local_path, "rb") as f:
        r = requests.put(
            f"{base}/v3.0/pdf/storage/file/{remote_path}",
            headers={"Authorization": f"Bearer {tok}"},
            params={"storageName": storage},
            data=f,
            timeout=timeout,
        )
    r.raise_for_status()

def aspose_convert_xfa_to_acroform(
    input_remote_path: str,
    out_remote_path: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 600,
) -> None:
    """Convert XFA PDF to AcroForm; writes to out_remote_path on Aspose storage."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    name_in  = op.basename(input_remote_path)
    folder_in = op.dirname(input_remote_path)
    r = requests.put(
        f"{base}/v3.0/pdf/{name_in}/convert/xfatoacroform",
        headers={"Authorization": f"Bearer {tok}"},
        params={"folder": folder_in, "outPath": out_remote_path, "storageName": storage},
        timeout=timeout,
    )
    r.raise_for_status()

def aspose_flatten_in_place(
    remote_path: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 600,
) -> None:
    """Flatten form fields in place on the given remote PDF path."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    name_conv   = op.basename(remote_path)
    folder_conv = op.dirname(remote_path)
    r = requests.put(
        f"{base}/v3.0/pdf/{name_conv}/fields/flatten",
        headers={"Authorization": f"Bearer {tok}"},
        params={"folder": folder_conv, "storageName": storage},
        timeout=timeout,
    )
    r.raise_for_status()

def aspose_download_file(
    remote_path: str,
    local_out: str,
    storage_name: Optional[str] = None,
    token: Optional[str] = None,
    base: str = ASPOSE_BASE,
    timeout: int = 600,
) -> str:
    """Download a file from Aspose storage to local_out. Returns local_out."""
    import requests
    storage = storage_name or ASPOSE_STORAGE_NAME
    tok = token or aspose_get_token()
    r = requests.get(
        f"{base}/v3.0/pdf/storage/file/{remote_path}",
        headers={"Authorization": f"Bearer {tok}"},
        params={"storageName": storage},
        stream=True, timeout=timeout,
    )
    r.raise_for_status()
    os.makedirs(op.dirname(local_out) or ".", exist_ok=True)
    with open(local_out, "wb") as out:
        for chunk in r.iter_content(1 << 20):
            out.write(chunk)
    return local_out

def flatten_xfa_pdf_with_aspose(
    local_in: str,
    in_remote: str = "incoming/input.pdf",
    out_remote: str = "converted/xfa_to_acroform.pdf",
    local_out: str = "your_xfa_flattened.pdf",
    storage_name: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> str:
    """
    End-to-end: upload -> XFA->AcroForm -> flatten in place -> download.
    Returns local_out path.
    """
    tok = aspose_get_token(client_id=client_id, client_secret=client_secret)
    aspose_upload_file(local_in, in_remote, storage_name=storage_name, token=tok)
    aspose_convert_xfa_to_acroform(in_remote, out_remote, storage_name=storage_name, token=tok)
    aspose_flatten_in_place(out_remote, storage_name=storage_name, token=tok)
    return aspose_download_file(out_remote, local_out, storage_name=storage_name, token=tok)
yout_boxes(image: Image.Image, layout: lp.Layout, box_width: int = 2, color=(0, 128, 255)) -> Image.Image:
    """Return a PIL.Image with LayoutParser boxes/labels drawn."""
    img2 = image.copy()
    draw = ImageDraw.Draw(img2)
    font = ImageFont.load_default()
    for i, ele in enumerate(layout):
        x1, y1, x2, y2 = ele.coordinates
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        label = f"{i}: ocr"
        lb, tb, rb, bb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([lb, tb, rb, bb], fill=color)
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
    return img2

import io
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
import layoutparser as lp

# === Optional installer cell ===
import sys, subprocess

def install_ocr_requirements(kind: str = "paddle"):
    """
    Install minimal dependencies for chosen OCR backend.
    kind: 'tesseract' | 'paddle' | 'publaynet'
    """
    pkgs = []
    kind = kind.lower()
    if kind == "tesseract":
        # pytesseract wrapper; assumes system tesseract is installed
        pkgs = ["pytesseract", "pillow", "pandas", "layoutparser", "pymupdf"]
        print("⚠️ Requires system Tesseract installed separately (brew/apt/etc).")
    elif kind == "paddle":
        pkgs = ["paddleocr", "paddlepaddle", "pillow", "pandas", "layoutparser", "pymupdf"]
    elif kind == "publaynet":
        pkgs = ["layoutparser[detectron2]", "pillow", "pandas", "pymupdf"]
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    print(f"Installing for {kind}: {pkgs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

# Example usage:
# install_ocr_requirements("tesseract")
# install_ocr_requirements("paddle")
# install_ocr_requirements("publaynet")


def render_pdf_page(pdf_path: str, page_idx: int = 0, dpi: int = 240):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    pix  = page.get_pixmap(dpi=dpi, alpha=False)
    img  = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img, np.array(img)


def make_ocr_engine(kind: str = "paddle_v4_mobile", **kwargs):
    """
    kind: 'tesseract' | 'paddle_v4_mobile' | 'paddle_v5_mobile' | 'paddle_v5_server' | 'publaynet'
    Returns a dict: {'kind': kind, 'engine': obj}
    """
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
    """
    Run OCR (or layout detect) on one page depending on engine kind.
    Returns (layout, df) for OCR kinds; for 'publaynet' returns (layout, df_with_types).
    """
    kind = engine_bundle["kind"]
    eng  = engine_bundle["engine"]
    img, img_np = render_pdf_page(pdf_path, page_idx=page_idx, dpi=dpi)

    if kind == "tesseract":
        # Tesseract likes ~300 DPI
        layout, df = ocr_tesseract_to_layout(img_np, eng, level="line")
    elif kind.startswith("paddle_"):
        layout, df = ocr_paddle_to_layout(img_np, eng)
    elif kind == "publaynet":
        layout = eng.detect(img_np)
        df = layout.to_dataframe()
    else:
        raise ValueError(f"Unsupported kind '{kind}'")

    # Attach page for convenience
    if not df.empty:
        if "page" not in df.columns:
            df.insert(0, "page", page_idx)
        else:
            df["page"] = page_idx
    return layout, df

def run_ocr(pdf_path: str, pages: list[int] | None = None, dpi: int = 240, engine_bundle=None):
    """
    Multi-page runner. Returns (layouts_per_page, dfs_per_page, agg_df).
    """
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
        print(f"[{engine_bundle['kind']}] page {i} → {len(layout)} blocks")

    agg_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return layouts, per_dfs, agg_df

from typing import Sequence, Mapping, Union, Any, Dict
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

def _display_inline(img):
    """Display a PIL image inline in notebooks without matplotlib."""
    try:
        from IPython.display import display  # lazy import; only in notebooks
        display(img)
    except Exception:
        # Fallback: opens a viewer window (may be fine in local dev, not in headless)
        img.show()


def overlay_bounds(
    pdf_path: str,
    bounds_list,
    page_idx: int = 0,
    dpi: int = 240,
    color: tuple = (255, 0, 0),
    width: int = 2,
    normalized: bool = False,
    adobe_top_left: bool = True,
    clamp: bool = True,
    *,
    show: bool = False,
) -> Image.Image:
    """
    Draw rectangles over a rendered PDF page.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF.
    bounds_list : sequence
        Each item is either a JSON string "[x1,y1,x2,y2]" or an iterable (x1,y1,x2,y2).
        Coordinates can be absolute (PDF points) or normalized (0..1) depending on `normalized`.
    page_idx : int
        Zero-based page index to render and annotate.
    dpi : int
        Render resolution for the page image.
    color : (R,G,B)
        Outline color for rectangles.
    width : int
        Outline width in pixels.
    normalized : bool
        If True, interpret coords as 0..1 normalized relative to page width/height.
        If False, interpret coords in PDF points (72 dpi).
    adobe_top_left : bool
        If True (default), flip Y assuming source coords have origin at top-left (Adobe Extract / most OCR).
        If False, assume origin at bottom-left (PDF coordinate system) and do not flip.
    clamp : bool
        If True, clamp rectangles to image bounds to avoid errors.

    Returns
    -------
    PIL.Image.Image
        Annotated RGB image of the page.
    """
    import io, json

    def _coerce_box(b) -> tuple[float, float, float, float]:
        if isinstance(b, str):
            b = json.loads(b)
        x1, y1, x2, y2 = map(float, b)
        return x1, y1, x2, y2

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Page size in points and rendered image size in pixels
    w_pt, h_pt = page.rect.width, page.rect.height
    img_w, img_h = img.size

    # Scale factors: normalized uses img size directly; absolute uses points→pixels
    if normalized:
        sx, sy = img_w, img_h
    else:
        sx, sy = img_w / float(w_pt), img_h / float(h_pt)

    for raw in bounds_list:
        try:
            x1, y1, x2, y2 = _coerce_box(raw)
        except Exception:
            continue

        # Y handling
        if adobe_top_left:
            # Convert top-left-origin Y to bottom-left-origin Y in page space
            y1p = h_pt - y1 if not normalized else 1.0 - y1
            y2p = h_pt - y2 if not normalized else 1.0 - y2
        else:
            y1p, y2p = y1, y2

        # Scale to pixels
        X1, Y1 = x1 * sx, y1p * sy
        X2, Y2 = x2 * sx, y2p * sy

        # Normalize for PIL (left<=right, top<=bottom)
        left, right = sorted((X1, X2))
        top, bottom = sorted((Y1, Y2))

        # Guard tiny/degenerate boxes (ensure visibility)
        if right - left < 1:
            right = left + 1
        if bottom - top < 1:
            bottom = top + 1

        # Clamp to image bounds if requested
        if clamp:
            left   = max(0, min(img_w - 1, left))
            right  = max(0, min(img_w - 1, right))
            top    = max(0, min(img_h - 1, top))
            bottom = max(0, min(img_h - 1, bottom))

        draw.rectangle([left, top, right, bottom], outline=color, width=width)

    if show:
        _display_inline(img)
    return img

def overlay_from_dataframe(
    df: pd.DataFrame,
    pdf_path: str,
    page: int,
    bounds_col: str = "Bounds",
    page_col: str = "page",
    dpi: int = 240,
    color: tuple = (255, 0, 0),
    width: int = 2,
    normalized: bool = False,
    adobe_top_left: bool = True,
    clamp: bool = True,
    *,
    show: bool = False,
) -> Image.Image:
    """
    Convenience wrapper: filter a DataFrame to one page and draw its boxes.

    Parameters
    ----------
    df : pandas.DataFrame
        Your elements_df or similar, with a bounds column.
    pdf_path : str
        Path to the PDF.
    page : int
        Page number to render (must match df[page_col] values).
    bounds_col : str
        Column containing boxes; values can be JSON strings or 4-length iterables.
    page_col : str
        Column that stores page numbers in df.
    Other args
        Passed through to overlay_bounds.

    Returns
    -------
    PIL.Image.Image
        Annotated image for that page.
    """
    if page_col not in df.columns:
        raise KeyError(f"DataFrame has no '{page_col}' column.")
    if bounds_col not in df.columns:
        # Try common alternatives
        for alt in ("bounds", "BBox", "bbox", "Rect", "rect"):
            if alt in df.columns:
                bounds_col = alt
                break
        else:
            raise KeyError(f"DataFrame has no '{bounds_col}' (or common alternatives).")

    page_rows = df[df[page_col] == page]
    bounds_list = page_rows[bounds_col].dropna().tolist()
    img = overlay_bounds(
        pdf_path=pdf_path,
        bounds_list=bounds_list,
        page_idx=page,
        dpi=dpi,
        color=color,
        width=width,
        normalized=normalized,
        adobe_top_left=adobe_top_left,
        clamp=clamp,
        show=show,
    )
    return img
