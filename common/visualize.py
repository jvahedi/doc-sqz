"""
Visualization helpers for docsqz.
Moved from stray cells in the notebook.
"""

from typing import Sequence
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pandas as pd


def _display_inline(img):
    """Display a PIL image inline in notebooks without matplotlib."""
    try:
        from IPython.display import display  # lazy import; only in notebooks
        display(img)
    except Exception:
        # Fallback: open in system viewer
        img.show()


def overlay_bounds(
    pdf_path: str,
    bounds_list: Sequence,
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
        Each item is a JSON string "[x1,y1,x2,y2]" or iterable (x1,y1,x2,y2).
    dpi : int
        Render resolution for page.
    normalized : bool
        If True, coords are normalized (0..1). If False, coords are in PDF points.
    adobe_top_left : bool
        Flip Y if source coords assume top-left origin (Adobe Extract).
    clamp : bool
        Clamp rectangles to image bounds.
    """
    import io, json

    def _coerce_box(b) -> tuple[float, float, float, float]:
        if isinstance(b, str):
            b = json.loads(b)
        return tuple(map(float, b))

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(img)

    w_pt, h_pt = page.rect.width, page.rect.height
    img_w, img_h = img.size

    sx, sy = (img_w, img_h) if normalized else (img_w / w_pt, img_h / h_pt)

    for raw in bounds_list:
        try:
            x1, y1, x2, y2 = _coerce_box(raw)
        except Exception:
            continue

        # Adjust Y
        if adobe_top_left:
            y1p = h_pt - y1 if not normalized else 1.0 - y1
            y2p = h_pt - y2 if not normalized else 1.0 - y2
        else:
            y1p, y2p = y1, y2

        X1, Y1 = x1 * sx, y1p * sy
        X2, Y2 = x2 * sx, y2p * sy

        left, right = sorted((X1, X2))
        top, bottom = sorted((Y1, Y2))

        # Guard degenerate boxes
        if right - left < 1: right = left + 1
        if bottom - top < 1: bottom = top + 1

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
    """
    if page_col not in df.columns:
        raise KeyError(f"DataFrame has no '{page_col}' column.")

    if bounds_col not in df.columns:
        for alt in ("bounds", "BBox", "bbox", "Rect", "rect"):
            if alt in df.columns:
                bounds_col = alt
                break
        else:
            raise KeyError(f"No '{bounds_col}' (or alternatives) in DataFrame.")

    page_rows = df[df[page_col] == page]
    bounds_list = page_rows[bounds_col].dropna().tolist()
    return overlay_bounds(
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
