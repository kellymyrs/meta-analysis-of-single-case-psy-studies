"""Convert PDF pages to images, keeping only pages that contain figures/charts.

Usage (from repo root):
    python scripts/pdf_to_images.py

Behavior:
- Scans `input/` for PDF files.
- Uses PyMuPDF to detect pages containing raster images or vector drawings.
- Renders qualifying pages directly with PyMuPDF at 300 DPI (no Poppler needed).
- Saves to `processed_images/` as `<paper_name>_page_<n>.png` (1-based page index).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "processed_images"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def page_has_figure(page: fitz.Page) -> bool:
    """Heuristic: return True if page likely contains a figure/chart, not just text.

    Signals used (conservative to avoid saving all pages):
    - Raster images: count images that are not tiny and not extreme-aspect headers/footers.
    - Vector drawings: require a higher path count to indicate charts/plots.
    - Text cues: presence of words like "Figure", "Fig.", "Graph" on the page.
    - Scanned full-page image suppression: if the page is a single full-page scan and no other
      signals are present, skip it.
    """
    page_rect = page.rect
    page_area = max(1.0, page_rect.width * page_rect.height)

    # --- Text cues ---
    text = (page.get_text() or "").lower()
    has_text_cue = any(k in text for k in ("figure", "fig.", "fig ", "graph", "chart"))

    # --- Raster images ---
    images = page.get_images(full=True)
    qualifying_imgs = []
    for img in images:
        # img tuple: (xref, smask, width, height, bpc, colorspace, alt-colorspace, name, filter, referer)
        width = img[2]
        height = img[3]
        filt = str(img[8]) if len(img) > 8 else ""
        area = float(width) * float(height)
        aspect_ratio = (max(width, height) / max(1, min(width, height))) if min(width, height) > 0 else 100.0
        # Keep medium+ sized images that are not long thin lines
        if area > 40000 and aspect_ratio < 6:
            qualifying_imgs.append({
                "area": area,
                "aspect": aspect_ratio,
                "filter": filt,
            })

    # --- Vector drawings ---
    drawings = page.get_drawings()
    drawings_count = len(drawings)

    # Quick positives:
    # - Many vector paths (likely charts)
    if drawings_count >= 60:
        return True
    # - Multiple medium+ images can indicate composite figure panels
    if len(qualifying_imgs) >= 2:
        return True
    # - Text cue plus any structural signal
    if has_text_cue and (len(qualifying_imgs) >= 1 or drawings_count >= 15):
        return True

    # Suppress scanned full-page text pages:
    # If exactly one raster image that covers most of the page and uses scan-like filter
    # and there are few drawings and no text cue, treat as non-figure.
    if len(images) == 1:
        w, h = images[0][2], images[0][3]
        area = float(w) * float(h)
        coverage = area / page_area
        img_filter = str(images[0][8]) if len(images[0]) > 8 else ""
        is_scan_like = ("ccitt" in img_filter.lower()) or ("fax" in img_filter.lower()) or ("dct" in img_filter.lower())
        if coverage >= 0.6 and drawings_count < 15 and not has_text_cue:
            return False

    # Fallback: a single qualifying image may be a figure, but to avoid false positives
    # require at least a modest vector presence or a text cue.
    if len(qualifying_imgs) == 1 and (drawings_count >= 15 or has_text_cue):
        return True

    return False


def save_pages_with_figures(pdf_path: Path) -> None:
    logging.info("Processing %s", pdf_path.name)

    with fitz.open(pdf_path) as doc:
        stem = pdf_path.stem.replace(" ", "_")
        matrix = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI

        figure_pages: List[int] = []
        for i, page in enumerate(doc):
            if page_has_figure(page):
                figure_pages.append(i)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                out_name = f"{stem}_page_{i + 1}.png"
                out_path = OUTPUT_DIR / out_name
                pix.save(out_path)
                logging.info("Saved %s", out_path.relative_to(ROOT))

    if not figure_pages:
        logging.info("No figures detected in %s", pdf_path.name)


def main() -> None:
    if OUTPUT_DIR.exists():
        logging.info("Cleaning output directory: %s", OUTPUT_DIR)
        for f in OUTPUT_DIR.iterdir():
            if f.is_file():
                f.unlink()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDF files found in %s", INPUT_DIR)
        return

    for pdf_path in pdf_files:
        try:
            save_pages_with_figures(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_path.name, exc)


if __name__ == "__main__":
    main()
