"""Extract text blocks with layout metadata from PDFs in /input.

Usage (from repo root):
    python scripts/pdf_text_blocks.py

Behavior:
- Scans `input/` for `.pdf` files.
- For each page, collects text blocks (as returned by PyMuPDF) with bounding box coords.
- Writes one JSON per PDF to `extracted_text/` named `<paper_name>_blocks.json`.

JSON structure (list of entries):
{
  "page": 1,               # 1-based page number
  "bbox": [x0, y0, x1, y1], # float coordinates in points
  "text": "..."            # concatenated text of the block
}

This layout-preserving output can be fed to LLMs for variable localization.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "extracted_text"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def extract_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):  # page_index is 1-based for readability
            # PyMuPDF page.get_text("blocks") returns tuples:
            # (x0, y0, x1, y1, text, block_no, block_type, block_flags)
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text, *_ = block
                text = (text or "").strip()
                if not text:
                    continue
                entries.append(
                    {
                        "page": page_index,
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "text": text,
                    }
                )
    return entries


def save_json(data: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        logging.info("Emptying output directory: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def process_pdf(pdf_path: Path) -> None:
    logging.info("Extracting blocks from %s", pdf_path.name)
    blocks = extract_blocks(pdf_path)
    if not blocks:
        logging.warning("No text blocks found in %s", pdf_path.name)
        return
    out_name = f"{pdf_path.stem.replace(' ', '_')}_blocks.json"
    out_path = OUTPUT_DIR / out_name
    save_json(blocks, out_path)
    logging.info("Saved %s", out_path.relative_to(ROOT))


def main() -> None:
    empty_output_dir(OUTPUT_DIR)
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDF files found in %s", INPUT_DIR)
        return

    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_path.name, exc)


if __name__ == "__main__":
    main()
