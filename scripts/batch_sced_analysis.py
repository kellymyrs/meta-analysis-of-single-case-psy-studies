"""
Batch SCED extraction across all PDFs in /input.

Usage (from repo root):
    python scripts/batch_sced_analysis.py

Behavior:
- For every PDF in input/, ensure a text-block JSON exists (reuses extracted_text/<paper>_blocks.json,
  or creates it via pdf_text_blocks.extract_blocks).
- Runs the LLM extractor (run_sced_extraction) for each paper.
- Saves per-paper results to extracted_text/<paper>_sced.json and a combined JSONL summary
  at extracted_text/sced_results.jsonl.

LLM model is configured via env (see scripts/sced_extraction.py):
    MODEL_PATH=/absolute/path/to/model.gguf
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Any

# Ensure project root is on sys.path so `scripts` package resolves when invoked as a file
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pdf_text_blocks import extract_blocks
from scripts.sced_extraction import run_sced_extraction


PDF_DIR = ROOT / "input"
TEXT_DIR = ROOT / "extracted_text"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def ensure_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    """Load block JSON if present; otherwise extract and write it."""
    TEXT_DIR.mkdir(exist_ok=True)
    blocks_path = TEXT_DIR / f"{pdf_path.stem}_blocks.json"
    if blocks_path.exists():
        return json.loads(blocks_path.read_text())
    blocks = extract_blocks(pdf_path)
    blocks_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2))
    logging.info("Created %s", blocks_path.relative_to(ROOT))
    return blocks


def main() -> None:
    model_path = os.getenv("MODEL_PATH", "").strip()
    if not model_path:
        logging.error("MODEL_PATH is not set. Example: export MODEL_PATH=/path/to/model.gguf")
        return
    if model_path == "/absolute/path/to/model.gguf":
        logging.error(
            "MODEL_PATH is still the placeholder value. Set it to the real path of your GGUF file."
        )
        return
    if not Path(model_path).expanduser().exists():
        logging.error("MODEL_PATH does not exist: %s", Path(model_path).expanduser())
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDFs found in %s", PDF_DIR)
        return

    results_path = TEXT_DIR / "sced_results.jsonl"
    lines_written = 0

    for pdf_path in pdf_files:
        logging.info("Processing %s", pdf_path.name)
        try:
            blocks = ensure_blocks(pdf_path)
            result = run_sced_extraction(blocks)
            if result is None:
                logging.warning("No result for %s", pdf_path.name)
                continue
            out_path = TEXT_DIR / f"{pdf_path.stem}_sced.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pdf": pdf_path.name, **result}, ensure_ascii=False) + "\n")
            lines_written += 1
            logging.info("Saved %s and appended to %s", out_path.relative_to(ROOT), results_path.relative_to(ROOT))
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_path.name, exc)

    logging.info("Done. Wrote %d result lines to %s", lines_written, results_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
