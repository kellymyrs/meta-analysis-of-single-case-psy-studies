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

LLM backend is configured via env (see scripts/sced_extraction.py):
    Proxy mode (preferred): LITELLM_KEY (+ optional LITELLM_MODEL, LITELLM_BASE_URL)
    Local fallback mode: MODEL_PATH=/absolute/path/to/model.gguf
"""

from __future__ import annotations

import argparse
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
from scripts.evaluate_sced_results import evaluate_jsonl_files
from scripts.sced_extraction import (
    get_runtime_backend,
    run_sced_extraction,
    run_sced_extraction_from_pdf,
    run_sced_extraction_full_text,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCED extraction across PDFs.")
    parser.add_argument(
        "--mode",
        choices=("blocks", "full_text", "full_pdf"),
        default="blocks",
        help="Use truncated blocks, full text via chunked chat completions, or attach the full PDF.",
    )
    parser.add_argument(
        "--pdf",
        help="Optional exact PDF filename in input/ to process instead of the whole folder.",
    )
    parser.add_argument(
        "--gold",
        help="Optional path to a gold-standard JSONL file. If provided, precision/recall evaluation runs after extraction.",
    )
    parser.add_argument(
        "--evaluation-dir",
        help="Optional directory for evaluation JSON outputs. Defaults next to the predictions JSONL file.",
    )
    return parser.parse_args()


def resolve_pdf_files(selected_pdf: str | None) -> List[Path]:
    if selected_pdf:
        pdf_path = PDF_DIR / selected_pdf
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return [pdf_path]
    return sorted(p for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def output_paths(mode: str) -> tuple[Path, str]:
    suffix = "" if mode == "blocks" else f"_{mode}"
    return TEXT_DIR / f"sced_results{suffix}.jsonl", suffix


def evaluation_output_path(results_path: Path, evaluation_dir: str | None) -> Path:
    if not evaluation_dir:
        return results_path.with_name(f"{results_path.stem}_evaluation.json")
    directory = Path(evaluation_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    return directory / f"{results_path.stem}_evaluation.json"


def main() -> None:
    args = parse_args()
    try:
        backend = get_runtime_backend()
    except RuntimeError as exc:
        logging.error(str(exc))
        return
    logging.info("Using LLM backend: %s", backend)
    logging.info("Extraction mode: %s", args.mode)

    if args.mode == "full_pdf" and backend != "proxy":
        logging.error("full_pdf mode requires proxy mode with LITELLM_KEY configured.")
        return

    try:
        pdf_files = resolve_pdf_files(args.pdf)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return

    if not pdf_files:
        logging.warning("No PDFs found in %s", PDF_DIR)
        return

    results_path, out_suffix = output_paths(args.mode)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")
    lines_written = 0

    for pdf_path in pdf_files:
        logging.info("Processing %s", pdf_path.name)
        try:
            if args.mode == "blocks":
                blocks = ensure_blocks(pdf_path)
                result = run_sced_extraction(blocks)
            elif args.mode == "full_text":
                blocks = ensure_blocks(pdf_path)
                result = run_sced_extraction_full_text(blocks)
            else:
                result = run_sced_extraction_from_pdf(pdf_path)
            if result is None:
                logging.warning("No result for %s", pdf_path.name)
                continue
            out_path = TEXT_DIR / f"{pdf_path.stem}_sced{out_suffix}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"pdf": pdf_path.name, **result}, ensure_ascii=False) + "\n")
            lines_written += 1
            logging.info("Saved %s and appended to %s", out_path.relative_to(ROOT), results_path.relative_to(ROOT))
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_path.name, exc)

    logging.info("Done. Wrote %d result lines to %s", lines_written, results_path.relative_to(ROOT))

    if args.gold:
        gold_path = Path(args.gold)
        evaluation = evaluate_jsonl_files(results_path, gold_path)
        evaluation_path = evaluation_output_path(results_path, args.evaluation_dir)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

        overall = evaluation["overall"]
        logging.info("Evaluation precision: %.3f", overall["precision"])
        logging.info("Evaluation recall: %.3f", overall["recall"])
        logging.info("Evaluation F1: %.3f", overall["f1"])
        logging.info("Saved evaluation summary to %s", evaluation_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
