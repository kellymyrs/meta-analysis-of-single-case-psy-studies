"""
Verify extracted SCED records against the full study PDFs with a vision model.

This is an independent groundedness check layered on top of extraction: a separate
verifier model (default Qwen-2.5-VL, a different vendor from the gpt-5.1 extractor)
re-reads the page images of each PDF and judges, per field, whether the extracted
value is supported by the document. It never sees the gold standard, so it gives a
source-grounded signal that complements evaluation against gold.

Usage (from repo root):
    python -m scripts.verify_sced_results \
        --predictions extracted_text/sced_results_full_pdf.jsonl

Behavior:
- Reads a predictions JSONL (one {"pdf": ..., <fields>} object per line).
- For each record, locates the PDF in input/, renders every page to an image, and
  asks the verifier model for a per-field verdict
  (supported / contradicted / not_in_text / inferred) plus a verbatim quote.
- Optionally string-matches each "supported" quote against the PDF text; a quote that
  is not found is flagged (a likely fabricated grounding).
- Writes extracted_text/sced_verification.jsonl with per-field verdicts and a summary.

Backend: requires proxy mode (LITELLM_KEY). The verifier model is set via
SCED_VERIFIER_MODEL (default Qwen-2.5-VL); rendering DPI via --dpi.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so `scripts` package resolves when invoked as a file
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sced_extraction import (
    DEFAULT_VERIFIER_MODEL,
    DEFAULT_VERIFY_DPI,
    get_runtime_backend,
    run_sced_verification_from_images,
    run_sced_verification_from_pdf,
)
from scripts.sced_fields import FIELDS, normalize_field_keys

PDF_DIR = ROOT / "input"
TEXT_DIR = ROOT / "extracted_text"
DEFAULT_OUTPUT = TEXT_DIR / "sced_verification.jsonl"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _display_path(path: Path) -> str:
    """Path relative to repo root when possible, else the absolute path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _normalize_for_match(text: str) -> str:
    """Lowercase and collapse whitespace so quotes match across line/page breaks."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _pdf_text(pdf_path: Path) -> str:
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() or "" for page in doc)


def load_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(normalize_field_keys(json.loads(line)))
    return records


def annotate_quote_matches(
    verdicts: Dict[str, Dict[str, Any]],
    pdf_path: Path,
) -> None:
    """Add a 'quote_matched' flag to each verdict by checking the quote against PDF text."""
    haystack = _normalize_for_match(_pdf_text(pdf_path))
    for entry in verdicts.values():
        quote = _normalize_for_match(entry.get("quote", ""))
        if entry.get("verdict") in ("supported", "contradicted") and quote:
            entry["quote_matched"] = quote in haystack
        else:
            entry["quote_matched"] = None


def summarize(verdicts: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    summary = {verdict: 0 for verdict in ("supported", "contradicted", "not_in_text", "inferred")}
    summary["quote_unmatched"] = 0
    for entry in verdicts.values():
        summary[entry["verdict"]] = summary.get(entry["verdict"], 0) + 1
        if entry.get("quote_matched") is False:
            summary["quote_unmatched"] += 1
    return summary


def resolve_pdf(record: Dict[str, Any], pdf_dir: Path) -> Optional[Path]:
    name = record.get("pdf")
    if not name:
        return None
    candidate = pdf_dir / name
    if candidate.exists():
        return candidate
    # Fall back to stem match (e.g. predictions stored without extension).
    stem = Path(name).stem
    for path in pdf_dir.glob("*.pdf"):
        if path.stem == stem:
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify SCED extractions against PDFs with a vision model.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a predictions JSONL file produced by extraction.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(PDF_DIR),
        help="Directory holding the source PDFs (default: input/).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL path (default: extracted_text/sced_verification.jsonl).",
    )
    parser.add_argument(
        "--mode",
        choices=("pdf", "images"),
        default="pdf",
        help="pdf: attach the PDF via Responses API (default, needs a doc-capable model "
        "like gpt-4.1). images: render pages and send to a vision model like Qwen-2.5-VL.",
    )
    parser.add_argument(
        "--pdf",
        help="Optional single PDF filename to verify instead of every record.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_VERIFY_DPI,
        help=f"Rendering DPI for page images (default: {DEFAULT_VERIFY_DPI}).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on pages sent per PDF (default: all pages).",
    )
    parser.add_argument(
        "--no-quote-check",
        action="store_true",
        help="Skip string-matching supported quotes against the PDF text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        backend = get_runtime_backend()
    except RuntimeError as exc:
        logging.error(str(exc))
        return
    if backend != "proxy":
        logging.error("Verification requires proxy mode with LITELLM_KEY configured.")
        return

    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = ROOT / predictions_path
    if not predictions_path.exists():
        logging.error("Predictions file not found: %s", predictions_path)
        return

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT / pdf_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_predictions(predictions_path)
    if args.pdf:
        records = [r for r in records if r.get("pdf") == args.pdf]
        if not records:
            logging.error("No prediction record found for --pdf %s", args.pdf)
            return

    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text("", encoding="utf-8")
    lines_written = 0

    for record in records:
        name = record.get("pdf", "<unknown>")
        pdf_path = resolve_pdf(record, pdf_dir)
        if pdf_path is None:
            logging.warning("PDF not found for record %s; skipping.", name)
            continue

        logging.info("Verifying %s", name)
        try:
            if args.mode == "pdf":
                verdicts = run_sced_verification_from_pdf(pdf_path, record)
            else:
                verdicts = run_sced_verification_from_images(
                    pdf_path,
                    record,
                    dpi=args.dpi,
                    max_pages=args.max_pages,
                )
        except Exception as exc:  # noqa: BLE001
            logging.error("Verification failed on %s: %s", name, exc)
            continue
        if verdicts is None:
            logging.warning("No verification result for %s", name)
            continue

        if not args.no_quote_check:
            annotate_quote_matches(verdicts, pdf_path)

        summary = summarize(verdicts)
        out_record = {
            "pdf": name,
            "verifier_model": os.getenv("SCED_VERIFIER_MODEL", DEFAULT_VERIFIER_MODEL).strip(),
            "verdicts": verdicts,
            "summary": summary,
        }
        with temp_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        lines_written += 1
        logging.info(
            "  supported=%d contradicted=%d not_in_text=%d inferred=%d quote_unmatched=%d",
            summary["supported"],
            summary["contradicted"],
            summary["not_in_text"],
            summary["inferred"],
            summary["quote_unmatched"],
        )

    if lines_written:
        temp_path.replace(output_path)
        logging.info("Done. Wrote %d verification lines to %s", lines_written, _display_path(output_path))
    else:
        temp_path.unlink(missing_ok=True)
        logging.warning("No verification lines written. %s left unchanged.", _display_path(output_path))


if __name__ == "__main__":
    main()
