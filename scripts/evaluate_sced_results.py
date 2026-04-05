"""
Evaluate SCED extraction results against a gold-standard JSONL file.

Expected JSONL schema for both predictions and gold:
    {"pdf": "Taylor 2011.pdf", "Participant ID": ..., "Baseline Mean": ..., ...}

Metrics are computed per field and overall using micro-averaged precision, recall,
and F1 across normalized value sets. This supports both scalar fields and list fields.

Usage:
    python -m scripts.evaluate_sced_results \
        --predictions extracted_text/sced_results.jsonl \
        --gold data/sced_gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FIELDS = [
    "Participant ID",
    "Baseline Mean",
    "Treatment Phase Slope",
    "Clinical Contradictions",
]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SCED extraction precision/recall.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file.")
    parser.add_argument("--gold", required=True, help="Path to gold-standard JSONL file.")
    parser.add_argument(
        "--output",
        help="Optional path to save a JSON summary. Defaults next to predictions as *_evaluation.json.",
    )
    return parser.parse_args()


def _normalize_atom(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return format(float(value), ".12g")
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na"}:
        return None
    return " ".join(text.lower().split())


def normalize_to_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        normalized = {_normalize_atom(item) for item in value}
        return {item for item in normalized if item is not None}
    atom = _normalize_atom(value)
    return {atom} if atom is not None else set()


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_jsonl_by_pdf(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: Dict[str, Dict[str, Any]] = {}
    duplicate_pdfs: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            pdf_name = str(record.get("pdf", "")).strip()
            if not pdf_name:
                raise ValueError(f"Missing 'pdf' key in {path} line {line_number}")
            if pdf_name in records:
                duplicate_pdfs.add(pdf_name)
            records[pdf_name] = record

    if duplicate_pdfs:
        logging.warning(
            "Found duplicate records in %s. Keeping the last record for: %s",
            path,
            ", ".join(sorted(duplicate_pdfs)),
        )

    return records


def evaluate_records(
    predictions: Dict[str, Dict[str, Any]],
    gold: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    common_pdfs = sorted(set(predictions) & set(gold))
    missing_predictions = sorted(set(gold) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(gold))

    per_field_counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in FIELDS}
    exact_match_count = 0
    per_pdf_results: List[Dict[str, Any]] = []

    for pdf_name in common_pdfs:
        pdf_exact = True
        field_details: Dict[str, Dict[str, Any]] = {}
        predicted_record = predictions[pdf_name]
        gold_record = gold[pdf_name]

        for field in FIELDS:
            predicted_values = normalize_to_set(predicted_record.get(field))
            gold_values = normalize_to_set(gold_record.get(field))

            tp = len(predicted_values & gold_values)
            fp = len(predicted_values - gold_values)
            fn = len(gold_values - predicted_values)

            per_field_counts[field]["tp"] += tp
            per_field_counts[field]["fp"] += fp
            per_field_counts[field]["fn"] += fn

            if fp or fn:
                pdf_exact = False

            field_details[field] = {
                "predicted": sorted(predicted_values),
                "gold": sorted(gold_values),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        if pdf_exact:
            exact_match_count += 1

        per_pdf_results.append(
            {
                "pdf": pdf_name,
                "exact_match": pdf_exact,
                "fields": field_details,
            }
        )

    per_field_metrics = {
        field: compute_metrics(**counts) for field, counts in per_field_counts.items()
    }

    total_tp = sum(counts["tp"] for counts in per_field_counts.values())
    total_fp = sum(counts["fp"] for counts in per_field_counts.values())
    total_fn = sum(counts["fn"] for counts in per_field_counts.values())

    return {
        "predictions_file_count": len(predictions),
        "gold_file_count": len(gold),
        "matched_file_count": len(common_pdfs),
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "overall": {
            **compute_metrics(total_tp, total_fp, total_fn),
            "exact_match_rate": safe_divide(exact_match_count, len(common_pdfs)),
        },
        "per_field": per_field_metrics,
        "per_pdf": per_pdf_results,
    }


def evaluate_jsonl_files(
    predictions_path: Path,
    gold_path: Path,
) -> Dict[str, Any]:
    predictions = load_jsonl_by_pdf(predictions_path)
    gold = load_jsonl_by_pdf(gold_path)
    return evaluate_records(predictions, gold)


def default_output_path(predictions_path: Path) -> Path:
    return predictions_path.with_name(f"{predictions_path.stem}_evaluation.json")


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    gold_path = Path(args.gold)
    output_path = Path(args.output) if args.output else default_output_path(predictions_path)

    summary = evaluate_jsonl_files(predictions_path, gold_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    logging.info("Matched files: %s", summary["matched_file_count"])
    logging.info("Precision: %.3f", overall["precision"])
    logging.info("Recall: %.3f", overall["recall"])
    logging.info("F1: %.3f", overall["f1"])
    logging.info("Exact match rate: %.3f", overall["exact_match_rate"])
    logging.info("Saved evaluation summary to %s", output_path)


if __name__ == "__main__":
    main()
