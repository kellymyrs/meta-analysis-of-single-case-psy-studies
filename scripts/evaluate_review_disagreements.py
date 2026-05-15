"""
Evaluate only the disagreement rows from review/disagreements_for_review.csv.

This deliberately ignores train/test splits. The preferred use is after manual
review, when `reviewed_value_json` has been filled. It can also compare LLM
outputs against the legacy values for exploratory diagnostics, but that should
not be reported as final accuracy if the legacy coding may contain errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_sced_results import compute_metrics, normalize_to_set, safe_divide
from scripts.sced_review import from_json_cell


DEFAULT_REVIEW_CSV = ROOT / "review" / "disagreements_for_review.csv"
DEFAULT_OUTPUT = ROOT / "review" / "disagreement_evaluation.json"

SOURCE_COLUMNS = {
    "legacy": "legacy_value_json",
    "llm_a": "llm_a_value_json",
    "llm_b": "llm_b_value_json",
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate only disagreement rows from the SCED review sheet."
    )
    parser.add_argument(
        "--review-csv",
        default=str(DEFAULT_REVIEW_CSV),
        help="Path to disagreements_for_review.csv.",
    )
    parser.add_argument(
        "--reference",
        choices=("reviewed", "legacy"),
        default="reviewed",
        help=(
            "Reference values to score against. Use 'reviewed' for final metrics "
            "after filling reviewed_value_json; use 'legacy' only for exploratory diagnostics."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("all", "legacy", "llm_a", "llm_b"),
        default="all",
        help="Which source to evaluate against the reference.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write JSON evaluation summary.",
    )
    return parser.parse_args()


def load_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Review CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def json_cell(row: dict[str, str], column: str) -> Any:
    return from_json_cell(row.get(column, ""))


def selected_sources(source: str, reference: str) -> list[str]:
    if source != "all":
        return [source]
    sources = list(SOURCE_COLUMNS)
    if reference == "legacy":
        sources.remove("legacy")
    return sources


def score_source(
    rows: list[dict[str, str]],
    *,
    source: str,
    reference: str,
) -> dict[str, Any]:
    source_column = SOURCE_COLUMNS[source]
    reference_column = "reviewed_value_json" if reference == "reviewed" else "legacy_value_json"

    per_field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    per_row: list[dict[str, Any]] = []
    exact_count = 0
    included = 0
    skipped_empty_reference = 0

    for row in rows:
        field = row["field"]
        reference_raw = json_cell(row, reference_column)
        if reference == "reviewed" and not row.get(reference_column, "").strip():
            skipped_empty_reference += 1
            continue

        reference_values = normalize_to_set(reference_raw, field)
        source_values = normalize_to_set(json_cell(row, source_column), field)

        tp = len(source_values & reference_values)
        fp = len(source_values - reference_values)
        fn = len(reference_values - source_values)

        per_field_counts[field]["tp"] += tp
        per_field_counts[field]["fp"] += fp
        per_field_counts[field]["fn"] += fn
        included += 1

        exact = source_values == reference_values
        if exact:
            exact_count += 1

        per_row.append(
            {
                "pdf": row["pdf"],
                "field": field,
                "status": row.get("status", ""),
                "source": source,
                "reference": reference,
                "source_values": sorted(source_values),
                "reference_values": sorted(reference_values),
                "exact": exact,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    total_tp = sum(counts["tp"] for counts in per_field_counts.values())
    total_fp = sum(counts["fp"] for counts in per_field_counts.values())
    total_fn = sum(counts["fn"] for counts in per_field_counts.values())

    return {
        "included_rows": included,
        "skipped_empty_reference_rows": skipped_empty_reference,
        "overall": {
            **compute_metrics(total_tp, total_fp, total_fn),
            "exact_match_rate": safe_divide(exact_count, included),
        },
        "per_field": {
            field: compute_metrics(counts["tp"], counts["fp"], counts["fn"])
            for field, counts in sorted(per_field_counts.items())
        },
        "per_row": per_row,
    }


def evaluate_disagreements(
    review_csv: Path,
    *,
    reference: str,
    source: str,
) -> dict[str, Any]:
    rows = load_review_rows(review_csv)
    sources = selected_sources(source, reference)
    source_results = {
        source_name: score_source(rows, source=source_name, reference=reference)
        for source_name in sources
    }
    reviewed_rows = sum(1 for row in rows if row.get("reviewed_value_json", "").strip())
    return {
        "review_csv": str(review_csv),
        "reference": reference,
        "total_disagreement_rows": len(rows),
        "reviewed_rows": reviewed_rows,
        "unreviewed_rows": len(rows) - reviewed_rows,
        "sources": source_results,
    }


def main() -> None:
    args = parse_args()
    review_csv = Path(args.review_csv)
    output = Path(args.output)
    if not review_csv.is_absolute():
        review_csv = ROOT / review_csv
    if not output.is_absolute():
        output = ROOT / output

    summary = evaluate_disagreements(
        review_csv,
        reference=args.reference,
        source=args.source,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Disagreement rows: %d", summary["total_disagreement_rows"])
    logging.info("Reviewed rows: %d", summary["reviewed_rows"])
    if args.reference == "reviewed" and summary["reviewed_rows"] == 0:
        logging.warning(
            "No reviewed rows found. Fill reviewed_value_json before using this for final metrics."
        )
    for source_name, result in summary["sources"].items():
        overall = result["overall"]
        logging.info(
            "%s vs %s: included=%d precision=%.3f recall=%.3f f1=%.3f exact=%.3f",
            source_name,
            args.reference,
            result["included_rows"],
            overall["precision"],
            overall["recall"],
            overall["f1"],
            overall["exact_match_rate"],
        )
    logging.info("Saved disagreement evaluation to %s", output)


if __name__ == "__main__":
    main()
