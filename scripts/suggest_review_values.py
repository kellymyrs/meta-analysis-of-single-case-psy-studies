"""
Create a suggested worklist for improving the SCED gold dataset.

This script reads review/disagreements_for_review.csv and writes an LLM-B-first
review file. Clean rows where legacy coding and the full-PDF extraction agree
are normally moved to silver by scripts.sced_review before this script runs; any
remaining rows are suggestions for human review. Rows with PDF alignment issues
are never auto-confirmed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sced_review import from_json_cell


DEFAULT_REVIEW_CSV = ROOT / "review" / "disagreements_for_review.csv"
DEFAULT_OUTPUT = ROOT / "review" / "disagreements_with_suggestions.csv"
DEFAULT_BATCH_OUTPUT = ROOT / "review" / "high_priority_review_batch.csv"
DEFAULT_ALIGNMENT_CSV = ROOT / "review" / "pdf_alignment_issues.csv"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


SUGGESTION_COLUMNS = [
    "suggested_value_json",
    "suggested_source",
    "suggestion_confidence",
    "suggested_decision_status",
    "review_action",
    "suggestion_reason",
    "needs_pdf_check",
    "alignment_issue",
    "alignment_evidence",
]

SOURCE_COLUMNS = {
    "legacy": "legacy_value_json",
    "llm_a": "llm_a_value_json",
    "llm_b": "llm_b_value_json",
}

NORMALIZED_COLUMNS = {
    "legacy": "legacy_normalized_json",
    "llm_a": "llm_a_normalized_json",
    "llm_b": "llm_b_normalized_json",
}

STATUS_RANK = {
    "two_llms_agree_legacy_diff": 0,
    "all_different": 1,
    "legacy_llm_b_agree": 2,
    "legacy_llm_a_agree": 3,
    "one_source_only": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create suggested values for manual SCED disagreement review."
    )
    parser.add_argument(
        "--review-csv",
        default=str(DEFAULT_REVIEW_CSV),
        help="Path to disagreements_for_review.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to write the full suggestion CSV.",
    )
    parser.add_argument(
        "--batch-output",
        default=str(DEFAULT_BATCH_OUTPUT),
        help="Path to write a smaller high-priority review batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Maximum rows to include in the high-priority batch.",
    )
    parser.add_argument(
        "--alignment-csv",
        default=str(DEFAULT_ALIGNMENT_CSV),
        help=(
            "Optional PDF alignment issue CSV. If present, affected rows are "
            "annotated but not marked reviewed."
        ),
    )
    return parser.parse_args()


def to_json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_alignment_issues(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        pdf = row.get("pdf", "").strip()
        if pdf:
            grouped.setdefault(pdf, []).append(row)
    return grouped


def non_empty_sources(row: dict[str, str]) -> list[str]:
    sources: list[str] = []
    for source, column in NORMALIZED_COLUMNS.items():
        raw = row.get(column, "").strip()
        if raw and raw != "[]":
            sources.append(source)
    return sources


def source_value(row: dict[str, str], source: str) -> str:
    return row.get(SOURCE_COLUMNS[source], "null").strip() or "null"


def suggest_row(row: dict[str, str]) -> dict[str, str]:
    status = row.get("status", "")
    suggestion = {
        "suggested_value_json": "",
        "suggested_source": "",
        "suggestion_confidence": "",
        "suggested_decision_status": "",
        "review_action": "",
        "suggestion_reason": "",
        "needs_pdf_check": "yes",
        "alignment_issue": "",
        "alignment_evidence": "",
    }

    if status == "two_llms_agree_legacy_diff":
        suggestion.update(
            {
                "suggested_value_json": source_value(row, "llm_b"),
                "suggested_source": "llm_a+llm_b",
                "suggestion_confidence": "high",
                "suggested_decision_status": "candidate_legacy_error",
                "review_action": "check_pdf_llm_consensus_against_legacy",
                "suggestion_reason": "Both independent LLM extractions agree after normalization and differ from legacy coding.",
            }
        )
        return suggestion

    if status == "legacy_llm_b_agree":
        suggestion.update(
            {
                "suggested_value_json": source_value(row, "legacy"),
                "suggested_source": "legacy+llm_b",
                "suggestion_confidence": "medium_high",
                "suggested_decision_status": "llm_b_confirmed",
                "review_action": "auto_accept_legacy_llm_b",
                "suggestion_reason": (
                    "Legacy coding and full-PDF extraction agree after normalization; "
                    "auto-confirm unless there is a PDF alignment issue."
                ),
            }
        )
        return suggestion

    if status == "legacy_llm_a_agree":
        suggestion.update(
            {
                "suggested_value_json": source_value(row, "legacy"),
                "suggested_source": "legacy+llm_a",
                "suggestion_confidence": "medium",
                "suggested_decision_status": "candidate_gold_reviewed",
                "review_action": "check_pdf_llm_b_disagreement",
                "suggestion_reason": "Legacy coding and block-text extraction agree after normalization.",
            }
        )
        return suggestion

    if status == "one_source_only":
        sources = non_empty_sources(row)
        if len(sources) == 1:
            source = sources[0]
            suggestion.update(
                {
                    "suggested_value_json": source_value(row, source),
                    "suggested_source": source,
                    "suggestion_confidence": "low",
                    "suggested_decision_status": f"candidate_{source}_only",
                    "review_action": "check_pdf_one_source_only",
                    "suggestion_reason": "Only one source has a non-empty value; source PDF must be checked.",
                }
            )
        return suggestion

    if status == "all_different":
        suggestion.update(
            {
                "suggestion_confidence": "none",
                "review_action": "manual_pdf_review",
                "suggestion_reason": "All available source values differ; source PDF review required before choosing a value.",
            }
        )
        return suggestion

    return suggestion


def annotate_alignment_issue(
    row: dict[str, str],
    alignment_issues: dict[str, list[dict[str, str]]],
) -> None:
    issues = alignment_issues.get(row.get("pdf", ""))
    if not issues:
        return

    issue_types = sorted({issue.get("issue_type", "") for issue in issues if issue.get("issue_type", "")})
    evidence = []
    for issue in issues:
        detail = issue.get("evidence", "")
        related = issue.get("related_pdf", "")
        if related:
            detail = f"{detail} Related: {related}"
        if detail:
            evidence.append(detail)

    row["alignment_issue"] = "; ".join(issue_types)
    row["alignment_evidence"] = " | ".join(evidence)
    row["needs_pdf_check"] = "alignment_issue"
    row["review_action"] = "fix_pdf_alignment_first"
    prefix = "PDF alignment issue: fix file mapping or extracted text before field-level review."
    reason = row.get("suggestion_reason", "")
    row["suggestion_reason"] = f"{prefix} {reason}".strip()
    existing_notes = row.get("notes", "").strip()
    alignment_note = f"PDF alignment issue: {row['alignment_issue']}; see review/pdf_alignment_issues.csv."
    row["notes"] = f"{existing_notes} {alignment_note}".strip() if existing_notes else alignment_note


def auto_confirm_legacy_llm_b(row: dict[str, str]) -> None:
    if row.get("status") != "legacy_llm_b_agree":
        return
    if row.get("alignment_issue"):
        return

    row["reviewed_value_json"] = source_value(row, "legacy")
    row["decision_status"] = "llm_b_confirmed"
    row["reviewer"] = "auto_llm_b_workflow"
    row["decision_reason"] = (
        "Accepted by LLM-B-first workflow: legacy coding and full-PDF extraction "
        "agree after normalization."
    )
    row["needs_pdf_check"] = "no"


def priority_key(row: dict[str, str]) -> tuple[int, int, int, str, str]:
    alignment_rank = 0 if row.get("alignment_issue") else 1
    status_rank = STATUS_RANK.get(row.get("status", ""), 99)
    confidence_rank = {
        "high": 0,
        "medium_high": 1,
        "medium": 2,
        "low": 3,
        "none": 4,
        "": 5,
    }.get(row.get("suggestion_confidence", ""), 5)
    return alignment_rank, status_rank, confidence_rank, row.get("field", ""), row.get("pdf", "")


def add_suggestions(
    rows: list[dict[str, str]],
    alignment_issues: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    suggested_rows: list[dict[str, str]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(suggest_row(row))
        annotate_alignment_issue(enriched, alignment_issues)
        auto_confirm_legacy_llm_b(enriched)
        suggested_rows.append(enriched)
    return sorted(suggested_rows, key=priority_key)


def main() -> None:
    args = parse_args()
    review_csv = Path(args.review_csv)
    output = Path(args.output)
    batch_output = Path(args.batch_output)
    alignment_csv = Path(args.alignment_csv)
    if not review_csv.is_absolute():
        review_csv = ROOT / review_csv
    if not output.is_absolute():
        output = ROOT / output
    if not batch_output.is_absolute():
        batch_output = ROOT / batch_output
    if not alignment_csv.is_absolute():
        alignment_csv = ROOT / alignment_csv

    fieldnames, rows = load_rows(review_csv)
    alignment_issues = load_alignment_issues(alignment_csv)
    suggested_rows = add_suggestions(rows, alignment_issues)
    output_fields = fieldnames + [col for col in SUGGESTION_COLUMNS if col not in fieldnames]
    write_csv(output, output_fields, suggested_rows)

    batch_rows = [
        row
        for row in suggested_rows
        if row.get("status") in {"two_llms_agree_legacy_diff", "all_different"}
    ][: args.batch_size]
    write_csv(batch_output, output_fields, batch_rows)

    counts = Counter(row.get("suggestion_confidence", "") for row in suggested_rows)
    alignment_count = sum(1 for row in suggested_rows if row.get("alignment_issue"))
    auto_confirmed_count = sum(
        1 for row in suggested_rows if row.get("decision_status") == "llm_b_confirmed"
    )
    logging.info("Wrote %d suggested rows to %s", len(suggested_rows), output)
    logging.info("Wrote %d high-priority rows to %s", len(batch_rows), batch_output)
    logging.info("Suggestion confidence counts: %s", dict(sorted(counts.items())))
    logging.info("Auto-confirmed %d legacy+LLM-B agreement rows", auto_confirmed_count)
    if alignment_count:
        logging.info("Annotated %d rows with PDF alignment issues", alignment_count)


if __name__ == "__main__":
    main()
