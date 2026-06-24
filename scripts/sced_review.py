"""
Create a silver/gold review workflow for SCED extraction outputs.

The workflow compares three independent sources at the field level:
1. legacy human coding, usually data/sced_gold.jsonl generated from Supplement 3
2. LLM A, for example extracted_text/sced_results.jsonl
3. LLM B, for example extracted_text/sced_results_full_pdf.jsonl

It writes:
- silver_candidates.csv: non-empty normalized three-way agreements, plus
  legacy/full-PDF agreements where block-text extraction differs
- disagreements_for_review.csv: field-level rows that need human review
- review_summary.json: counts by status and field

After review, fill `reviewed_value_json` in disagreements_for_review.csv and run
with --build-gold to create an evidence-reconciled gold JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_sced_results import load_jsonl_by_pdf, normalize_to_set
from scripts.sced_fields import FIELDS


DEFAULT_LEGACY = ROOT / "data" / "sced_gold.jsonl"
DEFAULT_LLM_A = ROOT / "extracted_text" / "sced_results.jsonl"
DEFAULT_LLM_B = ROOT / "extracted_text" / "sced_results_full_pdf.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "review"
DEFAULT_GOLD_OUTPUT = ROOT / "data" / "sced_gold_reviewed_v1.jsonl"
DEFAULT_SKIPPED_FIELDS = {"Study"}

SILVER_COLUMNS = [
    "pdf",
    "study",
    "field",
    "status",
    "silver_value_json",
    "normalized_value_json",
    "legacy_value_json",
    "llm_a_value_json",
    "llm_b_value_json",
    "decision_reason",
]

REVIEW_COLUMNS = [
    "pdf",
    "study",
    "field",
    "status",
    "priority",
    "legacy_value_json",
    "legacy_normalized_json",
    "llm_a_value_json",
    "llm_a_normalized_json",
    "llm_b_value_json",
    "llm_b_normalized_json",
    "agreement_pattern",
    "reviewed_value_json",
    "decision_status",
    "reviewer",
    "decision_reason",
    "evidence_quote",
    "page",
    "notes",
]

GOLD_ALLOWED_REVIEW_STATUSES = {
    "gold_reviewed",
    "legacy_error",
    "llm_a_error",
    "llm_b_error",
    "llm_error",
    "llm_b_confirmed",
    "source_ambiguous",
    "codebook_ambiguous",
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass(frozen=True)
class SourceValue:
    raw: Any
    normalized: tuple[str, ...]
    evidence_quote: str | None = None
    page: str | None = None

    @property
    def has_value(self) -> bool:
        return bool(self.normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create silver candidates and gold review sheets for SCED fields."
    )
    parser.add_argument(
        "--legacy",
        default=str(DEFAULT_LEGACY),
        help="Legacy human-coded JSONL, usually data/sced_gold.jsonl.",
    )
    parser.add_argument(
        "--llm-a",
        default=str(DEFAULT_LLM_A),
        help="First LLM JSONL output, usually block-text extraction.",
    )
    parser.add_argument(
        "--llm-b",
        default=str(DEFAULT_LLM_B),
        help="Second LLM JSONL output, usually full-PDF extraction.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for silver/review CSVs and summary JSON.",
    )
    parser.add_argument(
        "--build-gold",
        action="store_true",
        help="Build a reviewed gold JSONL from the review CSV plus silver candidates.",
    )
    parser.add_argument(
        "--review-csv",
        help="Path to completed disagreements CSV. Defaults to <output-dir>/disagreements_for_review.csv.",
    )
    parser.add_argument(
        "--silver-csv",
        help="Path to silver candidates CSV. Defaults to <output-dir>/silver_candidates.csv.",
    )
    parser.add_argument(
        "--gold-output",
        default=str(DEFAULT_GOLD_OUTPUT),
        help="Path to write reviewed gold JSONL when --build-gold is used.",
    )
    parser.add_argument(
        "--allow-partial-gold",
        action="store_true",
        help=(
            "Allow building gold even if some review rows are not completed. "
            "Unreviewed fields fall back to the legacy value."
        ),
    )
    return parser.parse_args()


def to_json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def from_json_cell(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def normalized_json(value: Iterable[str]) -> str:
    return to_json_cell(sorted(value))


def extract_wrapped_value(value: Any) -> tuple[Any, str | None, str | None]:
    """
    Support future provenance-aware values such as:
    {"value": "...", "evidence_quote": "...", "page": 3}

    Current extraction outputs are plain scalars/lists, so this returns the original
    value unless a clear wrapper is present.
    """
    if isinstance(value, dict) and "value" in value:
        page = value.get("page")
        return value.get("value"), value.get("evidence_quote"), str(page) if page is not None else None
    return value, None, None


def make_source_value(record: dict[str, Any] | None, field: str) -> SourceValue:
    if record is None:
        return SourceValue(raw=None, normalized=())
    raw, evidence_quote, page = extract_wrapped_value(record.get(field))
    normalized = tuple(sorted(normalize_to_set(raw, field)))
    return SourceValue(raw=raw, normalized=normalized, evidence_quote=evidence_quote, page=page)


def preferred_silver_value(legacy: SourceValue, llm_a: SourceValue, llm_b: SourceValue) -> Any:
    """
    Keep the legacy coding style when all sources agree after normalization.
    If legacy is empty for some reason, fall back to LLM A, then LLM B.
    """
    if legacy.raw is not None:
        return legacy.raw
    if llm_a.raw is not None:
        return llm_a.raw
    return llm_b.raw


def silver_decision_reason(status: str) -> str:
    if status == "legacy_llm_b_agree":
        return (
            "Normalized legacy/full-PDF agreement; accepted as silver "
            "despite block-text extraction differing."
        )
    return "Normalized three-way agreement; accepted as silver."


def classify_agreement(
    legacy: SourceValue,
    llm_a: SourceValue,
    llm_b: SourceValue,
) -> tuple[str, str, str]:
    legacy_norm = legacy.normalized
    a_norm = llm_a.normalized
    b_norm = llm_b.normalized
    non_empty_count = sum(source.has_value for source in (legacy, llm_a, llm_b))

    if not non_empty_count:
        return "all_missing", "low", "legacy=missing; llm_a=missing; llm_b=missing"

    if legacy_norm == a_norm == b_norm and legacy.has_value:
        return "silver_auto", "low", "legacy=llm_a=llm_b"

    if a_norm == b_norm and llm_a.has_value and legacy_norm != a_norm:
        return "two_llms_agree_legacy_diff", "high", "llm_a=llm_b; legacy differs"

    if legacy_norm == a_norm and legacy.has_value and legacy_norm != b_norm:
        return "legacy_llm_a_agree", "medium", "legacy=llm_a; llm_b differs"

    if legacy_norm == b_norm and legacy.has_value and legacy_norm != a_norm:
        return "legacy_llm_b_agree", "medium", "legacy=llm_b; llm_a differs"

    if non_empty_count == 1:
        return "one_source_only", "medium", "only one source has a non-empty value"

    return "all_different", "high", "all available non-empty values differ"


def study_name(*records: dict[str, Any] | None) -> str:
    for record in records:
        if not record:
            continue
        study = record.get("Study")
        if study is not None and str(study).strip():
            return str(study).strip()
    return ""


def collect_rows(
    legacy_records: dict[str, dict[str, Any]],
    llm_a_records: dict[str, dict[str, Any]],
    llm_b_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    all_pdfs = sorted(set(legacy_records) | set(llm_a_records) | set(llm_b_records))
    silver_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    field_status_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for pdf in all_pdfs:
        legacy_record = legacy_records.get(pdf)
        llm_a_record = llm_a_records.get(pdf)
        llm_b_record = llm_b_records.get(pdf)
        study = study_name(legacy_record, llm_a_record, llm_b_record)

        for field in FIELDS:
            if field in DEFAULT_SKIPPED_FIELDS:
                continue
            legacy = make_source_value(legacy_record, field)
            llm_a = make_source_value(llm_a_record, field)
            llm_b = make_source_value(llm_b_record, field)
            status, priority, pattern = classify_agreement(legacy, llm_a, llm_b)
            status_counts[status] += 1
            field_status_counts[field][status] += 1

            if status == "all_missing":
                continue

            if status in {"silver_auto", "legacy_llm_b_agree"}:
                normalized = legacy.normalized
                silver_rows.append(
                    {
                        "pdf": pdf,
                        "study": study,
                        "field": field,
                        "status": status,
                        "silver_value_json": to_json_cell(preferred_silver_value(legacy, llm_a, llm_b)),
                        "normalized_value_json": normalized_json(normalized),
                        "legacy_value_json": to_json_cell(legacy.raw),
                        "llm_a_value_json": to_json_cell(llm_a.raw),
                        "llm_b_value_json": to_json_cell(llm_b.raw),
                        "decision_reason": silver_decision_reason(status),
                    }
                )
                continue

            review_rows.append(
                {
                    "pdf": pdf,
                    "study": study,
                    "field": field,
                    "status": status,
                    "priority": priority,
                    "legacy_value_json": to_json_cell(legacy.raw),
                    "legacy_normalized_json": normalized_json(legacy.normalized),
                    "llm_a_value_json": to_json_cell(llm_a.raw),
                    "llm_a_normalized_json": normalized_json(llm_a.normalized),
                    "llm_b_value_json": to_json_cell(llm_b.raw),
                    "llm_b_normalized_json": normalized_json(llm_b.normalized),
                    "agreement_pattern": pattern,
                    "reviewed_value_json": "",
                    "decision_status": "",
                    "reviewer": "",
                    "decision_reason": "",
                    "evidence_quote": llm_a.evidence_quote or llm_b.evidence_quote or "",
                    "page": llm_a.page or llm_b.page or "",
                    "notes": "",
                }
            )

    summary = {
        "legacy_records": len(legacy_records),
        "llm_a_records": len(llm_a_records),
        "llm_b_records": len(llm_b_records),
        "matched_or_union_pdfs": len(all_pdfs),
        "fields_per_pdf": len(FIELDS) - len(DEFAULT_SKIPPED_FIELDS),
        "skipped_fields": sorted(DEFAULT_SKIPPED_FIELDS),
        "silver_rows": len(silver_rows),
        "review_rows": len(review_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "field_status_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in sorted(field_status_counts.items())
        },
        "input_coverage": {
            "missing_from_legacy": sorted(set(all_pdfs) - set(legacy_records)),
            "missing_from_llm_a": sorted(set(all_pdfs) - set(llm_a_records)),
            "missing_from_llm_b": sorted(set(all_pdfs) - set(llm_b_records)),
        },
    }
    return silver_rows, review_rows, summary


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_review_files(args: argparse.Namespace) -> None:
    legacy_path = Path(args.legacy)
    llm_a_path = Path(args.llm_a)
    llm_b_path = Path(args.llm_b)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    legacy_records = load_jsonl_by_pdf(legacy_path)
    llm_a_records = load_jsonl_by_pdf(llm_a_path)
    llm_b_records = load_jsonl_by_pdf(llm_b_path)

    silver_rows, review_rows, summary = collect_rows(legacy_records, llm_a_records, llm_b_records)

    silver_path = output_dir / "silver_candidates.csv"
    review_path = output_dir / "disagreements_for_review.csv"
    summary_path = output_dir / "review_summary.json"
    write_csv(silver_path, silver_rows, SILVER_COLUMNS)
    write_csv(review_path, review_rows, REVIEW_COLUMNS)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Wrote %d silver rows to %s", len(silver_rows), display_path(silver_path))
    logging.info("Wrote %d review rows to %s", len(review_rows), display_path(review_path))
    logging.info("Wrote summary to %s", display_path(summary_path))


def group_rows_by_pdf_field(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["pdf"], row["field"])
        if key in grouped:
            raise ValueError(f"Duplicate review row for {key[0]} / {key[1]}")
        grouped[key] = row
    return grouped


def should_use_reviewed_value(row: dict[str, str]) -> bool:
    value = row.get("reviewed_value_json", "").strip()
    if not value:
        return False
    status = row.get("status", "").strip()
    decision_status = row.get("decision_status", "").strip()
    if decision_status:
        return decision_status in GOLD_ALLOWED_REVIEW_STATUSES
    return status in {
        "two_llms_agree_legacy_diff",
        "legacy_llm_a_agree",
        "legacy_llm_b_agree",
        "one_source_only",
        "all_different",
    } | GOLD_ALLOWED_REVIEW_STATUSES


def build_gold(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    silver_path = Path(args.silver_csv) if args.silver_csv else output_dir / "silver_candidates.csv"
    review_path = Path(args.review_csv) if args.review_csv else output_dir / "disagreements_for_review.csv"
    gold_output = Path(args.gold_output)
    if not gold_output.is_absolute():
        gold_output = ROOT / gold_output

    legacy_records = load_jsonl_by_pdf(Path(args.legacy))
    silver_rows = read_csv(silver_path)
    review_rows = read_csv(review_path)
    silver_by_key = group_rows_by_pdf_field(silver_rows)
    review_by_key = group_rows_by_pdf_field(review_rows)

    all_pdfs = sorted(
        set(legacy_records)
        | {pdf for pdf, _ in silver_by_key}
        | {pdf for pdf, _ in review_by_key}
    )

    missing_reviews: list[str] = []
    records: list[dict[str, Any]] = []

    for pdf in all_pdfs:
        legacy_record = legacy_records.get(pdf, {})
        output_record: dict[str, Any] = {"pdf": pdf}
        for field in FIELDS:
            key = (pdf, field)
            if key in silver_by_key:
                output_record[field] = from_json_cell(silver_by_key[key]["silver_value_json"])
                continue

            review_row = review_by_key.get(key)
            if review_row and should_use_reviewed_value(review_row):
                output_record[field] = from_json_cell(review_row["reviewed_value_json"])
                continue

            if review_row and not args.allow_partial_gold:
                missing_reviews.append(f"{pdf} :: {field}")
                continue

            output_record[field] = legacy_record.get(field)

        records.append(output_record)

    if missing_reviews:
        preview = "\n".join(f"- {item}" for item in missing_reviews[:20])
        remainder = len(missing_reviews) - min(len(missing_reviews), 20)
        if remainder:
            preview += f"\n... and {remainder} more"
        raise ValueError(
            "Cannot build full gold file because some review rows do not have "
            f"reviewed_value_json. Fill them or pass --allow-partial-gold.\n{preview}"
        )

    gold_output.parent.mkdir(parents=True, exist_ok=True)
    with gold_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("Wrote %d reviewed gold records to %s", len(records), display_path(gold_output))


def main() -> None:
    args = parse_args()
    try:
        if args.build_gold:
            build_gold(args)
        else:
            generate_review_files(args)
    except Exception as exc:  # noqa: BLE001
        logging.error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
