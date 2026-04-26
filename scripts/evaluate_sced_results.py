"""
Evaluate extraction results against a gold-standard JSONL file.

Expected JSONL schema for both predictions and gold:
    {"pdf": "Taylor 2011.pdf", "Country": ..., "Number of Cases": ..., ...}

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
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FIELDS = [
    "Country",
    "Number of Cases",
    "Gender",
    "Age",
    "Ethnicity / Race",
    "Type of treatments",
    "Total Number of Observations",
]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


COUNTRY_ALIASES = {
    "usa": "united states",
    "u.s.a.": "united states",
    "u.s.a": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "u.k": "united kingdom",
    "aus": "australia",
    "nl": "netherlands",
    "the netherlands": "netherlands",
    "can": "canada",
    "ger": "germany",
}

VALUE_ALIASES = {
    "f": "female",
    "female": "female",
    "girl": "female",
    "girls": "female",
    "m": "male",
    "male": "male",
    "boy": "male",
    "boys": "male",
    "cbt": "cognitive behavioral therapy",
    "cognitive behavioural therapy": "cognitive behavioral therapy",
    "cognitive behavior therapy": "cognitive behavioral therapy",
    "cognitive behavioural treatment": "cognitive behavioral therapy",
    "behavior therapy": "behavioral therapy",
    "behaviour therapy": "behavioral therapy",
    "behavioral treatment": "behavioral therapy",
    "behavioural treatment": "behavioral therapy",
    "tf-cbt": "trauma focused cognitive behavioral therapy",
    "trauma-focused cbt": "trauma focused cognitive behavioral therapy",
    "trauma focused cbt": "trauma focused cognitive behavioral therapy",
    "vr exp": "virtual reality exposure",
    "virtual reality exposure therapy": "virtual reality exposure",
    "parent training": "behavioral parent training",
    "behavioral parent training": "behavioral parent training",
    "behavioural parent training": "behavioral parent training",
    "pcit": "parent child interaction therapy",
    "parent-child interaction therapy": "parent child interaction therapy",
    "parent child interaction therapy": "parent child interaction therapy",
}


def _normalize_age_text(text: str) -> str:
    age = text.lower().strip()
    age = age.replace("years old", "")
    age = age.replace("year old", "")
    age = age.replace("years", "")
    age = age.replace("year", "")
    age = age.replace("yrs", "")
    age = age.replace("yr", "")
    age = age.replace("y/o", "")
    age = age.replace("yo", "")
    age = age.replace("to", "-")
    age = age.replace("–", "-")
    age = age.replace("—", "-")
    age = re.sub(r"\s+", " ", age).strip()
    age = re.sub(r"\s*-\s*", "-", age)
    if re.fullmatch(r"\d+(?:\.\d+)?", age):
        return format(float(age), ".12g")
    if re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", age):
        start, end = age.split("-", 1)
        return f"{format(float(start), '.12g')}-{format(float(end), '.12g')}"
    return text


def _normalize_gender_counts(text: str) -> str:
    gender = text.lower().strip()
    gender = gender.replace("females", "female")
    gender = gender.replace("males", "male")
    gender = gender.replace("girls", "female")
    gender = gender.replace("girl", "female")
    gender = gender.replace("boys", "male")
    gender = gender.replace("boy", "male")
    gender = gender.replace("women", "female")
    gender = gender.replace("woman", "female")
    gender = gender.replace("men", "male")
    gender = gender.replace("man", "male")
    gender = re.sub(r"(\d+)\s*f\b", r"\1 female", gender)
    gender = re.sub(r"(\d+)\s*m\b", r"\1 male", gender)
    gender = re.sub(r"\bf\s*(\d+)", r"\1 female", gender)
    gender = re.sub(r"\bm\s*(\d+)", r"\1 male", gender)
    female_before = [int(x) for x in re.findall(r"(\d+)\s*female\b", gender)]
    male_before = [int(x) for x in re.findall(r"(\d+)\s*male\b", gender)]
    female_after = [int(x) for x in re.findall(r"\bfemale\s*(\d+)", gender)]
    male_after = [int(x) for x in re.findall(r"\bmale\s*(\d+)", gender)]
    if female_after or male_after:
        female_counts = female_after
        male_counts = male_after
    else:
        female_counts = female_before
        male_counts = male_before
    if not female_counts and not male_counts:
        return text
    parts = []
    if female_counts:
        parts.append(f"female:{sum(female_counts)}")
    if male_counts:
        parts.append(f"male:{sum(male_counts)}")
    return "|".join(parts)


def _normalize_treatment_text(text: str) -> list[str] | None:
    treatment = text.lower().strip()
    if not treatment:
        return None
    if not re.search(r"\+|;|/|\band\b|\(", treatment):
        return None

    treatment = re.sub(r"\(([^)]*)\)", "", treatment)
    treatment = treatment.replace("trauma-focused cbt", "trauma focused cognitive behavioral therapy")
    treatment = treatment.replace("tf-cbt", "trauma focused cognitive behavioral therapy")
    treatment = treatment.replace("trauma-focused cognitive behavioral therapy", "trauma focused cognitive behavioral therapy")
    treatment = treatment.replace("cognitive behavioural therapy", "cognitive behavioral therapy")
    treatment = treatment.replace("compassion-focused therapy", "compassion focused therapy")
    treatment = treatment.replace("cft", "compassion focused therapy")
    treatment = re.sub(r"\s+", " ", treatment).strip()

    parts = re.split(r"\s*(?:\+|;|/|\band\b)\s*", treatment)
    normalized_parts: list[str] = []
    for part in parts:
        part = part.strip(" ,")
        if not part:
            continue
        part = VALUE_ALIASES.get(part, part)
        part = re.sub(r"\s+", " ", part).strip()
        normalized_parts.append(part)

    if not normalized_parts:
        return None
    return normalized_parts


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
    normalized = " ".join(text.lower().split())
    normalized = re.sub(r"\s+", " ", normalized)
    gender_normalized = _normalize_gender_counts(normalized)
    if gender_normalized != normalized:
        normalized = gender_normalized
    age_normalized = _normalize_age_text(normalized)
    if age_normalized != normalized:
        normalized = age_normalized
    normalized = COUNTRY_ALIASES.get(normalized, normalized)
    normalized = VALUE_ALIASES.get(normalized, normalized)
    return normalized


def normalize_to_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        normalized_items: set[str] = set()
        for item in value:
            if isinstance(item, str):
                treatments = _normalize_treatment_text(item)
                if treatments:
                    normalized_items.update(treatments)
                    continue
            atom = _normalize_atom(item)
            if atom is not None:
                normalized_items.add(atom)
        return normalized_items
    if isinstance(value, str):
        treatments = _normalize_treatment_text(value)
        if treatments:
            return set(treatments)
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
