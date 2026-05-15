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
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sced_fields import FIELDS, normalize_field_keys

EVALUATION_DIR = ROOT / "evaluation_results"
NORMALIZATION_ALIASES_PATH = ROOT / "data" / "normalization_aliases.json"

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
    "espana": "spain",
    "españa": "spain",
    "columbia": "colombia",
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
    "cognitive-behavior therapy": "cognitive behavioral therapy",
    "cognitive-behavioral therapy": "cognitive behavioral therapy",
    "post-traumatic stress disorder": "posttraumatic stress disorder",
}

FIELD_VALUE_ALIASES = {
    "Sample type": {
        "ref": "referred",
        "referred": "referred",
        "rec": "recruited",
        "recruited": "recruited",
        "com": "combination",
        "combination": "combination",
        "oth": "other",
        "other": "other",
        "nr": "not reported",
        "not reported": "not reported",
    },
    "Setting": {
        "uc": "university clinic",
        "university clinic": "university clinic",
        "opc": "outpatient psychiatric center",
        "outpatient psychiatric center": "outpatient psychiatric center",
        "s": "school",
        "school": "school",
        "oth": "other",
        "other": "other",
        "nr": "not reported",
        "not reported": "not reported",
    },
    "Diagnosis screening or diagnostic interview?": {
        "diagnostic interviews": "diagnostic interview",
        "diagnostic interview": "diagnostic interview",
        "screening": "screening",
        "nr": "not reported",
        "not reported": "not reported",
    },
    "Treatment directed at": {
        "children": "child",
        "child": "child",
        "parents": "parent",
        "parent": "parent",
        "child and parent": "child and parent",
    },
    "Data availability": {
        "f.a.": "frequent assessment",
        "fa": "frequent assessment",
        "frequent assessment": "frequent assessment",
        "diagnosis": "diagnosis",
        "diagnosis/ f.a.": "diagnosis and frequent assessment",
        "diagnosis/f.a.": "diagnosis and frequent assessment",
        "diagnosis and frequent assessment": "diagnosis and frequent assessment",
    },
}


SPLITTABLE_FIELDS = {
    "Country",
    "Data availability",
    "Ethnicity / Race",
    "Primary diagnosis",
    "Setting",
    "Type of treatments",
}

NUMERIC_UNIT_FIELDS = {
    "Treatment length",
    "Number of sessions",
    "Total Number of Observations",
}


@lru_cache(maxsize=1)
def load_normalization_aliases() -> dict[str, dict[str, str]]:
    if not NORMALIZATION_ALIASES_PATH.exists():
        return {}
    with NORMALIZATION_ALIASES_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    aliases: dict[str, dict[str, str]] = {}
    for section, values in data.items():
        if section == "metadata" or not isinstance(values, dict):
            continue
        aliases[section] = {
            _basic_text_normalize(key): _basic_text_normalize(str(value))
            for key, value in values.items()
            if isinstance(key, str)
        }
    return aliases


def _strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _basic_text_normalize(text: str) -> str:
    normalized = _strip_accents(text.lower().strip())
    normalized = normalized.replace("–", "-").replace("—", "-").replace("‐", "-")
    normalized = normalized.replace("behaviour", "behavior")
    normalized = normalized.replace("behavioural", "behavioral")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _strip_parenthetical_text(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def _apply_alias(text: str, field: str | None = None) -> str:
    aliases = load_normalization_aliases()
    candidates = [text]
    stripped = _strip_parenthetical_text(text)
    if stripped != text:
        candidates.append(stripped)

    if field:
        field_aliases = aliases.get(field, {})
        for candidate in candidates:
            if candidate in field_aliases:
                return field_aliases[candidate]

    for candidate in candidates:
        if candidate in FIELD_VALUE_ALIASES.get(field or "", {}):
            return FIELD_VALUE_ALIASES[field or ""][candidate]
        if candidate in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[candidate]
        if candidate in VALUE_ALIASES:
            return VALUE_ALIASES[candidate]
        if candidate in aliases.get("global", {}):
            return aliases["global"][candidate]

    return text


def _canonical_country(text: str) -> str:
    normalized = _apply_alias(text, "Country")
    try:
        import pycountry  # type: ignore[import-not-found]

        country = pycountry.countries.lookup(normalized)
        return _basic_text_normalize(country.name)
    except Exception:
        return normalized


def _normalize_numeric_with_units(text: str, field: str | None) -> str:
    if field not in NUMERIC_UNIT_FIELDS:
        return text

    value = text.strip()
    value = re.sub(r"\bapproximately\b|\babout\b|\baround\b|\bapprox\.?\b", "", value).strip()
    value = re.sub(r"\s+", " ", value)

    if field == "Treatment length":
        leading_week_match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:week|weeks|wk|wks)\b", value)
        if leading_week_match:
            return format(float(leading_week_match.group(1)), ".12g")
        week_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:week|weeks|wk|wks)", value)
        if week_match:
            return format(float(week_match.group(1)), ".12g")
        week_range_match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:week|weeks|wk|wks)",
            value,
        )
        if week_range_match:
            start, end = week_range_match.groups()
            return f"{format(float(start), '.12g')}-{format(float(end), '.12g')}"
        session_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:session|sessions)", value)
        if session_match:
            return format(float(session_match.group(1)), ".12g")

    if field == "Number of sessions":
        session_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:session|sessions)", value)
        if session_match:
            return format(float(session_match.group(1)), ".12g")
        range_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:session|sessions)?", value)
        if range_match:
            start, end = range_match.groups()
            return f"{format(float(start), '.12g')}-{format(float(end), '.12g')}"

    leading_number = re.match(r"^(\d+(?:\.\d+)?)\s*\(", value)
    if leading_number:
        return format(float(leading_number.group(1)), ".12g")
    leading_session_number = re.match(r"^(\d+(?:\.\d+)?)\s+sessions?\b", value)
    if field == "Number of sessions" and leading_session_number:
        return format(float(leading_session_number.group(1)), ".12g")
    return text


def _normalize_age_text(text: str) -> str:
    age = _basic_text_normalize(text)
    age = re.sub(r"^(?:age|ages|aged)\s+", "", age)
    age = re.sub(r"^age range\s+", "", age)
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


def _parse_age_token(token: str) -> tuple[float, float] | None:
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        age = float(token)
        return age, age
    if re.fullmatch(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", token):
        start, end = token.split("-", 1)
        return float(start), float(end)
    return None


def _format_age_bound(value: float) -> str:
    return format(value, ".12g")


def _canonical_age_range(values: set[str]) -> set[str]:
    if len(values) <= 1:
        return values

    intervals: list[tuple[float, float]] = []
    for value in values:
        interval = _parse_age_token(value)
        if interval is None:
            return values
        intervals.append(interval)

    lower = min(start for start, _ in intervals)
    upper = max(end for _, end in intervals)
    if lower == upper:
        return {_format_age_bound(lower)}

    # Participant-level ages are often recorded with half years, while papers
    # report a whole-year age range. Compare those forms by the covered range.
    lower = float(int(lower // 1))
    upper = float(int(upper) if upper.is_integer() else int(upper) + 1)
    return {f"{_format_age_bound(lower)}-{_format_age_bound(upper)}"}


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


def _split_value_text(text: str, field: str | None = None) -> list[str] | None:
    if field not in SPLITTABLE_FIELDS:
        return None

    normalized = _basic_text_normalize(text)
    if not normalized:
        return None

    if field == "Primary diagnosis":
        if not re.search(r";", normalized):
            return None
        return [part.strip() for part in normalized.split(";") if part.strip()]

    if field == "Country":
        normalized = normalized.replace("new zealand-aus", "new zealand/australia")
        normalized = normalized.replace("aus-new zealand", "australia/new zealand")
        if not re.search(r",|/|\band\b", normalized):
            return None
        return [part.strip() for part in re.split(r"\s*(?:,|/|\band\b)\s*", normalized) if part.strip()]

    if field in {"Ethnicity / Race", "Setting"}:
        if not re.search(r",|;|/", normalized):
            return None
        return [part.strip() for part in re.split(r"\s*(?:,|;|/)\s*", normalized) if part.strip()]

    if field == "Data availability":
        data_alias = _apply_alias(normalized, field)
        if data_alias == "diagnosis and frequent assessment":
            return ["diagnosis", "frequent assessment"]
        if not re.search(r"/|\band\b|\+", normalized):
            return None
        return [part.strip() for part in re.split(r"\s*(?:/|\band\b|\+)\s*", normalized) if part.strip()]

    if field == "Type of treatments":
        return _normalize_treatment_text(normalized)

    return None


def _canonical_treatment_phrase(text: str) -> str | None:
    treatment = _strip_parenthetical_text(_basic_text_normalize(text))
    direct_alias = _apply_alias(treatment, "Type of treatments")
    if direct_alias != treatment:
        return direct_alias

    if "trauma focused cognitive behavioral therapy" in treatment:
        return "trauma focused cognitive behavioral therapy"
    if "parent child interaction therapy" in treatment:
        return "parent child interaction therapy"
    if re.search(r"\bcognitive behavioral therapy\b|\bcognitive behavior therapy\b", treatment):
        return "cognitive behavioral therapy"
    if "interpersonal psychotherapy" in treatment:
        return "interpersonal psychotherapy"
    if "attention bias modification" in treatment:
        return "attention bias modification"
    return None


def _normalize_treatment_text(text: str) -> list[str] | None:
    treatment = _basic_text_normalize(text)
    if not treatment:
        return None
    direct_alias = _apply_alias(treatment, "Type of treatments")
    if direct_alias != treatment:
        return [direct_alias]
    stripped = _strip_parenthetical_text(treatment)
    stripped_alias = _apply_alias(stripped, "Type of treatments")
    if stripped_alias != stripped:
        return [stripped_alias]
    canonical_phrase = _canonical_treatment_phrase(treatment)
    if canonical_phrase:
        return [canonical_phrase]
    if not re.search(r"\+|;|/|,|\(", treatment):
        return None

    treatment = _strip_parenthetical_text(treatment)
    treatment = treatment.replace("trauma-focused cbt", "trauma focused cognitive behavioral therapy")
    treatment = treatment.replace("tf-cbt", "trauma focused cognitive behavioral therapy")
    treatment = treatment.replace("trauma-focused cognitive behavioral therapy", "trauma focused cognitive behavioral therapy")
    treatment = treatment.replace("cognitive behavioural therapy", "cognitive behavioral therapy")
    treatment = treatment.replace("compassion-focused therapy", "compassion focused therapy")
    treatment = treatment.replace("cft", "compassion focused therapy")
    treatment = re.sub(r"\s+", " ", treatment).strip()

    parts = re.split(r"\s*(?:\+|;|/|,)\s*", treatment)
    normalized_parts: list[str] = []
    for part in parts:
        part = part.strip(" ,")
        if not part:
            continue
        part = _canonical_treatment_phrase(part) or _apply_alias(part, "Type of treatments")
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


def _normalize_atom(value: Any, field: str | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return format(float(value), ".12g")
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na"}:
        return None
    normalized = _basic_text_normalize(text)
    gender_normalized = _normalize_gender_counts(normalized)
    if gender_normalized != normalized:
        normalized = gender_normalized
    age_normalized = _normalize_age_text(normalized)
    if age_normalized != normalized:
        normalized = age_normalized
    normalized = _normalize_numeric_with_units(normalized, field)
    if field == "Country":
        normalized = _canonical_country(normalized)
    if field == "Primary diagnosis":
        diagnosis_alias = _apply_alias(normalized, field)
        if diagnosis_alias != normalized:
            normalized = diagnosis_alias
        else:
            normalized = _strip_parenthetical_text(normalized)
    if field == "Ethnicity / Race":
        normalized = re.sub(r"\s+\d+$", "", normalized).strip()
    normalized = _apply_alias(normalized, field)
    if field == "Type of treatments":
        normalized = re.sub(r"\s+(?:treatment|intervention|medication)$", "", normalized).strip()
        normalized = _apply_alias(normalized, field)
    if field == "Type of SCED design":
        normalized = normalized.replace("multiple-baseline", "multiple baseline")
        normalized = _apply_alias(normalized, field)
    return normalized


def normalize_to_set(value: Any, field: str | None = None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        normalized_items: set[str] = set()
        for item in value:
            if isinstance(item, str):
                split_values = _split_value_text(item, field)
                if split_values:
                    normalized_items.update(
                        atom
                        for part in split_values
                        if (atom := _normalize_atom(part, field)) is not None
                    )
                    continue
            atom = _normalize_atom(item, field)
            if atom is not None:
                normalized_items.add(atom)
        if field == "Age":
            return _canonical_age_range(normalized_items)
        return normalized_items
    if isinstance(value, str):
        split_values = _split_value_text(value, field)
        if split_values:
            return {
                atom
                for part in split_values
                if (atom := _normalize_atom(part, field)) is not None
            }
    atom = _normalize_atom(value, field)
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
            record = normalize_field_keys(json.loads(line))
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
            predicted_values = normalize_to_set(predicted_record.get(field), field)
            gold_values = normalize_to_set(gold_record.get(field), field)

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
    return EVALUATION_DIR / f"{predictions_path.stem}_evaluation.json"


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
    if summary["missing_predictions"]:
        logging.warning(
            "Missing predictions for %d gold files. First few: %s",
            len(summary["missing_predictions"]),
            ", ".join(summary["missing_predictions"][:5]),
        )
    if summary["extra_predictions"]:
        logging.warning(
            "Predictions contain %d files outside the gold set. First few: %s",
            len(summary["extra_predictions"]),
            ", ".join(summary["extra_predictions"][:5]),
        )
    logging.info("Precision: %.3f", overall["precision"])
    logging.info("Recall: %.3f", overall["recall"])
    logging.info("F1: %.3f", overall["f1"])
    logging.info("Exact match rate: %.3f", overall["exact_match_rate"])
    logging.info("Saved evaluation summary to %s", output_path)


if __name__ == "__main__":
    main()
