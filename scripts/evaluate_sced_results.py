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
import csv
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sced_fields import FIELDS, normalize_field_keys

# Fields excluded from the headline metric, each for a documented reason (not to
# inflate the score). The two reasons are distinct:
#   "reviewer-assigned": the value is assigned by the meta-analysts during coding
#       and is not present in the source paper at all, so there is no extraction
#       target. The model correctly returns None and scoring it forces a flat
#       F1 = 0.0 that drags the overall metric down equally for every setup.
#   "figure-dependent": the value exists in the paper but only in a plotted figure
#       (e.g. the per-case observation count = number of data points on the
#       time-series graph). The text / full-PDF-text pipeline cannot read plotted
#       markers, so the value is unrecoverable without multimodal (figure-image)
#       extraction; instructed not to guess, the model abstains and scores ~0.
# Excluded fields are still emitted by the extractor (kept in FIELDS / the prompt);
# they just do not count toward precision/recall/F1. Pass --include-excluded-fields
# (or excluded_fields=set()) to score all 24 fields for a sensitivity analysis.
FIELD_EXCLUSION_REASONS = {
    "Quality rating RoBiNT scale": "reviewer-assigned",
    "Total Number of Observations": "figure-dependent",
}
EVAL_EXCLUDED_FIELDS = set(FIELD_EXCLUSION_REASONS)
SCORED_FIELDS = [field for field in FIELDS if field not in EVAL_EXCLUDED_FIELDS]

# Identifier fields skipped in the verification cross-tab only (not the headline
# metric): "Study" is a citation key, so a full-title-vs-short-cite mismatch is
# expected and uninformative for spotting gold-standard errors.
CROSSTAB_EXCLUDED_FIELDS = {"Study"}

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

# Fields where "NR"/"not reported" means the paper omitted a numeric value, so it
# should normalize to absent (an empty set) and compare equal to a null gold.
# This removes the phantom FP/FN seen on these fields where gold is null/NR and
# the model emits "NR" (or vice versa). Categorical fields such as Sample type
# and Setting are deliberately excluded — there "not reported" is a real value.
NR_AS_ABSENT_FIELDS = NUMERIC_UNIT_FIELDS | {"Drop-outs"}

DIAGNOSIS_SIMILARITY_FIELDS = {
    "Primary diagnosis",
    "Comorbid diagnosis / problems",
}

DIAGNOSIS_SIMILARITY_THRESHOLD = 0.80

KNOWN_DIAGNOSIS_LABELS = {
    "attention deficit hyperactivity disorder",
    "attention-deficit hyperactivity disorder",
    "anxiety disorder not otherwise specified",
    "autism spectrum disorder",
    "behavioral sleep disorder",
    "depression",
    "dysthymic disorder",
    "encopresis",
    "enuresis",
    "generalized anxiety disorder",
    "learning disorder",
    "major depressive disorder",
    "obsessive-compulsive disorder",
    "oppositional defiant disorder",
    "panic disorder",
    "posttraumatic stress disorder",
    "selective mutism",
    "separation anxiety disorder",
    "social anxiety disorder",
    "specific phobia",
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


def _diagnosis_similarity_tokens(text: str) -> list[str]:
    normalized = _strip_parenthetical_text(_basic_text_normalize(text))
    normalized = normalized.replace("attention-deficit", "attention deficit")
    normalized = normalized.replace("obsessive-compulsive", "obsessive compulsive")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    cleaned: list[str] = []
    for token in tokens:
        if token in {"diagnosis", "diagnoses", "problem", "problems", "symptom", "symptoms"}:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        cleaned.append(token)
    return cleaned


def _token_cosine_similarity(left: str, right: str) -> float:
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in _diagnosis_similarity_tokens(left):
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in _diagnosis_similarity_tokens(right):
        right_counts[token] = right_counts.get(token, 0) + 1
    if not left_counts or not right_counts:
        return 0.0

    common = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = sum(count * count for count in left_counts.values()) ** 0.5
    right_norm = sum(count * count for count in right_counts.values()) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


# --- Partial-credit (soft) matching ----------------------------------------
#
# Exact set matching treats a value as correct only when it normalizes to the
# identical string. The findings in review/model_vs_reviewed_gold_discrepancies.csv
# show many rows where predicted and gold differ but clearly share content:
# descriptive SCED phrases vs a canonical design family, ranges ("16-20") vs the
# per-case values they cover ("18", "19", "20"), or multi-token symptom labels
# that overlap. Soft matching awards each predicted value the token/interval
# similarity (0-1) of its best gold counterpart, so "some of the values are the
# same" earns proportional credit instead of a flat zero.

# Default minimum similarity for a predicted/gold pair to count toward partial TP.
PARTIAL_SIMILARITY_THRESHOLD = 0.5

# Fields whose values are numbers or numeric ranges; compared by interval overlap.
NUMERIC_INTERVAL_FIELDS = NUMERIC_UNIT_FIELDS | {"Age"}

# Light stopwords stripped before token-overlap similarity (kept small so that
# domain words like "design", "baseline", "anxiety" still drive the score).
_PARTIAL_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "with", "in", "on", "to",
    "by", "at", "is", "as", "per", "vs", "via",
}


def _general_tokens(text: str) -> list[str]:
    normalized = _strip_parenthetical_text(_basic_text_normalize(text))
    tokens = re.findall(r"[a-z0-9]+", normalized)
    cleaned: list[str] = []
    for token in tokens:
        if token in _PARTIAL_STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        cleaned.append(token)
    return cleaned


def _token_overlap_similarity(left: str, right: str) -> float:
    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in _general_tokens(left):
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in _general_tokens(right):
        right_counts[token] = right_counts.get(token, 0) + 1
    if not left_counts or not right_counts:
        return 0.0
    common = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _parse_numeric_interval(text: str) -> tuple[float, float] | None:
    point = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text)
    if point:
        value = float(point.group(1))
        return value, value
    span = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", text)
    if span:
        low, high = float(span.group(1)), float(span.group(2))
        return (low, high) if low <= high else (high, low)
    return None


def _numeric_interval_similarity(left: str, right: str) -> float | None:
    left_interval = _parse_numeric_interval(left)
    right_interval = _parse_numeric_interval(right)
    if left_interval is None or right_interval is None:
        return None

    (l0, l1), (r0, r1) = left_interval, right_interval
    integral = all(float(x).is_integer() for x in (l0, l1, r0, r1))
    if integral and (max(l1, r1) - min(l0, r0)) <= 1000:
        left_span = set(range(int(l0), int(l1) + 1))
        right_span = set(range(int(r0), int(r1) + 1))
        union = left_span | right_span
        return len(left_span & right_span) / len(union) if union else 1.0

    intersection = max(0.0, min(l1, r1) - max(l0, r0))
    union = max(l1, r1) - min(l0, r0)
    if union == 0:
        return 1.0 if l0 == r0 else 0.0
    return intersection / union


def _partial_similarity(left: str, right: str, field: str | None) -> float:
    if left == right:
        return 1.0
    if field in NUMERIC_INTERVAL_FIELDS:
        numeric = _numeric_interval_similarity(left, right)
        if numeric is not None:
            return numeric
    return _token_overlap_similarity(left, right)


def score_value_sets(
    predicted: set[str],
    gold: set[str],
    field: str | None = None,
    threshold: float = PARTIAL_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Score one field's predicted vs gold value sets.

    Returns exact set-based counts plus partial (soft) counts. Partial matching
    greedily pairs each predicted value with at most one gold value (highest
    similarity first); the summed similarity is the partial TP, and the
    remaining mass on each side becomes partial FP / FN. This keeps
    partial_tp + partial_fp == |predicted| and partial_tp + partial_fn == |gold|,
    so precision/recall reduce to soft_tp / |predicted| and soft_tp / |gold|.
    """
    exact_tp = len(predicted & gold)
    exact_fp = len(predicted - gold)
    exact_fn = len(gold - predicted)

    pairs: list[tuple[float, str, str]] = []
    for predicted_value in predicted:
        for gold_value in gold:
            similarity = _partial_similarity(predicted_value, gold_value, field)
            if similarity >= threshold:
                pairs.append((similarity, predicted_value, gold_value))
    # Highest similarity first; tie-break on the values for deterministic output.
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_predicted: set[str] = set()
    used_gold: set[str] = set()
    soft_tp = 0.0
    matches: list[Dict[str, Any]] = []
    for similarity, predicted_value, gold_value in pairs:
        if predicted_value in used_predicted or gold_value in used_gold:
            continue
        used_predicted.add(predicted_value)
        used_gold.add(gold_value)
        soft_tp += similarity
        if similarity < 1.0:
            matches.append(
                {
                    "predicted": predicted_value,
                    "gold": gold_value,
                    "similarity": round(similarity, 4),
                }
            )

    return {
        "exact": {"tp": exact_tp, "fp": exact_fp, "fn": exact_fn},
        "partial": {
            "tp": round(soft_tp, 6),
            "fp": round(len(predicted) - soft_tp, 6),
            "fn": round(len(gold) - soft_tp, 6),
        },
        "partial_matches": matches,
    }


@lru_cache(maxsize=1)
def _diagnosis_similarity_vocabulary() -> tuple[str, ...]:
    aliases = load_normalization_aliases()
    labels = set(KNOWN_DIAGNOSIS_LABELS)
    labels.update(aliases.get("Primary diagnosis", {}).values())
    return tuple(sorted(_basic_text_normalize(label) for label in labels if label))


def _canonical_diagnosis_by_similarity(text: str) -> str:
    best_label = text
    best_score = 0.0
    for label in _diagnosis_similarity_vocabulary():
        score = _token_cosine_similarity(text, label)
        if score > best_score:
            best_label = label
            best_score = score
    if best_score >= DIAGNOSIS_SIMILARITY_THRESHOLD:
        return best_label
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

    if re.search(r"(?:^|[^a-z0-9])c\.?\s*b\.?\s*t\.?(?:$|[^a-z0-9])", treatment):
        return "cognitive behavioral therapy"
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
    if not re.search(r"\+|;|/|,|\(", treatment):
        canonical_phrase = _canonical_treatment_phrase(treatment)
        if canonical_phrase:
            return [canonical_phrase]
        return None

    treatment = _strip_parenthetical_text(treatment)
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
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=PARTIAL_SIMILARITY_THRESHOLD,
        help=(
            "Minimum predicted/gold token-or-interval similarity (0-1) that counts "
            "toward partial credit. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--include-excluded-fields",
        action="store_true",
        help=(
            "Score all 24 fields, including the documented exclusions "
            f"({', '.join(sorted(EVAL_EXCLUDED_FIELDS))}). Use for a sensitivity "
            "analysis reporting metrics with and without those fields."
        ),
    )
    parser.add_argument(
        "--verification",
        help=(
            "Optional path to a sced_verification.jsonl file. When given, cross-tabs "
            "the source-grounded verifier verdict against model-vs-gold agreement and "
            "writes a *_gold_suspect.csv of fields the model and gold disagree on but "
            "the verifier judged supported (candidate gold errors)."
        ),
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
    if field in NR_AS_ABSENT_FIELDS and _basic_text_normalize(text) in {"nr", "not reported"}:
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
    if field in DIAGNOSIS_SIMILARITY_FIELDS:
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
    if field in DIAGNOSIS_SIMILARITY_FIELDS:
        normalized = _canonical_diagnosis_by_similarity(normalized)
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
    partial_threshold: float = PARTIAL_SIMILARITY_THRESHOLD,
    excluded_fields: Iterable[str] = EVAL_EXCLUDED_FIELDS,
) -> Dict[str, Any]:
    excluded = set(excluded_fields)
    scored_fields = [field for field in FIELDS if field not in excluded]

    common_pdfs = sorted(set(predictions) & set(gold))
    missing_predictions = sorted(set(gold) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(gold))

    per_field_counts = {field: {"tp": 0, "fp": 0, "fn": 0} for field in scored_fields}
    per_field_partial = {field: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for field in scored_fields}
    exact_match_count = 0
    per_pdf_results: List[Dict[str, Any]] = []

    for pdf_name in common_pdfs:
        pdf_exact = True
        field_details: Dict[str, Dict[str, Any]] = {}
        predicted_record = predictions[pdf_name]
        gold_record = gold[pdf_name]

        for field in scored_fields:
            predicted_values = normalize_to_set(predicted_record.get(field), field)
            gold_values = normalize_to_set(gold_record.get(field), field)

            scored = score_value_sets(
                predicted_values, gold_values, field, threshold=partial_threshold
            )
            exact = scored["exact"]
            partial = scored["partial"]

            for key in ("tp", "fp", "fn"):
                per_field_counts[field][key] += exact[key]
                per_field_partial[field][key] += partial[key]

            if exact["fp"] or exact["fn"]:
                pdf_exact = False

            field_details[field] = {
                "predicted": sorted(predicted_values),
                "gold": sorted(gold_values),
                "tp": exact["tp"],
                "fp": exact["fp"],
                "fn": exact["fn"],
                "partial": partial,
                "partial_matches": scored["partial_matches"],
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
        field: {
            **compute_metrics(**per_field_counts[field]),
            "partial": compute_metrics(**per_field_partial[field]),
        }
        for field in scored_fields
    }

    total_tp = sum(counts["tp"] for counts in per_field_counts.values())
    total_fp = sum(counts["fp"] for counts in per_field_counts.values())
    total_fn = sum(counts["fn"] for counts in per_field_counts.values())

    partial_tp = sum(counts["tp"] for counts in per_field_partial.values())
    partial_fp = sum(counts["fp"] for counts in per_field_partial.values())
    partial_fn = sum(counts["fn"] for counts in per_field_partial.values())

    return {
        "predictions_file_count": len(predictions),
        "gold_file_count": len(gold),
        "matched_file_count": len(common_pdfs),
        "scored_field_count": len(scored_fields),
        "excluded_fields": {
            field: FIELD_EXCLUSION_REASONS.get(field, "excluded")
            for field in sorted(excluded)
        },
        "partial_similarity_threshold": partial_threshold,
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "overall": {
            **compute_metrics(total_tp, total_fp, total_fn),
            "exact_match_rate": safe_divide(exact_match_count, len(common_pdfs)),
            "partial": compute_metrics(partial_tp, partial_fp, partial_fn),
        },
        "per_field": per_field_metrics,
        "per_pdf": per_pdf_results,
    }


def evaluate_jsonl_files(
    predictions_path: Path,
    gold_path: Path,
    partial_threshold: float = PARTIAL_SIMILARITY_THRESHOLD,
    excluded_fields: Iterable[str] = EVAL_EXCLUDED_FIELDS,
) -> Dict[str, Any]:
    predictions = load_jsonl_by_pdf(predictions_path)
    gold = load_jsonl_by_pdf(gold_path)
    return evaluate_records(
        predictions,
        gold,
        partial_threshold=partial_threshold,
        excluded_fields=excluded_fields,
    )


def load_verification_by_pdf(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load sced_verification.jsonl into {pdf: {field: {verdict, quote, ...}}}."""
    if not path.exists():
        raise FileNotFoundError(f"Verification file not found: {path}")
    verdicts_by_pdf: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pdf_name = str(record.get("pdf", "")).strip()
            if pdf_name:
                verdicts_by_pdf[pdf_name] = normalize_field_keys(record.get("verdicts", {}))
    return verdicts_by_pdf


def build_verification_crosstab(
    summary: Dict[str, Any],
    verification_by_pdf: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Cross-tab the verifier verdict against model-vs-gold field agreement.

    A field "agrees" with gold when its exact fp and fn are both zero. The two
    diagnostic cells:
      - gold_suspect: model and gold DISAGREE but the verifier judged the model's
        value "supported" by the source -> candidate gold-standard errors.
      - both_suspect: model and gold AGREE but the verifier "contradicted" the value
        -> candidate cases where model and gold are both wrong.
    """
    verdict_keys = ("supported", "contradicted", "not_in_text", "inferred")
    crosstab = {v: {"agree": 0, "disagree": 0} for v in verdict_keys}
    gold_suspect: List[Dict[str, Any]] = []
    both_suspect: List[Dict[str, Any]] = []
    covered_pdfs = 0

    for pdf_result in summary["per_pdf"]:
        pdf_name = pdf_result["pdf"]
        verdicts = verification_by_pdf.get(pdf_name)
        if not verdicts:
            continue
        covered_pdfs += 1
        for field, detail in pdf_result["fields"].items():
            if field in CROSSTAB_EXCLUDED_FIELDS:
                continue
            entry = verdicts.get(field)
            if not isinstance(entry, dict):
                continue
            verdict = str(entry.get("verdict", "")).strip().lower()
            if verdict not in crosstab:
                continue
            agree = detail["fp"] == 0 and detail["fn"] == 0
            crosstab[verdict]["agree" if agree else "disagree"] += 1
            row = {
                "pdf": pdf_name,
                "field": field,
                "predicted": "; ".join(detail["predicted"]),
                "gold": "; ".join(detail["gold"]),
                "verdict": verdict,
                "quote": entry.get("quote", ""),
            }
            if not agree and verdict == "supported":
                gold_suspect.append(row)
            elif agree and verdict == "contradicted":
                both_suspect.append(row)

    return {
        "verification_covered_pdfs": covered_pdfs,
        "crosstab": crosstab,
        "gold_suspect_count": len(gold_suspect),
        "both_suspect_count": len(both_suspect),
        "gold_suspect": gold_suspect,
        "both_suspect": both_suspect,
    }


def write_gold_suspect_csv(crosstab: Dict[str, Any], path: Path) -> None:
    fieldnames = ["pdf", "field", "predicted", "gold", "verdict", "quote"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in crosstab["gold_suspect"]:
            writer.writerow(row)


def default_output_path(predictions_path: Path) -> Path:
    return EVALUATION_DIR / f"{predictions_path.stem}_evaluation.json"


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    gold_path = Path(args.gold)
    output_path = Path(args.output) if args.output else default_output_path(predictions_path)

    summary = evaluate_jsonl_files(
        predictions_path,
        gold_path,
        partial_threshold=args.partial_threshold,
        excluded_fields=set() if args.include_excluded_fields else EVAL_EXCLUDED_FIELDS,
    )

    crosstab = None
    if args.verification:
        verification_by_pdf = load_verification_by_pdf(Path(args.verification))
        crosstab = build_verification_crosstab(summary, verification_by_pdf)
        summary["verification_crosstab"] = crosstab

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if crosstab is not None:
        suspect_path = output_path.with_name(f"{output_path.stem}_gold_suspect.csv")
        write_gold_suspect_csv(crosstab, suspect_path)

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
    partial = overall["partial"]
    logging.info(
        "Partial (>=%.2f) | Precision: %.3f  Recall: %.3f  F1: %.3f",
        summary["partial_similarity_threshold"],
        partial["precision"],
        partial["recall"],
        partial["f1"],
    )
    logging.info("Saved evaluation summary to %s", output_path)

    if crosstab is not None:
        logging.info("--- Verification cross-tab (verifier verdict x model-vs-gold) ---")
        logging.info("Verifier coverage: %d/%d matched PDFs", crosstab["verification_covered_pdfs"], summary["matched_file_count"])
        logging.info("%-13s %8s %8s", "verdict", "agree", "disagree")
        for verdict, cells in crosstab["crosstab"].items():
            logging.info("%-13s %8d %8d", verdict, cells["agree"], cells["disagree"])
        logging.info(
            "Gold-suspect (model!=gold but verifier supported): %d  ->  %s",
            crosstab["gold_suspect_count"],
            f"{output_path.stem}_gold_suspect.csv",
        )
        logging.info(
            "Both-suspect (model==gold but verifier contradicted): %d",
            crosstab["both_suspect_count"],
        )


if __name__ == "__main__":
    main()
