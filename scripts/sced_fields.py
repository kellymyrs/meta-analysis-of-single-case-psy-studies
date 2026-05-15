from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScedField:
    name: str
    excel_column: int
    excel_header: str
    value_hint: str
    guidance: str = ""


SUPPLEMENT_3_FIELDS = [
    ScedField("Study", 1, "Study", "string or null"),
    ScedField("Type of publication", 2, "Type of publication", "article, dissertation, or null"),
    ScedField(
        "Country",
        3,
        "Country",
        "string, list, or null",
        (
            "Code the country where the study sample was recruited, assessed, "
            "or treated. Do not infer Country from author affiliations, journal "
            "publisher location, correspondence address, or cited studies unless "
            "the paper explicitly states that this is also the sample/treatment "
            "location. If no study/sample/treatment country is reported anywhere "
            "in the paper, use the country or countries from the author "
            "affiliations as a fallback. If multiple study/sample/treatment "
            "countries or fallback author-affiliation countries are reported, "
            "return a list."
        ),
    ),
    ScedField("Number of Cases", 4, "N cases", "number or null"),
    ScedField("Drop-outs", 5, "Drop-outs ", "number, NR, or null"),
    ScedField(
        "Sample type",
        6,
        "Sample type  Referred, Recruited, Combination, Other, Not reported",
        "Ref, Rec, Com, Oth, NR, full label, or null",
        "Allowed values: Referred/Ref, Recruited/Rec, Combination/Com, Other/Oth, Not reported/NR.",
    ),
    ScedField(
        "Setting",
        7,
        "Setting  University clinic, Outpatient Psychiatric Center, School, Other, Not reported",
        "UC, OPC, S, OTH, NR, mixed code, full label, or null",
        "Allowed values: University clinic/UC, Outpatient Psychiatric Center/OPC, School/S, Other/OTH, Not reported/NR.",
    ),
    ScedField("Gender", 8, "Gender", "string, list, or null"),
    ScedField("Age", 9, "Age", "number, string, list, or null"),
    ScedField("Ethnicity / Race", 10, "Ethnicity / Race", "string, list, or null"),
    ScedField(
        "Primary diagnosis",
        11,
        "Primary diagnosis (outcome variable I)",
        "string, list, or null",
        "Use the diagnosis label or code from the paper when possible, such as AD subtype, PTSD, or MDD.",
    ),
    ScedField(
        "Diagnosis screening or diagnostic interview?",
        12,
        "Diagnosis screening or diagnostic interview?",
        "screening, diagnostic interview, mixed description, NR, or null",
    ),
    ScedField("Comorbid diagnosis / problems", 13, "Comorbid diagnosis / problems", "string, list, or null"),
    ScedField("Type of treatments", 14, "Type of treatments(s)", "string, list, or null"),
    ScedField("Treatment protocol", 15, "Treatment protocol", "string, list, or null"),
    ScedField("Treatment length", 16, "Treatment length", "number, string, list, or null"),
    ScedField("Number of sessions", 17, "Number of sessions", "number, string, list, or null"),
    ScedField(
        "Treatment directed at",
        18,
        "Treatment directed at",
        "child, parent, child and parent, or null",
    ),
    ScedField("Type of SCED design", 19, "Type of SCED design", "string or null"),
    ScedField(
        "Data availability",
        20,
        "Data availability",
        "diagnosis, f.a., diagnosis/ f.a., or null",
        "Use diagnosis, frequent assessment/f.a., or both if both are available.",
    ),
    ScedField(
        "Frequent assessment variable broad category",
        21,
        "Frequent assessment variable broad category",
        "string, NA, or null",
    ),
    ScedField(
        "Frequent assessment specific symptom",
        22,
        "Frequent assessment specific symptom",
        "string, NA, or null",
    ),
    ScedField(
        "Total Number of Observations",
        23,
        "Total number of observations",
        "number, string, list, NA, or null",
    ),
    ScedField(
        "Quality rating RoBiNT scale",
        24,
        "Quality rating RoBiNT scale         Total, Internal validity, External validity",
        "string formatted as 'Total, Internal validity, External validity', list, or null",
        "The three possible sub-scores are Total, Internal validity, and External validity; keep that order when reporting one combined value.",
    ),
]

FIELDS = [field.name for field in SUPPLEMENT_3_FIELDS]

FIELD_ALIASES = {
    field.excel_header: field.name
    for field in SUPPLEMENT_3_FIELDS
    if field.excel_header != field.name
}
FIELD_ALIASES.update(
    {
        "N cases": "Number of Cases",
        "Drop-outs ": "Drop-outs",
        "Primary diagnosis (outcome variable I)": "Primary diagnosis",
        "Type of treatments(s)": "Type of treatments",
        "Total number of observations": "Total Number of Observations",
        "Quality rating RoBiNT scale         Total, Internal validity, External validity": "Quality rating RoBiNT scale",
    }
)


def normalize_field_keys(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        normalized[FIELD_ALIASES.get(key, key)] = value
    return normalized


def build_schema_prompt() -> str:
    lines = ["{"]
    for index, field in enumerate(SUPPLEMENT_3_FIELDS):
        suffix = "," if index < len(SUPPLEMENT_3_FIELDS) - 1 else ""
        lines.append(f'  "{field.name}": "<{field.value_hint}>"{suffix}')
    lines.append("}")
    return "\n".join(lines)


def build_guidance_prompt() -> str:
    guidance_lines = [
        f"- {field.name}: {field.guidance}"
        for field in SUPPLEMENT_3_FIELDS
        if field.guidance
    ]
    return "\n".join(guidance_lines)
