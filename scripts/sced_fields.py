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
    ScedField(
        "Drop-outs",
        5,
        "Drop-outs ",
        "number, NR, or null",
        (
            "Report the number of participants who dropped out. Use 0 only when the "
            "paper explicitly states there were no drop-outs. If the paper does not "
            "report drop-outs at all, output 'NR' — do not assume zero."
        ),
    ),
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
        (
            "Report the standard disorder name, expanding abbreviations to the full "
            "DSM/ICD-style label (e.g. 'SP' -> 'specific phobia', 'OCD' -> "
            "'obsessive-compulsive disorder', 'GAD' or 'OAD' -> 'generalized "
            "anxiety disorder', 'NOS' -> 'not otherwise specified'). Do not include "
            "qualifiers describing the phobic object or subtype (e.g. report "
            "'specific phobia', not 'specific phobia of dogs'). Return a list with "
            "one diagnosis per case when cases differ."
        ),
    ),
    ScedField(
        "Diagnosis screening or diagnostic interview?",
        12,
        "Diagnosis screening or diagnostic interview?",
        "screening, diagnostic interview, mixed description, NR, or null",
    ),
    ScedField(
        "Comorbid diagnosis / problems",
        13,
        "Comorbid diagnosis / problems",
        "string, list, or null",
        (
            "Use the meta-analysis coding scheme, one entry per case. Code 'N' when "
            "a case has no comorbidity. Otherwise begin with 'Y' and append, in this "
            "order: 'int' if any internalizing comorbidity is present, 'ext' if any "
            "externalizing comorbidity is present, then the abbreviations of specific "
            "named comorbid disorders (e.g. ADHD, ASD, ODD). Examples: 'N', 'Y, int', "
            "'Y, int, ADHD', 'Y, int, ext, ADHD'. Return a list with one value per "
            "case when cases differ. Do not list full clinical diagnosis names here."
        ),
    ),
    ScedField(
        "Type of treatments",
        14,
        "Type of treatments(s)",
        "string, list, or null",
        (
            "Report the established treatment name (e.g. 'cognitive behavioral "
            "therapy', 'parent child interaction therapy', 'behavioral parent "
            "training'), not the study-specific protocol name or acronym (those go "
            "in Treatment protocol). Give one entry per distinct treatment component; "
            "do not split a single named therapy into its sub-techniques."
        ),
    ),
    ScedField(
        "Treatment protocol",
        15,
        "Treatment protocol",
        "string, list, or null",
        (
            "Report the specific manual, protocol, or program name as stated in the "
            "paper (e.g. 'Coping Cat', 'BIACA/AFIYA'). This is the protocol label, "
            "not a prose description of the procedure."
        ),
    ),
    ScedField(
        "Treatment length",
        16,
        "Treatment length",
        "number, string, list, or null",
        (
            "Report the treatment length as a single bare number in the unit the "
            "paper uses (usually weeks or sessions), or a list of numbers with one "
            "per case when they differ. Give the number only, with no units or "
            "words (e.g. '4', not '4 weeks' or '5 treatment days per week'). Use a "
            "range string such as '6 to 7' only when the paper gives a range. Do "
            "not return a prose description of the schedule, follow-ups, or session "
            "timing. If the paper does not report treatment length, output 'NR'; do "
            "not infer or compute it from start/end dates, the assessment schedule, "
            "or follow-up timing."
        ),
    ),
    ScedField(
        "Number of sessions",
        17,
        "Number of sessions",
        "number, string, list, or null",
        (
            "Report the number of treatment sessions as a bare integer, or a list of "
            "integers with one per case when they differ. Use a range string such "
            "as '16 to 20' only when the paper reports a range. Do not include "
            "follow-up or booster contacts in the count unless the paper counts "
            "them, and do not return prose. If the number of sessions is not "
            "reported, output 'NR'; do not infer it from the schedule or treatment "
            "length."
        ),
    ),
    ScedField(
        "Treatment directed at",
        18,
        "Treatment directed at",
        "child, parent, child and parent, or null",
    ),
    ScedField(
        "Type of SCED design",
        19,
        "Type of SCED design",
        "string or null",
        (
            "Classify the design into one canonical family and report only that "
            "label. Allowed values: 'multiple baseline design', 'AB design', "
            "'ABAB design', 'changing-criterion design', 'alternating-treatments "
            "design', 'pre-post design', 'one-phase design', 'case series', "
            "'case study'. Strip descriptive qualifiers such as 'non-concurrent', "
            "'randomized', 'across participants/behaviors', or phase letters "
            "(e.g. 'non-concurrent multiple baseline across participants' -> "
            "'multiple baseline design')."
        ),
    ),
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
        (
            "Report a short symptom-domain label, not the name of the measure or "
            "instrument. Use forms such as 'anxiety symptoms', 'depression "
            "symptoms', 'PTSD symptoms', 'somatic complaints', 'physiological "
            "symptoms'. Return a list when several distinct domains are tracked."
        ),
    ),
    ScedField(
        "Frequent assessment specific symptom",
        22,
        "Frequent assessment specific symptom",
        "string, NA, or null",
        (
            "Report the specific symptom or target behavior being tracked as a "
            "short label (e.g. 'spontaneous speech', 'separation anxiety "
            "incidents', 'trait anxiety'), not the questionnaire/scale name or its "
            "abbreviation. Return a list when several distinct symptoms are tracked."
        ),
    ),
    ScedField(
        "Total Number of Observations",
        23,
        "Total number of observations",
        "number, string, list, NA, or null",
        (
            "Report the total count of repeated measurement time points (data "
            "points) per case as an integer, or a list of integers with one per "
            "case when they differ. Count the observations; do not describe the "
            "measurement schedule in prose, and do not infer a count from the "
            "design, treatment length, or session count. Use NA only when frequent "
            "assessment data are not available. If frequent assessment data exist "
            "but the per-case observation counts are not reported in the text, "
            "output 'NR'."
        ),
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
