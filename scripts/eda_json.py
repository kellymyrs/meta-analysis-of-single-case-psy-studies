"""EDA helper for study metadata.

Primary mode: scan `../input` for `.json` files containing study records.
Fallback: if no JSON is found, read metadata from `data/Supplement 3. Case and study characteristics.xlsx`.

Produces frequency counts for intervention types and demographics, a bar chart of study phases/designs,
and a missing-values report for key categorical fields. Outputs are written to `../eda_output`:
    intervention_types_frequency.csv
    demographics_frequency.csv
    study_phase_distribution.png
    missing_values_report.txt

Run from the project root:
    python scripts/eda_json.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "eda_output"
EXCEL_FALLBACK = ROOT / "data" / "Supplement 3. Case and study characteristics.xlsx"

# Keys to probe in the study dictionaries
INTERVENTION_KEYS = ["intervention_types", "intervention_type", "intervention"]
DEMOGRAPHIC_KEYS = ["demographics", "demographic", "participant_demographics"]
PHASE_KEYS = ["study_phase", "phase_sequence", "phase", "design", "sced_phase", "design_type"]
CATEGORICAL_MISSING_KEYS = ["participant_age", "clinical_setting"]

# Column names used when falling back to the Excel metadata file
EXCEL_INTERVENTION_COLS = ["Type of treatments(s)"]
EXCEL_DEMOGRAPHIC_COLS = ["Gender", "Ethnicity / Race"]
EXCEL_PHASE_COLS = ["Type of SCED design"]
EXCEL_MISSING_MAP = {
    "participant_age": "Age",
    "clinical_setting": "Setting  University clinic, Outpatient Psychiatric Center, School, Other, Not reported",
}


def coerce_list(value: Any) -> List[str]:
    """Normalize a value to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def flatten_demographics(value: Any) -> List[str]:
    """Flatten demographics payloads into label strings for counting."""
    if isinstance(value, dict):
        flattened = []
        for key, val in value.items():
            if isinstance(val, (list, tuple, set)):
                for inner in val:
                    inner_text = str(inner).strip()
                    if inner_text:
                        flattened.append(f"{key}:{inner_text}")
            else:
                val_text = str(val).strip()
                if val_text:
                    flattened.append(f"{key}:{val_text}")
        return flattened
    return coerce_list(value)


def iter_studies(payload: Any) -> Iterable[Dict[str, Any]]:
    """Yield study dicts from various JSON layouts."""
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        if "studies" in payload and isinstance(payload["studies"], list):
            for item in payload["studies"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield payload


def first_present(study: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in study:
            return study[key]
    return None


def coerce_split(value: Any) -> List[str]:
    """Split cell contents on common delimiters into clean labels."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tokens = []
        for item in value:
            tokens.extend(coerce_split(item))
        return tokens
    text = str(value)
    for delim in [";", ",", "/", "|"]:
        text = text.replace(delim, "|")
    return [token.strip() for token in text.split("|") if token.strip()]


def process_json_files(json_files: List[Path]) -> None:
    intervention_counter: Counter[str] = Counter()
    demographics_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    missing_records: defaultdict[str, List[str]] = defaultdict(list)

    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {json_path.name} (could not parse JSON): {exc}")
            continue

        for idx, study in enumerate(iter_studies(payload), start=1):
            interventions = coerce_list(first_present(study, INTERVENTION_KEYS))
            demographics = flatten_demographics(first_present(study, DEMOGRAPHIC_KEYS))
            phase_value = first_present(study, PHASE_KEYS)

            intervention_counter.update(interventions)
            demographics_counter.update(demographics)
            if phase_value:
                phase_counter[str(phase_value).strip()] += 1

            for field in CATEGORICAL_MISSING_KEYS:
                value = study.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_records[field].append(f"{json_path.name} [record {idx}]")

    write_outputs(intervention_counter, demographics_counter, phase_counter, missing_records)


def process_excel_metadata(excel_path: Path) -> None:
    print(f"Using Excel fallback: {excel_path}")
    df = pd.read_excel(excel_path)
    df = df.dropna(how="all")
    if "Study" in df.columns:
        df = df[df["Study"].notna()]

    intervention_counter: Counter[str] = Counter()
    demographics_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    missing_records: defaultdict[str, List[str]] = defaultdict(list)

    for _, row in df.iterrows():
        study_label = str(row.get("Study", "(unknown study)")).strip()

        # Interventions
        for col in EXCEL_INTERVENTION_COLS:
            if col in df.columns:
                intervention_counter.update(coerce_split(row.get(col)))

        # Demographics
        for col in EXCEL_DEMOGRAPHIC_COLS:
            if col in df.columns:
                prefix = col.split(" ")[0].lower()
                values = coerce_split(row.get(col))
                demographics_counter.update([f"{prefix}:{val}" for val in values])

        # Study phase / design
        for col in EXCEL_PHASE_COLS:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    phase_counter[str(val).strip()] += 1

        # Missing values for key categorical fields
        for missing_key, excel_col in EXCEL_MISSING_MAP.items():
            if excel_col in df.columns:
                val = row.get(excel_col)
                if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                    missing_records[missing_key].append(study_label)

    write_outputs(intervention_counter, demographics_counter, phase_counter, missing_records)


def write_outputs(
    intervention_counter: Counter[str],
    demographics_counter: Counter[str],
    phase_counter: Counter[str],
    missing_records: defaultdict[str, List[str]],
) -> None:
    # --- Frequency tables ---
    def write_counter(counter: Counter[str], name: str) -> None:
        if not counter:
            print(f"No data for {name}.")
            return
        csv_path = OUTPUT_DIR / f"{name}_frequency.csv"
        csv_lines = ["label,count"] + [f"{label},{count}" for label, count in counter.most_common()]
        csv_path.write_text("\n".join(csv_lines))
        print(f"Top {name.replace('_', ' ')}:")
        for label, count in counter.most_common(10):
            print(f"  {label}: {count}")
        print(f"Saved full counts to {csv_path}")

    write_counter(intervention_counter, "intervention_types")
    write_counter(demographics_counter, "demographics")

    # --- Study phase distribution plot ---
    if phase_counter:
        labels, counts = zip(*phase_counter.most_common())
        plt.figure(figsize=(8, 4))
        plt.bar(labels, counts, color="#4C72B0")
        plt.title("Study Phase / Design Distribution")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "study_phase_distribution.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved study phase plot to {plot_path}")
    else:
        print("No study phase information found to plot.")

    # --- Missing values report ---
    report_lines: List[str] = []
    for field in CATEGORICAL_MISSING_KEYS:
        files = missing_records.get(field, [])
        if files:
            report_lines.append(f"{field}: missing in {len(files)} records")
            report_lines.extend(f"  - {entry}" for entry in files)
        else:
            report_lines.append(f"{field}: no missing values detected")

    report_path = OUTPUT_DIR / "missing_values_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"Missing values report saved to {report_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_files = sorted(INPUT_DIR.glob("*.json"))

    if json_files:
        process_json_files(json_files)
    elif EXCEL_FALLBACK.exists():
        process_excel_metadata(EXCEL_FALLBACK)
    else:
        print(
            "No JSON files in /input and no Excel fallback found. "
            "Add JSON metadata files to /input or place the Excel file at data/Supplement 3. Case and study characteristics.xlsx."
        )


if __name__ == "__main__":
    main()
