from __future__ import annotations

import json
import re
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "input"
XLSX_PATH = ROOT / "data" / "Supplement 3. Case and study characteristics.xlsx"
OUT_PATH = ROOT / "data" / "sced_gold.jsonl"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def clean_value(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.upper() == "NA":
            return None
        return stripped
    return value


def build_pdf_lookup() -> dict[str, str]:
    pdfs = sorted(
        p.name for p in PDF_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    lookup: dict[str, str] = {}
    for pdf in pdfs:
        stem = Path(pdf).stem
        norm = normalize(stem)
        lookup[norm] = pdf
    return lookup


def match_pdf(study: str, pdf_lookup: dict[str, str]) -> str | None:
    study_norm = normalize(study)
    exact = pdf_lookup.get(study_norm)
    if exact:
        return exact

    candidates = [pdf for norm, pdf in pdf_lookup.items() if study_norm in norm or norm in study_norm]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return sorted(candidates, key=len)[0]
    return None


def collapse(values: list[object]) -> object | None:
    cleaned: list[object] = []
    seen: set[str] = set()
    for value in values:
        value = clean_value(value)
        if value is None:
            continue
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        cleaned.append(value)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned


def load_excel_records() -> dict[str, dict[str, object]]:
    wb = load_workbook(XLSX_PATH, data_only=True)
    ws = wb[wb.sheetnames[0]]

    current_study: str | None = None
    rows_by_study: dict[str, list[dict[str, object]]] = {}

    for row_idx in range(2, ws.max_row + 1):
        study = clean_value(ws.cell(row_idx, 1).value)
        if study is not None:
            current_study = str(study)
            rows_by_study.setdefault(current_study, [])
        if current_study is None:
            continue

        rows_by_study[current_study].append(
            {
                "Country": ws.cell(row_idx, 3).value,
                "Number of Cases": ws.cell(row_idx, 4).value,
                "Gender": ws.cell(row_idx, 8).value,
                "Age": ws.cell(row_idx, 9).value,
                "Ethnicity / Race": ws.cell(row_idx, 10).value,
                "Type of treatments": ws.cell(row_idx, 14).value,
                "Total Number of Observations": ws.cell(row_idx, 23).value,
            }
        )

    records: dict[str, dict[str, object]] = {}
    for study, rows in rows_by_study.items():
        countries = [row["Country"] for row in rows]
        num_cases = [row["Number of Cases"] for row in rows]
        genders = [row["Gender"] for row in rows]
        ages = [row["Age"] for row in rows]
        ethnicity = [row["Ethnicity / Race"] for row in rows]
        treatments = [row["Type of treatments"] for row in rows]
        observations = [row["Total Number of Observations"] for row in rows]

        records[study] = {
            "Country": collapse(countries),
            "Number of Cases": collapse(num_cases),
            "Gender": collapse(genders),
            "Age": collapse(ages),
            "Ethnicity / Race": collapse(ethnicity),
            "Type of treatments": collapse(treatments),
            "Total Number of Observations": collapse(observations),
        }
    return records


def main() -> None:
    pdf_lookup = build_pdf_lookup()
    study_records = load_excel_records()
    per_pdf: dict[str, dict[str, object]] = {}
    unmatched_studies: set[str] = set()
    for study, record in study_records.items():
        pdf_name = match_pdf(study, pdf_lookup)
        if not pdf_name:
            unmatched_studies.add(study)
            continue
        per_pdf[pdf_name] = record

    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for pdf_name in sorted(per_pdf):
            handle.write(json.dumps({"pdf": pdf_name, **per_pdf[pdf_name]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(per_pdf)} records to {OUT_PATH}")
    if unmatched_studies:
        print("Unmatched studies:")
        for study in sorted(unmatched_studies):
            print(f"- {study}")


if __name__ == "__main__":
    main()
