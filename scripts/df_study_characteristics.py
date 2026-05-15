"""
Quick EDA for case/study characteristics.

Reads the INT Excel metadata, reports missing PDFs, and plots a few key distributions.
"""

from pathlib import Path
import difflib
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "eda_output"
EXCEL_PATH = DATA_DIR / "Supplement 3. Case and study characteristics.xlsx"
DIAGNOSIS_MAP = {
    "AD (SM)": "Anxiety Disorder",
    "AD (separation anx)": "Anxiety Disorder",
    "AD (social phobia)": "Anxiety Disorder",
    "AD (NOS)": "Anxiety Disorder",
    "AD (generalized anx)": "Anxiety Disorder",
    "AD (social anx)": "Anxiety Disorder",
    "AD (simple phobia)": "Anxiety Disorder",
    "AD (specific phobia)": "Anxiety Disorder",
    "AD (overanxious disorder)": "Anxiety Disorder",
    "AD (GAD, SOP, or SAD)": "Anxiety Disorder",
    "AD (panic disorder with agoraphobia)": "Anxiety Disorder",
    "PTSD": "PTSD",
    "MDD": "Depression",
}


def _slugify(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _find_pdf_match(study_name: str, pdf_files: list[Path]) -> tuple[bool, str | None]:
    """Return (matched?, pdf_name) using exact, slug, author/year, or fuzzy match."""
    study_slug = _slugify(study_name)
    if not study_slug:
        return False, None

    # 1. Check for exact match in filenames (ignoring extension)
    for pdf_path in pdf_files:
        if study_name.lower() == pdf_path.stem.lower():
            return True, pdf_path.name

    # 2. Check for slug match
    for pdf_path in pdf_files:
        if study_slug == _slugify(pdf_path.stem):
            return True, pdf_path.name

    # 3. Check whether the year and any surname token before the year are in the filename.
    year_match = re.search(r"(19|20)\d{2}", study_name)
    if year_match:
        author_part = study_name[: year_match.start()]
        author_tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z]+", author_part)
            if len(token) > 2
        ]
        year = year_match.group(0)
        author_year_matches: list[tuple[int, Path]] = []
        for pdf_path in pdf_files:
            stem_lower = pdf_path.stem.lower()
            if year not in stem_lower:
                continue
            matched_tokens = sum(
                1
                for token in author_tokens
                if re.search(rf"\b{re.escape(token)}\b", stem_lower)
            )
            if matched_tokens:
                author_year_matches.append((matched_tokens, pdf_path))
        if author_year_matches:
            best_score = max(score for score, _ in author_year_matches)
            best_matches = [
                pdf_path
                for score, pdf_path in author_year_matches
                if score == best_score
            ]
            if len(best_matches) == 1:
                return True, best_matches[0].name

    # 4. Fuzzy similarity fallback
    best_pdf = None
    best_score = 0.0
    for pdf_path in pdf_files:
        score = difflib.SequenceMatcher(None, study_slug, _slugify(pdf_path.stem)).ratio()
        if score > best_score:
            best_score = score
            best_pdf = pdf_path.name
    if best_score >= 0.8:
        return True, best_pdf

    return False, None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    available_pdfs = [f for f in PDF_DIR.iterdir() if f.suffix.lower() == ".pdf"]
    print(f"Found {len(available_pdfs)} PDF files for processing in {PDF_DIR}.")

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel metadata not found: {EXCEL_PATH}")

    # --- 1. Load the dataset ---
    df = pd.read_excel(EXCEL_PATH)

    # --- 2. Basic Cleaning ---
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if "Study" in df.columns:
        df["Study"] = df["Study"].ffill()

    # --- 2b. Map Diagnoses ---
    diag_col = "Primary diagnosis (outcome variable I)"
    if diag_col in df.columns:
        df["Diagnosis_Category"] = df[diag_col].map(DIAGNOSIS_MAP).fillna(df[diag_col])
    else:
        df["Diagnosis_Category"] = "N/A"

    # Check: Which studies in the Excel are missing their actual PDF file?
    match_results = df["Study"].apply(lambda x: _find_pdf_match(str(x), available_pdfs))
    df["PDF_Exists"] = match_results.apply(lambda t: t[0])
    df["PDF_Match"] = match_results.apply(lambda t: t[1])
    missing_studies = df.loc[~df["PDF_Exists"], "Study"].unique()
    missing_count = len(missing_studies)
    print(
        f"Dataset loaded: {df.shape[0]} rows. "
        f"{missing_count} studies are missing PDF files in /input."
    )
    if missing_count > 0:
        print("\n--- List of Missing Studies ---")
        for s in sorted(missing_studies):
            print(f" - {s}")
    # Debugging: print study-to-PDF matches
    print("\nStudy to PDF matching (first 30):")
    for study, matched in df[["Study", "PDF_Match"]].drop_duplicates().head(30).itertuples(index=False):
        print(f" - Study: {study} -> PDF: {matched}")

    print("\nAvailable PDF filenames:")
    for f in sorted(available_pdfs, key=lambda x: x.name):
        print(f"   {f.name}")

    matched_pdf_names = set(df["PDF_Match"].dropna())
    extra_pdfs = sorted(set(f.name for f in available_pdfs) - matched_pdf_names)
    print("\nPDFs with no matching study entry (extras):")
    if extra_pdfs:
        for name in extra_pdfs:
            print(f"   {name}")
    else:
        print("   None")

    # --- 3. Distribution of Primary Diagnoses ---
    if "Diagnosis_Category" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y="Diagnosis_Category", data=df, order=df["Diagnosis_Category"].value_counts().index)
        plt.title("Distribution of Primary Diagnoses (Mapped Categories)")
        plt.xlabel("Number of Cases")
        plt.ylabel("Diagnosis")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "primary_diagnoses.png", dpi=300)
        plt.close()

    # --- 4. Age Distribution ---
    if "Age" in df.columns:
        df["Age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Age_numeric"].dropna(), bins=15, kde=True, color="skyblue")
        plt.title("Age Distribution of Participants")
        plt.xlabel("Age")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "age_distribution.png", dpi=300)
        plt.close()

    # --- 5. Study Design Types ---
    design_col = "Type of SCED design"
    if design_col in df.columns:
        plt.figure(figsize=(8, 8))
        df[design_col].value_counts().plot(
            kind="pie", autopct="%1.1f%%", startangle=140
        )
        plt.title("Breakdown of Single-Case Design Types")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "sced_design_types.png", dpi=300)
        plt.close()

    # --- 6. Summary Table ---
    if "Diagnosis_Category" in df.columns:
        summary = df.groupby("Diagnosis_Category").agg(
            {"Study": "nunique", "Age_numeric": ["mean", "min", "max"]}
        )
        print("\n--- Summary Table for Thesis Introduction ---")
        print(summary)


if __name__ == "__main__":
    main()
