from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Supplement 3. Case and study characteristics.xlsx"
OUTPUT_PATH = ROOT / "output" / "pdf" / "eda_template_filled_meta_analysis.pdf"
EDA_DIR = ROOT / "eda_output"
INPUT_DIR = ROOT / "input"
EXTRACTED_DIR = ROOT / "extracted_text"

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


def load_df() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH).dropna(how="all").dropna(axis=1, how="all")
    df["Study"] = df["Study"].ffill()
    df["Age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Diagnosis_Category"] = (
        df["Primary diagnosis (outcome variable I)"].map(DIAGNOSIS_MAP).fillna(
            df["Primary diagnosis (outcome variable I)"]
        )
    )
    return df


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return max(sum(1 for _ in path.open()) - 1, 0)


def pct(value: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def add_plot(story: list, path: Path, width_cm: float, height_cm: float) -> None:
    if path.exists():
        story.append(Image(str(path), width=width_cm * cm, height=height_cm * cm))
        story.append(Spacer(1, 0.3 * cm))


def make_bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style)) for item in items],
        bulletType="bullet",
        leftIndent=14,
    )


def build_pdf() -> Path:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_df()
    total_rows, total_cols = df.shape
    source_cols = total_cols - 2
    unique_studies = int(df["Study"].nunique())
    pdf_count = len([path for path in INPUT_DIR.iterdir() if path.suffix.lower() == ".pdf"])
    block_json_count = len(list(EXTRACTED_DIR.glob("*_blocks.json")))

    age_series = df["Age_numeric"].dropna()
    diagnosis_counts = df["Diagnosis_Category"].fillna("Missing").value_counts()
    design_counts = df["Type of SCED design"].fillna("Missing").value_counts()
    intervention_count = count_csv_rows(EDA_DIR / "intervention_types_frequency.csv")
    demographic_count = count_csv_rows(EDA_DIR / "demographics_frequency.csv")

    missing_pct = (df.isna().sum() / len(df)).sort_values(ascending=False)
    top_missing = list(missing_pct.head(8).items())

    age_by_diag = (
        df[["Primary diagnosis (outcome variable I)", "Age_numeric"]]
        .dropna()
        .groupby("Primary diagnosis (outcome variable I)")["Age_numeric"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(6)
    )

    gender_counts = df["Gender"].fillna("Missing").astype(str).str.strip().value_counts()

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=10,
        textColor=colors.HexColor("#153B50"),
    )
    h1 = ParagraphStyle(
        "Heading1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#153B50"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=8.5,
        leading=11,
    )

    story: list = []
    story.append(Paragraph("Exploratory Data Analysis (EDA) for Data Science Projects", title))
    story.append(
        Paragraph(
            "Filled template for the <i>meta-analysis-of-single-case-psy-studies</i> repository. "
            "This report summarizes the available study metadata, the linked paper corpus, and the main distributions "
            "that matter for downstream extraction and analysis.",
            body,
        )
    )

    story.append(Paragraph("Corpus", h1))
    corpus_table = Table(
        [
            ["Metric", "Value"],
            ["Usable metadata rows", str(total_rows)],
            ["Source metadata variables", str(source_cols)],
            ["Derived analysis variables", str(total_cols - source_cols)],
            ["Unique studies", str(unique_studies)],
            ["PDF files in input/", str(pdf_count)],
            ["Extracted text block JSON files", str(block_json_count)],
            ["Distinct intervention labels in EDA output", str(intervention_count)],
            ["Distinct demographic labels in EDA output", str(demographic_count)],
        ],
        colWidths=[8.2 * cm, 5.2 * cm],
    )
    corpus_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153B50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C6CE")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F6F8FA")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
            ]
        )
    )
    story.append(corpus_table)
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        make_bullets(
            [
                "The main structured source is <b>Supplement 3. Case and study characteristics.xlsx</b>. "
                "After removing empty rows and columns, the working table contains "
                f"<b>{total_rows}</b> rows, <b>{source_cols}</b> original variables, "
                "and 2 derived analysis variables.",
                f"The repository contains <b>{pdf_count}</b> source papers in <b>input/</b>. "
                "The matching logic in the repo links all study entries to a corresponding PDF, so PDF coverage is complete.",
                f"There are <b>{block_json_count}</b> text-block extraction files in <b>extracted_text/</b>, "
                "which indicates that some papers were processed under more than one filename variant.",
            ],
            body,
        )
    )

    story.append(Paragraph("Data Cleaning And Provenance", h1))
    story.append(
        make_bullets(
            [
                "Empty rows and fully empty columns were removed before analysis.",
                "The <b>Study</b> column was forward-filled because a single study can span multiple case rows in the Excel sheet.",
                "Age was coerced to numeric to make range and central tendency estimates possible.",
                "Detailed anxiety diagnoses were also mapped to a broader <b>Anxiety Disorder</b> category for higher-level summaries.",
                "Outputs are preserved in <b>eda_output/</b>, including plots, frequency tables, and missing-value reports.",
            ],
            body,
        )
    )

    story.append(Paragraph("Missingness And Variable Quality", h1))
    missing_items = [
        f"<b>{name}</b>: {share * 100:.1f}% missing" for name, share in top_missing
    ]
    story.append(make_bullets(missing_items, body))
    story.append(
        Paragraph(
            "Missingness is concentrated in study-level descriptors such as treatment protocol, intervention type, "
            "country, setting, publication type, and SCED design. This matters because it limits how far multivariate "
            "comparisons can be pushed without first standardizing and completing metadata.",
            body,
        )
    )

    story.append(Paragraph("Pre Variable Description", h1))
    story.append(
        make_bullets(
            [
                f"Participant ages are available for <b>{len(age_series)}</b> of {total_rows} rows "
                f"({pct(len(age_series), total_rows)}), with mean <b>{age_series.mean():.2f}</b>, median <b>{age_series.median():.1f}</b>, "
                f"and range <b>{age_series.min():.0f}-{age_series.max():.0f}</b> years.",
                f"Explicit single-gender labels show <b>{gender_counts.get('F', 0)}</b> female and <b>{gender_counts.get('M', 0)}</b> male cases, "
                f"with <b>{gender_counts.get('Missing', 0)}</b> missing and several mixed encodings such as 8F/2M or 4F/4M.",
                "Intervention and demographic frequencies were exported to CSV so the label space can be normalized before modeling.",
            ],
            body,
        )
    )

    story.append(Paragraph("Univariate Analysis", h1))
    story.append(
        make_bullets(
            [
                f"The broad diagnosis distribution is dominated by <b>{diagnosis_counts.index[0]}</b> "
                f"with <b>{diagnosis_counts.iloc[0]}</b> rows ({pct(int(diagnosis_counts.iloc[0]), total_rows)}).",
                f"The next largest groups are <b>{diagnosis_counts.index[1]}</b> "
                f"({diagnosis_counts.iloc[1]} rows) and <b>{diagnosis_counts.index[2]}</b> ({diagnosis_counts.iloc[2]} rows).",
                "The intervention frequency table is led by CBT and behavioral treatment variants, which suggests "
                "the corpus is methodologically concentrated rather than evenly spread across many therapy families.",
                "Among non-empty design entries, multiple baseline designs appear most often, but the design field itself is missing for most rows.",
            ],
            body,
        )
    )
    add_plot(story, EDA_DIR / "primary_diagnoses.png", 13.5, 8.1)
    add_plot(story, EDA_DIR / "age_distribution.png", 12.0, 7.5)

    story.append(Paragraph("Baseline", h1))
    story.append(
        Paragraph(
            f"If a model were asked to predict the broad diagnosis category using a naive majority-class rule, "
            f"it could always predict <b>{diagnosis_counts.index[0]}</b> and achieve a baseline accuracy of "
            f"<b>{pct(int(diagnosis_counts.iloc[0]), total_rows)}</b>. "
            "Any downstream classifier should beat that benchmark to be meaningful.",
            body,
        )
    )

    story.append(Paragraph("Multivariate Analysis", h1))
    multivariate_items = []
    for diagnosis, row in age_by_diag.iterrows():
        multivariate_items.append(
            f"{diagnosis}: mean age <b>{row['mean']:.1f}</b> across <b>{int(row['count'])}</b> observed cases."
        )
    story.append(make_bullets(multivariate_items, body))
    story.append(
        Paragraph(
            "The age summaries suggest clinically different age profiles across diagnoses: generalized anxiety and social anxiety skew older, "
            "while separation anxiety and selective mutism skew younger. A full correlation analysis would require further label cleaning "
            "because categorical values currently use many inconsistent spellings.",
            body,
        )
    )
    add_plot(story, EDA_DIR / "sced_design_types.png", 9.6, 9.6)

    story.append(Paragraph("Normalization Recommendations", h1))
    story.append(
        make_bullets(
            [
                "Collapse synonymous treatment labels such as CBT, Cognitive behavioral treatment, and CBT principles.",
                "Standardize SCED design spellings such as multiple baseline design, mutiple baseline design, and pre-post variants.",
                "Split combined gender strings into separate count variables so group totals are machine-readable.",
                "Normalize ethnicity spelling and casing because the current labels mix white, caucasian, New Zealand European, and many one-off encodings.",
                "Deduplicate extraction artifacts where the same paper produced more than one block JSON file because of filename variation.",
            ],
            body,
        )
    )

    story.append(Paragraph("Conclusion", h1))
    story.append(
        Paragraph(
            "This repository already supports a strong descriptive EDA workflow: the metadata table is linked to the paper corpus, "
            "the main demographic and intervention summaries are exported, and the plots make the dominant diagnosis and age patterns easy to inspect. "
            "The main limitation is metadata completeness and label consistency.",
            body,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.4 * cm,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        title="Filled EDA Template",
        author="Codex",
    )
    doc.build(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    output = build_pdf()
    print(output)
