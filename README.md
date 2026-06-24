# Meta-Analysis Pipeline

This repo contains scripts for:
1. Metadata EDA from Supplement 3.
2. PDF text/layout extraction.
3. Figure-page image extraction.
4. LLM-based SCED variable extraction across all papers.

Run all commands from the project root.

## 1. Setup

1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Put files in the expected folders:
   - PDFs: `input/`
   - Excel metadata: `data/Supplement 3. Case and study characteristics.xlsx`

## 2. End-to-End Run Order

1. Study characteristics EDA (Excel + PDF presence check):
   ```bash
   python scripts/df_study_characteristics.py
   ```

2. Extract text blocks (with page + bounding boxes) from all PDFs:
   ```bash
   python scripts/pdf_text_blocks.py
   ```

3. Optional: run the smarter text extractor that rebuilds line order and merges paragraph-like blocks:
   ```bash
   python scripts/pdf_text_blocks_smart.py
   ```

4. Optional: run the GROBID extractor for study/article-aware structure:
   ```bash
   python scripts/pdf_text_blocks_grobid.py
   ```

5. Extract figure/chart pages as 300 DPI PNGs:
   ```bash
   python scripts/pdf_to_images.py
   ```

6. Optional metadata EDA from JSON files (or Excel fallback):
   ```bash
   python scripts/eda_json.py
   ```

7. Run SCED LLM extraction across all papers:
   ```bash
   python -m scripts.batch_sced_analysis
   ```

8. Try sending the full PDF directly to the model:
   ```bash
   python -m scripts.batch_sced_analysis --mode full_pdf
   ```

9. If your proxy does not support PDF file input, send the full extracted paper text instead:
   ```bash
   python -m scripts.batch_sced_analysis --mode full_text
   ```

10. Evaluate extraction results against a gold-standard JSONL file:
   ```bash
   python -m scripts.evaluate_sced_results \
     --predictions extracted_text/sced_results.jsonl \
     --gold data/sced_gold.jsonl
   ```

11. (Optional) Independently verify the extractions against the source PDFs, then fold the verdicts into evaluation. A separate verifier model (default `gpt-4.1`, which reads the full PDF natively) judges each field as `supported` / `contradicted` / `not_in_text` / `inferred` against the document, never against gold. Passing `--verification` to the evaluator cross-tabs that verdict with model-vs-gold agreement and writes a `*_gold_suspect.csv` of fields the model and gold disagree on but the verifier judged supported (candidate gold-standard errors):
   ```bash
   python -m scripts.verify_sced_results \
     --predictions extracted_text/sced_results_full_pdf.jsonl

   python -m scripts.evaluate_sced_results \
     --predictions extracted_text/sced_results_full_pdf.jsonl \
     --gold data/sced_gold.jsonl \
     --verification extracted_text/sced_verification.jsonl
   ```

12. Create silver candidates and a field-level review sheet from legacy coding plus two LLM runs:
   ```bash
   python -m scripts.sced_review \
     --legacy data/sced_gold.jsonl \
     --llm-a extracted_text/sced_results.jsonl \
     --llm-b extracted_text/sced_results_full_pdf.jsonl
   ```

13. Check for likely PDF-to-gold alignment problems before reviewing field values:
   ```bash
   python -m scripts.report_pdf_alignment_issues
   ```

14. Add LLM-B-first suggested values and a smaller high-priority batch for manual checking:
   ```bash
   python -m scripts.suggest_review_values
   ```

15. After checking PDF evidence and filling any remaining blank `reviewed_value_json` rows in `review/disagreements_with_suggestions.csv`, build a reviewed gold file:
   ```bash
   python -m scripts.sced_review --build-gold \
     --review-csv review/disagreements_with_suggestions.csv \
     --gold-output data/sced_gold_reviewed_v1.jsonl
   ```

16. Evaluate only reviewed disagreement rows, without any train/test split:
   ```bash
   python -m scripts.evaluate_review_disagreements \
     --reference reviewed \
     --output review/disagreement_evaluation.json
   ```

17. Split the SCED gold dataset into train/test JSONL files:
   ```bash
   python -m scripts.split_sced_dataset
   ```

18. Split the SCED gold dataset and run the model on one split:
   ```bash
   python -m scripts.run_sced_split_experiment --split test --evaluate
   ```

## 3. Script Reference

- `scripts/df_study_characteristics.py`
  - What it does: runs EDA on Supplement 3, checks which studies have matching PDFs, reports extra PDFs with no study match, and maps detailed diagnoses into broader categories (`Diagnosis_Category`, e.g., groups anxiety subtypes under "Anxiety Disorder" and maps "MDD" to "Depression"). Matching prioritizes exact filename matches (e.g., `Bechor 2014.pdf`) with slug/fuzzy fallbacks.
  - Reads: `data/Supplement 3. Case and study characteristics.xlsx`, PDFs in `input/`
  - Writes: `eda_output/primary_diagnoses.png`, `eda_output/age_distribution.png`, `eda_output/sced_design_types.png`
  - Run: `python scripts/df_study_characteristics.py`

- `scripts/pdf_text_blocks.py`
  - What it does: extracts text block-by-block from each PDF with layout metadata (`page`, `bbox`, `text`).
  - Reads: PDFs in `input/`
  - Writes: `extracted_text/<paper>_blocks.json`
  - Run: `python scripts/pdf_text_blocks.py`

- `scripts/pdf_to_images.py`
  - What it does: finds pages likely containing figures/charts and renders those pages to 300 DPI PNGs.
  - Reads: PDFs in `input/`
  - Writes: `processed_images/<paper>_page_<n>.png`
  - Run: `python scripts/pdf_to_images.py`

- `scripts/pdf_text_blocks_smart.py`
  - What it does: reconstructs page text from line/span data, applies a simple column-aware reading order, filters likely header/footer noise, and merges nearby lines into paragraph-like blocks.
  - Reads: PDFs in `input/`
  - Writes: `extracted_text/<paper>_smart_blocks.json`
  - Run: `python scripts/pdf_text_blocks_smart.py`
  - Single paper: `python scripts/pdf_text_blocks_smart.py --pdf "Taylor 2011.pdf"`

- `scripts/pdf_text_blocks_grobid.py`
  - What it does: sends each PDF to a running GROBID server, saves the raw TEI XML, and converts the scholarly structure into JSON blocks for titles, abstract paragraphs, section headings, body paragraphs, figure/table text, and references.
  - Reads: PDFs in `input/`
  - Writes: `extracted_text/<paper>_grobid.tei.xml`, `extracted_text/<paper>_grobid_blocks.json`
  - Requires: a running GROBID service, default `http://localhost:8070`
  - Run: `python scripts/pdf_text_blocks_grobid.py`
  - Single paper: `python scripts/pdf_text_blocks_grobid.py --pdf "Taylor 2011.pdf"`

- `scripts/eda_json.py`
  - What it does: calculates metadata frequencies and missing-value reports from JSON study files; if no JSON is found, falls back to the Excel file.
  - Reads: `input/*.json` (preferred), fallback `data/Supplement 3. Case and study characteristics.xlsx`
  - Writes: `eda_output/intervention_types_frequency.csv`, `eda_output/demographics_frequency.csv`, `eda_output/study_phase_distribution.png`, `eda_output/missing_values_report.txt`
  - Run: `python scripts/eda_json.py`

- `scripts/sced_extraction.py`
  - What it does: provides `run_sced_extraction()` for truncated block-text extraction, `run_sced_extraction_full_text()` for whole-paper chunked text extraction, and `run_sced_extraction_from_pdf()` for full-PDF extraction via the Responses API. The target JSON schema mirrors the Supplement 3 columns, using concise keys for headers that include possible values (for example `Sample type`, `Setting`, and `Quality rating RoBiNT scale`).
  - Reads: block JSON objects (from `pdf_text_blocks.py`)
  - Writes: none by itself (used by other scripts)
  - Model config: proxy mode via `LITELLM_KEY` for hosted models; local GGUF via `MODEL_PATH` (plus optional `LLAMA_THREADS`, `LLAMA_CTX`) for block-text mode only

- `scripts/batch_sced_analysis.py`
  - What it does: runs SCED extraction across all PDFs, either from extracted blocks or by attaching the full PDF to the model. Optionally evaluates predictions against a gold-standard JSONL file.
  - Reads: PDFs in `input/`, optional existing `extracted_text/<paper>_blocks.json`
  - Writes: `extracted_text/<paper>_sced.json`, `extracted_text/sced_results.jsonl`, plus `_full_pdf` variants when `--mode full_pdf` is used, and evaluation summaries in `evaluation_results/` by default
  - Run: `python -m scripts.batch_sced_analysis`
  - Full-text run: `python -m scripts.batch_sced_analysis --mode full_text`
  - Full-PDF run: `python -m scripts.batch_sced_analysis --mode full_pdf`
  - With evaluation: `python -m scripts.batch_sced_analysis --gold data/sced_gold.jsonl`
  - Optional custom evaluation folder: `python -m scripts.batch_sced_analysis --gold data/sced_gold.jsonl --evaluation-dir custom_eval_dir`
  - Single paper: `python -m scripts.batch_sced_analysis --mode full_pdf --pdf "Taylor 2011.pdf"`

- `scripts/sced_extraction.py` (verification)
  - What it does: beyond extraction, exposes `run_sced_verification_from_pdf()` (preferred: attaches the full PDF to the Responses API so the verifier reads the whole document with no page rendering) and `run_sced_verification_from_images()` (renders pages and sends them to a vision model such as `Qwen-2.5-VL`). Both apply a refute-style prompt and return a per-field verdict (`supported`/`contradicted`/`not_in_text`/`inferred`) with a verbatim quote for `supported` and `contradicted`.
  - Model config: proxy mode (`LITELLM_KEY`). Verifier model via `SCED_VERIFIER_MODEL` (default `gpt-4.1`); image-mode DPI via the orchestrator's `--dpi`.

- `scripts/verify_sced_results.py`
  - What it does: an independent, source-grounded check on the extractions. For each predicted record it locates the PDF in `input/`, runs the verifier, optionally string-matches each quote against the PDF text (flagging non-verbatim quotes as `quote_unmatched`), and records per-field verdicts. The verifier never sees the gold standard, so it complements evaluation against gold. `--mode pdf` (default) reads the PDF natively and handles long documents (e.g. dissertations); `--mode images` renders pages for a vision model.
  - Reads: a predictions JSONL file, PDFs in `input/`
  - Writes: `extracted_text/sced_verification.jsonl` (per-paper `{verdicts, summary}`)
  - Run: `python -m scripts.verify_sced_results --predictions extracted_text/sced_results_full_pdf.jsonl`
  - Single paper: `python -m scripts.verify_sced_results --predictions extracted_text/sced_results_full_pdf.jsonl --pdf "Taylor 2011.pdf"`
  - Image mode (Qwen): `SCED_VERIFIER_MODEL=Qwen-2.5-VL python -m scripts.verify_sced_results --predictions extracted_text/sced_results_full_pdf.jsonl --mode images --dpi 150`

- `scripts/evaluate_sced_results.py`
  - What it does: compares predicted SCED JSONL records against gold-standard JSONL records and reports micro-averaged precision, recall, F1, per-field metrics, and per-paper details.
  - Verification cross-tab: with `--verification extracted_text/sced_verification.jsonl`, it cross-tabs the verifier verdict against per-field model-vs-gold agreement (a field "agrees" when its exact `fp` and `fn` are both zero), adds a `verification_crosstab` section to the JSON, and writes a `<predictions>_evaluation_gold_suspect.csv` listing fields where the model and gold disagree but the verifier judged the model supported (candidate gold-standard errors). The identifier field `Study` is skipped in the cross-tab only.
  - Normalization: uses reusable aliases from `data/normalization_aliases.json`, plus field-specific cleanup for countries, diagnosis abbreviations, treatment names, SCED design labels, numeric units, age ranges/lists, and common frequent-assessment typos. For the numeric/count fields (`Treatment length`, `Number of sessions`, `Total Number of Observations`, `Drop-outs`), `NR`/`not reported` normalizes to absent so it matches a null gold instead of scoring as a mismatch; categorical fields keep "not reported" as a real value.
  - Excluded fields: two of the 24 fields are excluded from the headline metric for documented reasons (`FIELD_EXCLUSION_REASONS`). `Quality rating RoBiNT scale` is *reviewer-assigned* (the meta-analysts code it during review; it is not stated in the source paper). `Total Number of Observations` is *figure-dependent* (it equals the number of data points plotted on the per-case time-series graph, which the text/full-PDF-text pipeline cannot read). Both are still emitted by the extractor; they just do not count toward precision/recall/F1. The output JSON records `scored_field_count` and `excluded_fields` with their reasons.
  - Reads: a predictions JSONL file and a gold JSONL file with one record per PDF
  - Writes: `evaluation_results/<predictions>_evaluation.json` by default
  - Run: `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl`
  - Custom output path: `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl --output custom_eval_dir/sced_results_evaluation.json`
  - Sensitivity analysis (score all 24 fields, including the two exclusions): `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl --include-excluded-fields`

- `scripts/sced_review.py`
  - What it does: treats `data/sced_gold.jsonl` as legacy human coding, compares it with two independent LLM JSONL outputs field-by-field, accepts non-empty normalized three-way agreement and legacy plus full-PDF agreement as silver, and writes remaining disagreements to a human review sheet.
  - Normalization: reuses the same `data/normalization_aliases.json` layer as evaluation. `Study` is treated as an identifier and skipped from manual review.
  - Reads: legacy JSONL plus two LLM JSONL files, defaulting to `data/sced_gold.jsonl`, `extracted_text/sced_results.jsonl`, and `extracted_text/sced_results_full_pdf.jsonl`
  - Writes: `review/silver_candidates.csv`, `review/disagreements_for_review.csv`, and `review/review_summary.json`
  - Run: `python -m scripts.sced_review`
  - Build final gold after review: `python -m scripts.sced_review --build-gold --gold-output data/sced_gold_reviewed_v1.jsonl`
  - Review-sheet columns to fill: `reviewed_value_json`, `decision_status`, `reviewer`, `decision_reason`, `evidence_quote`, `page`, and `notes`
  - Recommended `decision_status` values: `llm_b_confirmed`, `gold_reviewed`, `legacy_error`, `llm_error`, `source_ambiguous`, or `codebook_ambiguous`

- `scripts/suggest_review_values.py`
  - What it does: reads `review/disagreements_for_review.csv`, applies an LLM-B-first review workflow, and writes a smaller batch that prioritizes likely legacy errors and hard disagreements.
  - Auto-confirmation: legacy plus full-PDF agreement rows are already handled as silver by `scripts.sced_review`; this script now annotates the remaining disagreement rows with suggested review actions.
  - Manual review: when legacy and LLM B disagree, LLM A is used to route the row as LLM consensus against legacy, LLM-B disagreement, all-different, or one-source-only.
  - Reads: `review/disagreements_for_review.csv`
  - Writes: `review/disagreements_with_suggestions.csv` and `review/high_priority_review_batch.csv`
  - Run: `python -m scripts.suggest_review_values`
  - Important: alignment-issue rows are never auto-confirmed. Fix or exclude those before accepting field-level values.

- `scripts/report_pdf_alignment_issues.py`
  - What it does: detects likely file-to-gold alignment problems before manual review, including duplicate input PDFs under different filenames and cases where the legacy study author is absent from the first pages of extracted text.
  - Reads: `data/sced_gold.jsonl`, `input/*.pdf`, and `extracted_text/*_blocks.json`
  - Writes: `review/pdf_alignment_issues.csv`
  - Run: `python -m scripts.report_pdf_alignment_issues`

- `scripts/evaluate_review_disagreements.py`
  - What it does: evaluates only rows in `review/disagreements_for_review.csv`; train/test split files are not used.
  - Final review mode: scores legacy, LLM A, and LLM B against `reviewed_value_json`.
  - Exploratory mode: `--reference legacy` scores LLM A and LLM B against legacy values, but this should not be reported as final accuracy if legacy coding may be wrong.
  - Run after manual review: `python -m scripts.evaluate_review_disagreements --reference reviewed --output review/disagreement_evaluation.json`
  - Exploratory run before review: `python -m scripts.evaluate_review_disagreements --reference legacy --output review/disagreement_legacy_diagnostic.json`

- `scripts/split_sced_dataset.py`
  - What it does: splits `data/sced_gold.jsonl` into deterministic train/test JSONL files for downstream experiments.
  - Reads: `data/sced_gold.jsonl`
  - Writes: `data/sced_gold_train.jsonl`, `data/sced_gold_test.jsonl`
  - Run: `python -m scripts.split_sced_dataset`
  - Custom split: `python -m scripts.split_sced_dataset --train-ratio 0.75 --seed 7`

- `scripts/run_sced_split_experiment.py`
  - What it does: creates the deterministic training/test split, runs SCED extraction for one split, and can evaluate predictions after extraction.
  - Reads: `data/sced_gold.jsonl`, PDFs in `input/`
  - Writes: `data/sced_gold_train.jsonl`, `data/sced_gold_test.jsonl`, `extracted_text/sced_<split>_results*.jsonl`, and evaluation summaries in `evaluation_results/` by default
  - Run: `python -m scripts.run_sced_split_experiment --split test`
  - With evaluation: `python -m scripts.run_sced_split_experiment --split test --evaluate`
  - Optional custom evaluation folder: `python -m scripts.run_sced_split_experiment --split test --evaluate --evaluation-dir custom_eval_dir`
  - Full-PDF mode: `python -m scripts.run_sced_split_experiment --split test --mode full_pdf --evaluate`
  - Full-PDF mode with fallback for oversized PDFs: `python -m scripts.run_sced_split_experiment --split test --mode full_pdf --full-pdf-context-fallback full_text --full-pdf-fallback-pdf "Girling-Butcher 2009.pdf" --model-setup few_shot --evaluate`
  - Few-shot examples (default `--few-shot-count 5`) are selected from the training split stratified by `Type of SCED design`: the picker round-robins across design buckets (largest first, seeded tie-break and within-bucket shuffle) so the example set spans distinct designs and value shapes instead of clustering on the most common design. Selection is deterministic for a given seed. Use `--few-shot-seed` to change only the example selection.
  - Few-shot with selected training examples: `python -m scripts.run_sced_split_experiment --split test --mode full_pdf --model-setup few_shot --few-shot-pdf "Ooi 2012.pdf" --few-shot-pdf "Cooper-Vince 2016.pdf" --evaluate`

## 4. LLM Model Configuration

The SCED extractor supports two modes.

1. Proxy mode for hosted models and full-PDF input:
   ```bash
   export LITELLM_KEY=your_token_here
   export LITELLM_MODEL=gpt-5.1
   # optional
   export LITELLM_BASE_URL=https://llmproxy.uva.nl
   ```

2. Local GGUF mode for block-text extraction only:
   ```bash
   export MODEL_PATH=/absolute/path/to/model.gguf
   ```
   Verify it exists:
   ```bash
   ls "$MODEL_PATH"
   ```

3. Optional performance tuning for local mode:
   ```bash
   export LLAMA_THREADS=8
   export LLAMA_CTX=4096
   ```

4. Run block-text extraction:
   ```bash
   python -m scripts.batch_sced_analysis
   ```

5. Run full-text extraction through chat completions:
   ```bash
   python -m scripts.batch_sced_analysis --mode full_text
   ```

6. Run full-PDF extraction:
   ```bash
   python -m scripts.batch_sced_analysis --mode full_pdf
   ```

If you see `Model path does not exist: /absolute/path/to/model.gguf`, your shell still has the placeholder value.
Set `MODEL_PATH` to your actual `.gguf` file and rerun.

## 5. Outputs

- `python scripts/df_study_characteristics.py`
  - `eda_output/primary_diagnoses.png`
  - `eda_output/age_distribution.png`
  - `eda_output/sced_design_types.png`

- `python scripts/pdf_text_blocks.py`
  - `extracted_text/<paper>_blocks.json`

- `python scripts/pdf_text_blocks_smart.py`
  - `extracted_text/<paper>_smart_blocks.json`

- `python scripts/pdf_text_blocks_grobid.py`
  - `extracted_text/<paper>_grobid.tei.xml`
  - `extracted_text/<paper>_grobid_blocks.json`

- `python scripts/pdf_to_images.py`
  - `processed_images/<paper>_page_<n>.png`

- `python scripts/eda_json.py`
  - `eda_output/intervention_types_frequency.csv`
  - `eda_output/demographics_frequency.csv`
  - `eda_output/study_phase_distribution.png`
  - `eda_output/missing_values_report.txt`

- `python -m scripts.batch_sced_analysis`
  - `extracted_text/<paper>_sced.json`
  - `extracted_text/sced_results.jsonl`

- `python -m scripts.batch_sced_analysis --mode full_pdf`
  - `extracted_text/<paper>_sced_full_pdf.json`
  - `extracted_text/sced_results_full_pdf.jsonl`

- `python -m scripts.batch_sced_analysis --mode full_text`
  - `extracted_text/<paper>_sced_full_text.json`
  - `extracted_text/sced_results_full_text.jsonl`

- `python -m scripts.verify_sced_results --predictions extracted_text/sced_results_full_pdf.jsonl`
  - `extracted_text/sced_verification.jsonl`

- `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl`
  - `evaluation_results/sced_results_evaluation.json`

- `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results_full_pdf.jsonl --gold data/sced_gold.jsonl --verification extracted_text/sced_verification.jsonl`
  - `evaluation_results/sced_results_full_pdf_evaluation.json` (includes `verification_crosstab` section)
  - `evaluation_results/sced_results_full_pdf_evaluation_gold_suspect.csv`

- `python -m scripts.sced_review`
  - `review/silver_candidates.csv`
  - `review/disagreements_for_review.csv`
  - `review/review_summary.json`

- `python -m scripts.suggest_review_values`
  - `review/disagreements_with_suggestions.csv`
  - `review/high_priority_review_batch.csv`

- `python -m scripts.report_pdf_alignment_issues`
  - `review/pdf_alignment_issues.csv`

- `python -m scripts.sced_review --build-gold --review-csv review/disagreements_with_suggestions.csv --gold-output data/sced_gold_reviewed_v1.jsonl`
  - `data/sced_gold_reviewed_v1.jsonl`

- `python -m scripts.evaluate_review_disagreements --review-csv review/disagreements_with_suggestions.csv --reference reviewed --output review/disagreement_evaluation.json`
  - `review/disagreement_evaluation.json`

- `python -m scripts.split_sced_dataset`
  - `data/sced_gold_train.jsonl`
  - `data/sced_gold_test.jsonl`

- `python -m scripts.run_sced_split_experiment --split test --evaluate`
  - `data/sced_gold_train.jsonl`
  - `data/sced_gold_test.jsonl`
  - `extracted_text/sced_test_results.jsonl`
  - `evaluation_results/sced_test_results_evaluation.json`

## 6. Single-Paper LLM Extraction (Optional)

```python
from scripts.sced_extraction import run_sced_extraction
import json
import pathlib

blocks = json.loads(pathlib.Path("extracted_text/<paper>_blocks.json").read_text())
result = run_sced_extraction(blocks)
print(result)
```

## 7. Full-PDF Single-Paper Extraction (Optional)

```python
from pathlib import Path
from scripts.sced_extraction import run_sced_extraction_from_pdf

result = run_sced_extraction_from_pdf(Path("input/<paper>.pdf"))
print(result)
```

## 8. Evaluation Format

Gold and prediction files should be JSONL with one record per paper and a `pdf` key:

```json
{"pdf": "Taylor 2011.pdf", "Study": "Taylor 2011", "Type of publication": "article", "Country": "UK", "Number of Cases": 1, "Sample type": "Ref", "Setting": "UC", "Primary diagnosis": "AD (specific phobia)", "Type of treatments": "CBT", "Type of SCED design": "multiple baseline design"}
```

Evaluation treats each field as a normalized set of values:
- scalar fields count as one predicted/gold item
- list fields are compared item-by-item
- metrics reported are precision, recall, F1, and exact-match rate
- by default 22 of the 24 fields are scored; `Quality rating RoBiNT scale` (reviewer-assigned) and `Total Number of Observations` (figure-dependent) are excluded with recorded reasons, unless `--include-excluded-fields` is passed
