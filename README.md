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

11. Split the SCED gold dataset into train/test JSONL files:
   ```bash
   python -m scripts.split_sced_dataset
   ```

12. Split the SCED gold dataset and run the model on one split:
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
  - What it does: provides `run_sced_extraction()` for truncated block-text extraction, `run_sced_extraction_full_text()` for whole-paper chunked text extraction, and `run_sced_extraction_from_pdf()` for full-PDF extraction via the Responses API.
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

- `scripts/evaluate_sced_results.py`
  - What it does: compares predicted SCED JSONL records against gold-standard JSONL records and reports micro-averaged precision, recall, F1, per-field metrics, and per-paper details.
  - Reads: a predictions JSONL file and a gold JSONL file with one record per PDF
  - Writes: `evaluation_results/<predictions>_evaluation.json` by default
  - Run: `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl`
  - Custom output path: `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl --output custom_eval_dir/sced_results_evaluation.json`

- `scripts/split_sced_dataset.py`
  - What it does: splits `data/sced_gold.jsonl` into deterministic train/test JSONL files for downstream experiments.
  - Reads: `data/sced_gold.jsonl`
  - Writes: `data/sced_gold_train.jsonl`, `data/sced_gold_test.jsonl`
  - Run: `python -m scripts.split_sced_dataset`
  - Custom split: `python -m scripts.split_sced_dataset --train-ratio 0.75 --seed 7`

- `scripts/run_sced_split_experiment.py`
  - What it does: creates the train/test split and runs SCED extraction over either the train or test split, with optional evaluation.
  - Reads: `data/sced_gold.jsonl`, PDFs in `input/`
  - Writes: `data/sced_gold_train.jsonl`, `data/sced_gold_test.jsonl`, `extracted_text/sced_train_results*.jsonl` or `extracted_text/sced_test_results*.jsonl`, and evaluation summaries in `evaluation_results/` by default
  - Run: `python -m scripts.run_sced_split_experiment --split test`
  - With evaluation: `python -m scripts.run_sced_split_experiment --split test --evaluate`
  - Optional custom evaluation folder: `python -m scripts.run_sced_split_experiment --split test --evaluate --evaluation-dir custom_eval_dir`
  - Full-PDF mode: `python -m scripts.run_sced_split_experiment --split test --mode full_pdf --evaluate`

## 4. LLM Model Configuration

The SCED extractor supports two modes.

1. Proxy mode for hosted models and full-PDF input:
   ```bash
   export LITELLM_KEY=your_token_here
   export LITELLM_MODEL=gpt-5.1
   # optional
   export LITELLM_BASE_URL=https://ai-research-proxy.azurewebsites.net
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

- `python -m scripts.evaluate_sced_results --predictions extracted_text/sced_results.jsonl --gold data/sced_gold.jsonl`
  - `evaluation_results/sced_results_evaluation.json`

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
{"pdf": "Taylor 2011.pdf", "Participant ID": "P1", "Baseline Mean": 12.4, "Treatment Phase Slope": -0.8, "Clinical Contradictions": ["No contradiction reported"]}
```

Evaluation treats each field as a normalized set of values:
- scalar fields count as one predicted/gold item
- list fields are compared item-by-item
- metrics reported are precision, recall, F1, and exact-match rate
