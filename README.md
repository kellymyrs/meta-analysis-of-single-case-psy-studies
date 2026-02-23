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

3. Extract figure/chart pages as 300 DPI PNGs:
   ```bash
   python scripts/pdf_to_images.py
   ```

4. Optional metadata EDA from JSON files (or Excel fallback):
   ```bash
   python scripts/eda_json.py
   ```

5. Run SCED LLM extraction across all papers:
   ```bash
   python -m scripts.batch_sced_analysis
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

- `scripts/eda_json.py`
  - What it does: calculates metadata frequencies and missing-value reports from JSON study files; if no JSON is found, falls back to the Excel file.
  - Reads: `input/*.json` (preferred), fallback `data/Supplement 3. Case and study characteristics.xlsx`
  - Writes: `eda_output/intervention_types_frequency.csv`, `eda_output/demographics_frequency.csv`, `eda_output/study_phase_distribution.png`, `eda_output/missing_values_report.txt`
  - Run: `python scripts/eda_json.py`

- `scripts/sced_extraction.py`
  - What it does: provides `run_sced_extraction()` for LLM extraction of SCED fields from block JSON.
  - Reads: block JSON objects (from `pdf_text_blocks.py`)
  - Writes: none by itself (used by other scripts)
  - Model config: local GGUF via `MODEL_PATH` (plus optional `LLAMA_THREADS`, `LLAMA_CTX`)

- `scripts/batch_sced_analysis.py`
  - What it does: runs SCED extraction across all PDFs, reusing existing block JSONs or creating them if missing.
  - Reads: PDFs in `input/`, optional existing `extracted_text/<paper>_blocks.json`
  - Writes: `extracted_text/<paper>_sced.json`, `extracted_text/sced_results.jsonl`
  - Run: `python -m scripts.batch_sced_analysis`

## 4. LLM Model Configuration

The SCED extractor uses one local `llama.cpp` model only.

1. Set your GGUF model path:
   ```bash
   export MODEL_PATH=/absolute/path/to/model.gguf
   ```
   Verify it exists:
   ```bash
   ls "$MODEL_PATH"
   ```

2. Optional performance tuning:
   ```bash
   export LLAMA_THREADS=8
   export LLAMA_CTX=4096
   ```

3. Run extraction:
   ```bash
   python -m scripts.batch_sced_analysis
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

## 6. Single-Paper LLM Extraction (Optional)

```python
from scripts.sced_extraction import run_sced_extraction
import json
import pathlib

blocks = json.loads(pathlib.Path("extracted_text/<paper>_blocks.json").read_text())
result = run_sced_extraction(blocks)
print(result)
```
