from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "input"
TEXT_DIR = ROOT / "extracted_text"
EVALUATION_DIR = ROOT / "evaluation_results"
DEFAULT_INPUT = DATA_DIR / "sced_gold.jsonl"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the SCED gold dataset and run the extraction model on one split."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the source gold JSONL dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of records assigned to the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before splitting.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Which split to run the model on.",
    )
    parser.add_argument(
        "--mode",
        choices=("blocks", "full_text", "full_pdf"),
        default="blocks",
        help="Extraction mode to use.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the predictions against the selected split after extraction.",
    )
    parser.add_argument(
        "--evaluation-dir",
        help="Optional directory for evaluation JSON outputs. Defaults to evaluation_results/.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep input order instead of shuffling before splitting.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def validate_ratio(train_ratio: float, total_records: int) -> int:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    train_count = int(total_records * train_ratio)
    return max(1, min(train_count, total_records - 1))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, Any]], train_ratio: float, seed: int, shuffle: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    working_records = list(records)
    if shuffle:
        random.Random(seed).shuffle(working_records)
    train_count = validate_ratio(train_ratio, len(working_records))
    return working_records[:train_count], working_records[train_count:]


def result_suffix(mode: str) -> str:
    return "" if mode == "blocks" else f"_{mode}"


def evaluation_output_path(predictions_path: Path, evaluation_dir: str | None) -> Path:
    if not evaluation_dir:
        return EVALUATION_DIR / f"{predictions_path.stem}_evaluation.json"
    directory = Path(evaluation_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    return directory / f"{predictions_path.stem}_evaluation.json"


def ensure_blocks(pdf_path: Path) -> list[dict[str, Any]]:
    from scripts.pdf_text_blocks import extract_blocks

    TEXT_DIR.mkdir(exist_ok=True)
    blocks_path = TEXT_DIR / f"{pdf_path.stem}_blocks.json"
    if blocks_path.exists():
        return json.loads(blocks_path.read_text(encoding="utf-8"))
    blocks = extract_blocks(pdf_path)
    blocks_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Created %s", blocks_path.relative_to(ROOT))
    return blocks


def run_model_for_pdf(pdf_name: str, mode: str) -> dict[str, Any] | None:
    from scripts.sced_extraction import (
        run_sced_extraction,
        run_sced_extraction_from_pdf,
        run_sced_extraction_full_text,
    )

    pdf_path = PDF_DIR / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if mode == "blocks":
        blocks = ensure_blocks(pdf_path)
        return run_sced_extraction(blocks)
    if mode == "full_text":
        blocks = ensure_blocks(pdf_path)
        return run_sced_extraction_full_text(blocks)
    return run_sced_extraction_from_pdf(pdf_path)


def main() -> None:
    args = parse_args()
    from scripts.evaluate_sced_results import evaluate_jsonl_files
    from scripts.sced_extraction import get_runtime_backend

    try:
        backend = get_runtime_backend()
    except RuntimeError as exc:
        logging.error(str(exc))
        return

    if args.mode == "full_pdf" and backend != "proxy":
        logging.error("full_pdf mode requires proxy mode with LITELLM_KEY configured.")
        return

    input_path = Path(args.input)
    all_records = load_jsonl(input_path)
    train_records, test_records = split_records(
        all_records,
        train_ratio=args.train_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    train_path = DATA_DIR / "sced_gold_train.jsonl"
    test_path = DATA_DIR / "sced_gold_test.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)

    selected_records = train_records if args.split == "train" else test_records
    selected_gold_path = train_path if args.split == "train" else test_path

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = result_suffix(args.mode)
    predictions_path = TEXT_DIR / f"sced_{args.split}_results{suffix}.jsonl"
    predictions_path.write_text("", encoding="utf-8")

    logging.info("Using LLM backend: %s", backend)
    logging.info("Extraction mode: %s", args.mode)
    logging.info("Train records: %d", len(train_records))
    logging.info("Test records: %d", len(test_records))
    logging.info("Running model on %s split (%d PDFs)", args.split, len(selected_records))

    lines_written = 0
    for record in selected_records:
        pdf_name = record.get("pdf")
        if not isinstance(pdf_name, str) or not pdf_name.strip():
            logging.warning("Skipping record without a valid pdf field: %s", record)
            continue
        logging.info("Processing %s", pdf_name)
        try:
            result = run_model_for_pdf(pdf_name, args.mode)
            if result is None:
                logging.warning("No result for %s", pdf_name)
                continue
            out_path = TEXT_DIR / f"{Path(pdf_name).stem}_sced{suffix}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pdf": pdf_name, **result}, ensure_ascii=False) + "\n")
            lines_written += 1
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_name, exc)

    logging.info("Saved predictions to %s", predictions_path.relative_to(ROOT))

    if args.evaluate:
        evaluation = evaluate_jsonl_files(predictions_path, selected_gold_path)
        evaluation_path = evaluation_output_path(predictions_path, args.evaluation_dir)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        overall = evaluation["overall"]
        logging.info("Evaluation precision: %.3f", overall["precision"])
        logging.info("Evaluation recall: %.3f", overall["recall"])
        logging.info("Evaluation F1: %.3f", overall["f1"])
        logging.info("Saved evaluation summary to %s", evaluation_path.relative_to(ROOT))

    logging.info("Done. Wrote %d prediction lines.", lines_written)


if __name__ == "__main__":
    main()
