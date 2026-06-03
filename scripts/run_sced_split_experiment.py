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
DEFAULT_FROZEN_TEST_GOLD = DATA_DIR / "sced_gold_test_reviewed_frozen.jsonl"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the SCED gold dataset and run zero-shot or few-shot extraction "
            "on the training or held-out test split."
        )
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
        "--full-pdf-context-fallback",
        choices=("none", "full_text", "blocks"),
        default="none",
        help=(
            "Fallback mode to use when --mode full_pdf exceeds the model context window. "
            "Use 'full_text' to retry only oversized PDFs with extracted text chunks."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the predictions against the selected split after extraction.",
    )
    parser.add_argument(
        "--model-setup",
        choices=("zero_shot", "few_shot", "both"),
        default="zero_shot",
        help="Prompt setup to run. Few-shot examples are drawn only from the training split.",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=3,
        help="Number of training-set examples to include for few-shot extraction.",
    )
    parser.add_argument(
        "--few-shot-pdf",
        action="append",
        default=[],
        help=(
            "Exact training-set PDF filename to use as a few-shot example. "
            "Can be passed multiple times. When set, these examples are used instead "
            "of randomly selected training records."
        ),
    )
    parser.add_argument(
        "--few-shot-seed",
        type=int,
        help=(
            "Random seed for selecting few-shot examples from the training split. "
            "Defaults to --seed."
        ),
    )
    parser.add_argument(
        "--frozen-test-gold",
        default=str(DEFAULT_FROZEN_TEST_GOLD),
        help=(
            "Frozen reviewed test JSONL used for final test evaluation. "
            "Required when --final-evaluation is set."
        ),
    )
    parser.add_argument(
        "--final-evaluation",
        action="store_true",
        help=(
            "Require held-out test evaluation against --frozen-test-gold. "
            "Use after manual review and label freezing."
        ),
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


def result_suffix(mode: str, full_pdf_context_fallback: str = "none") -> str:
    suffix = "" if mode == "blocks" else f"_{mode}"
    if mode == "full_pdf" and full_pdf_context_fallback != "none":
        suffix = f"{suffix}_context_fallback_{full_pdf_context_fallback}"
    return suffix


def model_setup_suffix(model_setup: str) -> str:
    return f"_{model_setup}"


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


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def select_few_shot_examples(
    training_records: list[dict[str, Any]],
    count: int,
    selected_pdfs: list[str] | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("--few-shot-count must be >= 0.")
    if selected_pdfs:
        records_by_pdf = {
            record["pdf"]: record
            for record in training_records
            if isinstance(record.get("pdf"), str) and record.get("pdf")
        }
        missing = [pdf_name for pdf_name in selected_pdfs if pdf_name not in records_by_pdf]
        if missing:
            preview = ", ".join(missing)
            raise ValueError(
                "Requested few-shot PDFs are not in the training split: "
                f"{preview}"
            )
        return [records_by_pdf[pdf_name] for pdf_name in selected_pdfs]
    if count == 0:
        return []
    examples = list(training_records)
    random.Random(seed).shuffle(examples)
    return examples[:count]


def run_model_for_pdf(
    pdf_name: str,
    mode: str,
    few_shot_examples: list[dict[str, Any]] | None = None,
    full_pdf_context_fallback: str = "none",
) -> dict[str, Any] | None:
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
        return run_sced_extraction(blocks, few_shot_examples=few_shot_examples)
    if mode == "full_text":
        blocks = ensure_blocks(pdf_path)
        return run_sced_extraction_full_text(blocks, few_shot_examples=few_shot_examples)
    try:
        return run_sced_extraction_from_pdf(pdf_path, few_shot_examples=few_shot_examples)
    except Exception as exc:
        if full_pdf_context_fallback == "none" or "context_length_exceeded" not in str(exc):
            raise
        logging.warning(
            "Full-PDF extraction exceeded context for %s; retrying with %s fallback.",
            pdf_name,
            full_pdf_context_fallback,
        )
        blocks = ensure_blocks(pdf_path)
        if full_pdf_context_fallback == "full_text":
            return run_sced_extraction_full_text(blocks, few_shot_examples=few_shot_examples)
        return run_sced_extraction(blocks, few_shot_examples=few_shot_examples)


def run_selected_model_setup(
    *,
    selected_records: list[dict[str, Any]],
    selected_split_name: str,
    mode: str,
    model_setup: str,
    training_records: list[dict[str, Any]],
    evaluate: bool,
    selected_gold_path: Path,
    evaluation_dir: str | None,
    few_shot_count: int,
    few_shot_pdfs: list[str],
    few_shot_seed: int,
    full_pdf_context_fallback: str,
) -> None:
    from scripts.evaluate_sced_results import evaluate_jsonl_files

    few_shot_examples: list[dict[str, Any]] | None = None
    if model_setup == "few_shot":
        few_shot_examples = select_few_shot_examples(
            training_records,
            count=few_shot_count,
            selected_pdfs=few_shot_pdfs,
            seed=few_shot_seed,
        )
        logging.info("Few-shot examples: %d training records", len(few_shot_examples))
        for example in few_shot_examples:
            logging.info("Few-shot example: %s", example.get("pdf"))

    suffix = f"{model_setup_suffix(model_setup)}{result_suffix(mode, full_pdf_context_fallback)}"
    predictions_path = TEXT_DIR / f"sced_{selected_split_name}_results{suffix}.jsonl"
    predictions_path.write_text("", encoding="utf-8")

    logging.info("Model setup: %s", model_setup)
    logging.info("Extraction mode: %s", mode)
    logging.info("Running model on %s split (%d PDFs)", selected_split_name, len(selected_records))

    lines_written = 0
    for record in selected_records:
        pdf_name = record.get("pdf")
        if not isinstance(pdf_name, str) or not pdf_name.strip():
            logging.warning("Skipping record without a valid pdf field: %s", record)
            continue
        logging.info("Processing %s", pdf_name)
        try:
            result = run_model_for_pdf(
                pdf_name,
                mode,
                few_shot_examples=few_shot_examples,
                full_pdf_context_fallback=full_pdf_context_fallback,
            )
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

    if evaluate:
        evaluation = evaluate_jsonl_files(predictions_path, selected_gold_path)
        evaluation_path = evaluation_output_path(predictions_path, evaluation_dir)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        overall = evaluation["overall"]
        logging.info("Evaluation precision: %.3f", overall["precision"])
        logging.info("Evaluation recall: %.3f", overall["recall"])
        logging.info("Evaluation F1: %.3f", overall["f1"])
        logging.info("Gold reference: %s", selected_gold_path.relative_to(ROOT))
        logging.info("Saved evaluation summary to %s", evaluation_path.relative_to(ROOT))

    logging.info("Done with %s. Wrote %d prediction lines.", model_setup, lines_written)


def main() -> None:
    args = parse_args()

    if args.final_evaluation and args.split != "test":
        logging.error("--final-evaluation requires --split test.")
        return

    input_path = Path(args.input)
    all_records = load_jsonl(input_path)
    training_records, test_records = split_records(
        all_records,
        train_ratio=args.train_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    train_path = DATA_DIR / "sced_gold_train.jsonl"
    test_path = DATA_DIR / "sced_gold_test.jsonl"
    write_jsonl(train_path, training_records)
    write_jsonl(test_path, test_records)

    selected_split_name = args.split
    selected_records = training_records if selected_split_name == "train" else test_records
    selected_gold_path = train_path if selected_split_name == "train" else test_path

    if args.final_evaluation:
        frozen_gold_path = resolve_project_path(args.frozen_test_gold)
        if not frozen_gold_path.exists():
            logging.error(
                "Frozen reviewed test gold file not found: %s. "
                "Create it after manual review before final evaluation.",
                frozen_gold_path,
            )
            return
        selected_gold_path = frozen_gold_path

    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Training records: %d", len(training_records))
    logging.info("Test records: %d", len(test_records))
    logging.info("Evaluation gold: %s", selected_gold_path.relative_to(ROOT))

    from scripts.sced_extraction import get_runtime_backend

    try:
        backend = get_runtime_backend()
    except RuntimeError as exc:
        logging.error(str(exc))
        return

    if args.mode == "full_pdf" and backend != "proxy":
        logging.error("full_pdf mode requires proxy mode with LITELLM_KEY configured.")
        return

    logging.info("Using LLM backend: %s", backend)

    setups = ["zero_shot", "few_shot"] if args.model_setup == "both" else [args.model_setup]
    for setup in setups:
        run_selected_model_setup(
            selected_records=selected_records,
            selected_split_name=selected_split_name,
            mode=args.mode,
            model_setup=setup,
            training_records=training_records,
            evaluate=args.evaluate or args.final_evaluation,
            selected_gold_path=selected_gold_path,
            evaluation_dir=args.evaluation_dir,
            few_shot_count=args.few_shot_count,
            few_shot_pdfs=args.few_shot_pdf,
            few_shot_seed=args.few_shot_seed if args.few_shot_seed is not None else args.seed,
            full_pdf_context_fallback=args.full_pdf_context_fallback,
        )


if __name__ == "__main__":
    main()
