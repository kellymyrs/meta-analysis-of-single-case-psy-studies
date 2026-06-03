from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "sced_gold.jsonl"
DEFAULT_TRAIN_OUTPUT = ROOT / "data" / "sced_gold_train.jsonl"
DEFAULT_TEST_OUTPUT = ROOT / "data" / "sced_gold_test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the SCED gold dataset into train and test JSONL files."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the source JSONL dataset.",
    )
    parser.add_argument(
        "--train-output",
        default=str(DEFAULT_TRAIN_OUTPUT),
        help="Path to write the training JSONL file.",
    )
    parser.add_argument(
        "--test-output",
        default=str(DEFAULT_TEST_OUTPUT),
        help="Path to write the test JSONL file.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of records to place in the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before splitting.",
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
    train_count = max(1, min(train_count, total_records - 1))
    return train_count


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    train_output = Path(args.train_output)
    test_output = Path(args.test_output)

    records = load_jsonl(input_path)
    working_records = list(records)
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(working_records)

    train_count = validate_ratio(args.train_ratio, len(working_records))
    train_records = working_records[:train_count]
    test_records = working_records[train_count:]

    write_jsonl(train_output, train_records)
    write_jsonl(test_output, test_records)

    print(f"Input records: {len(working_records)}")
    print(f"Training records: {len(train_records)} -> {train_output}")
    print(f"Test records: {len(test_records)} -> {test_output}")
    print(f"Seed: {args.seed}")
    print(f"Shuffled: {not args.no_shuffle}")


if __name__ == "__main__":
    main()
