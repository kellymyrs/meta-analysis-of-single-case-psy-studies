from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_SOURCE = ROOT / "data" / "sced_gold.jsonl"
DEFAULT_REVIEWED_GOLD = ROOT / "data" / "sced_gold_reviewed_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "sced_gold_test_reviewed_frozen.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a frozen reviewed held-out test gold JSONL."
    )
    parser.add_argument(
        "--split-source",
        default=str(DEFAULT_SPLIT_SOURCE),
        help="Source JSONL used to reproduce the training/test PDF split.",
    )
    parser.add_argument(
        "--reviewed-gold",
        default=str(DEFAULT_REVIEWED_GOLD),
        help="Reviewed gold JSONL containing corrected labels.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the frozen reviewed test JSONL.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction assigned to the training split before the held-out test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the original split.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep input order instead of shuffling before splitting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing frozen output file.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


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


def test_pdf_names(
    records: list[dict[str, Any]],
    train_ratio: float,
    seed: int,
    shuffle: bool,
) -> list[str]:
    working_records = list(records)
    if shuffle:
        random.Random(seed).shuffle(working_records)
    train_count = validate_ratio(train_ratio, len(working_records))
    test_records = working_records[train_count:]
    names: list[str] = []
    for record in test_records:
        pdf_name = record.get("pdf")
        if not isinstance(pdf_name, str) or not pdf_name.strip():
            raise ValueError(f"Test record is missing a valid pdf field: {record}")
        names.append(pdf_name)
    return names


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    split_source_path = resolve_path(args.split_source)
    reviewed_gold_path = resolve_path(args.reviewed_gold)
    output_path = resolve_path(args.output)

    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Frozen output already exists: {output_path}. "
            "Use --force only if you intentionally want to create a new frozen version."
        )

    split_records = load_jsonl(split_source_path)
    reviewed_records = load_jsonl(reviewed_gold_path)
    reviewed_by_pdf = {
        record["pdf"]: record
        for record in reviewed_records
        if isinstance(record.get("pdf"), str) and record.get("pdf")
    }

    names = test_pdf_names(
        split_records,
        train_ratio=args.train_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    missing = [name for name in names if name not in reviewed_by_pdf]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} test PDFs are missing from reviewed gold. First missing: {preview}"
        )

    frozen_records = [reviewed_by_pdf[name] for name in names]
    write_jsonl(output_path, frozen_records)

    print(f"Frozen test records: {len(frozen_records)} -> {output_path}")
    print(f"Split source: {split_source_path}")
    print(f"Reviewed gold: {reviewed_gold_path}")
    print(f"Seed: {args.seed}")
    print(f"Shuffled: {not args.no_shuffle}")


if __name__ == "__main__":
    main()
