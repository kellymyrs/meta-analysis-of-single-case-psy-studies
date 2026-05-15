"""Report likely PDF-to-gold alignment problems.

The SCED review workflow assumes each `pdf` name points to the paper described
by the legacy gold row. If two input filenames contain the same PDF, or if the
legacy study author is absent from the first pages of the extracted text, many
field disagreements are likely caused by file alignment rather than extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sced_fields import normalize_field_keys


DEFAULT_GOLD = ROOT / "data" / "sced_gold.jsonl"
DEFAULT_INPUT_DIR = ROOT / "input"
DEFAULT_BLOCKS_DIR = ROOT / "extracted_text"
DEFAULT_OUTPUT = ROOT / "review" / "pdf_alignment_issues.csv"

OUTPUT_COLUMNS = [
    "issue_type",
    "pdf",
    "study",
    "related_pdf",
    "related_study",
    "pdf_sha256",
    "evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report likely SCED PDF alignment issues.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Path to legacy gold JSONL.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory with PDF files.")
    parser.add_argument(
        "--blocks-dir",
        default=str(DEFAULT_BLOCKS_DIR),
        help="Directory with *_blocks.json extracted text files.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write CSV report.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def file_key(path_or_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(path_or_name).stem.lower()).strip("_")


def study_author(study: Any) -> str:
    if study is None:
        return ""
    first = str(study).strip().split(maxsplit=1)[0]
    return normalize_text(first)


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = normalize_field_keys(json.loads(line))
            pdf = str(record.get("pdf", "")).strip()
            if pdf:
                records[pdf] = record
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_blocks_map(blocks_dir: Path) -> dict[str, Path]:
    return {
        file_key(path.name.replace("_blocks.json", "")): path
        for path in blocks_dir.glob("*_blocks.json")
    }


def find_blocks_path(pdf: str, blocks_map: dict[str, Path]) -> Path | None:
    key = file_key(pdf)
    if key in blocks_map:
        return blocks_map[key]
    for candidate_key, path in blocks_map.items():
        if candidate_key.startswith(key[:50]) or key.startswith(candidate_key[:50]):
            return path
    return None


def first_pages_text(blocks_path: Path, max_page: int = 2) -> str:
    blocks = json.loads(blocks_path.read_text(encoding="utf-8"))
    text_parts = [
        str(block.get("text", ""))
        for block in blocks
        if int(block.get("page") or 0) <= max_page
    ]
    return normalize_text(" ".join(text_parts))


def duplicate_pdf_rows(
    gold: dict[str, dict[str, Any]],
    input_dir: Path,
) -> list[dict[str, str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    for path in input_dir.glob("*.pdf"):
        hashes[sha256(path)].append(path.name)

    rows: list[dict[str, str]] = []
    for digest, pdfs in sorted(hashes.items()):
        if len(pdfs) < 2:
            continue
        related = [
            f"{pdf} ({gold.get(pdf, {}).get('Study', '')})"
            for pdf in sorted(pdfs)
        ]
        for pdf in sorted(pdfs):
            related_for_pdf = [item for item in related if not item.startswith(f"{pdf} (")]
            rows.append(
                {
                    "issue_type": "duplicate_pdf_file",
                    "pdf": pdf,
                    "study": str(gold.get(pdf, {}).get("Study", "")),
                    "related_pdf": "; ".join(related_for_pdf),
                    "related_study": "",
                    "pdf_sha256": digest,
                    "evidence": "Same SHA-256 hash as another input PDF filename.",
                }
            )
    return rows


def author_mismatch_rows(
    gold: dict[str, dict[str, Any]],
    input_dir: Path,
    blocks_dir: Path,
) -> list[dict[str, str]]:
    blocks_map = build_blocks_map(blocks_dir)
    rows: list[dict[str, str]] = []
    for pdf, record in sorted(gold.items()):
        pdf_path = input_dir / pdf
        if not pdf_path.exists():
            rows.append(
                {
                    "issue_type": "missing_input_pdf",
                    "pdf": pdf,
                    "study": str(record.get("Study", "")),
                    "related_pdf": "",
                    "related_study": "",
                    "pdf_sha256": "",
                    "evidence": "No matching PDF file found in input directory.",
                }
            )
            continue

        author = study_author(record.get("Study"))
        if not author:
            continue
        blocks_path = find_blocks_path(pdf, blocks_map)
        if blocks_path is None:
            continue
        text = first_pages_text(blocks_path)
        if not text:
            rows.append(
                {
                    "issue_type": "empty_extracted_blocks",
                    "pdf": pdf,
                    "study": str(record.get("Study", "")),
                    "related_pdf": "",
                    "related_study": "",
                    "pdf_sha256": sha256(pdf_path),
                    "evidence": f"No text found in the first two pages of {blocks_path.name}.",
                }
            )
            continue
        if author and author not in text:
            rows.append(
                {
                    "issue_type": "legacy_author_not_in_first_pages",
                    "pdf": pdf,
                    "study": str(record.get("Study", "")),
                    "related_pdf": "",
                    "related_study": "",
                    "pdf_sha256": sha256(pdf_path),
                    "evidence": (
                        f"Legacy study author token '{author}' was not found in "
                        f"the first two pages of {blocks_path.name}."
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gold_path = Path(args.gold)
    input_dir = Path(args.input_dir)
    blocks_dir = Path(args.blocks_dir)
    output = Path(args.output)
    if not gold_path.is_absolute():
        gold_path = ROOT / gold_path
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    if not blocks_dir.is_absolute():
        blocks_dir = ROOT / blocks_dir
    if not output.is_absolute():
        output = ROOT / output

    gold = load_gold(gold_path)
    rows = duplicate_pdf_rows(gold, input_dir)
    rows.extend(author_mismatch_rows(gold, input_dir, blocks_dir))
    write_csv(output, rows)
    print(f"Wrote {len(rows)} likely alignment issue rows to {output}")


if __name__ == "__main__":
    main()
