"""Smarter PDF text extraction with line reconstruction and column-aware ordering.

Usage (from repo root):
    python scripts/pdf_text_blocks_smart.py
    python scripts/pdf_text_blocks_smart.py --pdf "Taylor 2011.pdf"

Behavior:
- Scans `input/` for `.pdf` files unless `--pdf` is provided.
- Reconstructs text from line/span data rather than using raw PyMuPDF blocks.
- Attempts a simple column-aware reading order per page.
- Drops likely header/footer noise near page edges.
- Merges nearby lines into paragraph-like blocks.
- Writes one JSON per PDF to `extracted_text/` named `<paper_name>_smart_blocks.json`.

JSON structure:
{
  "page": 1,
  "bbox": [x0, y0, x1, y1],
  "text": "...",
  "line_count": 4,
  "column": 0
}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import fitz  # PyMuPDF


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "extracted_text"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

WHITESPACE_RE = re.compile(r"\s+")
HYPHEN_LINE_END_RE = re.compile(r"(?<=[A-Za-z])-$")


@dataclass
class Line:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    font_size: float


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\n", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def iter_lines(page: fitz.Page, page_number: int) -> Iterable[Line]:
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            parts: List[str] = []
            font_sizes: List[float] = []
            for span in spans:
                text = normalize_text(span.get("text", ""))
                if not text:
                    continue
                parts.append(text)
                size = span.get("size")
                if isinstance(size, (int, float)):
                    font_sizes.append(float(size))
            if not parts:
                continue
            x0, y0, x1, y1 = line["bbox"]
            yield Line(
                page=page_number,
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                text=" ".join(parts),
                font_size=sum(font_sizes) / len(font_sizes) if font_sizes else 0.0,
            )


def looks_like_margin_noise(line: Line, page_rect: fitz.Rect) -> bool:
    top_margin = page_rect.height * 0.06
    bottom_margin = page_rect.height * 0.94
    short_text = len(line.text) <= 80
    small_font = line.font_size <= 11.5 if line.font_size else True
    if line.y1 <= top_margin and short_text and small_font:
        return True
    if line.y0 >= bottom_margin and len(line.text) <= 140 and small_font:
        return True
    if re.fullmatch(r"\d+", line.text):
        return True
    return False


def assign_columns(lines: List[Line], page_width: float) -> List[List[Line]]:
    if len(lines) < 6:
        return [sorted(lines, key=lambda line: (line.y0, line.x0))]

    mid_x = page_width / 2
    gutter = page_width * 0.05
    spanning = [
        line
        for line in lines
        if (line.x0 < (mid_x - gutter) and line.x1 > (mid_x + gutter))
        or (line.x1 - line.x0) >= (page_width * 0.45)
    ]
    remaining = [line for line in lines if line not in spanning]
    left = [line for line in remaining if line.x0 < (mid_x - gutter) and line.x1 <= (mid_x + gutter)]
    right = [line for line in remaining if line.x0 >= (mid_x - gutter)]
    spill = [line for line in remaining if line not in left and line not in right]

    if not left or not right:
        merged = sorted(lines, key=lambda line: (line.y0, line.x0))
        return [merged]

    for line in spill:
        left_gap = abs(line.x0 - (sum(item.x0 for item in left) / len(left)))
        right_gap = abs(line.x0 - (sum(item.x0 for item in right) / len(right)))
        if left_gap <= right_gap:
            left.append(line)
        else:
            right.append(line)

    ordered_groups = [group for group in (spanning, left, right) if group]
    return [sorted(group, key=lambda line: (line.y0, line.x0)) for group in ordered_groups]


def merge_lines_to_blocks(lines: List[Line], column_index: int) -> List[Dict[str, Any]]:
    if not lines:
        return []

    blocks: List[Dict[str, Any]] = []
    current_lines: List[Line] = [lines[0]]

    for line in lines[1:]:
        prev = current_lines[-1]
        vertical_gap = line.y0 - prev.y1
        indent_shift = abs(line.x0 - prev.x0)
        median_font = max((prev.font_size + line.font_size) / 2, 1.0)

        same_paragraph = vertical_gap <= max(10.0, median_font * 0.9) and indent_shift <= 24.0
        if same_paragraph:
            current_lines.append(line)
            continue

        blocks.append(build_block(current_lines, column_index))
        current_lines = [line]

    blocks.append(build_block(current_lines, column_index))
    return blocks


def build_block(lines: List[Line], column_index: int) -> Dict[str, Any]:
    merged_parts: List[str] = []
    for line in lines:
        if merged_parts and HYPHEN_LINE_END_RE.search(merged_parts[-1]):
            merged_parts[-1] = HYPHEN_LINE_END_RE.sub("", merged_parts[-1]) + line.text
        else:
            merged_parts.append(line.text)

    text = " ".join(part.strip() for part in merged_parts if part.strip())
    return {
        "page": lines[0].page,
        "bbox": [
            min(line.x0 for line in lines),
            min(line.y0 for line in lines),
            max(line.x1 for line in lines),
            max(line.y1 for line in lines),
        ],
        "text": normalize_text(text),
        "line_count": len(lines),
        "column": column_index,
    }


def extract_blocks_smart(pdf_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            page_rect = page.rect
            page_lines = [
                line
                for line in iter_lines(page, page_number)
                if line.text and not looks_like_margin_noise(line, page_rect)
            ]
            columns = assign_columns(page_lines, page_rect.width)
            for column_index, column_lines in enumerate(columns):
                entries.extend(merge_lines_to_blocks(column_lines, column_index))
    return [entry for entry in entries if entry["text"]]


def save_json(data: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_pdf(pdf_path: Path) -> None:
    logging.info("Smart-extracting %s", pdf_path.name)
    blocks = extract_blocks_smart(pdf_path)
    if not blocks:
        logging.warning("No text blocks found in %s", pdf_path.name)
        return
    out_name = f"{pdf_path.stem.replace(' ', '_')}_smart_blocks.json"
    out_path = OUTPUT_DIR / out_name
    save_json(blocks, out_path)
    logging.info("Saved %s", out_path.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smarter PDF text extraction with reconstructed reading order.")
    parser.add_argument("--pdf", help="Optional exact PDF filename in input/ to process.")
    return parser.parse_args()


def resolve_pdf_files(selected_pdf: str | None) -> List[Path]:
    if selected_pdf:
        pdf_path = INPUT_DIR / selected_pdf
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        return [pdf_path]
    return sorted(INPUT_DIR.glob("*.pdf"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    try:
        pdf_files = resolve_pdf_files(args.pdf)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return

    if not pdf_files:
        logging.warning("No PDF files found in %s", INPUT_DIR)
        return

    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_path.name, exc)


if __name__ == "__main__":
    main()
