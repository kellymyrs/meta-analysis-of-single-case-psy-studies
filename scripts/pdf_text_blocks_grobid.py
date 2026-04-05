"""Extract structured study/article text from PDFs using a GROBID server.

Usage (from repo root):
    python scripts/pdf_text_blocks_grobid.py
    python scripts/pdf_text_blocks_grobid.py --pdf "Taylor 2011.pdf"

Requirements:
- A running GROBID service, by default at http://localhost:8070

Behavior:
- Sends each PDF in `input/` to GROBID's `/api/processFulltextDocument`.
- Saves the raw TEI XML to `extracted_text/<paper>_grobid.tei.xml`.
- Parses the TEI into structured JSON blocks and writes
  `extracted_text/<paper>_grobid_blocks.json`.

The JSON output is study-oriented and keeps article structure such as title,
abstract, section headings, paragraphs, figure/table captions, and references.
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "extracted_text"
DEFAULT_GROBID_URL = "http://localhost:8070"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def iter_text(node: ET.Element) -> Iterable[str]:
    if node.text:
        yield node.text
    for child in node:
        yield from iter_text(child)
        if child.tail:
            yield child.tail


def node_text(node: ET.Element) -> str:
    return normalize_text("".join(iter_text(node)))


def make_multipart_body(
    file_field_name: str,
    file_path: Path,
    extra_fields: Dict[str, str],
) -> tuple[bytes, str]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    for key, value in extra_fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/pdf"
    file_bytes = file_path.read_bytes()
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field_name}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


def grobid_process_fulltext(
    pdf_path: Path,
    grobid_url: str,
    timeout: int = 180,
) -> str:
    body, boundary = make_multipart_body(
        file_field_name="input",
        file_path=pdf_path,
        extra_fields={
            "consolidateHeader": "0",
            "consolidateCitations": "0",
            "includeRawCitations": "1",
        },
    )
    endpoint = urljoin(grobid_url.rstrip("/") + "/", "api/processFulltextDocument")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GROBID HTTP {exc.code} for {pdf_path.name}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach GROBID at {grobid_url}. Start the server and retry."
        ) from exc


def tei_texts(nodes: Iterable[ET.Element], block_type: str, section: str | None = None) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for order, node in enumerate(nodes, start=1):
        text = node_text(node)
        if not text:
            continue
        block: Dict[str, Any] = {
            "block_type": block_type,
            "text": text,
            "order": order,
        }
        if section:
            block["section"] = section
        blocks.append(block)
    return blocks


def parse_tei_to_blocks(tei_xml: str) -> Dict[str, Any]:
    root = ET.fromstring(tei_xml)

    title = node_text(root.find(".//tei:titleStmt/tei:title", NS) or ET.Element("empty"))
    abstract_blocks = tei_texts(root.findall(".//tei:profileDesc/tei:abstract//tei:p", NS), "abstract")

    body_blocks: List[Dict[str, Any]] = []
    divs = root.findall(".//tei:text/tei:body//tei:div", NS)
    body_order = 0
    for div in divs:
        head = div.find("./tei:head", NS)
        head_text = node_text(head) if head is not None else ""
        if head_text:
            body_order += 1
            body_blocks.append(
                {
                    "block_type": "section_heading",
                    "section": head_text,
                    "text": head_text,
                    "order": body_order,
                }
            )
        for paragraph in div.findall("./tei:p", NS):
            paragraph_text = node_text(paragraph)
            if not paragraph_text:
                continue
            body_order += 1
            block: Dict[str, Any] = {
                "block_type": "paragraph",
                "text": paragraph_text,
                "order": body_order,
            }
            if head_text:
                block["section"] = head_text
            body_blocks.append(block)

        for figure in div.findall("./tei:figure", NS):
            figure_text = node_text(figure)
            if not figure_text:
                continue
            body_order += 1
            block = {
                "block_type": "figure_or_table",
                "text": figure_text,
                "order": body_order,
            }
            if head_text:
                block["section"] = head_text
            body_blocks.append(block)

    if not body_blocks:
        paragraphs = root.findall(".//tei:text/tei:body//tei:p", NS)
        body_blocks = tei_texts(paragraphs, "paragraph")

    references: List[Dict[str, Any]] = []
    for order, ref in enumerate(root.findall(".//tei:listBibl/tei:biblStruct", NS), start=1):
        reference_text = node_text(ref)
        if not reference_text:
            continue
        references.append({"block_type": "reference", "text": reference_text, "order": order})

    keywords = [
        node_text(keyword)
        for keyword in root.findall(".//tei:profileDesc//tei:keywords//tei:term", NS)
        if node_text(keyword)
    ]

    authors = []
    for author in root.findall(".//tei:titleStmt//tei:author", NS):
        author_text = node_text(author)
        if author_text:
            authors.append(author_text)

    return {
        "title": title or None,
        "authors": authors,
        "keywords": keywords,
        "abstract_blocks": abstract_blocks,
        "body_blocks": body_blocks,
        "reference_blocks": references,
    }


def process_pdf(pdf_path: Path, grobid_url: str) -> None:
    logging.info("GROBID-extracting %s", pdf_path.name)
    tei_xml = grobid_process_fulltext(pdf_path, grobid_url=grobid_url)
    parsed = parse_tei_to_blocks(tei_xml)

    tei_out = OUTPUT_DIR / f"{pdf_path.stem.replace(' ', '_')}_grobid.tei.xml"
    json_out = OUTPUT_DIR / f"{pdf_path.stem.replace(' ', '_')}_grobid_blocks.json"
    tei_out.write_text(tei_xml, encoding="utf-8")
    json_out.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved %s", tei_out.relative_to(ROOT))
    logging.info("Saved %s", json_out.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured PDF text using GROBID.")
    parser.add_argument("--pdf", help="Optional exact PDF filename in input/ to process.")
    parser.add_argument(
        "--grobid-url",
        default=DEFAULT_GROBID_URL,
        help=f"GROBID base URL. Default: {DEFAULT_GROBID_URL}",
    )
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
            process_pdf(pdf_path, grobid_url=args.grobid_url)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed on %s: %s", pdf_path.name, exc)


if __name__ == "__main__":
    main()
