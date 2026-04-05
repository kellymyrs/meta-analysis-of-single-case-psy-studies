"""
SCED variable extraction with either:
1) AI research proxy via OpenAI-compatible API, or
2) local GGUF model via llama.cpp fallback.

Usage example:
    from scripts.sced_extraction import run_sced_extraction
    import json, pathlib
    blocks = json.loads(pathlib.Path("extracted_text/example_blocks.json").read_text())
    result = run_sced_extraction(blocks)
    print(result)

Environment configuration (proxy mode, preferred):
    LITELLM_KEY       = <proxy token>                        (required for proxy mode)
    LITELLM_BASE_URL  = https://ai-research-proxy.azurewebsites.net (optional)
    LITELLM_MODEL     = nf-gpt-4o                            (optional)

Environment configuration (local fallback):
    MODEL_PATH        = /absolute/path/to/model.gguf         (required for local fallback)
    LLAMA_THREADS     = 4                                    (optional)
    LLAMA_CTX         = 4096                                 (optional)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from llama_cpp import Llama


_LLM: Optional[Llama] = None
_CLIENT: Optional[OpenAI] = None


def get_runtime_backend() -> str:
    """Return 'proxy' if proxy key is configured, else 'local' if MODEL_PATH is configured."""
    if os.getenv("LITELLM_KEY", "").strip():
        return "proxy"
    if os.getenv("MODEL_PATH", "").strip():
        return "local"
    raise RuntimeError("Set either LITELLM_KEY (proxy mode) or MODEL_PATH (local GGUF mode).")


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("LITELLM_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LITELLM_KEY env var is required for proxy mode.")

    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LITELLM_BASE_URL", "https://ai-research-proxy.azurewebsites.net").strip(),
    )
    return _CLIENT


def _get_llm() -> Llama:
    global _LLM
    if _LLM is not None:
        return _LLM

    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH env var is required (path to GGUF model).")
    if model_path == "/absolute/path/to/model.gguf":
        raise RuntimeError(
            "MODEL_PATH is still set to the README placeholder. Set it to a real .gguf file path."
        )

    resolved_model_path = Path(model_path).expanduser()
    if not resolved_model_path.exists():
        raise RuntimeError(f"Model path does not exist: {resolved_model_path}")

    _LLM = Llama(
        model_path=str(resolved_model_path),
        n_ctx=int(os.getenv("LLAMA_CTX", "4096")),
        n_threads=int(os.getenv("LLAMA_THREADS", "4")),
        verbose=False,
    )
    return _LLM


def _call_llm(messages: List[Dict[str, str]]) -> str:
    backend = get_runtime_backend()
    if backend == "proxy":
        client = _get_client()
        response = client.chat.completions.create(
            model=os.getenv("LITELLM_MODEL", "nf-gpt-4o").strip(),
            messages=messages,
            temperature=0.2,
            max_completion_tokens=600,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content

    llm = _get_llm()
    system_msgs = [m["content"] for m in messages if m["role"] == "system"]
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    prompt = "\n\n".join(
        [
            "<s>[INST] <<SYS>>",
            "\n".join(system_msgs),
            "<</SYS>>",
            "\n".join(user_msgs),
            "[/INST]",
        ]
    )
    resp = llm(
        prompt,
        temperature=0.2,
        max_tokens=600,
        top_p=0.9,
        repeat_penalty=1.05,
        echo=False,
    )
    return resp["choices"][0]["text"]


def _build_system_prompt() -> str:
    return (
        "You are an expert psychologist specializing in single-case experimental designs. "
        "Extract a structured summary from the provided study PDF or PDF text blocks. "
        "Respond with pure JSON matching this schema:\n"
        "{\n"
        '  "Participant ID": "<string or list>",\n'
        '  "Baseline Mean": "<number or null>",\n'
        '  "Treatment Phase Slope": "<number or null>",\n'
        '  "Clinical Contradictions": ["<string>", ...]\n'
        "}\n"
        "If information is missing, use null or an empty list. Do NOT add extra keys or prose."
    )


def _parse_json_with_retries(
    *,
    invoke: Any,
    retries: int,
) -> Optional[Dict[str, Any]]:
    last_error: Optional[Exception] = None

    for _ in range(retries):
        raw = invoke()
        try:
            parsed = json.loads(raw)
            parsed.setdefault("Participant ID", None)
            parsed.setdefault("Baseline Mean", None)
            parsed.setdefault("Treatment Phase Slope", None)
            parsed.setdefault("Clinical Contradictions", [])
            return parsed
        except Exception as exc:
            last_error = exc

    print(f"LLM extraction failed after {retries} attempts: {last_error}")
    return None


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def run_sced_extraction(
    pdf_text_json: List[Dict[str, Any]],
    max_chars: int = 8000,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    pdf_text_json: list of {page, bbox, text} entries from pdf_text_blocks.py
    Returns parsed JSON dict or None if all retries fail.
    """
    joined_text = "\n".join(
        f"[p{b['page']} bbox{b['bbox']}] {b['text']}"
        for b in pdf_text_json
        if "text" in b and "page" in b and "bbox" in b
    )[:max_chars]

    base_user_prompt = (
        "Extract the fields from the provided PDF text blocks. "
        "Only output the JSON object, nothing else.\n\n"
        f"TEXT:\n{joined_text}"
    )
    system_prompt = _build_system_prompt()

    def invoke() -> str:
        return _call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_user_prompt},
            ]
        )

    return _parse_json_with_retries(invoke=invoke, retries=retries)


def run_sced_extraction_full_text(
    pdf_text_json: List[Dict[str, Any]],
    chunk_chars: int = 20000,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Extract from the full text-block corpus by chunking long papers across multiple calls,
    then synthesize one final JSON answer. This stays on chat/completions and avoids
    Responses API file-input support, which some proxies do not implement reliably.
    """
    joined_text = "\n".join(
        f"[p{b['page']} bbox{b['bbox']}] {b['text']}"
        for b in pdf_text_json
        if "text" in b and "page" in b and "bbox" in b
    )
    if not joined_text.strip():
        return None

    system_prompt = _build_system_prompt()
    chunks = _chunk_text(joined_text, chunk_chars)
    chunk_summaries: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_prompt = (
            f"This is chunk {index} of {len(chunks)} from one study PDF. "
            "Extract any evidence relevant to the target JSON fields. "
            "If a field is not supported by this chunk, use null or an empty list. "
            "Only output JSON.\n\n"
            f"TEXT:\n{chunk}"
        )
        raw = _call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_prompt},
            ]
        )
        chunk_summaries.append(raw)

    synthesis_prompt = (
        "You are given per-chunk JSON extractions from one full study PDF. "
        "Merge them into one final JSON object using the required schema only. "
        "Prefer values supported repeatedly or most specifically. "
        "Deduplicate Clinical Contradictions. "
        "Only output JSON.\n\n"
        f"CHUNK_JSONS:\n{json.dumps(chunk_summaries, ensure_ascii=False)}"
    )

    def invoke() -> str:
        return _call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": synthesis_prompt},
            ]
        )

    return _parse_json_with_retries(invoke=invoke, retries=retries)


def run_sced_extraction_from_pdf(
    pdf_path: Path,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Send the full PDF file to the OpenAI Responses API and return the parsed JSON summary.
    This requires proxy mode because the local GGUF fallback cannot ingest PDFs directly.
    """
    if get_runtime_backend() != "proxy":
        raise RuntimeError("Full-PDF extraction requires proxy mode with LITELLM_KEY configured.")

    client = _get_client()
    system_prompt = _build_system_prompt()
    instruction = (
        "Extract the fields from the attached full PDF. "
        "Use the complete document, not a partial excerpt. "
        "Only output the JSON object, nothing else."
    )
    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("ascii")

    def invoke() -> str:
        response = client.responses.create(
            model=os.getenv("LITELLM_MODEL", "gpt-5.1").strip(),
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": pdf_path.name,
                            "file_data": encoded_pdf,
                        },
                        {
                            "type": "input_text",
                            "text": instruction,
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
        return getattr(response, "output_text", "") or ""

    return _parse_json_with_retries(invoke=invoke, retries=retries)


__all__ = [
    "run_sced_extraction",
    "run_sced_extraction_full_text",
    "run_sced_extraction_from_pdf",
    "get_runtime_backend",
]
