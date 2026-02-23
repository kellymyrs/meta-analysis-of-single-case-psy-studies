"""
SCED variable extraction with one configurable local model (llama.cpp).

Usage example:
    from scripts.sced_extraction import run_sced_extraction
    import json, pathlib
    blocks = json.loads(pathlib.Path("extracted_text/example_blocks.json").read_text())
    result = run_sced_extraction(blocks)
    print(result)

Environment configuration:
    MODEL_PATH      = /absolute/path/to/model.gguf  (required)
    LLAMA_THREADS   = 4    (optional)
    LLAMA_CTX       = 4096 (optional)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama


_LLM: Optional[Llama] = None


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

    system_prompt = (
        "You are an expert psychologist specializing in single-case experimental designs. "
        "Given PDF text with bounding boxes, extract a structured summary. "
        "Respond with pure JSON matching this schema:\n"
        "{\n"
        '  \"Participant ID\": \"<string or list>\",\n'
        '  \"Baseline Mean\": \"<number or null>\",\n'
        '  \"Treatment Phase Slope\": \"<number or null>\",\n'
        '  \"Clinical Contradictions\": [\"<string>\", ...]\n'
        "}\n"
        "If information is missing, use null or an empty list. Do NOT add extra keys or prose."
    )

    user_prompt = base_user_prompt
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        raw = _call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        try:
            parsed = json.loads(raw)
            # Minimal shape enforcement
            parsed.setdefault("Participant ID", None)
            parsed.setdefault("Baseline Mean", None)
            parsed.setdefault("Treatment Phase Slope", None)
            parsed.setdefault("Clinical Contradictions", [])
            return parsed
        except Exception as exc:  # invalid JSON or shape
            last_error = exc
            user_prompt = (
                "Your last reply was not valid JSON. "
                "Return ONLY the JSON object per schema, no extra text. "
                "Schema keys: Participant ID, Baseline Mean, Treatment Phase Slope, Clinical Contradictions."
            )

    print(f"LLM extraction failed after {retries} attempts: {last_error}")
    return None


__all__ = ["run_sced_extraction"]
