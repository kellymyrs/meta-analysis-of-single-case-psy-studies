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
    LITELLM_BASE_URL  = https://llmproxy.uva.nl              (optional)
    LITELLM_MODEL     = gpt-5.1                              (optional)

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

from scripts.sced_fields import (
    FIELDS,
    build_guidance_prompt,
    build_schema_prompt,
    normalize_field_keys,
)


_LLM: Optional[Llama] = None
_CLIENT: Optional[OpenAI] = None
DEFAULT_MAX_OUTPUT_TOKENS = 3000
DEFAULT_LITELLM_BASE_URL = "https://llmproxy.uva.nl"
DEFAULT_VERIFIER_MODEL = "gpt-4.1"
DEFAULT_VERIFY_DPI = 150
VERIFICATION_VERDICTS = ("supported", "contradicted", "not_in_text", "inferred")


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def get_runtime_backend() -> str:
    """Return 'proxy' if proxy key is configured, else 'local' if MODEL_PATH is configured."""
    if os.getenv("LITELLM_KEY", "").strip():
        return "proxy"
    if os.getenv("MODEL_PATH", "").strip():
        return "local"
    raise RuntimeError("Set either LITELLM_KEY (proxy mode) or MODEL_PATH (local GGUF mode).")


def _max_output_tokens() -> int:
    return int(os.getenv("SCED_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)))


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("LITELLM_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LITELLM_KEY env var is required for proxy mode.")

    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LITELLM_BASE_URL", DEFAULT_LITELLM_BASE_URL).strip(),
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
            model=os.getenv("LITELLM_MODEL", "gpt-5.1").strip(),
            messages=messages,
            temperature=0.2,
            max_completion_tokens=_max_output_tokens(),
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
        max_tokens=_max_output_tokens(),
        top_p=0.9,
        repeat_penalty=1.05,
        echo=False,
    )
    return resp["choices"][0]["text"]


def _few_shot_prompt(few_shot_examples: Optional[List[Dict[str, Any]]]) -> str:
    if not few_shot_examples:
        return ""

    normalized_examples: List[Dict[str, Any]] = []
    for example in few_shot_examples:
        normalized = normalize_field_keys(dict(example))
        normalized_examples.append(
            {
                key: normalized.get(key)
                for key in ["pdf", *FIELDS]
                if key in normalized
            }
        )

    return (
        "\nDevelopment-set examples:\n"
        "The following labelled examples show how source studies should be mapped "
        "into the target schema. Use them as formatting and coding examples only. "
        "Do not copy values from an example unless the attached study explicitly "
        "supports the same value.\n"
        f"{json.dumps(normalized_examples, ensure_ascii=False, indent=2)}\n"
    )


def _build_system_prompt(few_shot_examples: Optional[List[Dict[str, Any]]] = None) -> str:
    guidance = build_guidance_prompt()
    guidance_section = f"\nImportant field guidance:\n{guidance}\n" if guidance else "\n"
    examples_section = _few_shot_prompt(few_shot_examples)
    return (
        "You are an expert psychologist specializing in single-case experimental designs. "
        "Extract a structured summary from the provided study PDF or PDF text blocks. "
        "Respond with pure JSON matching this schema:\n"
        f"{build_schema_prompt()}\n"
        f"{guidance_section}"
        f"{examples_section}"
        "If information is missing, use null or an empty list. Do NOT add extra keys or prose."
    )


def _extract_json_text(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_with_retries(
    *,
    invoke: Any,
    retries: int,
    debug_label: str | None = None,
) -> Optional[Dict[str, Any]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        raw = invoke()
        try:
            parsed = normalize_field_keys(json.loads(_extract_json_text(raw)))
            for field in FIELDS:
                parsed.setdefault(field, None)
            return parsed
        except Exception as exc:
            last_error = exc
            label = f" [{debug_label}]" if debug_label else ""
            preview = raw[:500] if isinstance(raw, str) else repr(raw)[:500]
            print(
                f"JSON parse failed{label} attempt {attempt}/{retries}: {exc}\n"
                f"Raw response preview:\n{preview}\n"
            )

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
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
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
    system_prompt = _build_system_prompt(few_shot_examples)

    def invoke() -> str:
        return _call_llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_user_prompt},
            ]
        )

    return _parse_json_with_retries(invoke=invoke, retries=retries, debug_label="blocks")


def run_sced_extraction_full_text(
    pdf_text_json: List[Dict[str, Any]],
    chunk_chars: int = 20000,
    retries: int = 3,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
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

    system_prompt = _build_system_prompt(few_shot_examples)
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
        "Keep each field concise: use canonical labels rather than long excerpts, "
        "and limit list fields to the distinct values needed for coding. "
        "Deduplicate list-like fields such as Gender, Age, Type of treatments, treatment protocol, "
        "number of sessions, total observations, and frequent assessment symptoms. "
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

    return _parse_json_with_retries(invoke=invoke, retries=retries, debug_label="full_text")


def run_sced_extraction_from_pdf(
    pdf_path: Path,
    retries: int = 3,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send the full PDF file to the OpenAI Responses API and return the parsed JSON summary.
    This requires proxy mode because the local GGUF fallback cannot ingest PDFs directly.
    """
    if get_runtime_backend() != "proxy":
        raise RuntimeError("Full-PDF extraction requires proxy mode with LITELLM_KEY configured.")

    client = _get_client()
    system_prompt = _build_system_prompt(few_shot_examples)
    instruction = (
        "Extract the fields from the attached full PDF. "
        "Use the complete document, not a partial excerpt. "
        "Only output the JSON object, nothing else."
    )
    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    file_data = f"data:application/pdf;base64,{encoded_pdf}"

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
                            "file_data": file_data,
                        },
                        {
                            "type": "input_text",
                            "text": instruction,
                        },
                    ],
                },
            ],
            temperature=0.2,
            max_output_tokens=_max_output_tokens(),
        )
        return getattr(response, "output_text", "") or ""

    return _parse_json_with_retries(
        invoke=invoke,
        retries=retries,
        debug_label=f"full_pdf:{pdf_path.name}",
    )


def _verifier_model() -> str:
    return os.getenv("SCED_VERIFIER_MODEL", DEFAULT_VERIFIER_MODEL).strip()


def render_pdf_pages_to_data_urls(
    pdf_path: Path,
    dpi: int = DEFAULT_VERIFY_DPI,
    max_pages: Optional[int] = None,
) -> List[str]:
    """Render every PDF page to a PNG and return base64 data: URLs (one per page).

    Used to feed a vision verifier model that cannot ingest a PDF directly. Pages
    are rendered in document order; ``max_pages`` caps how many are sent.
    """
    import fitz  # PyMuPDF; already a dependency via pdf_to_images.py

    data_urls: List[str] = []
    scale = dpi / 72.0
    with fitz.open(pdf_path) as doc:
        matrix = fitz.Matrix(scale, scale)
        for index, page in enumerate(doc):
            if max_pages is not None and index >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            encoded = base64.b64encode(pix.tobytes("png")).decode("ascii")
            data_urls.append(f"data:image/png;base64,{encoded}")
    return data_urls


def _build_verifier_system_prompt() -> str:
    guidance = build_guidance_prompt()
    guidance_section = f"\nField coding rules the extractor followed:\n{guidance}\n" if guidance else "\n"
    field_list = "\n".join(f"- {name}" for name in FIELDS)
    return (
        "You are an expert psychologist auditing a single-case experimental design "
        "(SCED) data extraction. You are shown the page images of the full study PDF "
        "and a set of field values another system extracted from it. Verify each value "
        "STRICTLY against the page images. Try to REFUTE each value rather than assume "
        "it is correct.\n\n"
        "For every field return one verdict:\n"
        '- "supported": the pages contain explicit text supporting the value. You MUST '
        "copy a verbatim supporting quote from the pages into the \"quote\" field.\n"
        '- "contradicted": the pages state something that conflicts with the value. You '
        "MUST copy the verbatim conflicting text from the pages into the \"quote\" "
        "field, so the disagreement can be checked.\n"
        '- "not_in_text": the value cannot be located in the pages. Use this as the '
        "default whenever you are uncertain.\n"
        '- "inferred": the value is a reasonable inference but is not stated explicitly.\n\n'
        "Fields to verify:\n"
        f"{field_list}\n"
        f"{guidance_section}"
        "Respond with pure JSON of the exact form:\n"
        '{ "<field name>": {"verdict": "<one verdict>", "quote": "<verbatim quote or empty>"}, ... }\n'
        "Provide a verbatim quote for both \"supported\" and \"contradicted\" verdicts; "
        "use an empty string for \"not_in_text\" and \"inferred\". The quote must be a "
        "SINGLE short span copied from the document: do not concatenate multiple quotes, "
        "do not add commentary, and do not include double-quote characters inside the "
        "quote value. Do not add extra keys, commentary, or prose."
    )


def _parse_verification_with_retries(
    *,
    invoke: Any,
    retries: int,
    debug_label: str | None = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        raw = invoke()
        try:
            parsed = normalize_field_keys(json.loads(_extract_json_text(raw)))
            verdicts: Dict[str, Dict[str, Any]] = {}
            for field in FIELDS:
                entry = parsed.get(field)
                if not isinstance(entry, dict):
                    verdicts[field] = {"verdict": "not_in_text", "quote": ""}
                    continue
                verdict = str(entry.get("verdict", "")).strip().lower()
                if verdict not in VERIFICATION_VERDICTS:
                    verdict = "not_in_text"
                quote = entry.get("quote") or ""
                verdicts[field] = {"verdict": verdict, "quote": str(quote)}
            return verdicts
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            label = f" [{debug_label}]" if debug_label else ""
            preview = raw[:500] if isinstance(raw, str) else repr(raw)[:500]
            print(
                f"Verification JSON parse failed{label} attempt {attempt}/{retries}: {exc}\n"
                f"Raw response preview:\n{preview}\n"
            )

    print(f"LLM verification failed after {retries} attempts: {last_error}")
    return None


def run_sced_verification_from_pdf(
    pdf_path: Path,
    record: Dict[str, Any],
    retries: int = 3,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Verify an extracted record by attaching the full PDF to the Responses API.

    This is the preferred verification path: the verifier model (default gpt-4.1,
    override via SCED_VERIFIER_MODEL) ingests the PDF natively, so there is no page
    rendering and no context-window blow-up on long documents. Requires proxy mode.
    Returns a dict mapping each field to {"verdict", "quote"}, or None on failure.
    """
    if get_runtime_backend() != "proxy":
        raise RuntimeError("SCED verification requires proxy mode with LITELLM_KEY configured.")

    client = _get_client()
    model = _verifier_model()
    system_prompt = _build_verifier_system_prompt()
    extracted = {field: record.get(field) for field in FIELDS}
    instruction = (
        "Verify the following extracted field values against the attached full study "
        "PDF. Use the complete document. Only output the JSON object.\n\n"
        f"EXTRACTED_VALUES:\n{json.dumps(extracted, ensure_ascii=False, indent=2)}"
    )
    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    file_data = f"data:application/pdf;base64,{encoded_pdf}"

    def invoke() -> str:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": pdf_path.name,
                            "file_data": file_data,
                        },
                        {"type": "input_text", "text": instruction},
                    ],
                },
            ],
            temperature=0.0,
            max_output_tokens=_max_output_tokens(),
            text={"format": {"type": "json_object"}},
        )
        return getattr(response, "output_text", "") or ""

    return _parse_verification_with_retries(
        invoke=invoke,
        retries=retries,
        debug_label=f"verify_pdf:{pdf_path.name}",
    )


def _is_context_length_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "context length" in text or "context_length" in text or "maximum context" in text


def run_sced_verification_from_images(
    pdf_path: Path,
    record: Dict[str, Any],
    dpi: int = DEFAULT_VERIFY_DPI,
    retries: int = 3,
    max_pages: Optional[int] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Verify an extracted record against the PDF page images with a vision model.

    Returns a dict mapping each field to {"verdict", "quote"}, or None if all
    retries fail. Requires proxy mode; the verifier model defaults to Qwen-2.5-VL
    (override via SCED_VERIFIER_MODEL) and is reached over chat.completions with
    image content, since it cannot ingest a PDF directly.

    Long documents (e.g. dissertations) can exceed the model's context window. On a
    context-length error the call is retried with progressively lower DPI, then with
    a page cap, so very long PDFs degrade to a legible-but-smaller subset rather than
    failing outright. The page subset used is reported by the caller as a partial run.
    """
    if get_runtime_backend() != "proxy":
        raise RuntimeError("SCED verification requires proxy mode with LITELLM_KEY configured.")

    client = _get_client()
    model = _verifier_model()
    system_prompt = _build_verifier_system_prompt()
    extracted = {field: record.get(field) for field in FIELDS}
    instruction = (
        "Verify the following extracted field values against the attached page "
        "images of the full study PDF. Only output the JSON object.\n\n"
        f"EXTRACTED_VALUES:\n{json.dumps(extracted, ensure_ascii=False, indent=2)}"
    )

    # Fallback ladder for documents that overflow the context window: first shrink
    # the images (lower DPI), then cap the number of pages sent.
    attempts: List[Dict[str, Optional[int]]] = [{"dpi": dpi, "max_pages": max_pages}]
    for fallback_dpi in (max(dpi // 2, 72), 72):
        if fallback_dpi < dpi:
            attempts.append({"dpi": fallback_dpi, "max_pages": max_pages})
    attempts.append({"dpi": 72, "max_pages": min(max_pages or 40, 40)})

    last_error: Optional[Exception] = None
    for attempt in attempts:
        data_urls = render_pdf_pages_to_data_urls(
            pdf_path, dpi=attempt["dpi"], max_pages=attempt["max_pages"]
        )
        if not data_urls:
            print(f"No pages rendered for {pdf_path.name}; skipping verification.")
            return None

        user_content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
        for url in data_urls:
            user_content.append({"type": "image_url", "image_url": {"url": url}})

        def invoke() -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_completion_tokens=_max_output_tokens(),
            )
            content = response.choices[0].message.content
            return content or ""

        try:
            return _parse_verification_with_retries(
                invoke=invoke,
                retries=retries,
                debug_label=f"verify:{pdf_path.name}",
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if _is_context_length_error(exc):
                print(
                    f"Context overflow for {pdf_path.name} at dpi={attempt['dpi']} "
                    f"max_pages={attempt['max_pages']}; retrying smaller."
                )
                continue
            raise

    if last_error is not None:
        raise last_error
    return None


__all__ = [
    "run_sced_extraction",
    "run_sced_extraction_full_text",
    "run_sced_extraction_from_pdf",
    "run_sced_verification_from_pdf",
    "run_sced_verification_from_images",
    "render_pdf_pages_to_data_urls",
    "get_runtime_backend",
]
