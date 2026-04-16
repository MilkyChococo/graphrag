# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Shared helpers for local Hugging Face runtimes."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from graphrag_llm.types import LLMCompletionMessagesParam


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_COMPLETION_RUNTIME_CACHE: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
_EMBEDDING_RUNTIME_CACHE: dict[tuple[str, str, bool], tuple[Any, Any, str]] = {}


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Missing local Hugging Face dependencies. Install torch, transformers, and accelerate into the Python env running GraphRAG."
        ) from exc
    return torch


def resolve_torch_dtype(torch_dtype_value: str | None) -> Any:
    """Resolve a torch dtype string into the matching torch dtype object."""
    if torch_dtype_value in (None, "", "auto"):
        return "auto"

    torch = _import_torch()
    normalized = str(torch_dtype_value).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(normalized, "auto")


def ensure_hf_token_env() -> None:
    """Mirror HUGGINGFACE_API_KEY into HF_TOKEN for hub downloads when available."""
    if "HF_TOKEN" in os.environ:
        return
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
    if api_key:
        os.environ["HF_TOKEN"] = api_key


def resolve_local_device(device_preference: str | None) -> str:
    """Resolve a local device preference to cpu or cuda."""
    if device_preference in {"cpu", "cuda"}:
        return str(device_preference)

    torch = _import_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=8)
def load_local_hf_tokenizer(model_name: str, trust_remote_code: bool = False):
    """Load and cache a Hugging Face tokenizer."""
    from transformers import AutoTokenizer

    ensure_hf_token_env()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_local_completion_runtime(
    *,
    model_name: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Load and cache a local completion runtime."""
    cache_key = (model_name, device_map, torch_dtype, trust_remote_code)
    cached = _COMPLETION_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resolved_dtype = resolve_torch_dtype(torch_dtype)

    try:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Missing local Hugging Face dependencies. Install torch, transformers, and accelerate into the Python env running GraphRAG."
        ) from exc

    ensure_hf_token_env()
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    is_vision_language_model = (
        hasattr(config, "vision_config")
        or "vl" in config.__class__.__name__.lower()
        or "vision" in config.__class__.__name__.lower()
    )

    if is_vision_language_model:
        load_errors: list[Exception] = []
        for loader in (
            Qwen2_5_VLForConditionalGeneration.from_pretrained,
            AutoModelForImageTextToText.from_pretrained,
        ):
            try:
                model = loader(
                    model_name,
                    torch_dtype=resolved_dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                tokenizer = load_local_hf_tokenizer(
                    model_name,
                    trust_remote_code=trust_remote_code,
                )
                runtime = {
                    "kind": "text_generation",
                    "model": model,
                    "tokenizer": tokenizer,
                }
                _COMPLETION_RUNTIME_CACHE[cache_key] = runtime
                return runtime
            except Exception as exc:
                load_errors.append(exc)

        raise load_errors[-1]

    else:
        tokenizer = load_local_hf_tokenizer(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        runtime = {
            "kind": "text_generation",
            "model": model,
            "tokenizer": tokenizer,
        }
        _COMPLETION_RUNTIME_CACHE[cache_key] = runtime
        return runtime


def load_local_embedding_runtime(
    *,
    model_name: str,
    device_preference: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any, str]:
    """Load and cache a local embedding runtime."""
    cache_key = (model_name, device_preference, trust_remote_code)
    cached = _EMBEDDING_RUNTIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Missing local Hugging Face dependencies. Install torch, transformers, and accelerate into the Python env running GraphRAG."
        ) from exc

    device = resolve_local_device(device_preference)
    ensure_hf_token_env()
    tokenizer = load_local_hf_tokenizer(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    runtime = (model, tokenizer, device)
    _EMBEDDING_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def normalize_completion_messages(
    messages: "LLMCompletionMessagesParam",
) -> list[dict[str, str]]:
    """Normalize messages into text-only chat messages."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    normalized: list[dict[str, str]] = []
    for message in messages:
        payload = message if isinstance(message, dict) else message.model_dump()
        role = str(payload.get("role") or "user")
        content = payload.get("content") or ""
        if isinstance(content, str):
            normalized.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if "text" in part and part["text"]:
                        text_parts.append(str(part["text"]))
                    elif part.get("type") == "image_url":
                        text_parts.append("[image omitted]")
                    elif part.get("type") == "input_audio":
                        text_parts.append("[audio omitted]")
        normalized.append({"role": role, "content": "\n".join(text_parts).strip()})
    return normalized


def append_json_instruction(
    messages: list[dict[str, str]],
    response_format: type["BaseModel"] | None,
) -> list[dict[str, str]]:
    """Append JSON-only instructions when structured output is required."""
    if response_format is None:
        return messages

    schema = json.dumps(
        response_format.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    instruction = (
        "Return ONLY valid JSON. Do not include markdown, comments, or explanation.\n"
        "The JSON must satisfy this schema:\n"
        f"{schema}"
    )

    patched = [dict(message) for message in messages]
    for index in range(len(patched) - 1, -1, -1):
        if patched[index]["role"] == "user":
            patched[index]["content"] = (
                patched[index]["content"].rstrip() + "\n\n" + instruction
            ).strip()
            return patched

    patched.append({"role": "user", "content": instruction})
    return patched


def render_prompt_from_messages(
    *,
    templater: Any,
    messages: list[dict[str, str]],
) -> str:
    """Render a text prompt from normalized chat messages."""
    if hasattr(templater, "apply_chat_template"):
        return templater.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}".strip()
        for message in messages
        if message["content"]
    ) + "\n\nASSISTANT:\n"


def get_text_model_device(model: Any) -> str:
    """Resolve the device used for text model inputs."""
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)

    try:
        first_param = next(model.parameters())
    except (AttributeError, StopIteration):
        return resolve_local_device("auto")
    return str(first_param.device)


def extract_json_string(text: str) -> str:
    """Extract a valid JSON object or array from model output."""
    stripped = str(text or "").strip()
    if not stripped:
        return stripped

    fenced_match = _JSON_FENCE_PATTERN.search(stripped)
    if fenced_match:
        stripped = fenced_match.group(1).strip()

    decoder = json.JSONDecoder()
    for start, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            payload, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        return json.dumps(payload, ensure_ascii=False)

    return stripped


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean-pool hidden states using the attention mask."""
    torch = _import_torch()

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
