from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROVIDER = "huggingface"
DEFAULT_API_KEY_ENV = "HUGGINGFACE_API_KEY"


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_api_key(env_file: Path, env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    file_values = parse_env_file(env_file)
    env_value = file_values.get(env_name)
    if env_value:
        os.environ[env_name] = env_value
        return env_value
    if env_name == "HUGGINGFACE_API_KEY":
        fallback = os.environ.get("HF_TOKEN") or file_values.get("HF_TOKEN")
        if fallback:
            os.environ[env_name] = fallback
            os.environ.setdefault("HF_TOKEN", fallback)
            return fallback
    raise RuntimeError(f"Missing API key. Expected {env_name} in environment or {env_file}")


def build_completion_model(
    *,
    provider: str,
    model_name: str,
    api_key: str,
    api_base: str | None,
    max_retries: int,
) -> Any:
    from graphrag_llm.completion import create_completion
    from graphrag_llm.config.model_config import ModelConfig
    from graphrag_llm.config.types import AuthMethod

    model_config = ModelConfig(
        model_provider=provider,
        model=model_name,
        api_key=api_key,
        api_base=api_base,
        auth_method=AuthMethod.ApiKey,
        retry={"type": "exponential_backoff", "max_retries": max_retries},
    )
    return create_completion(model_config)


async def complete_text(
    *,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    from graphrag_llm.utils import CompletionMessagesBuilder

    messages = (
        CompletionMessagesBuilder()
        .add_system_message(system_prompt)
        .add_user_message(user_prompt)
        .build()
    )
    response = await model.completion_async(messages=messages, temperature=temperature)
    return str(response.content or "").strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        payload = json.loads(cleaned[start : end + 1])
        return payload if isinstance(payload, dict) else {}
    return {}
