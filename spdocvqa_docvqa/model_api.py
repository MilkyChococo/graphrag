from __future__ import annotations

import json
import base64
import asyncio
import mimetypes
import os
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROVIDER = "local_hf"
DEFAULT_API_KEY_ENV = "HUGGINGFACE_API_KEY"
DEFAULT_MAX_NEW_TOKENS = 512


class LocalQwenVLCompletion:
    def __init__(self, model_name: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._runtime: tuple[Any, Any] | None = None

    def _load_runtime(self) -> tuple[Any, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Missing local Qwen-VL dependencies. Install torch, transformers, accelerate, and qwen-vl-utils."
            ) from exc

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._runtime = (model, processor)
        return self._runtime

    def _generate_sync(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: Path | None,
        temperature: float,
    ) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = self._load_runtime()
        user_content: list[dict[str, str]] = []
        if image_path is not None:
            user_content.append({"type": "image", "image": str(image_path.resolve())})
        user_content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generation_config = model.generation_config
        generation_config.do_sample = temperature > 0
        generation_config.temperature = temperature if temperature > 0 else 1.0

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                generation_config=generation_config,
            )
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        return str(
            processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ).strip()

    async def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=None,
            temperature=temperature,
        )

    async def complete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        temperature: float = 0.0,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            temperature=temperature,
        )


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
    if provider == "local_hf":
        if api_key:
            os.environ.setdefault("HF_TOKEN", api_key)
            os.environ.setdefault("HUGGINGFACE_API_KEY", api_key)
        return LocalQwenVLCompletion(model_name=model_name)

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
    if hasattr(model, "complete_text"):
        return await model.complete_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

    from graphrag_llm.utils import CompletionMessagesBuilder

    messages = (
        CompletionMessagesBuilder()
        .add_system_message(system_prompt)
        .add_user_message(user_prompt)
        .build()
    )
    response = await model.completion_async(messages=messages, temperature=temperature)
    return str(response.content or "").strip()


def image_to_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


async def complete_with_image(
    *,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    image_path: Path,
    image_detail: str = "high",
    temperature: float = 0.0,
) -> str:
    if hasattr(model, "complete_with_image"):
        return await model.complete_with_image(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            temperature=temperature,
        )

    from graphrag_llm.utils import CompletionContentPartBuilder, CompletionMessagesBuilder

    content = (
        CompletionContentPartBuilder()
        .add_text_part(user_prompt)
        .add_image_part(image_to_data_url(image_path), image_detail)  # type: ignore[arg-type]
        .build()
    )
    messages = (
        CompletionMessagesBuilder()
        .add_system_message(system_prompt)
        .add_user_message(content)
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
