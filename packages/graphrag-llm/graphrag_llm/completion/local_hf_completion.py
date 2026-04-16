# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLMCompletion backed by local Hugging Face models."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.completion.completion import LLMCompletion
from graphrag_llm.middleware import with_middleware_pipeline
from graphrag_llm.types import LLMCompletionChunk, LLMCompletionResponse
from graphrag_llm.utils import create_completion_response, structure_completion_response
from graphrag_llm.utils.local_hf import (
    append_json_instruction,
    extract_json_string,
    get_text_model_device,
    load_local_completion_runtime,
    normalize_completion_messages,
    render_prompt_from_messages,
)

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator

    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        AsyncLLMCompletionFunction,
        LLMCompletionArgs,
        LLMCompletionFunction,
        LLMCompletionMessagesParam,
        Metrics,
        ResponseFormat,
    )


class LocalHFCompletion(LLMCompletion):
    """LLMCompletion backed by local Hugging Face models."""

    _model_config: "ModelConfig"
    _model_id: str
    _track_metrics: bool = False
    _metrics_store: "MetricsStore"
    _metrics_processor: "MetricsProcessor | None"
    _cache: "Cache | None"
    _cache_key_creator: "CacheKeyCreator"
    _tokenizer: "Tokenizer"
    _rate_limiter: "RateLimiter | None"
    _retrier: "Retry | None"

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        metrics_processor: "MetricsProcessor | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        retrier: "Retry | None" = None,
        cache: "Cache | None" = None,
        cache_key_creator: "CacheKeyCreator",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize local Hugging Face completion."""
        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._cache = cache
        self._track_metrics = metrics_processor is not None
        self._cache_key_creator = cache_key_creator
        self._rate_limiter = rate_limiter
        self._retrier = retrier
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._default_max_new_tokens = max_new_tokens
        self._default_top_p = top_p

        self._completion, self._completion_async = _create_base_completions(
            model_config=model_config,
            tokenizer=tokenizer,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            default_max_new_tokens=max_new_tokens,
            default_top_p=top_p,
        )

        self._completion, self._completion_async = with_middleware_pipeline(
            model_config=self._model_config,
            model_fn=self._completion,
            async_model_fn=self._completion_async,
            request_type="chat",
            cache=self._cache,
            cache_key_creator=self._cache_key_creator,
            tokenizer=self._tokenizer,
            metrics_processor=self._metrics_processor,
            rate_limiter=self._rate_limiter,
            retrier=self._retrier,
        )

    def completion(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | Iterator[LLMCompletionChunk]":
        """Sync completion method."""
        messages: LLMCompletionMessagesParam = kwargs.pop("messages")
        response_format = kwargs.pop("response_format", None)
        is_streaming = kwargs.get("stream") or False

        if is_streaming:
            msg = "LocalHFCompletion does not support streaming completions."
            raise ValueError(msg)

        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            response = self._completion(
                messages=messages,
                metrics=request_metrics,
                response_format=response_format,
                **kwargs,  # type: ignore[arg-type]
            )
            if response_format is not None:
                structured_response = structure_completion_response(
                    extract_json_string(response.content),
                    response_format,
                )
                response.formatted_response = structured_response
            return response
        finally:
            if request_metrics is not None:
                self._metrics_store.update_metrics(metrics=request_metrics)

    async def completion_async(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | AsyncIterator[LLMCompletionChunk]":
        """Async completion method."""
        messages: LLMCompletionMessagesParam = kwargs.pop("messages")
        response_format = kwargs.pop("response_format", None)
        is_streaming = kwargs.get("stream") or False

        if is_streaming:
            msg = "LocalHFCompletion does not support streaming completions."
            raise ValueError(msg)

        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            response = await self._completion_async(
                messages=messages,
                metrics=request_metrics,
                response_format=response_format,
                **kwargs,  # type: ignore[arg-type]
            )
            if response_format is not None:
                structured_response = structure_completion_response(
                    extract_json_string(response.content),
                    response_format,
                )
                response.formatted_response = structured_response
            return response
        finally:
            if request_metrics is not None:
                self._metrics_store.update_metrics(metrics=request_metrics)

    @property
    def metrics_store(self) -> "MetricsStore":
        """Get metrics store."""
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        """Get tokenizer."""
        return self._tokenizer


def _create_base_completions(
    *,
    model_config: "ModelConfig",
    tokenizer: "Tokenizer",
    device_map: str,
    torch_dtype: str,
    trust_remote_code: bool,
    default_max_new_tokens: int,
    default_top_p: float,
) -> tuple["LLMCompletionFunction", "AsyncLLMCompletionFunction"]:
    """Create base completion functions for local Hugging Face models."""

    runtime = load_local_completion_runtime(
        model_name=model_config.model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = runtime["model"]

    def _base_completion(**kwargs: Any) -> LLMCompletionResponse:
        kwargs.pop("metrics", None)
        kwargs.pop("stream", None)
        messages = kwargs.pop("messages")
        response_format = kwargs.pop("response_format", None)

        temperature = kwargs.pop("temperature", None)
        if temperature is None:
            temperature = model_config.call_args.get("temperature", 0)
        max_new_tokens = (
            kwargs.pop("max_completion_tokens", None)
            or kwargs.pop("max_tokens", None)
            or model_config.call_args.get("max_completion_tokens")
            or model_config.call_args.get("max_tokens")
            or default_max_new_tokens
        )
        top_p = kwargs.pop("top_p", None)
        if top_p is None:
            top_p = model_config.call_args.get("top_p", default_top_p)

        response_text, prompt_tokens, completion_tokens = _generate_local_completion(
            model=model,
            runtime=runtime,
            tokenizer=tokenizer,
            messages=messages,
            response_format=response_format,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            top_p=float(top_p),
        )

        response = create_completion_response(response_text)
        response.id = f"local-hf-{int(time.time() * 1000)}"
        response.created = int(time.time())
        response.model = model_config.model
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        response.usage.total_tokens = prompt_tokens + completion_tokens
        return response

    async def _base_completion_async(**kwargs: Any) -> LLMCompletionResponse:
        import asyncio

        return await asyncio.to_thread(_base_completion, **kwargs)

    return (_base_completion, _base_completion_async)


def _generate_local_completion(
    *,
    model: Any,
    runtime: dict[str, Any],
    tokenizer: "Tokenizer",
    messages: "LLMCompletionMessagesParam",
    response_format: type["ResponseFormat"] | None,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
) -> tuple[str, int, int]:
    """Generate a local text completion and token counts."""
    import torch

    normalized_messages = normalize_completion_messages(messages)
    normalized_messages = append_json_instruction(normalized_messages, response_format)
    base_tokenizer = runtime["tokenizer"]
    prompt = render_prompt_from_messages(
        templater=base_tokenizer,
        messages=normalized_messages,
    )
    encoded = base_tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt",
    )
    decode = base_tokenizer.batch_decode

    model_device = get_text_model_device(model)
    encoded = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "pad_token_id": getattr(model.generation_config, "pad_token_id", None),
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            **{k: v for k, v in generation_kwargs.items() if v is not None},
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(encoded["input_ids"], generated_ids)
    ]
    output_text = decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    prompt_tokens = tokenizer.num_prompt_tokens(messages=normalized_messages)
    completion_tokens = tokenizer.num_tokens(output_text)
    return output_text, prompt_tokens, completion_tokens
