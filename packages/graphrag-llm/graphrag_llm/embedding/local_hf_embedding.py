# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLMEmbedding backed by local Hugging Face models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.middleware import with_middleware_pipeline
from graphrag_llm.types import (
    LLMEmbedding as LLMEmbeddingItem,
    LLMEmbeddingResponse,
    LLMEmbeddingUsage,
)
from graphrag_llm.utils.local_hf import (
    load_local_embedding_runtime,
    mean_pool,
)

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator

    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        AsyncLLMEmbeddingFunction,
        LLMEmbeddingArgs,
        LLMEmbeddingFunction,
        Metrics,
    )


class LocalHFEmbedding(LLMEmbedding):
    """LLMEmbedding backed by local Hugging Face models."""

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
        device: str = "auto",
        trust_remote_code: bool = False,
        batch_size: int = 8,
        max_length: int = 1024,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ):
        """Initialize local Hugging Face embedding."""
        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._track_metrics = metrics_processor is not None
        self._cache = cache
        self._cache_key_creator = cache_key_creator
        self._rate_limiter = rate_limiter
        self._retrier = retrier

        self._embedding, self._embedding_async = _create_base_embeddings(
            model_config=model_config,
            tokenizer=tokenizer,
            device=device,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
            max_length=max_length,
            normalize_embeddings=normalize_embeddings,
        )

        self._embedding, self._embedding_async = with_middleware_pipeline(
            model_config=self._model_config,
            model_fn=self._embedding,
            async_model_fn=self._embedding_async,
            request_type="embedding",
            cache=self._cache,
            cache_key_creator=self._cache_key_creator,
            tokenizer=self._tokenizer,
            metrics_processor=self._metrics_processor,
            rate_limiter=self._rate_limiter,
            retrier=self._retrier,
        )

    def embedding(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Sync embedding method."""
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return self._embedding(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    async def embedding_async(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Async embedding method."""
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return await self._embedding_async(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    @property
    def metrics_store(self) -> "MetricsStore":
        """Get metrics store."""
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        """Get tokenizer."""
        return self._tokenizer


def _create_base_embeddings(
    *,
    model_config: "ModelConfig",
    tokenizer: "Tokenizer",
    device: str,
    trust_remote_code: bool,
    batch_size: int,
    max_length: int,
    normalize_embeddings: bool,
) -> tuple["LLMEmbeddingFunction", "AsyncLLMEmbeddingFunction"]:
    """Create base embedding functions for local Hugging Face models."""
    model, base_tokenizer, resolved_device = load_local_embedding_runtime(
        model_name=model_config.model,
        device_preference=device,
        trust_remote_code=trust_remote_code,
    )

    def _base_embedding(**kwargs: Any) -> LLMEmbeddingResponse:
        import torch

        kwargs.pop("metrics", None)
        inputs = list(kwargs.pop("input"))
        runtime_batch_size = int(kwargs.pop("batch_size", batch_size))
        runtime_max_length = int(kwargs.pop("max_length", max_length))

        all_vectors: list[list[float]] = []
        for start in range(0, len(inputs), runtime_batch_size):
            batch = inputs[start : start + runtime_batch_size]
            encoded = base_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=runtime_max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(resolved_device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            with torch.no_grad():
                outputs = model(**encoded)
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None:
                last_hidden_state = outputs[0]
            pooled = mean_pool(last_hidden_state, encoded["attention_mask"])
            if normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vectors.extend(pooled.cpu().tolist())

        prompt_tokens = sum(tokenizer.num_tokens(text) for text in inputs)
        response = LLMEmbeddingResponse(
            object="list",
            data=[
                LLMEmbeddingItem(
                    object="embedding",
                    embedding=vector,
                    index=index,
                )
                for index, vector in enumerate(all_vectors)
            ],
            model=model_config.model,
            usage=LLMEmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
        response.model = model_config.model
        return response

    async def _base_embedding_async(**kwargs: Any) -> LLMEmbeddingResponse:
        import asyncio

        return await asyncio.to_thread(_base_embedding, **kwargs)

    return _base_embedding, _base_embedding_async
