from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from graphrag_llm.tokenizer import Tokenizer
from graphrag_llm.utils import CompletionMessagesBuilder

from graphrag.callbacks.query_callbacks import QueryCallbacks
from graphrag.prompts.query.basic_search_system_prompt import (
    BASIC_SEARCH_SYSTEM_PROMPT,
)
from graphrag.query.context_builder.builders import BasicContextBuilder
from graphrag.query.context_builder.conversation_history import ConversationHistory
from graphrag.query.structured_search.base import BaseSearch, SearchResult

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionChunk

logger = logging.getLogger(__name__)


class BasicSearch(BaseSearch[BasicContextBuilder]):
    """Search orchestration for basic search mode."""

    def __init__(
        self,
        model: "LLMCompletion",
        context_builder: BasicContextBuilder,
        tokenizer: Tokenizer | None = None,
        system_prompt: str | None = None,
        response_type: str = "multiple paragraphs",
        callbacks: list[QueryCallbacks] | None = None,
        model_params: dict[str, Any] | None = None,
        context_builder_params: dict | None = None,
    ):
        super().__init__(
            model=model,
            context_builder=context_builder,
            tokenizer=tokenizer,
            model_params=model_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt or BASIC_SEARCH_SYSTEM_PROMPT
        self.callbacks = callbacks or []
        self.response_type = response_type

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        content = getattr(response, "content", None)
        if content is not None:
            return str(content)

        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                msg_content = getattr(message, "content", None)
                if msg_content is not None:
                    return str(msg_content)

        return ""

    @staticmethod
    def _normalize_context_text(context_chunks: Any) -> str:
        if isinstance(context_chunks, list):
            return "\n\n".join(str(x) for x in context_chunks)
        return str(context_chunks)

    async def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        start_time = time.time()

        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )

        context_text = self._normalize_context_text(context_result.context_chunks)
        context_records = context_result.context_records

        logger.info("GENERATE ANSWER: %s. QUERY: %s", context_text, query)

        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text,
                response_type=self.response_type,
            )

            messages_builder = (
                CompletionMessagesBuilder()
                .add_system_message(search_prompt)
                .add_user_message(query)
                )

            for callback in self.callbacks:
                callback.on_context(context_records)

            response_obj = await self.model.completion_async(
                messages=messages_builder.build(),
                **self.model_params,
            )
            response = self._extract_response_text(response_obj)

            if response:
                for callback in self.callbacks:
                    callback.on_llm_new_token(response)

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1 + getattr(context_result, "llm_calls", 0),
                prompt_tokens=getattr(context_result, "prompt_tokens", 0),
                output_tokens=getattr(context_result, "output_tokens", 0),
            )
        except Exception:
            logger.exception("error generating answer")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1 + getattr(context_result, "llm_calls", 0),
                prompt_tokens=getattr(context_result, "prompt_tokens", 0),
                output_tokens=getattr(context_result, "output_tokens", 0),
            )

    async def stream_search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
    ) -> AsyncGenerator[str, None]:
        context_result = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **self.context_builder_params,
        )

        context_text = self._normalize_context_text(context_result.context_chunks)
        context_records = context_result.context_records

        logger.info("GENERATE ANSWER: %s. QUERY: %s", context_text, query)

        search_prompt = self.system_prompt.format(
            context_data=context_text,
            response_type=self.response_type,
        )

        messages_builder = (
                CompletionMessagesBuilder()
                .add_system_message(search_prompt)
                .add_user_message(query)
            )

        for callback in self.callbacks:
            callback.on_context(context_records)

        try:
            response_stream: AsyncIterator[LLMCompletionChunk] = await self.model.completion_async(
                messages=messages_builder.build(),
                stream=True,
                **self.model_params,
            )  # type: ignore

            async for chunk in response_stream:
                response_text = chunk.choices[0].delta.content or ""
                for callback in self.callbacks:
                    callback.on_llm_new_token(response_text)
                yield response_text

        except ValueError as e:
            if "does not support streaming completions" not in str(e):
                raise

            response_obj = await self.model.completion_async(
                messages=messages_builder.build(),
                **self.model_params,
            )
            response_text = self._extract_response_text(response_obj)

            if response_text:
                for callback in self.callbacks:
                    callback.on_llm_new_token(response_text)
                yield response_text