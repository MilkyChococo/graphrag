# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Local Hugging Face tokenizer."""

from functools import lru_cache
from typing import Any

from graphrag_llm.tokenizer.tokenizer import Tokenizer
from graphrag_llm.utils.local_hf import ensure_hf_token_env


@lru_cache(maxsize=8)
def _load_tokenizer(model_id: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    ensure_hf_token_env()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class LocalHFTokenizer(Tokenizer):
    """Tokenizer backed by Hugging Face AutoTokenizer."""

    _model_id: str
    _trust_remote_code: bool

    def __init__(
        self,
        *,
        model_id: str,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the local Hugging Face tokenizer."""
        self._model_id = model_id
        self._trust_remote_code = trust_remote_code
        self._tokenizer = _load_tokenizer(model_id, trust_remote_code)

    def encode(self, text: str) -> list[int]:
        """Encode the given text into a list of token ids."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token ids back into a string."""
        return self._tokenizer.decode(tokens, skip_special_tokens=False)
