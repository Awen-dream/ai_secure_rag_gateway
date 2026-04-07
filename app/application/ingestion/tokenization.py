from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Protocol

from app.core.config import settings

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class _Tokenizer(Protocol):
    def encode(self, text: str, *, disallowed_special: tuple[str, ...] = ()) -> list[int]:
        ...


def approximate_token_count(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0

    cjk_chars = len(_CJK_CHAR_RE.findall(cleaned))
    other_chars = len(re.sub(r"[\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", "", cleaned))
    return max(cjk_chars + math.ceil(other_chars / 4), 1)


@lru_cache(maxsize=4)
def _get_tokenizer(model: str, encoding_name: str | None) -> _Tokenizer | None:
    if tiktoken is None:
        return None

    if encoding_name:
        try:
            return tiktoken.get_encoding(encoding_name)
        except KeyError:
            return None

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except KeyError:
            return None


def estimate_token_count(
    text: str,
    *,
    model: str | None = None,
    encoding_name: str | None = None,
) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0

    resolved_model = model or settings.chunk_tokenizer_model
    resolved_encoding = encoding_name if encoding_name is not None else settings.chunk_tokenizer_encoding
    tokenizer = _get_tokenizer(resolved_model, resolved_encoding)
    if tokenizer is None:
        return approximate_token_count(cleaned)
    return len(tokenizer.encode(cleaned, disallowed_special=()))
