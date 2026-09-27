"""What every LLM backend returns, and the one method it must have."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    """A backend could not produce an answer. The message says why and what to do."""


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    seconds: float
    cached_tokens: int = 0    # of input_tokens, how many were read from the prompt cache


class LLM(Protocol):
    """A frozen model that turns a prompt into text. Backends: see `evoagent_no.llm.BACKENDS`."""

    def complete(self, prompt: str, *, system: str | None = None) -> Reply: ...
