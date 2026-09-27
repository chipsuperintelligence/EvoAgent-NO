"""OpenAI models, or any OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama), via the `openai` SDK.

Install with `uv sync --extra openai` (or `pip install evoagent-no[openai]`). The key comes from
OPENAI_API_KEY unless `api_key` is given; `base_url` points at a compatible endpoint. There is no
default model: name one in task.yaml (`agents.llm.model`).
"""

from __future__ import annotations

import time

from evoagent_no.llm.base import LLMError, Reply
from evoagent_no.llm.claude_cli import DEFAULT_SYSTEM, EFFORTS

# EvoAgent-NO's effort levels mapped onto OpenAI's reasoning_effort.
REASONING_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class OpenAIChat:
    def __init__(self, model: str | None = None, effort: str | None = "medium", timeout_s: float = 600.0,
                 base_url: str | None = None, api_key: str | None = None, max_tokens: int = 16000,
                 client=None, **_ignored):
        if not model:
            raise ValueError("the openai backend needs a model name (agents.llm.model in task.yaml)")
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)}; got {effort!r}")
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise LLMError("the openai package is not installed: uv sync --extra openai") from None
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self.client = client

    def complete(self, prompt: str, *, system: str | None = None) -> Reply:
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system or DEFAULT_SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        if self.effort is not None:
            request["reasoning_effort"] = REASONING_EFFORT[self.effort]
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as e:   # the SDK's error types, network errors, bad parameters
            raise LLMError(f"OpenAI request failed: {type(e).__name__}: {e}") from None
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return Reply(
            text=text,
            model=getattr(response, "model", self.model),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=None,
            seconds=time.monotonic() - start,
            cached_tokens=int(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0),
        )
