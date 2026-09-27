"""LLM backends for the slow, agentic loop.

The fast self-play loop never calls an LLM. Agents use one to write code: new problem
types, kernels, reward rules. Every backend has the same `complete()` method, so the
agents never know which provider answered.

Backends: `claude-cli` (default: Claude Sonnet 5 at medium effort, through the Claude Code CLI and
your own login) and `openai` (OpenAI or any OpenAI-compatible endpoint; name the model).
`from_settings` builds one backend, or a weighted `Ensemble` of several, from task.yaml.
"""

from __future__ import annotations

from evoagent_no.llm.base import LLM, LLMError, Reply
from evoagent_no.llm.claude_cli import ClaudeCLI
from evoagent_no.llm.ensemble import Ensemble
from evoagent_no.llm.openai_api import OpenAIChat

BACKENDS = {"claude-cli": ClaudeCLI, "openai": OpenAIChat}
DEFAULT_BACKEND = "claude-cli"


def make_llm(backend: str = DEFAULT_BACKEND, **options) -> LLM:
    """Build a backend by name. Options go to its constructor (model, effort, timeout_s, ...)."""
    try:
        cls = BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown LLM backend {backend!r}; choose from {', '.join(BACKENDS)}") from None
    return cls(**options)


def from_settings(settings: dict | list[dict], seed: int = 0) -> LLM:
    """task.yaml's `agents.llm`: one model ({backend, model, effort, budget_usd}) or a list of them,
    each with a `weight`, which becomes an Ensemble."""
    entries = settings if isinstance(settings, list) else [settings]
    members, weights = [], []
    for entry in entries:
        options = dict(entry)
        backend = options.pop("backend", DEFAULT_BACKEND)
        weights.append(float(options.pop("weight", 1.0)))
        if "budget_usd" in options:
            options["max_budget_usd"] = options.pop("budget_usd")
        members.append(make_llm(backend, **options))
    return members[0] if len(members) == 1 else Ensemble(members, weights, seed=seed)


__all__ = ["BACKENDS", "DEFAULT_BACKEND", "LLM", "LLMError", "Ensemble", "Reply", "from_settings", "make_llm"]
