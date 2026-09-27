"""Claude through the Claude Code CLI (`claude -p`).

Authentication is the CLI's own login (`claude auth login`), so no API key is needed.
Each call is a plain completion: tools are switched off (`--tools ""`, so there is nothing to
approve and no agent loop), the system prompt is ours, no MCP servers load, and the CLI runs in an
empty temporary directory so no project CLAUDE.md is loaded. `effort=None` leaves the flag out, for
models without effort levels.
Flags verified against `claude --help`, Claude Code 2.1.278.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time

from evoagent_no.llm.base import LLMError, Reply

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_EFFORT = "medium"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_SYSTEM = "You are a precise assistant. Answer the request directly, with no preamble."


class ClaudeCLI:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str | None = DEFAULT_EFFORT,
        timeout_s: float = 600.0,
        max_budget_usd: float = 1.0,
        executable: str = "claude",
        **_ignored,
    ):
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {', '.join(EFFORTS)} or None; got {effort!r}")
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.max_budget_usd = max_budget_usd
        self.executable = executable

    def command(self, system: str | None = None) -> list[str]:
        effort = ["--effort", self.effort] if self.effort is not None else []   # None: the model's default
        return [
            self.executable, "-p",
            "--model", self.model,
            *effort,
            "--tools", "",
            "--system-prompt", system or DEFAULT_SYSTEM,
            "--output-format", "json",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--max-budget-usd", str(self.max_budget_usd),
        ]

    def complete(self, prompt: str, *, system: str | None = None) -> Reply:
        # The prompt goes in on stdin: argv has a length limit, and prompts carry whole programs.
        start = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="evoagent-no-llm-") as cwd:
                proc = subprocess.run(
                    self.command(system), input=prompt, capture_output=True, text=True,
                    timeout=self.timeout_s, cwd=cwd,
                )
        except FileNotFoundError:
            raise LLMError(f"`{self.executable}` not found. Install Claude Code, then run `claude auth login`.") from None
        except subprocess.TimeoutExpired:
            raise LLMError(f"Claude CLI gave no answer within {self.timeout_s:.0f} s.") from None
        seconds = time.monotonic() - start

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            detail = (proc.stderr or proc.stdout).strip()[-500:]
            raise LLMError(f"Claude CLI exited {proc.returncode} without a JSON result: {detail}") from None

        if data.get("is_error"):
            message = str(data.get("result") or data.get("terminal_reason") or "unknown error")
            if "logged in" in message.lower() or "/login" in message:
                message += " (run `claude auth login` in a terminal)"
            raise LLMError(f"Claude CLI: {message}")

        usage = data.get("usage") or {}
        models = list((data.get("modelUsage") or {}).keys())
        return Reply(
            text=str(data.get("result", "")),
            model=models[0] if models else self.model,
            input_tokens=int(usage.get("input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            cost_usd=data.get("total_cost_usd"),
            seconds=seconds,
            cached_tokens=int(usage.get("cache_read_input_tokens", 0)),
        )
