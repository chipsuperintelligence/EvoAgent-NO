"""ClaudeCLI against a fake `claude` executable: command line, parsing, and every failure path."""

import json
import os
import stat
import textwrap

import pytest

from evoagent_no.cli import main
from evoagent_no.llm import LLMError, make_llm
from evoagent_no.llm.claude_cli import ClaudeCLI

OK = {
    "type": "result", "is_error": False, "result": "pong", "total_cost_usd": 0.0012,
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 0,
              "output_tokens": 3},
    "modelUsage": {"claude-sonnet-5": {}},
}


def fake_claude(tmp_path, stdout, *, exit_code=0, sleep=0.0):
    """Write an executable that records argv, stdin and cwd, then prints `stdout`."""
    log = tmp_path / "call.json"
    script = tmp_path / "claude"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import json, os, sys, time
        json.dump({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd()}},
                  open({str(log)!r}, "w"))
        time.sleep({sleep})
        sys.stdout.write({stdout!r})
        sys.exit({exit_code})
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), log


def test_reply_is_parsed_and_prompt_goes_through_stdin(tmp_path):
    exe, log = fake_claude(tmp_path, json.dumps(OK))
    reply = ClaudeCLI(executable=exe).complete("Reply with pong", system="Be terse.")

    assert reply.text == "pong"
    assert reply.model == "claude-sonnet-5"
    assert (reply.input_tokens, reply.output_tokens) == (15, 3)
    assert reply.cost_usd == pytest.approx(0.0012)

    call = json.loads(log.read_text())
    argv = call["argv"]
    assert call["stdin"] == "Reply with pong"
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"
    assert argv[argv.index("--effort") + 1] == "medium"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--system-prompt") + 1] == "Be terse."
    assert "--no-session-persistence" in argv
    # Runs in a throwaway directory so no project CLAUDE.md is picked up, and cleans it up.
    assert os.path.basename(call["cwd"]).startswith("evoagent-no-llm-")
    assert not os.path.exists(call["cwd"])


def test_not_logged_in_says_how_to_fix(tmp_path):
    exe, _ = fake_claude(tmp_path, json.dumps({"is_error": True, "result": "Not logged in · Please run /login"}),
                         exit_code=1)
    with pytest.raises(LLMError, match="claude auth login"):
        ClaudeCLI(executable=exe).complete("hi")


def test_non_json_output_is_an_error(tmp_path):
    exe, _ = fake_claude(tmp_path, "boom", exit_code=3)
    with pytest.raises(LLMError, match="exited 3"):
        ClaudeCLI(executable=exe).complete("hi")


def test_timeout_is_an_error(tmp_path):
    exe, _ = fake_claude(tmp_path, json.dumps(OK), sleep=5)
    with pytest.raises(LLMError, match="within 1 s"):
        ClaudeCLI(executable=exe, timeout_s=1).complete("hi")


def test_missing_executable_is_an_error(tmp_path):
    with pytest.raises(LLMError, match="not found"):
        ClaudeCLI(executable=str(tmp_path / "nope")).complete("hi")


def test_bad_effort_and_backend_are_rejected():
    with pytest.raises(ValueError, match="effort"):
        ClaudeCLI(effort="huge")
    with pytest.raises(ValueError, match="unknown LLM backend"):
        make_llm("nope")


def test_cli_check_reports_ok(tmp_path, monkeypatch, capsys):
    exe, _ = fake_claude(tmp_path, json.dumps(OK))
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert main(["llm", "check"]) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-5 (effort medium)" in out and out.strip().endswith("OK")


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        usage = type("Usage", (), {"prompt_tokens": 12, "completion_tokens": 3})()
        message = type("Message", (), {"content": "pong"})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice], "usage": usage, "model": "gpt-test"})()


def test_openai_backend_sends_system_prompt_and_effort():
    completions = FakeCompletions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    llm = make_llm("openai", model="gpt-test", effort="xhigh", client=client)
    reply = llm.complete("hello", system="Be terse.")
    request = completions.requests[0]
    assert request["messages"][0] == {"role": "system", "content": "Be terse."}
    assert request["reasoning_effort"] == "high"
    assert (reply.text, reply.model, reply.input_tokens, reply.output_tokens) == ("pong", "gpt-test", 12, 3)


def test_openai_backend_needs_a_model():
    with pytest.raises(ValueError, match="needs a model"):
        make_llm("openai", client=object())


def test_effort_none_leaves_the_flag_out_and_cached_tokens_are_reported(tmp_path):
    data = {**OK, "usage": {**OK["usage"], "cache_read_input_tokens": 700}}
    exe, log = fake_claude(tmp_path, json.dumps(data))
    reply = ClaudeCLI(executable=exe, model="claude-haiku-4-5", effort=None).complete("hi")
    argv = json.loads(log.read_text())["argv"]
    assert "--effort" not in argv and argv[argv.index("--model") + 1] == "claude-haiku-4-5"
    assert reply.cached_tokens == 700 and reply.input_tokens == 710
