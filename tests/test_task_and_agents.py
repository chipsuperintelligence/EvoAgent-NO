"""Task folders, the CLI around them, the agent loop (with a fake LLM), and a tiny end-to-end run."""

import json

import pytest
import torch

from evoagent_no.cli import main
from evoagent_no.llm.base import Reply
from evoagent_no.problems.generator import GeneratorProgram, split, splice
from evoagent_no.selfplay import run, training_report
from evoagent_no.task import load_task


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, *, system=None):
        self.prompts.append(prompt)
        return Reply(text=self.replies.pop(0), model="fake", input_tokens=0, output_tokens=0,
                     cost_usd=None, seconds=0.0)


def rewrite(source: str) -> str:
    """An LLM-style reply: a new generator.py whose block turns diffusion on more often."""
    block = split(source)[1].replace('"diffusion":      (0.8,', '"diffusion":      (0.9,') + "# MARK\n"
    return f"Diffusion on more often.\n```python\n{splice(source, block)}```\n"


@pytest.mark.parametrize("template", ["pde1d", "elliptic2d"])
def test_new_and_check_from_the_command_line(template, tmp_path, capsys):
    folder = tmp_path / "my-task"
    assert main(["new", template, str(folder)]) == 0
    assert {p.name for p in folder.iterdir()} >= {"task.yaml", "generator.py", "context.md"}
    assert load_task(folder).name == "my-task"                 # named after its folder
    assert main(["new", template, str(folder)]) == 2          # refuses to overwrite
    assert main(["check", str(folder), "--device", "cpu"]) == 0
    assert "ready." in capsys.readouterr().out


def test_task_yaml_only_needs_what_differs_from_the_defaults(task_dir):
    (task_dir / "task.yaml").write_text("name: tiny\ntraining:\n  rounds: 7\n")
    task = load_task(task_dir)
    assert task.name == "tiny"
    assert task.section("training")["rounds"] == 7
    assert task.section("training")["problems_per_round"] == 256      # default kept
    assert task.section("agents")["enabled"] and not task.section("selfplay")["enabled"]
    assert task.section("agents")["llm"]["model"] == "claude-sonnet-5"


def test_the_report_names_kinds_of_problem():
    records = [{"cell": (1, 0, 1), "origin": "fresh", "accepted": True, "score": 0.3, "loss": 0.1},
               {"cell": (0, 2, 0), "origin": "mutated", "accepted": False, "score": 0.0, "loss": 0.0}]

    class Task:
        from evoagent_no.physics import pde1d as physics

    report = training_report(Task, records, [0.2, 0.1])
    assert "burgers/diffusive/medium" in report and "50% accepted" in report
    assert "Kinds never posed:" in report


def toy_suite(task):
    io = task.physics.IO
    space = (128,) if io["dims"] == 1 else (32, 32)
    return {"toy": (torch.randn(4, io["in_channels"], *space), torch.randn(4, io["out_channels"], *space))}


@pytest.mark.parametrize("selfplay", [False, True])
def test_tiny_run_with_agents_end_to_end(any_small_task, selfplay, tmp_path):
    task = any_small_task
    task.settings["agents"].update(every=2, candidates=1, probe_problems=8)
    source = GeneratorProgram.load(task.generator_path).source
    llm = FakeLLM([f"No change.\n```python\n{source}```\n"])
    out = tmp_path / "run"

    final = run(task, out, device="cpu", suite=toy_suite(task), llm=llm, agents=True, selfplay=selfplay,
                log=lambda *_: None, rounds=3, problems_per_round=8, steps_per_round=2, batch=8, eval_every=2)

    assert "toy" in final and "toy/trivial" in final and (out / "student.pt").is_file()
    rows = [json.loads(line) for line in (out / "metrics.jsonl").open()]
    assert [r["round"] for r in rows if "train_loss" in r] == [0, 1, 2]
    assert len(llm.prompts) == 1 and (out / "agents.jsonl").is_file()
    config = json.loads((out / "config.json").read_text())
    assert config["agents"] is True and config["variant"] == ("selfplay+agents" if selfplay else "agents")


def test_tiny_baseline_run(small_task, tmp_path):
    run(small_task, tmp_path / "prior", device="cpu", suite=toy_suite(small_task), agents=False,
        log=lambda *_: None, rounds=2, problems_per_round=8, steps_per_round=2, batch=8, eval_every=2)
    config = json.loads((tmp_path / "prior" / "config.json").read_text())
    assert config["variant"] == "fixed" and config["agents"] is False and config["selfplay"] is False
