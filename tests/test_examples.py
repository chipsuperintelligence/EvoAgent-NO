"""The examples: every one passes `check`; the ones with their own physics.py and heldout.py also
run end to end (the bring-your-own-physics path)."""

import json
import shutil
from pathlib import Path

import pytest

from evoagent_no.cli import main
from evoagent_no.llm.base import Reply
from evoagent_no.problems.generator import GeneratorProgram, splice, split
from evoagent_no.selfplay import run
from evoagent_no.task import load_task

EXAMPLES = Path(__file__).parent.parent / "examples"


class EchoLLM:
    """Replies with the current generator plus a harmless edit, so the agent path runs end to end."""

    def __init__(self, source):
        self.source = splice(source, split(source)[1] + "# tweaked by the test\n")

    def complete(self, prompt, *, system=None):
        return Reply(text=f"No change.\n```python\n{self.source}```", model="fake", input_tokens=0,
                     output_tokens=0, cost_usd=None, seconds=0.0)


@pytest.mark.parametrize("example", ["transport1d", "heat3d", "broad-guess", "heat-only"])
def test_example_passes_check(example, tmp_path, capsys):
    folder = shutil.copytree(EXAMPLES / example, tmp_path / example)
    assert main(["check", str(folder), "--device", "cpu"]) == 0
    assert "ready." in capsys.readouterr().out


@pytest.mark.parametrize("example, families", [("transport1d", {"long_advection", "pure_diffusion"}),
                                               ("heat3d", {"long_time", "rough_fields"})])
def test_example_runs_with_agents_rewriting_its_generator(example, families, tmp_path):
    task = load_task(shutil.copytree(EXAMPLES / example, tmp_path / example))
    task.settings["student"].update(width=8, modes=4, layers=2)
    task.settings["agents"].update(every=2, candidates=1, probe_problems=16)
    source = GeneratorProgram.load(task.generator_path).source
    out = tmp_path / "run"

    final = run(task, out, device="cpu", agents=True, llm=EchoLLM(source), log=lambda *_: None,
                rounds=3, problems_per_round=16, steps_per_round=2, batch=8, eval_every=2, eval_count=4)

    assert families | {"mean"} <= set(final)
    attempt = json.loads((out / "agents.jsonl").read_text().splitlines()[0])
    assert attempt["score"] is not None, attempt["reason"]     # the rewrite loaded and was scored
