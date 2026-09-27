"""Using EvoAgent-NO end to end: the Python API, resuming runs, and the pretrained student."""

import json

import numpy as np
import pytest

import evoagent_no as evo
from evoagent_no.cli import main
from evoagent_no.llm.base import Reply
from evoagent_no.problems.generator import GeneratorProgram

TINY = dict(problems_per_round=8, steps_per_round=2, batch=8, eval_every=2, eval_count=4, checkpoint_every=2)


class EchoLLM:
    def __init__(self, source):
        self.source, self.calls = source, 0

    def complete(self, prompt, *, system=None):
        self.calls += 1
        return Reply(text=f"Change: none\n```python\n{self.source}```", model="fake", input_tokens=0,
                     output_tokens=0, cost_usd=None, seconds=0.0)


def tiny_suite(task):
    return task.heldout.build(count=4, device="cpu")


@pytest.fixture
def agent_task(small_task):
    small_task.settings["agents"].update(every=1, candidates=1, probe_problems=8, trigger="every")
    return small_task


def test_a_resumed_run_ends_where_an_uninterrupted_one_does(agent_task, tmp_path):
    source = GeneratorProgram.load(agent_task.generator_path).source
    suite = tiny_suite(agent_task)
    quiet = dict(device="cpu", suite=suite, agents=True, log=lambda *_: None)

    straight = evo.run(agent_task, tmp_path / "straight", llm=EchoLLM(source), rounds=4, **quiet, **TINY)
    evo.run(agent_task, tmp_path / "split", llm=EchoLLM(source), rounds=2, **quiet, **TINY)
    resumed = evo.run(agent_task, tmp_path / "split", llm=EchoLLM(source), rounds=4, resume=True, **quiet)

    assert resumed["mean"] == pytest.approx(straight["mean"], rel=1e-5)
    lines = lambda d: [json.loads(x) for x in (tmp_path / d / "agents.jsonl").read_text().splitlines()]
    assert len(lines("split")) == len(lines("straight")) == 3                # rounds 1, 2, 3
    rounds = [r["round"] for r in map(json.loads, (tmp_path / "split" / "metrics.jsonl").open()) if "train_loss" in r]
    assert rounds == [0, 1, 2, 3]


def test_the_cli_refuses_to_overwrite_a_run(small_task, tmp_path, capsys):
    evo.run(small_task, tmp_path / "r", device="cpu", suite=tiny_suite(small_task), log=lambda *_: None,
            rounds=2, **TINY)
    assert main(["run", str(small_task.root), "--out", str(tmp_path / "r"), "--device", "cpu"]) == 2
    assert "--resume" in capsys.readouterr().err


def test_the_pretrained_student_predicts_finetunes_and_saves(small_task, tmp_path):
    run_dir = tmp_path / "r"
    evo.run(small_task, run_dir, device="cpu", suite=tiny_suite(small_task), log=lambda *_: None, rounds=2, **TINY)
    student = evo.load_student(run_dir, device="cpu")
    assert student.io["in_channels"] == 4

    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 4, 128)).astype(np.float32)
    y = (x[:, 3:] * 0.9).astype(np.float32)                   # a simple rule to fit
    assert student.predict(x).shape == (16, 1, 128)
    history = student.finetune(x, y, steps=40, lr=3e-3)
    assert history[-1] < history[0]

    saved = student.save(tmp_path / "ft")
    again = evo.load_student(saved, device="cpu")
    assert np.allclose(again.predict(x), student.predict(x), atol=1e-6)
    with pytest.raises(ValueError, match="input channels"):
        student.predict(x[:, :2])


def test_cli_report_predict_and_finetune(small_task, tmp_path, capsys):
    run_dir = tmp_path / "r"
    evo.run(small_task, run_dir, device="cpu", suite=tiny_suite(small_task), log=lambda *_: None, rounds=2, **TINY)
    assert main(["report", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "held-out" in out and "--resume" in out

    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 4, 128)).astype(np.float32)
    np.save(tmp_path / "x.npy", x)
    assert main(["predict", str(run_dir), "--inputs", str(tmp_path / "x.npy"), "--out", str(tmp_path / "y.npy"),
                 "--device", "cpu"]) == 0
    assert np.load(tmp_path / "y.npy").shape == (10, 1, 128)

    np.savez(tmp_path / "mine.npz", inputs=x, targets=x[:, 3:])
    assert main(["finetune", str(run_dir), "--data", str(tmp_path / "mine.npz"), "--steps", "5",
                 "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "before" in out and "after" in out and (tmp_path / "r-finetuned" / "student.pt").is_file()


def test_the_plateau_trigger_skips_llm_calls_while_the_student_improves(small_task, tmp_path):
    small_task.settings["agents"].update(every=1, candidates=1, probe_problems=8, trigger="plateau")
    source = GeneratorProgram.load(small_task.generator_path).source
    llm = EchoLLM(source)
    evo.run(small_task, tmp_path / "r", device="cpu", suite=tiny_suite(small_task), agents=True, llm=llm,
            log=lambda *_: None, rounds=4, **TINY)
    rows = [json.loads(x) for x in (tmp_path / "r" / "metrics.jsonl").open()]
    skipped = [r for r in rows if r.get("agents") == "skipped"]
    assert [r["round"] for r in skipped][:1] == [2]            # right after an evolution step: wait
    assert llm.calls == 4 - 1 - len(skipped)                   # checks at rounds 1, 2, 3


def test_extending_a_finished_run_does_not_repeat_its_last_evaluation(small_task, tmp_path):
    quiet = dict(device="cpu", suite=tiny_suite(small_task), log=lambda *_: None)
    evo.run(small_task, tmp_path / "r", rounds=2, **quiet, **TINY)
    evo.run(small_task, tmp_path / "r", rounds=4, resume=True, **quiet)
    evals = [r["round"] for r in map(json.loads, (tmp_path / "r" / "metrics.jsonl").open()) if "eval" in r]
    assert evals == [0, 2, 4]
