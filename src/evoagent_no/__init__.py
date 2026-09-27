"""EvoAgent-NO: LLM agents + neural operators + self-play.

A generator program invents physics problems, an exact solver answers them, and a neural operator
(the student) trains on the answers, while LLM agents evolve the generator toward problems the
student learns more from. Self-play within each round is optional. The student is pretrained with
zero downloaded data.

    import evoagent_no as evo
    task = evo.load_task("tasks/my-pde")                 # a task folder (see docs/TUTORIAL.md)
    evo.run(task, "runs/my-pde/agents-s0")               # agents=False: the baseline; selfplay=True: add self-play
    student = evo.load_student("runs/my-pde/agents-s0")
    prediction = student.predict(inputs)

The command line (`evoagent-no --help`) does the same.
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def load_task(path):
    """A task folder: task.yaml, generator.py, context.md, and optionally physics.py / heldout.py."""
    from evoagent_no.task import load_task as _load
    return _load(path)


def run(task, out_dir=None, seed: int = 0, device: str | None = None, **options) -> dict:
    """Pretrain a student on a task or task folder; returns the final held-out scores. Options:
    agents, selfplay (both default to task.yaml), resume, rounds, problems_per_round, llm, log, ..."""
    from evoagent_no.selfplay import label, resolve
    from evoagent_no.selfplay import run as _run
    from evoagent_no.student.pretrained import _default_device
    task = load_task(task) if isinstance(task, (str, Path)) else task
    variant = label(*resolve(task, options.get("agents"), options.get("selfplay")))
    out_dir = Path(out_dir or f"runs/{task.name}/{variant}-s{seed}")
    return _run(task, out_dir, seed=seed, device=device or _default_device(), **options)


def load_student(run_dir, device: str | None = None):
    """The pretrained student from a run folder, ready to predict and fine-tune."""
    from evoagent_no.student.pretrained import load_student as _load
    return _load(run_dir, device)


__all__ = ["__version__", "load_student", "load_task", "run"]
