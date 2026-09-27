"""A task folder: everything EvoAgent-NO needs to pretrain a student on invented problems.

    my-task/
      task.yaml      settings; missing keys take the defaults below
      generator.py   the question-maker (EVOLVE-BLOCK), rewritten by agents when they are on
      context.md     what the agents are told about the physics
      physics.py     optional: your own answer key, if task.yaml points at ./physics.py
      heldout.py     optional: your own test problems, likewise

A physics backend is a module with `Problem`, `IO`, `check(problem)`, `describe(problem)` and
`solve(problems, device) -> Samples` (see `evoagent_no.physics`). A held-out module has
`build(count, device) -> {family: (inputs, targets)}` in the same layout.

A task's own files reach its physics as `import task_physics`, whichever backend task.yaml names;
that works wherever the file is loaded from, including generators the agents rewrite.

Templates: `pde1d` (1D time stepping) and `elliptic2d` (a 2D input → output operator).
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import yaml

TEMPLATES = Path(__file__).parent / "templates"

DEFAULTS = {
    "name": None,
    "description": "",
    "physics": "evoagent_no.physics.pde1d",
    "heldout": "evoagent_no.evals.pde1d",
    "generator": "generator.py",
    "context": "context.md",
    "student": {"width": 64, "modes": 32, "layers": 4, "padding": 0, "lr": 1e-3,
                "schedule": "cosine", "warmup_steps": 100, "min_lr_ratio": 0.05, "ema": 0.999},
    "training": {"rounds": 200, "problems_per_round": 256, "steps_per_round": 20, "batch": 256,
                 "eval_every": 10, "eval_count": 64, "checkpoint_every": 10},
    "selfplay": {"enabled": False, "pool": {"fresh": 0.4, "mutated": 0.4, "replay": 0.2},
                 "score": "relative", "too_hard": 4.0},
    "agents": {"enabled": True, "every": 20, "candidates": 2, "parallel": 2, "probe_problems": 128,
               "islands": 2, "migrate_every": 3, "rescore": 8, "inspirations": 2, "edits": "diff",
               "trigger": "plateau", "min_improvement": 0.05,
               "llm": {"backend": "claude-cli", "model": "claude-sonnet-5", "effort": "medium",
                       "budget_usd": 1.0}},
}


def load_module(ref: str, base: Path | None = None) -> ModuleType:
    """A module by dotted name, or a .py file (relative to `base` if not absolute). A file is loaded
    once per process; later calls return the same module."""
    if not ref.endswith(".py"):
        return importlib.import_module(ref)
    path = Path(ref) if base is None else (base / ref)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    name = "evoagent_no_user_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    if name in sys.modules:   # one module per file, so a task's generator and physics share classes
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@dataclass
class Task:
    root: Path
    settings: dict
    physics: ModuleType
    heldout: ModuleType

    @property
    def name(self) -> str:
        return self.settings["name"] or self.root.name

    @property
    def generator_path(self) -> Path:
        return self.root / self.settings["generator"]

    @property
    def context(self) -> str:
        path = self.root / self.settings["context"]
        return path.read_text() if path.is_file() else ""

    def section(self, name: str) -> dict:
        return self.settings[name]


def load_task(path: str | Path) -> Task:
    root = Path(path).resolve()
    config_file = root / "task.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"{config_file} not found. Start from a template: evoagent-no new pde1d {path}")
    settings = _merge(DEFAULTS, yaml.safe_load(config_file.read_text()) or {})
    physics = load_module(settings["physics"], root)
    sys.modules["task_physics"] = physics   # generators and held-out files: `import task_physics`
    missing = [f for f in ("Problem", "IO", "check", "describe", "solve") if not hasattr(physics, f)]
    if missing:
        raise ValueError(f"physics module {settings['physics']} lacks {', '.join(missing)}")
    heldout = load_module(settings["heldout"], root)
    if not hasattr(heldout, "build"):
        raise ValueError(f"held-out module {settings['heldout']} lacks build(count, device)")
    if not (root / settings["generator"]).is_file():
        raise FileNotFoundError(f"generator {root / settings['generator']} not found")
    return Task(root=root, settings=settings, physics=physics, heldout=heldout)


def templates() -> list[str]:
    return sorted(p.name for p in TEMPLATES.iterdir() if (p / "task.yaml").is_file())
