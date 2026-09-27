"""Generator programs: the Python file that poses problems, and the EVOLVE-BLOCK agents rewrite.

A generator defines `generate_problems(rng, n)` and `mutate_problem(problem, rng)`. Agents edit
only the code between the markers; `splice` puts their block into the current file, so the rest
of the file cannot drift. Before a rewrite is loaded into the run, `validate` exercises it in a
separate Python process with a time limit, so a broken or hanging rewrite cannot take the run down.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from evoagent_no.task import load_module

START, END = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"


def split(source: str) -> tuple[str, str, str]:
    """(before, block, after); the markers stay in `before` and `after`."""
    if source.count(START) != 1 or source.count(END) != 1:
        raise ValueError(f"expected exactly one {START} … {END} pair")
    head, rest = source.split(START, 1)
    block, tail = rest.split(END, 1)
    return head + START, block, END + tail


def splice(source: str, block: str) -> str:
    before, _, after = split(source)
    return before + "\n" + block.strip("\n") + "\n" + after


def extract_code(reply: str) -> str:
    """The last ```python fenced block in an LLM reply."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, flags=re.DOTALL)
    if not blocks:
        raise ValueError("the reply has no ```python code block")
    return blocks[-1]


@dataclass
class GeneratorProgram:
    path: Path
    source: str
    module: ModuleType

    @classmethod
    def load(cls, path: str | Path) -> "GeneratorProgram":
        path = Path(path).resolve()
        module = load_module(str(path))
        for name in ("generate_problems", "mutate_problem"):
            if not callable(getattr(module, name, None)):
                raise ValueError(f"{path.name} must define {name}()")
        return cls(path=path, source=path.read_text(), module=module)

    def generate(self, rng, n: int) -> list:
        problems = self.module.generate_problems(rng, n)
        if not isinstance(problems, list):
            raise ValueError(f"generate_problems returned {type(problems).__name__}, not a list")
        return problems

    def mutate(self, problem, rng):
        return self.module.mutate_problem(problem, rng)


_VALIDATE = """
import sys
import numpy as np
from evoagent_no.task import load_module
physics = load_module(sys.argv[2], __import__("pathlib").Path(sys.argv[3]))
sys.modules["task_physics"] = physics
generator = load_module(sys.argv[1])
rng = np.random.default_rng(0)
problems = generator.generate_problems(rng, 64)
if not isinstance(problems, list) or len(problems) != 64:
    raise SystemExit(f"generate_problems(rng, 64) returned {type(problems).__name__} of length {len(problems) if hasattr(problems, '__len__') else '?'}")
for p in problems:
    physics.check(p)
for p in problems[:16]:
    physics.check(generator.mutate_problem(p, rng))
print("ok")
"""


def validate(path: Path, physics_ref: str, task_root: Path, timeout_s: float = 60.0) -> None:
    """Raise ValueError unless the generator at `path` runs and produces well-formed problems."""
    try:
        proc = subprocess.run([sys.executable, "-c", _VALIDATE, str(path), physics_ref, str(task_root)],
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise ValueError(f"generator did not finish within {timeout_s:.0f} s") from None
    if proc.returncode != 0 or proc.stdout.strip() != "ok":
        detail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        raise ValueError(detail[-1] if detail else f"validation exited {proc.returncode}")
