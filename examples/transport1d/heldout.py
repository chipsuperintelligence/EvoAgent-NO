"""Held-out tests for transport1d: `build(count, device) -> {family: (inputs, targets)}`.

The answer key here is exact, so the references are too. What makes these tests held out is that
they come from regions a generator is unlikely to favour (pure advection over a long distance,
pure diffusion) and are never shown to the agents. For a numerical backend, references must come
from a different method than the solver; see `evoagent_no.evals.pde1d` and `.elliptic2d`.
"""

import numpy as np
import task_physics as physics   # this task's physics.py, as named in task.yaml
import torch


def build(count: int = 32, device=None, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    families = {
        "long_advection": lambda: physics.Problem(float(rng.uniform(2, 4) * rng.choice((-1, 1))), 0.0,
                                                  float(rng.uniform(0.5, 1.0)), float(rng.uniform(1, 3)),
                                                  int(rng.integers(2**31))),
        "pure_diffusion": lambda: physics.Problem(0.0, float(rng.uniform(0.005, 0.05)),
                                                  float(rng.uniform(0.2, 1.0)), float(rng.uniform(1, 3)),
                                                  int(rng.integers(2**31))),
    }
    out = {}
    for name, make in families.items():
        problems = [make() for _ in range(count)]
        u0, uT = physics.exact(problems)
        out[name] = (torch.as_tensor(physics.inputs_for(problems, u0), dtype=torch.float32, device=device),
                     torch.as_tensor(uT[:, None], dtype=torch.float32, device=device))
    return out
