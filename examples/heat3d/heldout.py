"""Held-out tests for heat3d: long times and very smooth or very rough fields, never shown to the
agents. The answer key is exact, so the references are too."""

import numpy as np
import task_physics as physics   # this task's physics.py, as named in task.yaml
import torch


def build(count: int = 16, device=None, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    families = {
        "long_time": lambda: physics.Problem(float(rng.uniform(0.005, 0.02)), float(rng.uniform(0.6, 1.0)),
                                             float(rng.uniform(1.5, 3.0)), int(rng.integers(2**31))),
        "rough_fields": lambda: physics.Problem(float(rng.uniform(0.001, 0.01)), float(rng.uniform(0.1, 0.4)),
                                                float(rng.uniform(0.5, 1.0)), int(rng.integers(2**31))),
    }
    out = {}
    for name, make in families.items():
        problems = [make() for _ in range(count)]
        u0, uT = physics.exact(problems)
        out[name] = (torch.as_tensor(physics.inputs_for(problems, u0), dtype=torch.float32, device=device),
                     torch.as_tensor(uT[:, None], dtype=torch.float32, device=device))
    return out
