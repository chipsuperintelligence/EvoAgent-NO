"""The question-maker for heat3d. The agents may rewrite the EVOLVE-BLOCK."""

import numpy as np

from task_physics import Problem   # this task's physics.py, as named in task.yaml

# EVOLVE-BLOCK-START
def generate_problems(rng: np.random.Generator, n: int) -> list:
    return [Problem(diffusivity=float(rng.uniform(0.0, 0.02)),
                    time=float(rng.uniform(0.05, 0.5)),
                    slope=float(rng.uniform(1.0, 4.0)),
                    seed=int(rng.integers(0, 2**31 - 1)))
            for _ in range(n)]


def mutate_problem(problem, rng: np.random.Generator):
    return Problem(diffusivity=float(np.clip(problem.diffusivity * np.exp(rng.normal(0, 0.3)), 0, 0.05)),
                   time=float(np.clip(problem.time + rng.normal(0, 0.05), 0.01, 1.0)),
                   slope=float(np.clip(problem.slope + rng.normal(0, 0.3), 0.5, 6.0)),
                   seed=int(rng.integers(0, 2**31 - 1)))
# EVOLVE-BLOCK-END
