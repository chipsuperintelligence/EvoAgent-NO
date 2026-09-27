"""The question-maker for this task: which steady diffusion problems to pose, and how to vary them.

EvoAgent-NO samples `generate_problems` thousands of times per round and uses `mutate_problem`
to breed variations of problems the student learned from. With agents on, an LLM rewrites the
code between the EVOLVE-BLOCK markers, and a rewrite is kept only if the student learns more
from its problems. Everything outside the markers stays as it is.

Problems must pass `evoagent_no.physics.elliptic2d.check` (see LIMITS and MAX_CONTRAST there).
"""

import math

import numpy as np

from evoagent_no.physics.elliptic2d import MAX_CONTRAST, Problem

# EVOLVE-BLOCK-START
def _log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def generate_problems(rng: np.random.Generator, n: int) -> list[Problem]:
    """n fresh problems: coefficients from constant to two-phase, sources from uniform to varying."""
    problems = []
    for _ in range(n):
        a_lo = _log_uniform(rng, 0.1, 1.0)
        contrast = _log_uniform(rng, 1.0, 100.0)
        f_mean = float(rng.uniform(-2.0, 2.0)) if rng.random() < 0.5 else 0.0
        f_amp = _log_uniform(rng, 0.1, 3.0) if rng.random() < 0.8 else 0.0
        if f_mean == 0.0 and f_amp == 0.0:
            f_amp = 1.0
        problems.append(Problem(
            a_lo=a_lo,
            a_hi=a_lo * contrast,
            a_slope=float(rng.uniform(1.5, 4.0)),
            sharpness=0.0 if rng.random() < 0.2 else _log_uniform(rng, 0.5, 15.0),
            f_mean=f_mean,
            f_amp=f_amp,
            f_slope=float(rng.uniform(1.5, 4.0)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return problems


def mutate_problem(problem: Problem, rng: np.random.Generator) -> Problem:
    """A nearby problem: nudged contrast, sharpness and source, often fresh random fields."""
    a_lo = float(np.clip(problem.a_lo * math.exp(rng.normal(0.0, 0.3)), 0.01, 10.0))
    contrast = float(np.clip(problem.a_hi / problem.a_lo * math.exp(rng.normal(0.0, 0.4)), 1.0, MAX_CONTRAST))
    return Problem(
        a_lo=a_lo,
        a_hi=min(a_lo * contrast, 100.0),
        a_slope=float(np.clip(problem.a_slope + rng.normal(0.0, 0.3), 0.5, 6.0)),
        sharpness=float(np.clip(problem.sharpness * math.exp(rng.normal(0.0, 0.3)), 0.0, 50.0)),
        f_mean=float(np.clip(problem.f_mean + rng.normal(0.0, 0.3), -10.0, 10.0)),
        f_amp=float(np.clip(problem.f_amp * math.exp(rng.normal(0.0, 0.3)), 0.0, 10.0)) or 0.1,
        f_slope=float(np.clip(problem.f_slope + rng.normal(0.0, 0.3), 0.5, 6.0)),
        seed=int(rng.integers(0, 2**31 - 1)) if rng.random() < 0.5 else problem.seed,
    )
# EVOLVE-BLOCK-END
