"""A generator that only knows the heat equation, u_t = ν u_xx.

Every problem it poses is pure diffusion. Self-play alone cannot leave that family: the pool
only mutates what the generator gives it. The student will later be tested on advection,
Burgers, KdV and reaction fronts, which it never sees here. With agents on, an LLM reads
context.md (the whole equation the answer key can solve) and rewrites this file; a rewrite is
kept only if the student learns more from its problems.
"""

import math

import numpy as np

from evoagent_no.physics.pde1d import MAX_T0_FRAMES, TERMS, Problem

# EVOLVE-BLOCK-START
def _heat(nu: float) -> tuple[float, ...]:
    return tuple(nu if term == "diffusion" else 0.0 for term in TERMS)


def generate_problems(rng: np.random.Generator, n: int) -> list[Problem]:
    """n heat-equation problems with random diffusivity and random initial fields."""
    return [Problem(
        coef=_heat(math.exp(rng.uniform(math.log(1e-3), math.log(0.3)))),
        ic_amp=math.exp(rng.uniform(math.log(0.1), math.log(2.0))),
        ic_slope=float(rng.uniform(1.0, 4.0)),
        ic_mean=float(rng.uniform(-0.5, 0.5)),
        ic_squash=False,
        t0_frames=int(rng.integers(0, MAX_T0_FRAMES + 1)),
        seed=int(rng.integers(0, 2**31 - 1)),
    ) for _ in range(n)]


def mutate_problem(problem: Problem, rng: np.random.Generator) -> Problem:
    """A nearby heat problem: nudged diffusivity and fields."""
    nu = float(np.clip(problem.term("diffusion") * math.exp(rng.normal(0.0, 0.3)), 1e-4, 1.0))
    return Problem(
        coef=_heat(nu),
        ic_amp=float(np.clip(problem.ic_amp * math.exp(rng.normal(0.0, 0.2)), 0.05, 3.0)),
        ic_slope=float(np.clip(problem.ic_slope + rng.normal(0.0, 0.3), 0.5, 5.0)),
        ic_mean=float(np.clip(problem.ic_mean + rng.normal(0.0, 0.1), -1.0, 1.0)),
        ic_squash=False,
        t0_frames=int(np.clip(problem.t0_frames + rng.integers(-1, 2), 0, MAX_T0_FRAMES)),
        seed=int(rng.integers(0, 2**31 - 1)) if rng.random() < 0.5 else problem.seed,
    )
# EVOLVE-BLOCK-END
