"""A first guess at a generator, written without knowing which problems are useful.

Every term is switched on with even odds and its coefficient is drawn uniformly over everything
`check` allows; the initial field's amplitude spans three orders of magnitude. Most of what this
poses blows up, is under-resolved, or decays to nothing within a frame. That is the point of this
example: the pool, not the person, finds the part of it worth learning from.
"""

import math

import numpy as np

from evoagent_no.physics.pde1d import COEF_LIMITS, FIELD_LIMITS, MAX_T0_FRAMES, TERMS, Problem

# EVOLVE-BLOCK-START
def generate_problems(rng: np.random.Generator, n: int) -> list[Problem]:
    """n fresh problems: every term on or off with even odds, anywhere inside the limits."""
    problems = []
    for _ in range(n):
        coef = tuple(float(rng.uniform(*COEF_LIMITS[t])) if rng.random() < 0.5 else 0.0 for t in TERMS)
        if not any(coef):   # `check` refuses u_t = 0
            coef = (float(rng.uniform(*COEF_LIMITS["diffusion"])),) + coef[1:]
        problems.append(Problem(
            coef=coef,
            ic_amp=math.exp(rng.uniform(math.log(1e-2), math.log(FIELD_LIMITS["ic_amp"][1]))),
            ic_slope=float(rng.uniform(*FIELD_LIMITS["ic_slope"])),
            ic_mean=float(rng.uniform(-1.0, 1.0)),
            ic_squash=bool(rng.random() < 0.5),
            t0_frames=int(rng.integers(0, MAX_T0_FRAMES + 1)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return problems


def mutate_problem(problem: Problem, rng: np.random.Generator) -> Problem:
    """A nearby problem: each coefficient nudged by a tenth of its range, sometimes switched."""
    coef = []
    for term, value in zip(TERMS, problem.coef):
        lo, hi = COEF_LIMITS[term]
        if rng.random() < 0.1:
            value = 0.0 if value != 0.0 else float(rng.uniform(lo, hi))
        elif value != 0.0:
            value += rng.normal(0.0, 0.1 * (hi - lo))
        coef.append(float(np.clip(value, lo, hi)))
    if not any(coef):
        coef[0] = float(rng.uniform(*COEF_LIMITS["diffusion"]))
    return Problem(
        coef=tuple(coef),
        ic_amp=float(np.clip(problem.ic_amp * math.exp(rng.normal(0.0, 0.3)), 1e-2, 5.0)),
        ic_slope=float(np.clip(problem.ic_slope + rng.normal(0.0, 0.5), 0.0, 8.0)),
        ic_mean=float(np.clip(problem.ic_mean + rng.normal(0.0, 0.2), -1.0, 1.0)),
        ic_squash=problem.ic_squash if rng.random() > 0.1 else not problem.ic_squash,
        t0_frames=int(np.clip(problem.t0_frames + rng.integers(-1, 2), 0, MAX_T0_FRAMES)),
        seed=int(rng.integers(0, 2**31 - 1)) if rng.random() < 0.5 else problem.seed,
    )
# EVOLVE-BLOCK-END
