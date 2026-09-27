"""The question-maker for this task: which 1D PDE problems to pose, and how to vary them.

EvoAgent-NO samples `generate_problems` thousands of times per round and uses `mutate_problem`
to breed variations of problems the student learned from. With agents on, an LLM rewrites the
code between the EVOLVE-BLOCK markers, and a rewrite is kept only if the student learns more
from its problems. Everything outside the markers stays as it is.

Problems must pass `evoagent_no.physics.pde1d.check` (see COEF_LIMITS and FIELD_LIMITS there).
"""

import math

import numpy as np

from evoagent_no.physics.pde1d import COEF_LIMITS, MAX_T0_FRAMES, TERMS, Problem

# EVOLVE-BLOCK-START
# Probability that a term is on, and how its coefficient is drawn when it is:
#   ("log", lo, hi, signed) log-uniform magnitude; ("lin", lo, hi) uniform.
PRIOR = {
    "diffusion":      (0.8, ("log", 1e-3, 3e-1, False)),
    "hyperdiffusion": (0.2, ("log", 1e-5, 1e-3, False)),
    "advection":      (0.5, ("lin", -2.0, 2.0)),
    "dispersion":     (0.25, ("log", 1e-3, 3e-2, True)),
    "growth":         (0.3, ("lin", -1.0, 1.0)),
    "burgers":        (0.5, ("log", 0.2, 2.0, True)),
    "quadratic":      (0.25, ("lin", -1.5, 1.5)),
    "cubic":          (0.25, ("lin", -1.5, 0.5)),
    "forcing":        (0.2, ("lin", -1.0, 1.0)),
}


def _draw(spec, rng):
    if spec[0] == "log":
        _, lo, hi, signed = spec
        value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        return value * (rng.choice((-1.0, 1.0)) if signed else 1.0)
    _, lo, hi = spec
    return float(rng.uniform(lo, hi))


def generate_problems(rng: np.random.Generator, n: int) -> list[Problem]:
    """n fresh problems."""
    problems = []
    for _ in range(n):
        coef = tuple(_draw(PRIOR[t][1], rng) if rng.random() < PRIOR[t][0] else 0.0 for t in TERMS)
        if not any(coef):   # all terms off would be u_t = 0: nothing to learn
            coef = (_draw(PRIOR["diffusion"][1], rng),) + coef[1:]
        problems.append(Problem(
            coef=coef,
            ic_amp=math.exp(rng.uniform(math.log(0.1), math.log(2.0))),
            ic_slope=float(rng.uniform(1.0, 4.0)),
            ic_mean=float(rng.uniform(-0.5, 0.5)),
            ic_squash=bool(rng.random() < 0.2),
            t0_frames=int(rng.integers(0, MAX_T0_FRAMES + 1)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return problems


def mutate_problem(problem: Problem, rng: np.random.Generator) -> Problem:
    """A nearby problem: nudged coefficients, sometimes a term switched, often fresh fields."""
    coef = list(problem.coef)
    for i, term in enumerate(TERMS):
        spec = PRIOR[term][1]
        if coef[i] == 0.0:
            if rng.random() < 0.1:
                coef[i] = _draw(spec, rng)
        elif rng.random() < 0.1:
            coef[i] = 0.0
        elif spec[0] == "log":
            coef[i] *= math.exp(rng.normal(0.0, 0.3))
        else:
            coef[i] += rng.normal(0.0, 0.15 * (spec[2] - spec[1]))
        coef[i] = float(np.clip(coef[i], *COEF_LIMITS[term]))   # repeated mutation must not drift out
    if not any(coef):
        coef[0] = _draw(PRIOR["diffusion"][1], rng)
    return Problem(
        coef=tuple(coef),
        ic_amp=float(np.clip(problem.ic_amp * math.exp(rng.normal(0.0, 0.2)), 0.05, 3.0)),
        ic_slope=float(np.clip(problem.ic_slope + rng.normal(0.0, 0.3), 0.5, 5.0)),
        ic_mean=float(np.clip(problem.ic_mean + rng.normal(0.0, 0.1), -1.0, 1.0)),
        ic_squash=problem.ic_squash if rng.random() > 0.1 else not problem.ic_squash,
        t0_frames=int(np.clip(problem.t0_frames + rng.integers(-1, 2), 0, MAX_T0_FRAMES)),
        seed=int(rng.integers(0, 2**31 - 1)) if rng.random() < 0.5 else problem.seed,
    )
# EVOLVE-BLOCK-END
