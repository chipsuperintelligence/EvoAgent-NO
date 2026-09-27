"""A physics backend written for this task: linear transport u_t + c u_x = ν u_xx on [0, 1).

This file is the whole answer key. EvoAgent-NO needs five things from it: `Problem`, `IO`,
`check`, `describe` and `solve`. Everything else here is ordinary code.

A problem is a velocity c, a diffusivity ν, a time T and an initial field. The student sees the
initial field plus two constant channels (c·T and ν·T, how far the field moves and spreads) and
predicts the field at time T. The equation is linear with constant coefficients, so each Fourier
mode is solved exactly and nothing is ever rejected for accuracy; `solve` still rejects problems
whose answer is numerically flat, which teach nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from evoagent_no.physics import Samples

GRID = 64
KMAX = 12

IO = {"dims": 1, "in_channels": 3, "out_channels": 1, "norm_groups": [[0], [1], [2]],
      "residual": 0, "scale": (0, "std"), "periodic": True}


@dataclass(frozen=True)
class Problem:
    velocity: float     # c
    diffusivity: float  # ν
    time: float         # T
    slope: float        # spectral decay of the initial field
    seed: int


def check(problem: Problem) -> None:
    if not isinstance(problem, Problem):
        raise ValueError(f"expected a transport1d Problem, got {type(problem).__name__}")
    limits = {"velocity": (-5.0, 5.0), "diffusivity": (0.0, 0.1), "time": (0.0, 1.0), "slope": (0.5, 6.0)}
    for name, (lo, hi) in limits.items():
        value = getattr(problem, name)
        if not (math.isfinite(value) and lo <= value <= hi):
            raise ValueError(f"{name} = {value} is outside [{lo}, {hi}]")
    if not (isinstance(problem.seed, (int, np.integer)) and 0 <= problem.seed < 2**32):
        raise ValueError(f"seed = {problem.seed} must be an int in [0, 2^32)")


CELL_NAMES = (("short shift", "long shift"), ("little spreading", "strong spreading"))


def describe(problem: Problem) -> tuple[int, int]:
    """(how far the field moves, how much it spreads): 2 × 2 cells."""
    return int(abs(problem.velocity * problem.time) > 0.25), int(problem.diffusivity * problem.time > 1e-3)


def cell_name(cell: tuple[int, int]) -> str:
    return "/".join(names[i] for names, i in zip(CELL_NAMES, cell))


def exact(problems: list[Problem], n: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """Initial and final fields (B, n): every Fourier mode moves and decays exactly."""
    x = np.arange(n) / n
    k = np.arange(1, KMAX + 1)
    u0, uT = [], []
    for p in problems:
        rng = np.random.default_rng(int(p.seed))
        amp = rng.standard_normal(KMAX) * (1.0 + k) ** (-p.slope)
        amp /= math.sqrt(0.5 * np.sum(amp**2))
        phase = rng.uniform(0, 2 * math.pi, KMAX)
        arg = 2 * math.pi * k[:, None] * x[None, :] + phase[:, None]
        decay = np.exp(-p.diffusivity * (2 * math.pi * k) ** 2 * p.time)[:, None]
        shift = (2 * math.pi * k * p.velocity * p.time)[:, None]
        u0.append((amp[:, None] * np.cos(arg)).sum(0))
        uT.append((amp[:, None] * decay * np.cos(arg - shift)).sum(0))
    return np.array(u0), np.array(uT)


def inputs_for(problems: list[Problem], u0: np.ndarray) -> np.ndarray:
    c_t = np.array([p.velocity * p.time for p in problems])[:, None] * np.ones_like(u0)
    nu_t = np.array([p.diffusivity * p.time for p in problems])[:, None] * np.ones_like(u0)
    return np.stack([u0, c_t, nu_t], axis=1)


def solve(problems: list[Problem], device=None) -> Samples:
    u0, uT = exact(problems)
    accepted = uT.std(axis=1) > 1e-3
    kept = np.nonzero(accepted)[0]
    as_tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    return Samples(inputs=as_tensor(inputs_for(problems, u0)[kept]), targets=as_tensor(uT[kept, None]),
                   problem=as_tensor(kept, torch.long), accepted=as_tensor(accepted, torch.bool))
