"""A 3D physics backend: the heat equation u_t = ν Δu on the periodic unit cube, solved exactly.

A problem is a diffusivity ν, a time T and a random initial field. The student sees the initial
field and ν·T (how far it spreads) on a GRID³ grid and predicts the field at time T. Each Fourier
mode decays exactly, so nothing is rejected for accuracy; problems whose answer is flat are
dropped, because they teach nothing. Small on purpose: it shows the 3D path end to end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from evoagent_no.physics import Samples

GRID = 16
KMAX = 3

IO = {"dims": 3, "in_channels": 2, "out_channels": 1, "norm_groups": [[0], [1]],
      "residual": 0, "scale": (0, "std"), "periodic": True}


@dataclass(frozen=True)
class Problem:
    diffusivity: float  # ν
    time: float         # T
    slope: float        # spectral decay of the initial field
    seed: int


def check(problem: Problem) -> None:
    if not isinstance(problem, Problem):
        raise ValueError(f"expected a heat3d Problem, got {type(problem).__name__}")
    for name, (lo, hi) in {"diffusivity": (0.0, 0.05), "time": (0.0, 1.0), "slope": (0.5, 6.0)}.items():
        value = getattr(problem, name)
        if not (math.isfinite(value) and lo <= value <= hi):
            raise ValueError(f"{name} = {value} is outside [{lo}, {hi}]")
    if not (isinstance(problem.seed, (int, np.integer)) and 0 <= problem.seed < 2**32):
        raise ValueError(f"seed = {problem.seed} must be an int in [0, 2^32)")


CELL_NAMES = (("little spreading", "some spreading", "strong spreading"), ("rough", "smooth"))


def describe(problem: Problem) -> tuple[int, int]:
    spread = problem.diffusivity * problem.time * (2 * math.pi) ** 2
    return int(np.digitize(spread, (0.05, 0.3))), int(problem.slope > 2.5)


def cell_name(cell: tuple[int, int]) -> str:
    return "/".join(names[i] for names, i in zip(CELL_NAMES, cell))


_K = np.array([(a, b, c) for a in range(-KMAX, KMAX + 1) for b in range(-KMAX, KMAX + 1)
               for c in range(0, KMAX + 1) if (a, b, c) != (0, 0, 0) and not (c == 0 and (b < 0 or (b == 0 and a < 0)))])


def exact(problems: list[Problem], n: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """Initial and final fields (B, n, n, n)."""
    x = np.arange(n) / n
    phase_x = np.exp(2j * math.pi * _K[:, 0, None] * x)   # (K, n)
    phase_y = np.exp(2j * math.pi * _K[:, 1, None] * x)
    phase_z = np.exp(2j * math.pi * _K[:, 2, None] * x)
    k2 = (2 * math.pi) ** 2 * (_K**2).sum(axis=1)
    u0, uT = [], []
    for p in problems:
        rng = np.random.default_rng(int(p.seed))
        amp = rng.standard_normal(len(_K)) * (1.0 + np.sqrt((_K**2).sum(axis=1))) ** (-p.slope)
        amp /= math.sqrt(0.5 * np.sum(amp**2))
        c = amp * np.exp(1j * rng.uniform(0, 2 * math.pi, len(_K)))
        field = lambda coef: np.einsum("k,kx,ky,kz->xyz", coef, phase_x, phase_y, phase_z).real
        u0.append(field(c))
        uT.append(field(c * np.exp(-p.diffusivity * k2 * p.time)))
    return np.array(u0), np.array(uT)


def inputs_for(problems: list[Problem], u0: np.ndarray) -> np.ndarray:
    spread = np.array([p.diffusivity * p.time for p in problems])[:, None, None, None] * np.ones_like(u0)
    return np.stack([u0, spread], axis=1)


def solve(problems: list[Problem], device=None) -> Samples:
    u0, uT = exact(problems)
    accepted = uT.reshape(len(problems), -1).std(axis=1) > 1e-3
    kept = np.nonzero(accepted)[0]
    as_tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    return Samples(inputs=as_tensor(inputs_for(problems, u0)[kept]), targets=as_tensor(uT[kept, None]),
                   problem=as_tensor(kept, torch.long), accepted=as_tensor(accepted, torch.bool))
