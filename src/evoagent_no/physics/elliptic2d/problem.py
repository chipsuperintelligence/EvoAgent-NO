"""The elliptic2d problem language: a steady diffusion (Darcy-type) problem on the unit square.

    −∇·(a(x, y) ∇u) = f(x, y)   on (0, 1)²,   u = 0 on the boundary

The coefficient a = a_lo + (a_hi − a_lo) · sigmoid(sharpness · g_a) ranges from a smooth field
(small sharpness) to two distinct phases (large sharpness); sharpness 0 gives a constant. The
source f = f_mean + f_amp · g_f. g_a and g_f are random fields drawn from the seed, band-limited
to KMAX modes, so a problem means the same thing at every resolution.

The student sees (log a, f) on a GRID × GRID grid and predicts u: an input → output operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KMAX = 6       # highest wavenumber in random fields
GRID = 32      # cells per side the student sees

# The equation's symmetries, given to the student: u scales with f (so f's size is hidden and u
# is in units of rms f) and inversely with a (so log a's level is hidden and u is in units of 1/a).
IO = {"dims": 2, "in_channels": 2, "out_channels": 1, "norm_groups": [[0], [1]],
      "residual": None, "invariant": {0: "shift", 1: "scale"},
      "units": [(1, "rms", 1), (0, "exp_mean", -1)], "periodic": False}

LIMITS = {"a_lo": (0.01, 10.0), "a_hi": (0.01, 100.0), "a_slope": (0.5, 6.0), "sharpness": (0.0, 50.0),
          "f_mean": (-10.0, 10.0), "f_amp": (0.0, 10.0), "f_slope": (0.5, 6.0)}
MAX_CONTRAST = 1000.0


@dataclass(frozen=True)
class Problem:
    a_lo: float        # lowest coefficient value
    a_hi: float        # highest coefficient value (contrast = a_hi / a_lo)
    a_slope: float     # spectral decay of g_a: larger is smoother
    sharpness: float   # 0: constant a; small: smooth; large: two phases with sharp interfaces
    f_mean: float
    f_amp: float
    f_slope: float
    seed: int


def check(problem: Problem) -> None:
    """Raise ValueError if the problem is malformed."""
    if not isinstance(problem, Problem):
        raise ValueError(f"expected an elliptic2d Problem, got {type(problem).__name__}")
    for name, (lo, hi) in LIMITS.items():
        value = getattr(problem, name)
        if not (isinstance(value, (int, float)) and math.isfinite(value) and lo <= value <= hi):
            raise ValueError(f"{name} = {value} is outside [{lo}, {hi}]")
    if problem.a_hi < problem.a_lo:
        raise ValueError(f"a_hi = {problem.a_hi} is below a_lo = {problem.a_lo}")
    if problem.a_hi / problem.a_lo > MAX_CONTRAST:
        raise ValueError(f"contrast a_hi / a_lo = {problem.a_hi / problem.a_lo:.0f} exceeds {MAX_CONTRAST:.0f}")
    if abs(problem.f_mean) + problem.f_amp < 1e-3:
        raise ValueError("f is zero everywhere (f_mean and f_amp both zero)")
    if not (isinstance(problem.seed, (int, np.integer)) and 0 <= problem.seed < 2**32):
        raise ValueError(f"seed = {problem.seed} must be an int in [0, 2^32)")


CELL_NAMES = (("low contrast", "medium contrast", "high contrast"),
              ("smooth", "graded", "sharp"),
              ("mean forcing", "varying forcing"))


def describe(problem: Problem) -> tuple[int, int, int]:
    """MAP-Elites cell: (contrast, interface sharpness, kind of forcing). 3 × 3 × 2 cells."""
    contrast = int(np.digitize(problem.a_hi / problem.a_lo, (3.0, 30.0)))
    sharp = int(np.digitize(problem.sharpness, (3.0, 10.0)))
    forcing = int(abs(problem.f_mean) < problem.f_amp)
    return contrast, sharp, forcing


def cell_name(cell: tuple[int, int, int]) -> str:
    return "/".join(names[i] for names, i in zip(CELL_NAMES, cell))


def wavevectors() -> np.ndarray:
    """(K, 2) integer wavevectors, one per independent cosine mode."""
    return np.array([(kx, ky) for kx in range(-KMAX, KMAX + 1) for ky in range(0, KMAX + 1)
                     if (kx, ky) != (0, 0) and not (ky == 0 and kx < 0)])


def random_field(rng: np.random.Generator, slope: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes and phases of a unit-RMS field Σ A cos(2π k·x + φ) over `wavevectors()`."""
    k = wavevectors()
    amp = rng.standard_normal(len(k)) * (1.0 + np.hypot(k[:, 0], k[:, 1])) ** (-slope)
    amp /= max(math.sqrt(0.5 * np.sum(amp**2)), 1e-12)
    return amp, rng.uniform(0, 2 * math.pi, len(k))


def fields(problems: list[Problem]) -> dict[str, np.ndarray]:
    """Per-problem field coefficients: a_amp, a_phase, f_amp, f_phase (B, K) and scalars (B,)."""
    out = {"a_amp": [], "a_phase": [], "f_amp_k": [], "f_phase": []}
    for p in problems:
        rng = np.random.default_rng(int(p.seed))
        amp, phase = random_field(rng, p.a_slope)
        out["a_amp"].append(amp)
        out["a_phase"].append(phase)
        amp, phase = random_field(rng, p.f_slope)
        out["f_amp_k"].append(amp)
        out["f_phase"].append(phase)
    out = {k: np.array(v) for k, v in out.items()}
    for name in ("a_lo", "a_hi", "sharpness", "f_mean", "f_amp"):
        out[name] = np.array([getattr(p, name) for p in problems], dtype=float)
    return out
