"""The pde1d problem language: a 1D PDE on the periodic interval [0, 2π), its fields and a seed.

    u_t = ν u_xx − μ u_xxxx − c u_x − δ u_xxx + r u − a u u_x + b u² + g u³ + φ F(x)

Each coefficient is one term; zero means the term is off. The initial field u0 and the forcing F
are random fields drawn from the seed, band-limited to KMAX modes, so a problem means the same
thing at every grid resolution. What a generator may produce is fixed here (`check`); which
problems it produces is up to the generator, and the agents may rewrite that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TERMS = ("diffusion", "hyperdiffusion", "advection", "dispersion", "growth",
         "burgers", "quadratic", "cubic", "forcing")
KMAX = 32            # highest wavenumber in random fields
FRAME_DT = 0.05      # time between the frames the student sees
FRAMES = 16          # frames per problem
MAX_T0_FRAMES = 10   # latest start, in frames
HISTORY = 4          # frames the student sees; it predicts the next one
GRID = 128           # points in a frame

IO = {"dims": 1, "in_channels": HISTORY, "out_channels": 1, "norm_groups": [list(range(HISTORY))],
      "residual": HISTORY - 1, "scale": (HISTORY - 1, "std"), "periodic": True}

# Hard limits. The answer key rejects what it cannot solve; these reject what is malformed.
COEF_LIMITS = {
    "diffusion": (0.0, 1.0), "hyperdiffusion": (0.0, 1e-2), "advection": (-5.0, 5.0),
    "dispersion": (-0.1, 0.1), "growth": (-3.0, 3.0), "burgers": (-5.0, 5.0),
    "quadratic": (-5.0, 5.0), "cubic": (-5.0, 5.0), "forcing": (-5.0, 5.0),
}
FIELD_LIMITS = {"ic_amp": (1e-3, 5.0), "ic_slope": (0.0, 8.0), "ic_mean": (-3.0, 3.0)}


@dataclass(frozen=True)
class Problem:
    coef: tuple[float, ...]   # one per TERMS entry
    ic_amp: float             # RMS amplitude of the initial field
    ic_slope: float           # spectral decay: mode k has amplitude ∝ (1 + k)^(−slope)
    ic_mean: float
    ic_squash: bool           # u0 = sigmoid(4·field), a field in (0, 1) for reaction fronts
    t0_frames: int            # start of the first frame, in frames
    seed: int

    def term(self, name: str) -> float:
        return self.coef[TERMS.index(name)]


def check(problem: Problem) -> None:
    """Raise ValueError if the problem is malformed. Generators written by agents pass through here."""
    if not isinstance(problem, Problem):
        raise ValueError(f"expected a pde1d Problem, got {type(problem).__name__}")
    if len(problem.coef) != len(TERMS):
        raise ValueError(f"coef needs {len(TERMS)} values ({', '.join(TERMS)}), got {len(problem.coef)}")
    for name, value in zip(TERMS, problem.coef):
        lo, hi = COEF_LIMITS[name]
        if not (math.isfinite(value) and lo <= value <= hi):
            raise ValueError(f"{name} = {value} is outside [{lo}, {hi}]")
    if not any(problem.coef):
        raise ValueError("every term is zero: u_t = 0, nothing evolves")
    for name, (lo, hi) in FIELD_LIMITS.items():
        value = getattr(problem, name)
        if not (math.isfinite(value) and lo <= value <= hi):
            raise ValueError(f"{name} = {value} is outside [{lo}, {hi}]")
    if not (isinstance(problem.t0_frames, (int, np.integer)) and 0 <= problem.t0_frames <= MAX_T0_FRAMES):
        raise ValueError(f"t0_frames = {problem.t0_frames} must be an int in [0, {MAX_T0_FRAMES}]")
    if not (isinstance(problem.seed, (int, np.integer)) and 0 <= problem.seed < 2**32):
        raise ValueError(f"seed = {problem.seed} must be an int in [0, 2^32)")


def describe(problem: Problem) -> tuple[int, int, int]:
    """MAP-Elites cell: (nonlinearity, highest-order linear term, amplitude). 4 × 3 × 3 cells.
    Fixed here, not in the generator, so an agent cannot redefine coverage."""
    burgers = problem.term("burgers") != 0.0
    reaction = problem.term("quadratic") != 0.0 or problem.term("cubic") != 0.0
    nonlinear = 1 * burgers + 2 * reaction
    if problem.term("hyperdiffusion") != 0.0:
        linear = 2
    elif problem.term("dispersion") != 0.0:
        linear = 1
    else:
        linear = 0
    amplitude = int(np.digitize(problem.ic_amp, (0.3, 1.0)))
    return nonlinear, linear, amplitude


CELL_NAMES = (("linear", "burgers", "reaction", "burgers+reaction"),
              ("diffusive", "dispersive", "hyperdiffusive"),
              ("small", "medium", "large"))


def cell_name(cell: tuple[int, int, int]) -> str:
    return "/".join(names[i] for names, i in zip(CELL_NAMES, cell))


def random_fields(problems: list[Problem]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fourier amplitudes and phases (float64, shape (B, KMAX)) for u0 and for the forcing F."""
    k = np.arange(1, KMAX + 1)
    amps, phases, f_amps, f_phases = [], [], [], []
    for p in problems:
        rng = np.random.default_rng(int(p.seed))
        a = rng.standard_normal(KMAX) * (1.0 + k) ** (-p.ic_slope)
        a *= p.ic_amp / max(math.sqrt(0.5 * np.sum(a**2)), 1e-12)   # RMS of Σ a_k cos(...) is √(Σa²/2)
        amps.append(a)
        phases.append(rng.uniform(0.0, 2 * math.pi, KMAX))
        f = rng.standard_normal(KMAX) * (1.0 + k) ** -2.0
        f /= max(math.sqrt(0.5 * np.sum(f**2)), 1e-12)
        f_amps.append(f)
        f_phases.append(rng.uniform(0.0, 2 * math.pi, KMAX))
    return np.array(amps), np.array(phases), np.array(f_amps), np.array(f_phases)
