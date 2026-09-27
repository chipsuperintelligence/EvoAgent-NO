"""Held-out 1D physics the student never trains on, with references from other methods.

No reference comes from our spectral solver, so its numerics cannot leak into the score:

    advection      u_t + c u_x = 0                exact shift
    heat           u_t = ν u_xx                   exact mode decay
    burgers        u_t + u u_x = ν u_xx           Cole–Hopf, heat-kernel integral in log space
    kdv            u_t + u u_x + δ u_xxx = 0      exact soliton (periodic images summed)
    fisher_kpp     u_t = D u_xx + r u (1 − u)     SciPy DOP853, 4th-order finite differences
    allen_cahn     u_t = ε u_xx + u − u³          SciPy DOP853, 4th-order finite differences

Each instance gives HISTORY frames and the next one, FRAME_DT apart, on a 128-point grid:
inputs (count, HISTORY, 128) and targets (count, 1, 128), the layout of `physics.pde1d.IO`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from evoagent_no.physics.pde1d import FRAME_DT, HISTORY, KMAX, MAX_T0_FRAMES

N = 128
FAMILIES = ("advection", "heat", "burgers", "kdv", "fisher_kpp", "allen_cahn")


def _grf(rng, count, amp_range, slope_range):
    """Band-limited random fields as (amplitudes, phases), each (count, KMAX)."""
    k = np.arange(1, KMAX + 1)
    amps = np.empty((count, KMAX))
    for i in range(count):
        a = rng.standard_normal(KMAX) * (1.0 + k) ** (-rng.uniform(*slope_range))
        amps[i] = a * rng.uniform(*amp_range) / math.sqrt(0.5 * np.sum(a**2))
    return amps, rng.uniform(0, 2 * math.pi, (count, KMAX))


def _series(amps, phases, x, shift=0.0):
    k = np.arange(1, amps.shape[-1] + 1)
    return np.cos(k[:, None] * (x[None, :] - shift) + phases[:, None]).T @ amps


def _frame_times(t0_frames):
    return (t0_frames + np.arange(HISTORY + 1)) * FRAME_DT


def _advection(rng, count, x):
    amps, phases = _grf(rng, count, (0.5, 1.0), (2.0, 3.0))
    out = []
    for i in range(count):
        c = rng.uniform(0.5, 1.5) * rng.choice((-1, 1))
        times = _frame_times(rng.integers(0, MAX_T0_FRAMES + 1))
        out.append([_series(amps[i], phases[i], x, shift=c * t) for t in times])
    return np.array(out)


def _heat(rng, count, x):
    amps, phases = _grf(rng, count, (0.5, 1.0), (1.5, 3.0))
    k = np.arange(1, KMAX + 1)
    out = []
    for i in range(count):
        nu = rng.uniform(0.02, 0.2)
        times = _frame_times(rng.integers(0, MAX_T0_FRAMES + 1))
        out.append([_series(amps[i] * np.exp(-nu * k**2 * t), phases[i], x) for t in times])
    return np.array(out)


def _burgers(rng, count, x, fine=4096):
    """Cole–Hopf in log space: u(x,t) = ∫ ((x−y)/t) e^(−Φ) dy / ∫ e^(−Φ) dy with
    Φ = (x−y)²/(4νt) + U0(y)/(2ν), U0' = u0. The weights are a softmax over y (periodic images
    included), so the huge dynamic range of e^(−Φ) never has to be represented."""
    amps, phases = _grf(rng, count, (0.5, 1.0), (2.0, 3.0))
    k = np.arange(1, KMAX + 1)
    y = np.arange(fine) * 2 * math.pi / fine
    images = y[None, :] + 2 * math.pi * np.arange(-1, 2)[:, None]            # (3, fine)
    out = []
    for i in range(count):
        nu = rng.uniform(0.03, 0.1)
        potential = _series(amps[i] / k, phases[i] - math.pi / 2, y)          # ∫u0 dx, periodic: zero mean
        frames = []
        for t in _frame_times(rng.integers(0, MAX_T0_FRAMES + 1)):
            if t == 0:
                frames.append(_series(amps[i], phases[i], x))
                continue
            gap = x[:, None, None] - images[None]                              # (N, 3, fine)
            phi = gap**2 / (4 * nu * t) + potential[None, None, :] / (2 * nu)
            w = np.exp(-(phi - phi.min(axis=(1, 2), keepdims=True)))
            frames.append((w * gap).sum(axis=(1, 2)) / (t * w.sum(axis=(1, 2))))
        out.append(frames)
    return np.array(out)


def _kdv(rng, count, x):
    out = []
    for _ in range(count):
        delta, amp, x0 = rng.uniform(0.005, 0.02), rng.uniform(0.5, 1.5), rng.uniform(0, 2 * math.pi)
        speed, width = amp / 3, math.sqrt(12 * delta / amp)
        frames = []
        for t in _frame_times(rng.integers(0, MAX_T0_FRAMES + 1)):
            xi = x[None, :] - x0 - speed * t + 2 * math.pi * np.arange(-3, 4)[:, None]
            frames.append((amp / np.cosh(xi / width) ** 2).sum(axis=0))
        out.append(frames)
    return np.array(out)


def _reaction_diffusion(rng, count, x, reaction, fine=1024):
    """u_t = D u_xx + f(u), all instances in one SciPy solve on a 4th-order FD grid."""
    xf = np.arange(fine) * 2 * math.pi / fine
    dx = xf[1]
    params = [reaction(rng) for _ in range(count)]
    u0 = np.stack([p["u0"](xf) for p in params])
    D = np.array([p["D"] for p in params])[:, None]

    def rhs(_, y):
        u = y.reshape(count, fine)
        lap = (-np.roll(u, 2, 1) + 16 * np.roll(u, 1, 1) - 30 * u + 16 * np.roll(u, -1, 1)
               - np.roll(u, -2, 1)) / (12 * dx**2)
        return (D * lap + np.stack([p["f"](u[i]) for i, p in enumerate(params)])).ravel()

    all_times = np.arange(MAX_T0_FRAMES + HISTORY + 1) * FRAME_DT
    sol = solve_ivp(rhs, (0, all_times[-1]), u0.ravel(), t_eval=all_times, method="DOP853",
                    rtol=1e-9, atol=1e-11)
    traj = sol.y.T.reshape(len(all_times), count, fine)[:, :, :: fine // N]
    out = []
    for i in range(count):
        t0 = rng.integers(0, MAX_T0_FRAMES + 1)
        out.append(traj[t0:t0 + HISTORY + 1, i])
    return np.array(out)


def _fisher(rng):
    amps, phases = _grf(rng, 1, (0.5, 1.5), (1.5, 3.0))
    r = rng.uniform(0.5, 1.0)
    return {"D": rng.uniform(0.005, 0.05), "f": lambda u: r * u * (1 - u),
            "u0": lambda xf: 1 / (1 + np.exp(-4 * _series(amps[0], phases[0], xf)))}


def _allen_cahn(rng):
    amps, phases = _grf(rng, 1, (0.3, 0.8), (1.5, 3.0))
    return {"D": rng.uniform(0.003, 0.02), "f": lambda u: u - u**3,
            "u0": lambda xf: _series(amps[0], phases[0], xf)}


def build(count: int = 64, seed: int = 12345, device=None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """family → (inputs (count, HISTORY, N), targets (count, 1, N)), float32."""
    rng = np.random.default_rng(seed)
    x = np.arange(N) * 2 * math.pi / N
    makers = {
        "advection": lambda: _advection(rng, count, x),
        "heat": lambda: _heat(rng, count, x),
        "burgers": lambda: _burgers(rng, count, x),
        "kdv": lambda: _kdv(rng, count, x),
        "fisher_kpp": lambda: _reaction_diffusion(rng, count, x, _fisher),
        "allen_cahn": lambda: _reaction_diffusion(rng, count, x, _allen_cahn),
    }
    out = {}
    for family in FAMILIES:
        frames = torch.as_tensor(makers[family](), dtype=torch.float32, device=device)
        out[family] = (frames[:, :HISTORY], frames[:, HISTORY:])
    return out
