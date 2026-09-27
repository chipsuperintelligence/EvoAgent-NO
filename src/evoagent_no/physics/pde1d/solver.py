"""The pde1d answer key: a batched pseudo-spectral solver.

Every problem in a batch has its own PDE: the linear part is a per-problem diagonal in
Fourier space, the nonlinear part is a per-problem mix of u·u_x, u², u³ and forcing. Time
stepping is ETDRK4 (Cox & Matthews 2002) with the contour-integral coefficients of Kassam &
Trefethen (2005), 2/3-rule dealiasing, float64 throughout.

`solve` is the answer key: it solves every problem twice, at 2N and at 4N with half the time
step, and keeps a problem only if the two agree and nothing blew up.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from evoagent_no.physics import Samples
from evoagent_no.physics.pde1d.problem import FRAME_DT, FRAMES, HISTORY, MAX_T0_FRAMES, Problem, random_fields

BLOWUP = 1e3   # |u| above this is treated as a blow-up


def grid(n: int, device=None) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=torch.float64) * (2 * math.pi / n)


def fourier_series(amp: torch.Tensor, phase: torch.Tensor, x: torch.Tensor, shift=None) -> torch.Tensor:
    """Σ_k amp_k cos(k (x − shift) + phase_k) on grid x; amp, phase (B, K); shift (B,) or None."""
    k = torch.arange(1, amp.shape[1] + 1, device=x.device, dtype=x.dtype)
    arg = k[None, :, None] * x[None, None, :] + phase[..., None]
    if shift is not None:
        arg = arg - k[None, :, None] * shift[:, None, None]
    return torch.einsum("bk,bkn->bn", amp, torch.cos(arg))


def compile_problems(problems: list[Problem], device) -> dict[str, torch.Tensor]:
    """Problems → tensors, independent of grid size."""
    amp, phase, f_amp, f_phase = random_fields(problems)
    t = lambda a, dtype=torch.float64: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
    return {
        "coef": t([p.coef for p in problems]),
        "amp": t(amp), "phase": t(phase), "f_amp": t(f_amp), "f_phase": t(f_phase),
        "mean": t([p.ic_mean for p in problems]),
        "squash": t([p.ic_squash for p in problems], torch.bool),
        "t0": t([p.t0_frames for p in problems], torch.long),
    }


def initial_field(c: dict[str, torch.Tensor], n: int) -> torch.Tensor:
    u = c["mean"][:, None] + fourier_series(c["amp"], c["phase"], grid(n, c["amp"].device))
    return torch.where(c["squash"][:, None], torch.sigmoid(4.0 * u), u)


def _etdrk4_coefficients(L: torch.Tensor, h: float, contour_points: int = 64):
    """E, E/2, Q, f1, f2, f3 for a batch of complex diagonal operators L (B, M)."""
    j = torch.arange(1, contour_points + 1, device=L.device, dtype=torch.float64)
    r = torch.exp(2j * math.pi * (j - 0.5) / contour_points)
    LR = h * L[..., None] + r
    mean = lambda z: h * z.mean(dim=-1)
    eLR = torch.exp(LR)
    return (
        torch.exp(h * L),
        torch.exp(h * L / 2),
        mean((torch.exp(LR / 2) - 1) / LR),
        mean((-4 - LR + eLR * (4 - 3 * LR + LR**2)) / LR**3),
        mean((2 + LR + eLR * (-2 + LR)) / LR**3),
        mean((-4 - 3 * LR - LR**2 + eLR * (4 - LR)) / LR**3),
    )


def integrate(c: dict[str, torch.Tensor], n: int, substeps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames u(t0 + j·FRAME_DT), j < FRAMES, on an n-point grid: (B, FRAMES, n) float64, and a
    (B,) bool mask of problems that stayed finite and bounded."""
    device = c["coef"].device
    nu, mu, cadv, delta, r, a, b, g, phi = c["coef"].T
    k = torch.arange(n // 2 + 1, device=device, dtype=torch.float64)
    ik = 1j * k
    L = (-nu[:, None] * k**2 - mu[:, None] * k**4 - cadv[:, None] * ik
         + 1j * delta[:, None] * k**3 + r[:, None])
    h = FRAME_DT / substeps
    E, E2, Q, f1, f2, f3 = _etdrk4_coefficients(L, h)
    dealias = (k <= n / 3).to(torch.float64)
    forcing = torch.fft.rfft(phi[:, None] * fourier_series(c["f_amp"], c["f_phase"], grid(n, device)))
    burgers_op = -0.5 * a[:, None] * ik

    def nonlinear(v):
        u = torch.fft.irfft(v, n=n)
        u2 = torch.fft.rfft(u * u)
        u3 = torch.fft.rfft(u * u * u)
        return dealias * ((burgers_op + b[:, None]) * u2 + g[:, None] * u3) + forcing

    v = torch.fft.rfft(initial_field(c, n))
    ok = torch.ones(v.shape[0], dtype=torch.bool, device=device)
    last = MAX_T0_FRAMES + FRAMES
    all_frames = torch.empty(v.shape[0], last, n, dtype=torch.float64, device=device)
    for frame in range(last):
        u = torch.fft.irfft(v, n=n)
        bad = ~torch.isfinite(u).all(dim=1) | (u.abs().amax(dim=1) > BLOWUP)
        ok &= ~bad
        v = torch.where(ok[:, None], v, torch.zeros_like(v))   # stop NaNs spreading through the batch
        all_frames[:, frame] = torch.where(ok[:, None], u, torch.zeros_like(u))
        if frame == last - 1:
            break
        for _ in range(substeps):
            Nv = nonlinear(v)
            va = E2 * v + Q * Nv
            Na = nonlinear(va)
            vb = E2 * v + Q * Na
            Nb = nonlinear(vb)
            vc = E2 * va + Q * (2 * Nb - Nv)
            Nc = nonlinear(vc)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    index = c["t0"][:, None] + torch.arange(FRAMES, device=device)[None, :]
    frames = torch.gather(all_frames, 1, index[..., None].expand(-1, -1, n))
    return frames, ok


def solve_with_disagreement(problems: list[Problem], n: int = 128, device=None, substeps: int = 20,
                            tol: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frames (B, FRAMES, n) float32, a (B,) bool mask of accepted problems, and each problem's
    relative disagreement between the two resolutions.

    The answer is solved at 2n and checked against a 4n solve with half the step; both are
    sampled onto the n-point grid (every 2nd / 4th point, exact on these nested grids)."""
    c = compile_problems(problems, device)
    fine, ok_fine = integrate(c, 2 * n, substeps)
    finer, ok_finer = integrate(c, 4 * n, 2 * substeps)
    fine, finer = fine[..., ::2], finer[..., ::4]
    scale = finer.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
    disagreement = (fine - finer).pow(2).mean(dim=(1, 2)).sqrt() / scale
    ok = ok_fine & ok_finer & (disagreement < tol)
    return fine.float(), ok, disagreement.float()


def windows(frames: torch.Tensor, history: int = HISTORY) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trajectories (P, T, N) → inputs (P·W, history, N), targets (P·W, 1, N), problem index (P·W,)."""
    p, t, n = frames.shape
    starts = torch.arange(t - history, device=frames.device)
    idx = starts[:, None] + torch.arange(history + 1, device=frames.device)[None, :]   # (W, H+1)
    chunks = frames[:, idx].reshape(-1, history + 1, n)
    problem = torch.arange(p, device=frames.device).repeat_interleave(len(starts))
    return chunks[:, :history], chunks[:, history:], problem


def solve(problems: list[Problem], device=None) -> Samples:
    """The answer key. Each accepted problem gives FRAMES − HISTORY samples: HISTORY frames in,
    the next frame out, on a 128-point grid."""
    frames, ok, _ = solve_with_disagreement(problems, device=device)
    kept = ok.nonzero().squeeze(1)
    inputs, targets, of = windows(frames[kept])
    return Samples(inputs=inputs, targets=targets, problem=kept[of], accepted=ok)
