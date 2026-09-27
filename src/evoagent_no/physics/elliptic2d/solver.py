"""The elliptic2d answer key: a batched, Jacobi-preconditioned conjugate-gradient solver.

Cell-centred finite differences on an n × n grid (h = 1/n): face coefficients are harmonic means
of the neighbouring cells, and the zero boundary value sits on the boundary faces. Every problem
in a batch has its own coefficient field, so the whole batch is one set of tensor operations.

`solve` computes the answer at 2·GRID and checks it against 4·GRID; both are averaged onto the
GRID × GRID grid the student sees. A problem is kept only if CG converged at both resolutions
and the two answers agree, which rejects coefficient fields too sharp to resolve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evoagent_no.physics import Samples
from evoagent_no.physics.elliptic2d.problem import GRID, Problem, fields, wavevectors


def centers(n: int, device=None) -> torch.Tensor:
    return (torch.arange(n, device=device, dtype=torch.float64) + 0.5) / n


def series(amp: torch.Tensor, phase: torch.Tensor, n: int) -> torch.Tensor:
    """Σ_k amp_k cos(2π k·x + phase_k) at the n × n cell centres: (B, n, n)."""
    k = torch.as_tensor(wavevectors(), dtype=torch.float64, device=amp.device)
    x = centers(n, amp.device)
    ex = torch.exp(2j * torch.pi * k[:, 0, None] * x[None, :])     # (K, n)
    ey = torch.exp(2j * torch.pi * k[:, 1, None] * x[None, :])
    c = amp * torch.exp(1j * phase)                                 # (B, K)
    return torch.einsum("bk,kx,ky->bxy", c, ex, ey).real


def coefficient_and_source(c: dict[str, torch.Tensor], n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """a and f at the n × n cell centres, each (B, n, n) float64."""
    g_a = series(c["a_amp"], c["a_phase"], n)
    lo, hi, sharp = c["a_lo"][:, None, None], c["a_hi"][:, None, None], c["sharpness"][:, None, None]
    a = lo + (hi - lo) * torch.sigmoid(sharp * g_a)
    f = c["f_mean"][:, None, None] + c["f_amp"][:, None, None] * series(c["f_amp_k"], c["f_phase"], n)
    return a, f


def operator(a: torch.Tensor):
    """(apply, diagonal) of the discrete −∇·(a∇·) with zero boundary values, for a (B, n, n)."""
    n = a.shape[-1]
    inv_h2 = float(n * n)
    fx = 2 * a[:, :-1] * a[:, 1:] / (a[:, :-1] + a[:, 1:])          # faces between cells along axis 1
    fy = 2 * a[:, :, :-1] * a[:, :, 1:] / (a[:, :, :-1] + a[:, :, 1:])
    east = torch.cat([fx, 2 * a[:, -1:]], dim=1)                     # boundary face: a over h/2
    west = torch.cat([2 * a[:, :1], fx], dim=1)
    north = torch.cat([fy, 2 * a[:, :, -1:]], dim=2)
    south = torch.cat([2 * a[:, :, :1], fy], dim=2)
    diagonal = (east + west + north + south) * inv_h2

    def apply(u):
        u_e = F.pad(u[:, 1:], (0, 0, 0, 1))
        u_w = F.pad(u[:, :-1], (0, 0, 1, 0))
        u_n = F.pad(u[:, :, 1:], (0, 1))
        u_s = F.pad(u[:, :, :-1], (1, 0))
        return diagonal * u - (east * u_e + west * u_w + north * u_n + south * u_s) * inv_h2

    return apply, diagonal


@torch.no_grad()
def conjugate_gradient(a: torch.Tensor, f: torch.Tensor, tol: float = 1e-10, max_iter: int | None = None):
    """Solve for u (B, n, n); also returns a (B,) mask of problems that converged."""
    apply, diagonal = operator(a)
    max_iter = max_iter or 25 * a.shape[-1]
    u = torch.zeros_like(f)
    r = f.clone()
    z = r / diagonal
    p = z.clone()
    rz = (r * z).sum(dim=(1, 2))
    f_norm = f.norm(dim=(1, 2)).clamp_min(1e-300)
    done = torch.zeros(f.shape[0], dtype=torch.bool, device=f.device)
    for _ in range(max_iter):
        ap = apply(p)
        denominator = (p * ap).sum(dim=(1, 2))
        alpha = torch.where(done | (denominator == 0), torch.zeros_like(rz), rz / denominator)
        u += alpha[:, None, None] * p
        r -= alpha[:, None, None] * ap
        done |= r.norm(dim=(1, 2)) / f_norm < tol
        if bool(done.all()):
            break
        z = r / diagonal
        rz_next = (r * z).sum(dim=(1, 2))
        beta = torch.where(done, torch.zeros_like(rz), rz_next / rz.clamp_min(1e-300))
        p = z + beta[:, None, None] * p
        rz = rz_next
    return u, done


def restrict(u: torch.Tensor, n: int) -> torch.Tensor:
    """Average an (B, m, m) cell-centred field onto n × n cells (m a multiple of n)."""
    b, m = u.shape[0], u.shape[-1]
    return u.reshape(b, n, m // n, n, m // n).mean(dim=(2, 4))


def compile_problems(problems: list[Problem], device=None) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in fields(problems).items()}


def solve_with_disagreement(problems: list[Problem], n: int = GRID, device=None, tol: float = 1e-2):
    """inputs (B, 2, n, n), targets (B, 1, n, n) float32, accepted (B,), disagreement (B,)."""
    c = compile_problems(problems, device)
    a2, f2 = coefficient_and_source(c, 2 * n)
    a4, f4 = coefficient_and_source(c, 4 * n)
    u2, ok2 = conjugate_gradient(a2, f2)
    u4, ok4 = conjugate_gradient(a4, f4)
    answer, check = restrict(u2, n), restrict(u4, n)
    scale = check.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-12)
    disagreement = (answer - check).pow(2).mean(dim=(1, 2)).sqrt() / scale
    finite = torch.isfinite(answer).all(dim=(1, 2)) & torch.isfinite(check).all(dim=(1, 2))
    accepted = ok2 & ok4 & finite & (disagreement < tol)
    a, f = coefficient_and_source(c, n)
    inputs = torch.stack([torch.log(a), f], dim=1).float()
    return inputs, answer[:, None].float(), accepted, disagreement.float()


def solve(problems: list[Problem], device=None) -> Samples:
    """The answer key: one sample per accepted problem, (log a, f) in and u out."""
    inputs, targets, accepted, _ = solve_with_disagreement(problems, device=device)
    kept = accepted.nonzero().squeeze(1)
    return Samples(inputs=inputs[kept], targets=targets[kept], problem=kept, accepted=accepted)
