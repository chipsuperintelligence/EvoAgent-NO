"""Held-out elliptic2d problems the student never trains on, with references from other methods.

    poisson        a = 1, u a random sine series                      exact
    smooth         a = exp(σ φ), σ in [0.2, 0.8] (contrast ≤ 5)        exact (manufactured solution)
    high_contrast  a = exp(σ φ), σ in [1.5, 3.0] (contrast 20–400)     exact (manufactured solution)
    two_phase      a = 1 + 9 · sigmoid(6 g), f = 1                    SciPy direct sparse solve at 4× resolution

For the manufactured families u is chosen first and f = −∇·(a∇u) is computed analytically, so the
reference owes nothing to any discretization. The two-phase reference uses the same kind of
finite-difference scheme as the answer key but a direct solver and four times the resolution.

Inputs are (log a, f) and targets u on the GRID × GRID cell centres, the layout of
`physics.elliptic2d.IO`.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from evoagent_no.physics.elliptic2d.problem import GRID, random_field, wavevectors

FAMILIES = ("poisson", "smooth", "high_contrast", "two_phase")


def _grid(n: int = GRID):
    x = (np.arange(n) + 0.5) / n
    return np.meshgrid(x, x, indexing="ij")


def _sine_solution(rng, modes: int, X, Y):
    """u = Σ b sin(mπx) sin(nπy) with its gradient and Laplacian."""
    u, ux, uy, lap = (np.zeros_like(X) for _ in range(4))
    for m in range(1, modes + 1):
        for n in range(1, modes + 1):
            b = rng.standard_normal() / (m * m + n * n)
            sx, cx = np.sin(m * math.pi * X), np.cos(m * math.pi * X)
            sy, cy = np.sin(n * math.pi * Y), np.cos(n * math.pi * Y)
            u += b * sx * sy
            ux += b * m * math.pi * cx * sy
            uy += b * n * math.pi * sx * cy
            lap -= b * math.pi**2 * (m * m + n * n) * sx * sy
    return u, ux, uy, lap


def _log_coefficient(rng, X, Y):
    """φ with max |φ| = 1 from a few low modes, with its gradient."""
    phi, px, py = (np.zeros_like(X) for _ in range(3))
    for _ in range(4):
        p, q = rng.integers(-2, 3, size=2)
        c, t = rng.standard_normal(), rng.uniform(0, 2 * math.pi)
        arg = 2 * math.pi * (p * X + q * Y) + t
        phi += c * np.cos(arg)
        px -= c * 2 * math.pi * p * np.sin(arg)
        py -= c * 2 * math.pi * q * np.sin(arg)
    top = max(np.abs(phi).max(), 1e-12)
    return phi / top, px / top, py / top


def _manufactured(rng, sigma_range):
    X, Y = _grid()
    u, ux, uy, lap = _sine_solution(rng, 3, X, Y)
    phi, px, py = _log_coefficient(rng, X, Y)
    sigma = rng.uniform(*sigma_range)
    a = np.exp(sigma * phi)
    f = -a * (lap + sigma * (px * ux + py * uy))
    return np.log(a), f, u


def _poisson(rng):
    X, Y = _grid()
    u, _, _, lap = _sine_solution(rng, 4, X, Y)
    return np.zeros_like(u), -lap, u


def _fd_matrix(a: np.ndarray) -> sp.csr_matrix:
    """The same cell-centred −∇·(a∇·) as the answer key, assembled for a direct solve."""
    n = a.shape[0]
    idx = np.arange(n * n).reshape(n, n)
    rows, cols, vals = [], [], []
    diag = np.zeros((n, n))
    for axis in (0, 1):
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(0, n - 1), slice(1, n)
        face = 2 * a[tuple(lo)] * a[tuple(hi)] / (a[tuple(lo)] + a[tuple(hi)])
        diag[tuple(lo)] += face
        diag[tuple(hi)] += face
        rows += [idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()]
        cols += [idx[tuple(hi)].ravel(), idx[tuple(lo)].ravel()]
        vals += [-face.ravel(), -face.ravel()]
        first, last = [slice(None)] * 2, [slice(None)] * 2
        first[axis], last[axis] = 0, n - 1
        diag[tuple(first)] += 2 * a[tuple(first)]
        diag[tuple(last)] += 2 * a[tuple(last)]
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    return sp.csr_matrix((np.concatenate(vals) * n * n, (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n * n, n * n))


def _two_phase(rng, factor: int = 4):
    amp, phase = random_field(rng, 3.0)
    k = wavevectors()

    def g(n):
        X, Y = _grid(n)
        arg = 2 * math.pi * (k[:, 0, None, None] * X + k[:, 1, None, None] * Y) + phase[:, None, None]
        return np.tensordot(amp, np.cos(arg), axes=1)

    a_fine = 1 + 9 / (1 + np.exp(-6 * g(GRID * factor)))
    u_fine = spla.spsolve(_fd_matrix(a_fine).tocsc(), np.ones(a_fine.size)).reshape(a_fine.shape)
    u = u_fine.reshape(GRID, factor, GRID, factor).mean(axis=(1, 3))
    a = 1 + 9 / (1 + np.exp(-6 * g(GRID)))
    return np.log(a), np.ones_like(a), u


def build(count: int = 64, seed: int = 54321, device=None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """family → (inputs (count, 2, GRID, GRID), targets (count, 1, GRID, GRID)), float32."""
    rng = np.random.default_rng(seed)
    makers = {
        "poisson": lambda: _poisson(rng),
        "smooth": lambda: _manufactured(rng, (0.2, 0.8)),
        "high_contrast": lambda: _manufactured(rng, (1.5, 3.0)),
        "two_phase": lambda: _two_phase(rng),
    }
    out = {}
    for family in FAMILIES:
        log_a, f, u = zip(*(makers[family]() for _ in range(count)))
        inputs = torch.as_tensor(np.stack([np.stack(log_a), np.stack(f)], axis=1), dtype=torch.float32, device=device)
        targets = torch.as_tensor(np.stack(u)[:, None], dtype=torch.float32, device=device)
        out[family] = (inputs, targets)
    return out
