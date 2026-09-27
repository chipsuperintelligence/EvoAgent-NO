"""The pde1d backend: the answer key against exact solutions, and what counts as well-formed."""

from dataclasses import replace

import pytest
import torch

from evoagent_no.physics.pde1d import FRAME_DT, FRAMES, KMAX, TERMS, Problem, cell_name, check, describe
from evoagent_no.physics.pde1d.solver import compile_problems, fourier_series, grid, solve_with_disagreement


def problem(**terms) -> Problem:
    return Problem(coef=tuple(terms.get(t, 0.0) for t in TERMS), ic_amp=0.8, ic_slope=2.0,
                   ic_mean=0.0, ic_squash=False, t0_frames=2, seed=11)


@pytest.mark.parametrize("name, terms", [
    ("heat", {"diffusion": 0.1}),
    ("advection", {"advection": 1.3}),
    ("dispersion", {"dispersion": 0.02}),
])
def test_linear_problems_match_exact_solutions(name, terms):
    p = problem(**terms)
    frames, ok, _ = solve_with_disagreement([p], n=64)
    c = compile_problems([p], None)
    k = torch.arange(1, KMAX + 1, dtype=torch.float64)
    exact = []
    for j in range(FRAMES):
        t = (p.t0_frames + j) * FRAME_DT
        amp, phase = c["amp"][0], c["phase"][0]
        if name == "heat":
            amp = amp * torch.exp(-0.1 * k**2 * t)
        elif name == "advection":
            phase = phase - 1.3 * k * t
        else:
            phase = phase + 0.02 * k**3 * t
        exact.append(fourier_series(amp[None], phase[None], grid(64))[0])
    exact = torch.stack(exact)
    assert bool(ok[0])
    assert (frames[0].double() - exact).norm() / exact.norm() < 1e-6


def test_blow_up_is_rejected():
    p = replace(problem(cubic=0.5, growth=1.0), ic_amp=3.0, ic_mean=2.0)
    _, ok, _ = solve_with_disagreement([p], n=64)
    assert not bool(ok[0])


def test_check_rejects_malformed_problems():
    check(problem(diffusion=0.1))
    with pytest.raises(ValueError, match="diffusion"):
        check(problem(diffusion=-0.1))
    with pytest.raises(ValueError, match="ic_amp"):
        check(replace(problem(diffusion=0.1), ic_amp=float("nan")))
    with pytest.raises(ValueError, match="t0_frames"):
        check(replace(problem(diffusion=0.1), t0_frames=11))
    with pytest.raises(ValueError, match="nothing evolves"):
        check(problem())
    with pytest.raises(ValueError, match="coef needs"):
        check(replace(problem(), coef=(0.1,)))
    with pytest.raises(ValueError, match="expected a pde1d Problem"):
        check({"coef": ()})


def test_cells_have_readable_names():
    p = replace(problem(burgers=1.0, dispersion=0.01), ic_amp=1.5)
    assert describe(p) == (1, 1, 2)
    assert cell_name(describe(p)) == "burgers/dispersive/large"
