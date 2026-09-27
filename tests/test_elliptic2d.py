"""The elliptic2d backend and its held-out suite."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from evoagent_no.evals import elliptic2d as heldout
from evoagent_no.physics.elliptic2d import IO, Problem, cell_name, check, describe, solve
from evoagent_no.physics.elliptic2d.solver import centers, conjugate_gradient, operator


def problem(**changes) -> Problem:
    base = Problem(a_lo=0.5, a_hi=2.0, a_slope=3.0, sharpness=2.0, f_mean=1.0, f_amp=0.5, f_slope=3.0, seed=3)
    return replace(base, **changes)


def manufactured(n):
    x = centers(n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    s = 0.5
    phi = torch.sin(2 * math.pi * X) * torch.cos(2 * math.pi * Y)
    a = torch.exp(s * phi)
    u = torch.sin(math.pi * X) * torch.sin(2 * math.pi * Y)
    ux = math.pi * torch.cos(math.pi * X) * torch.sin(2 * math.pi * Y)
    uy = 2 * math.pi * torch.sin(math.pi * X) * torch.cos(2 * math.pi * Y)
    px = 2 * math.pi * torch.cos(2 * math.pi * X) * torch.cos(2 * math.pi * Y)
    py = -2 * math.pi * torch.sin(2 * math.pi * X) * torch.sin(2 * math.pi * Y)
    f = -a * (-5 * math.pi**2 * u + s * (px * ux + py * uy))
    return a, f, u


def test_solver_converges_at_second_order_to_a_manufactured_solution():
    errors = []
    for n in (16, 32):
        a, f, u = manufactured(n)
        uh, ok = conjugate_gradient(a[None], f[None])
        assert bool(ok)
        errors.append(((uh[0] - u).norm() / u.norm()).item())
    assert errors[1] < 5e-3 and 3.5 < errors[0] / errors[1] < 4.5


def test_heldout_direct_solver_uses_the_same_discrete_operator():
    a = torch.rand(1, 8, 8, dtype=torch.float64) + 0.5
    u = torch.rand(1, 8, 8, dtype=torch.float64)
    apply, _ = operator(a)
    matrix = heldout._fd_matrix(a[0].numpy())
    assert np.allclose(matrix @ u[0].numpy().ravel(), apply(u)[0].numpy().ravel())


def test_solve_returns_one_sample_per_accepted_problem():
    samples = solve([problem(), problem(seed=4, sharpness=0.0)])
    assert samples.accepted.tolist() == [True, True]
    assert samples.inputs.shape == (2, IO["in_channels"], 32, 32)
    assert samples.targets.shape == (2, 1, 32, 32)


def test_unresolvable_interfaces_are_rejected():
    samples = solve([problem(a_lo=0.05, a_hi=50.0, sharpness=50.0, a_slope=1.0)])
    assert samples.accepted.tolist() == [False] and len(samples.problem) == 0


def test_check_and_cells():
    check(problem())
    with pytest.raises(ValueError, match="contrast"):
        check(problem(a_lo=0.01, a_hi=50.0))
    with pytest.raises(ValueError, match="below a_lo"):
        check(problem(a_lo=2.0, a_hi=1.0))
    with pytest.raises(ValueError, match="zero everywhere"):
        check(problem(f_mean=0.0, f_amp=0.0))
    assert cell_name(describe(problem())) == "medium contrast/smooth/mean forcing"


def test_heldout_suite_is_well_formed():
    suite = heldout.build(count=2)
    assert set(suite) == set(heldout.FAMILIES)
    for inputs, targets in suite.values():
        assert inputs.shape == (2, 2, 32, 32) and targets.shape == (2, 1, 32, 32)
        assert torch.isfinite(inputs).all() and torch.isfinite(targets).all()
