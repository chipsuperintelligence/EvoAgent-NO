"""The self-play score: a directional derivative, computed in one forward-mode pass, in 1D and 2D."""

import pytest
import torch
from torch.func import functional_call

from evoagent_no.physics import elliptic2d, pde1d
from evoagent_no.score.progress import adam_step_size, learning_progress
from evoagent_no.student.fno import per_sample_loss
from evoagent_no.student.learner import Learner

IO_3D = {"dims": 3, "in_channels": 2, "out_channels": 1, "residual": 0, "scale": (0, "std"), "periodic": True}
SHAPES = {"pde1d": (pde1d.IO, (32,)), "elliptic2d": (elliptic2d.IO, (12, 12)), "3d": (IO_3D, (8, 8, 6))}


def trained_learner(name):
    io, space = SHAPES[name]
    torch.manual_seed(0)
    learner = Learner("cpu", io, width=8, modes=4, layers=2, ema=0)
    learner.model.double()
    learner.optimizer = torch.optim.AdamW(learner.model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, io["in_channels"], *space, dtype=torch.float64)
    y = torch.randn(32, io["out_channels"], *space, dtype=torch.float64)
    learner.snapshot(0)
    learner.train(x, y, steps=3, batch=8, generator=g)
    learner.snapshot(1)
    learner.train(x, y, steps=3, batch=8, generator=g)
    return learner, x, y, torch.arange(32) // 8


@pytest.mark.parametrize("name", SHAPES)
def test_paper_score_is_the_directional_derivative_of_each_problems_loss(name):
    learner, x, y, prob = trained_learner(name)
    score, loss = learning_progress(learner.model, learner.optimizer, learner.lookback(2), x, y, prob, 4,
                                    normalize="paper")

    named = dict(learner.model.named_parameters())
    params = {k: v.detach() for k, v in named.items()}
    step = adam_step_size(learner.optimizer, params, named)
    back = learner.lookback(2)
    tangent = {k: step[k] * (back[k] - params[k]) for k in params}

    def problem_loss(eps):
        p = {k: params[k] + eps * tangent[k] for k in params}
        per = per_sample_loss(functional_call(learner.model, p, (x,)), y)
        return torch.zeros(4, dtype=per.dtype).index_add_(0, prob, per) / 8

    finite_difference = ((problem_loss(1e-5) - problem_loss(-1e-5)) / 2e-5).abs()
    assert torch.allclose(score, finite_difference, rtol=1e-5)
    assert torch.allclose(loss, problem_loss(0.0))


def test_relative_score_divides_by_loss_plus_the_median():
    learner, x, y, prob = trained_learner("pde1d")
    args = (learner.model, learner.optimizer, learner.lookback(2), x, y, prob, 4)
    paper, loss = learning_progress(*args, normalize="paper")
    relative, _ = learning_progress(*args, normalize="relative")
    assert torch.allclose(relative, paper / (loss + loss.median()))
    fixed, _ = learning_progress(*args, normalize="relative", floor=0.5)
    assert torch.allclose(fixed, paper / (loss + 0.5))


def test_score_is_zero_before_the_student_has_moved():
    learner = Learner("cpu", pde1d.IO, width=8, modes=4, layers=2)
    learner.snapshot(0)
    x, y = torch.randn(8, 4, 32), torch.randn(8, 1, 32)
    score, _ = learning_progress(learner.model, learner.optimizer, learner.lookback(0), x, y,
                                 torch.zeros(8, dtype=torch.long), 1)
    assert score.item() == 0.0
