"""The student adapts to any backend's IO: dimensions, channels, residual and scale."""

import pytest
import torch

from evoagent_no.physics import elliptic2d, pde1d
from evoagent_no.student.fno import FNO, SpectralConv, per_sample_loss


def test_shapes_follow_the_io():
    x1 = torch.randn(3, 4, 64)
    assert FNO(pde1d.IO, width=8, modes=4, layers=2)(x1).shape == (3, 1, 64)
    x2 = torch.randn(3, 2, 20, 20)
    assert FNO(elliptic2d.IO, width=8, modes=6, layers=2, padding=4)(x2).shape == (3, 1, 20, 20)


def test_time_stepping_predicts_a_change_from_the_last_frame():
    model = FNO(pde1d.IO, width=8, modes=4, layers=2)
    for p in model.head[-1].parameters():
        torch.nn.init.zeros_(p)            # the network outputs zero change
    x = torch.randn(2, 4, 32)
    assert torch.allclose(model(x), x[:, [3]])


def test_loss_is_scale_free():
    y, p = torch.randn(4, 1, 32), torch.randn(4, 1, 32)
    assert torch.allclose(per_sample_loss(p, y), per_sample_loss(10 * p, 10 * y), rtol=1e-5)


def test_operator_respects_the_equations_symmetries():
    """For −∇·(a∇u) = f: doubling f doubles u, and multiplying a by e^c divides u by e^c."""
    torch.manual_seed(0)
    model = FNO(elliptic2d.IO, width=8, modes=4, layers=2, padding=2).double()
    x = torch.randn(2, 2, 16, 16, dtype=torch.float64)
    louder, stiffer = x.clone(), x.clone()
    louder[:, 1] *= 40.0
    stiffer[:, 0] += 3.0
    assert torch.allclose(model(louder), 40.0 * model(x))
    assert torch.allclose(model(stiffer), model(x) * torch.exp(torch.tensor(-3.0, dtype=torch.float64)))
    flat = x.clone()
    flat[:, 1] = 1.0                                    # constant source: nothing blows up
    assert torch.isfinite(model(flat)).all()


@pytest.mark.parametrize("dims, space", [(1, (16,)), (2, (12, 10)), (3, (12, 10, 8))])
def test_spectral_layer_with_identity_weights_is_a_low_pass_filter(dims, space):
    """Places every frequency block where it belongs: with identity weights the layer must equal
    keeping frequencies −modes … modes−1 on every axis but the last, and 0 … modes−1 on the last
    (the standard FNO truncation)."""
    layer = SpectralConv(dims, width=3, modes=3).double()
    eye = torch.eye(3, dtype=torch.float64).view(3, 3, *([1] * dims))
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
            p[..., 0] = eye.expand(p.shape[:-1])
    x = torch.randn(2, 3, *space, dtype=torch.float64)
    axes = tuple(range(-dims, 0))
    xf = torch.fft.rfftn(x, dim=axes)
    keep = torch.ones_like(xf.real, dtype=torch.bool)
    for axis, n in zip(axes[:-1], space[:-1]):
        f = torch.fft.fftfreq(n, d=1 / n)
        shape = [1] * xf.ndim
        shape[axis] = n
        keep &= ((f >= -3) & (f < 3)).view(shape)
    last = torch.arange(xf.shape[-1])
    keep &= (last < 3).view(*([1] * (xf.ndim - 1)), -1)
    expected = torch.fft.irfftn(xf * keep, s=space, dim=axes)
    assert torch.allclose(layer(x), expected, atol=1e-10)


def test_3d_shapes():
    io = {"dims": 3, "in_channels": 2, "out_channels": 1, "residual": 0, "scale": (0, "std"), "periodic": True}
    assert FNO(io, width=8, modes=3, layers=2, padding=2)(torch.randn(2, 2, 8, 8, 6)).shape == (2, 1, 8, 8, 6)



def test_the_learning_rate_warms_up_then_decays_to_its_floor():
    from evoagent_no.student.learner import Learner

    learner = Learner("cpu", pde1d.IO, lr=1e-3, warmup_steps=10, min_lr_ratio=0.05, width=4, modes=2, layers=1)
    learner.plan(110)
    assert learner.lr_at(0) == pytest.approx(1e-4) and learner.lr_at(9) == pytest.approx(1e-3)
    assert learner.lr_at(60) == pytest.approx(0.05e-3 + 0.95e-3 * 0.5)          # halfway through the decay
    assert learner.lr_at(110) == pytest.approx(0.05e-3)
    assert Learner("cpu", pde1d.IO, schedule="constant", width=4, modes=2, layers=1).lr_at(500) == 1e-3


def test_predictions_and_the_saved_student_come_from_the_average():
    from evoagent_no.student.learner import Learner

    torch.manual_seed(0)
    learner = Learner("cpu", pde1d.IO, width=8, modes=4, layers=2, ema=0.9, warmup_steps=1)
    learner.plan(30)
    x, y = torch.randn(16, 4, 32), torch.randn(16, 1, 32)
    learner.train(x, y, steps=30, batch=8, generator=torch.Generator().manual_seed(0))
    trained = learner.model(x[:2]).detach()
    averaged = learner.predict(x[:2])
    assert not torch.allclose(trained, averaged)
    assert torch.allclose(averaged, learner.ema_model(x[:2]))
    assert all(torch.equal(learner.student_state()[k], v) for k, v in learner.ema_model.state_dict().items())
    off = Learner("cpu", pde1d.IO, width=8, modes=4, layers=2, ema=0)
    assert torch.allclose(off.predict(x[:2]), off.model(x[:2]).detach())


def test_trial_leaves_the_student_as_it_was():
    from evoagent_no.physics.pde1d import IO
    from evoagent_no.student.learner import Learner

    torch.manual_seed(0)
    learner = Learner("cpu", IO, width=8, modes=4, layers=1)
    x, y = torch.randn(16, 4, 128), torch.randn(16, 1, 128)
    g = torch.Generator().manual_seed(0)
    learner.train(x, y, steps=2, batch=8, generator=g)
    weights = {k: v.clone() for k, v in learner.model.state_dict().items()}
    steps = learner.step_count
    problem = torch.arange(8).repeat_interleave(2)
    before, after = learner.trial((x, y), (x, y, problem), 8, steps=5, batch=8, generator=g)
    assert before.shape == after.shape == (8,)
    assert after.mean() < before.mean()                      # it learned during the trial
    assert learner.step_count == steps
    assert all(torch.equal(weights[k], v) for k, v in learner.model.state_dict().items())
