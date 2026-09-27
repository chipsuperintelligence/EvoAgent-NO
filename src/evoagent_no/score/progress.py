"""Learning progress: how much the student is learning from each problem right now.

    r_i = | ⟨ ∇θ L_i(θ_e),  P_e ⊙ (θ_p(e) − θ_e) ⟩ |,     p(e) = ⌊e/2⌋,     P_e = lr / (√v̂_e + ε)

L_i is the student's loss on problem i, θ_p(e) is the student at the lookback round, and P_e is
Adam's per-parameter step size (Cowsik et al., arXiv 2609.30063, Eq. 2). A problem the student
has mastered has a small gradient; a problem that is pure noise has a gradient that does not
line up with where the student has been going. Both score low.

The inner product is a directional derivative, so a single forward-mode pass (torch.func.jvp)
gives it for every sample at once, without per-sample gradients.

Two variants, chosen by `normalize`:

    "paper"      r_i as above.
    "relative"   r_i / (L_i + median_j L_j)   (default)

The paper's reward grows with the size of the gradient, and in regression that tracks how hard a
problem is: in our first runs the pool drifted to hard problems and the student got worse on
held-out physics. Dividing by the problem's loss asks how fast the student improves on it relative
to how wrong it still is. Adding the round's median loss keeps problems the student has nearly
mastered from being inflated by a tiny denominator.
"""

from __future__ import annotations

import torch
from torch.func import functional_call, jvp

from evoagent_no.student.fno import per_sample_loss


def adam_step_size(optimizer: torch.optim.Optimizer, params: dict[str, torch.Tensor],
                   named: dict[str, torch.nn.Parameter]) -> dict[str, torch.Tensor]:
    """P = lr / (√v̂ + ε) per parameter, from the optimizer's second-moment state."""
    group = optimizer.param_groups[0]
    lr, (_, beta2), eps = group["lr"], group["betas"], group["eps"]
    out = {}
    for name, p in named.items():
        state = optimizer.state.get(p, {})
        if "exp_avg_sq" not in state:
            out[name] = torch.zeros_like(params[name])
            continue
        step = float(state["step"])
        v_hat = state["exp_avg_sq"] / (1 - beta2**step)
        out[name] = lr / (v_hat.sqrt() + eps)
    return out


def learning_progress(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      lookback: dict[str, torch.Tensor], inputs: torch.Tensor,
                      targets: torch.Tensor, problem: torch.Tensor, n_problems: int,
                      normalize: str = "relative", floor: float | None = None,
                      chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """(scores, losses), each (n_problems,). `problem` maps each (input, target) sample to its
    problem; a problem with no samples scores 0. `floor` replaces the median loss in the relative
    score, so batches from different generators can be compared on the same scale."""
    if normalize not in ("relative", "paper"):
        raise ValueError(f"normalize must be 'relative' or 'paper', got {normalize!r}")
    named = dict(model.named_parameters())
    params = {k: v.detach() for k, v in named.items()}
    step_size = adam_step_size(optimizer, params, named)
    tangent = {k: step_size[k] * (lookback[k].to(params[k].device) - params[k]) for k in params}

    losses, derivs = [], []
    for start in range(0, inputs.shape[0], chunk):
        x, y = inputs[start:start + chunk], targets[start:start + chunk]
        loss_of = lambda p: per_sample_loss(functional_call(model, p, (x,)), y)
        with torch.no_grad():
            loss, d = jvp(loss_of, (params,), (tangent,))
        losses.append(loss)
        derivs.append(d)
    loss, d = torch.cat(losses), torch.cat(derivs)

    zeros = lambda: torch.zeros(n_problems, device=d.device, dtype=d.dtype)
    count = zeros().index_add_(0, problem, torch.ones_like(d)).clamp_min(1)
    mean_loss = zeros().index_add_(0, problem, loss) / count
    score = (zeros().index_add_(0, problem, d) / count).abs()
    if normalize == "relative":
        if floor is None:
            present = zeros().index_add_(0, problem, torch.ones_like(d)) > 0
            floor = mean_loss[present].median() if present.any() else mean_loss.new_tensor(0.0)
        score = score / (mean_loss + floor).clamp_min(1e-12)
    return score, mean_loss
