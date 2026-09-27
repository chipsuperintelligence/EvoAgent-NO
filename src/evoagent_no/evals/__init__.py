"""Held-out physics the student never trains on, scored against references from other methods.

A held-out module (`evals.pde1d`, `evals.elliptic2d`, or a task's own) has
`build(count, device) -> {family: (inputs, targets)}` in its backend's IO layout. Held-out
scores are logged but never shown to the agents, so they cannot tune to the test.
"""

from __future__ import annotations

import numpy as np
import torch


def nrmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean over instances of ‖pred − target‖ / ‖target‖."""
    p, t = prediction.flatten(1), target.flatten(1)
    return ((p - t).norm(dim=1) / t.norm(dim=1).clamp_min(1e-8)).mean().item()


def trivial_prediction(io: dict):
    """The floor to beat: copy the residual channel (time stepping) or predict zero (operators)."""
    residual = io.get("residual")
    if residual is not None:
        return lambda x: x[:, [residual]]
    return lambda x: x.new_zeros(x.shape[0], io["out_channels"], *x.shape[2:])


def evaluate(predict, suite: dict[str, tuple[torch.Tensor, torch.Tensor]], trivial=None) -> dict[str, float]:
    """nRMSE per family and their mean; with `trivial`, the same for the trivial prediction."""
    scores = {}
    for family, (x, y) in suite.items():
        scores[family] = nrmse(predict(x), y)
        if trivial is not None:
            scores[f"{family}/trivial"] = nrmse(trivial(x), y)
    scores["mean"] = float(np.mean([scores[f] for f in suite]))
    if trivial is not None:
        scores["mean/trivial"] = float(np.mean([scores[f"{f}/trivial"] for f in suite]))
    return scores
