"""Answer keys: physics backends that solve problems exactly and reject what they cannot.

A backend is a module with:

    Problem                      the problem type (a frozen dataclass)
    IO                           what a training sample looks like (see below)
    check(problem)               raise ValueError if malformed
    describe(problem) -> tuple   its MAP-Elites cell (fixed here, so agents cannot redefine coverage)
    solve(problems, device) -> Samples

and optionally `cell_name(cell)` and `CELL_NAMES` for readable reports. Built in: `pde1d`
(1D time stepping) and `elliptic2d` (a 2D input → output operator). A task can bring its own by
pointing task.yaml at ./physics.py.

`IO` describes samples to the student, so any backend can use the same neural operator:

    dims           1, 2 or 3 spatial dimensions (regular grids)
    in_channels    input fields per sample
    out_channels   output fields per sample
    norm_groups    input channels normalized together (e.g. the frames of one history)
    residual       input channel the output is a change from (time stepping), or None
    invariant      {channel: "scale" | "shift"}: the physics does not depend on that group's size
                   ("scale") or level ("shift"), so the student is not shown it
    units          [(channel, "std" | "rms" | "exp_mean", power), ...]: the output's units, a
                   product of input statistics; `scale: (channel, kind)` means power 1
    periodic       whether space wraps around
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Samples:
    """What an answer key returns for a batch of problems."""
    inputs: torch.Tensor     # (M, in_channels, *space)
    targets: torch.Tensor    # (M, out_channels, *space)
    problem: torch.Tensor    # (M,) index of the problem each sample came from
    accepted: torch.Tensor   # (B,) bool, one per problem; rejected problems have no samples
