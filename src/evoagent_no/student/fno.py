"""The student: a Fourier neural operator in 1D, 2D or 3D, shaped by the physics backend's `IO`.

It maps input fields (M, in_channels, *space) to output fields (M, out_channels, *space) and is
never told which problem it is looking at, which is what lets one pretrained model transfer.

- Inputs are normalized per sample, one mean and std per `norm_groups` group (a history's frames
  share one, so the change between frames survives). The means and log-stds go back in as
  constant channels, because nonlinear physics depends on amplitude, unless `invariant` says the
  physics does not care: "scale" hides a group's size (only mean/RMS, which stays in [−1, 1], goes
  in), "shift" hides its level (only log std goes in).
- The output is in `units`: a product of input statistics, [(channel, "std" | "rms" | "exp_mean",
  power), ...]. With the right units and invariances the operator respects the equation's
  symmetries exactly (for −∇·(a∇u) = f: u scales with f and inversely with a).
- Coordinates go in as channels: sin/cos on periodic space, a [0, 1) ramp otherwise.
- The output is `base + units · network`: base is the `residual` input channel (time stepping)
  or zero (operators). `scale: (channel, kind)` is shorthand for units [(channel, kind, 1)].
- `padding` extends the grid before the Fourier layers, the usual fix for non-periodic domains.
- Spectral weights are real tensors (…, 2) viewed as complex, so every parameter is real and the
  optimizer state is plain.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _conv(dims: int, c_in: int, c_out: int) -> nn.Module:
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims](c_in, c_out, 1)


class SpectralConv(nn.Module):
    def __init__(self, dims: int, width: int, modes: int):
        super().__init__()
        if dims not in (1, 2, 3):
            raise ValueError(f"dims must be 1, 2 or 3, got {dims}")
        self.dims, self.modes = dims, modes
        shape = (width, width) + (modes,) * dims
        scale = 1.0 / (width * width)
        self.weight = nn.Parameter(scale * torch.randn(*shape, 2))
        if dims == 2:   # a second block for negative frequencies along the first axis
            self.weight_neg = nn.Parameter(scale * torch.randn(*shape, 2))
        if dims == 3:   # blocks for (−, +), (+, −) and (−, −) frequencies along the first two axes
            self.weight_np = nn.Parameter(scale * torch.randn(*shape, 2))
            self.weight_pn = nn.Parameter(scale * torch.randn(*shape, 2))
            self.weight_nn = nn.Parameter(scale * torch.randn(*shape, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Built with cat, not in-place writes, so forward-mode AD (the self-play score) works.
        if self.dims == 1:
            n = x.shape[-1]
            xf = torch.fft.rfft(x)
            m = min(self.modes, xf.shape[-1])
            w = torch.view_as_complex(self.weight)[..., :m]
            out = torch.einsum("bim,iom->bom", xf[..., :m], w)
            out = torch.cat([out, out.new_zeros(*out.shape[:2], xf.shape[-1] - m)], dim=-1)
            return torch.fft.irfft(out, n=n)
        if self.dims == 3:
            return self._forward_3d(x)
        h, w_ = x.shape[-2:]
        xf = torch.fft.rfft2(x)
        m1, m2 = min(self.modes, h // 2), min(self.modes, xf.shape[-1])
        pos = torch.einsum("bixy,ioxy->boxy", xf[..., :m1, :m2], torch.view_as_complex(self.weight)[..., :m1, :m2])
        neg = torch.einsum("bixy,ioxy->boxy", xf[..., -m1:, :m2], torch.view_as_complex(self.weight_neg)[..., :m1, :m2])
        middle = pos.new_zeros(*pos.shape[:2], h - 2 * m1, m2)
        out = torch.cat([pos, middle, neg], dim=-2)
        out = torch.cat([out, out.new_zeros(*out.shape[:3], xf.shape[-1] - m2)], dim=-1)
        return torch.fft.irfft2(out, s=(h, w_))


    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        h, w_, d = x.shape[-3:]
        xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
        m1, m2, m3 = min(self.modes, h // 2), min(self.modes, w_ // 2), min(self.modes, xf.shape[-1])
        mix = lambda block, weight: torch.einsum(
            "bixyz,ioxyz->boxyz", block, torch.view_as_complex(weight)[..., :m1, :m2, :m3])
        rows = []
        for sx, (w_pos, w_neg) in ((slice(0, m1), (self.weight, self.weight_pn)),
                                   (slice(h - m1, h), (self.weight_np, self.weight_nn))):
            pos = mix(xf[:, :, sx, :m2, :m3], w_pos)
            neg = mix(xf[:, :, sx, w_ - m2:, :m3], w_neg)
            rows.append(torch.cat([pos, pos.new_zeros(*pos.shape[:3], w_ - 2 * m2, m3), neg], dim=-2))
        out = torch.cat([rows[0], rows[0].new_zeros(*rows[0].shape[:2], h - 2 * m1, w_, m3), rows[1]], dim=-3)
        out = torch.cat([out, out.new_zeros(*out.shape[:4], xf.shape[-1] - m3)], dim=-1)
        return torch.fft.irfftn(out, s=(h, w_, d), dim=(-3, -2, -1))


class FNO(nn.Module):
    def __init__(self, io: dict, width: int = 64, modes: int = 16, layers: int = 4, padding: int = 0):
        super().__init__()
        self.io = io
        self.dims = io["dims"]
        self.groups = [list(g) for g in io.get("norm_groups") or [[c] for c in range(io["in_channels"])]]
        self.group_of = {c: i for i, g in enumerate(self.groups) for c in g}
        self.padding = padding
        invariant = {int(c): kind for c, kind in (io.get("invariant") or {}).items()}
        self.group_invariance = [invariant.get(g[0]) for g in self.groups]
        units = io.get("units")
        if units is None and io.get("scale") is not None:
            units = [(*io["scale"], 1)]
        self.units = [tuple(u) for u in units or []]
        constants = sum(2 if kind is None else 1 for kind in self.group_invariance)
        coords = 2 * self.dims if io.get("periodic") else self.dims
        self.lift = _conv(self.dims, io["in_channels"] + coords + constants, width)
        self.spectral = nn.ModuleList(SpectralConv(self.dims, width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(_conv(self.dims, width, width) for _ in range(layers))
        self.head = nn.Sequential(_conv(self.dims, width, 128), nn.GELU(), _conv(self.dims, 128, io["out_channels"]))

    def _coordinates(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, space = x.shape[0], x.shape[2:]
        out = []
        for axis, n in enumerate(space):
            t = torch.arange(n, device=x.device, dtype=x.dtype) / n
            shape = [1] * len(space)
            shape[axis] = n
            t = t.view(1, 1, *shape).expand(b, 1, *space)
            out += [torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)] if self.io.get("periodic") else [t]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, space = x.shape[0], x.shape[2:]
        axes = tuple(range(2, x.ndim))
        normed, stats = [], []
        for group in self.groups:
            xs = x[:, group]
            mean = xs.mean(dim=(1, *axes), keepdim=True)
            std = xs.std(dim=(1, *axes), keepdim=True).clamp_min(1e-4)
            normed.append((xs - mean) / std)
            stats.append((mean, std, xs.pow(2).mean(dim=(1, *axes), keepdim=True).sqrt().clamp_min(1e-4)))
        order = [c for g in self.groups for c in g]
        inverse = [order.index(c) for c in range(len(order))]
        channels = torch.cat(normed, dim=1)[:, inverse]
        constants = []
        for (mean, std, rms), kind in zip(stats, self.group_invariance):
            chosen = {"scale": (mean / rms,), "shift": (torch.log(std),)}.get(kind, (mean, torch.log(std)))
            constants += [t.expand(b, 1, *space) for t in chosen]
        h = self.lift(torch.cat([channels, *self._coordinates(x), *constants], dim=1))
        if self.padding:
            h = nn.functional.pad(h, (0, self.padding) * self.dims)
        for i, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            h = spectral(h) + pointwise(h)
            if i < len(self.spectral) - 1:
                h = nn.functional.gelu(h)
        if self.padding:
            h = h[(..., *[slice(0, n) for n in space])]
        out = self.head(h)
        for channel, kind, power in self.units:
            mean, std, rms = stats[self.group_of[channel]]
            value = {"std": std, "rms": rms, "exp_mean": torch.exp(mean)}[kind]
            out = out * value.pow(power)
        if self.io.get("residual") is not None:
            out = out + x[:, [self.io["residual"]]]
        return out


def per_sample_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error divided by the target's variance (plus a little of its mean square, so a
    flat target does not divide by zero): one scale-free value per sample."""
    axes = tuple(range(1, target.ndim))
    scale = target.var(dim=axes, unbiased=False) + 1e-3 * target.pow(2).mean(dim=axes) + 1e-8
    return (prediction - target).pow(2).mean(dim=axes) / scale
