"""The student's training: AdamW on the current round's samples, a learning-rate schedule, an
averaged copy of the student (EMA), and the parameter snapshots the self-play score looks back to.

- **Schedule.** `cosine` (default): a linear warm-up over `warmup_steps`, then a cosine decay to
  `min_lr_ratio` × lr at the end of the run, so the student settles instead of bouncing around at
  a constant step size. `constant` keeps lr throughout. Extending a finished run re-plans the
  schedule over the new length.
- **EMA.** An exponential moving average of the weights (`ema`, 0 to switch off), with the usual
  warm-up min(ema, (1 + t) / (10 + t)). Held-out tests, predictions and the saved student use the
  average, which smooths out the noise of individual steps. The self-play score keeps using the
  trained weights: learning progress is a property of the model being trained.
"""

from __future__ import annotations

import copy
import math

import torch

from evoagent_no.student.fno import FNO, per_sample_loss


class Learner:
    def __init__(self, device, io: dict, lr: float = 1e-3, weight_decay: float = 1e-4,
                 schedule: str = "cosine", warmup_steps: int = 100, min_lr_ratio: float = 0.05,
                 ema: float = 0.999, **model_options):
        if schedule not in ("cosine", "constant"):
            raise ValueError(f"student.schedule must be 'cosine' or 'constant', got {schedule!r}")
        self.model = FNO(io, **model_options).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.base_lr, self.schedule = lr, schedule
        self.warmup_steps, self.min_lr_ratio = warmup_steps, min_lr_ratio
        self.total_steps: int | None = None
        self.step_count = 0
        self.ema = ema
        self.ema_model = None
        if ema:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.requires_grad_(False)
        self._snapshots: dict[int, dict[str, torch.Tensor]] = {}

    @classmethod
    def for_task(cls, task, device) -> "Learner":
        s = task.section("student")
        return cls(device, task.physics.IO, lr=s["lr"], width=s["width"], modes=s["modes"],
                   layers=s["layers"], padding=s.get("padding", 0), schedule=s.get("schedule", "cosine"),
                   warmup_steps=s.get("warmup_steps", 100), min_lr_ratio=s.get("min_lr_ratio", 0.05),
                   ema=s.get("ema", 0.999))

    # ---- schedule and averaging -------------------------------------------------------------

    def plan(self, total_steps: int) -> None:
        """How many training steps the run will take, which the cosine schedule spans."""
        self.total_steps = total_steps

    def lr_at(self, step: int) -> float:
        if self.schedule == "constant":
            return self.base_lr
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        total = max(self.total_steps or step + 1, self.warmup_steps + 1)
        progress = min((step - self.warmup_steps) / (total - self.warmup_steps), 1.0)
        low = self.min_lr_ratio * self.base_lr
        return low + (self.base_lr - low) * 0.5 * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = min(self.ema, (1 + self.step_count) / (10 + self.step_count))
        for averaged, current in zip(self.ema_model.parameters(), self.model.parameters()):
            averaged.lerp_(current, 1 - decay)

    # ---- training -------------------------------------------------------------------------

    def train(self, inputs: torch.Tensor, targets: torch.Tensor, steps: int, batch: int,
              generator: torch.Generator) -> float:
        """`steps` AdamW steps on minibatches drawn uniformly from the round's samples."""
        self.model.train()
        total = 0.0
        for _ in range(steps):
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr_at(self.step_count)
            idx = torch.randint(0, inputs.shape[0], (batch,), device=inputs.device, generator=generator)
            loss = per_sample_loss(self.model(inputs[idx]), targets[idx]).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.step_count += 1
            if self.ema_model is not None:
                self._update_ema()
            total += loss.item()
        return total / max(steps, 1)

    def trial(self, train: tuple[torch.Tensor, torch.Tensor], test: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              n_problems: int, steps: int, batch: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """Train briefly on `train` = (inputs, targets), then put the student back exactly as it was.
        Returns each problem's mean loss on `test` = (inputs, targets, problem index) before and
        after, each (n_problems,): how much the student would learn from these problems, measured
        rather than extrapolated. Uses the current learning rate; the step count and EMA are untouched."""
        saved_model = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        per_problem = lambda: self.per_problem_loss(*test, n_problems)
        before = per_problem()
        lr = self.lr_at(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        for _ in range(steps):
            idx = torch.randint(0, train[0].shape[0], (batch,), device=train[0].device, generator=generator)
            loss = per_sample_loss(self.model(train[0][idx]), train[1][idx]).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        after = per_problem()
        self.model.load_state_dict(saved_model)
        self.optimizer.load_state_dict(saved_optimizer)
        return before, after

    @torch.no_grad()
    def per_problem_loss(self, inputs: torch.Tensor, targets: torch.Tensor, problem: torch.Tensor,
                         n_problems: int, chunk: int = 4096) -> torch.Tensor:
        """Mean loss of the trained student per problem, (n_problems,); 0 for problems with no samples."""
        self.model.eval()
        loss = torch.cat([per_sample_loss(self.model(inputs[i:i + chunk]), targets[i:i + chunk])
                          for i in range(0, len(inputs), chunk)])
        zeros = torch.zeros(n_problems, device=loss.device, dtype=loss.dtype)
        count = zeros.clone().index_add_(0, problem, torch.ones_like(loss)).clamp_min(1)
        return zeros.index_add_(0, problem, loss) / count

    def snapshot(self, round_index: int) -> None:
        self._snapshots[round_index] = {k: v.detach().to("cpu", copy=True)
                                        for k, v in self.model.named_parameters()}

    def lookback(self, round_index: int) -> dict[str, torch.Tensor]:
        """Parameters at round ⌊e/2⌋. Snapshots older than that are never needed again."""
        target = round_index // 2
        for old in [r for r in self._snapshots if r < target]:
            del self._snapshots[old]
        return self._snapshots[target]

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Predictions of the averaged student (or the trained one, with EMA off)."""
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()
        return torch.cat([model(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)])

    # ---- saving ---------------------------------------------------------------------------

    def student_state(self) -> dict:
        """The weights to ship: the average if there is one."""
        return (self.ema_model if self.ema_model is not None else self.model).state_dict()

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "ema_model": self.ema_model.state_dict() if self.ema_model is not None else None,
                "step_count": self.step_count, "snapshots": self._snapshots}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._snapshots = state["snapshots"]
        self.step_count = state.get("step_count", 0)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(state.get("ema_model") or state["model"])
