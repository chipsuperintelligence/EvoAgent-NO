"""Use a pretrained student: load it from a run folder, predict, fine-tune on your own data.

    from evoagent_no import load_student
    student = load_student("runs/my-pde/agents-s0")
    u_next = student.predict(inputs)                      # numpy in, numpy out
    history = student.finetune(inputs, targets, steps=500)
    student.save("runs/my-pde/finetuned")

Inputs and targets use the task's layout: (M, in_channels, *space) and (M, out_channels, *space);
`student.io` describes it. Nothing from the task folder is needed, only the run folder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from evoagent_no.student.fno import FNO, per_sample_loss


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Pretrained:
    def __init__(self, model: FNO, config: dict, device: str):
        self.model, self.config, self.device = model, config, device

    @property
    def io(self) -> dict:
        return self.config["io"]

    @classmethod
    def load(cls, run_dir: str | Path, device: str | None = None) -> "Pretrained":
        run_dir = Path(run_dir)
        config = json.loads((run_dir / "config.json").read_text())
        if "io" not in config:
            raise ValueError(f"{run_dir}/config.json has no `io`; it was written by an older version")
        device = device or _default_device()
        s = config["student"]
        model = FNO(config["io"], width=s["width"], modes=s["modes"], layers=s["layers"],
                    padding=s.get("padding", 0)).to(device)
        model.load_state_dict(torch.load(run_dir / "student.pt", map_location=device))
        return cls(model, config, device)

    def _tensor(self, array) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)
        expected = 2 + self.io["dims"]
        if x.ndim != expected:
            raise ValueError(f"expected {expected}-D arrays (samples, channels, *space), got shape {tuple(x.shape)}")
        return x

    @torch.no_grad()
    def predict(self, inputs, batch: int = 1024) -> np.ndarray:
        x = self._tensor(inputs)
        if x.shape[1] != self.io["in_channels"]:
            raise ValueError(f"expected {self.io['in_channels']} input channels, got {x.shape[1]}")
        self.model.eval()
        return torch.cat([self.model(x[i:i + batch]) for i in range(0, len(x), batch)]).cpu().numpy()

    def finetune(self, inputs, targets, steps: int = 500, batch: int = 32, lr: float = 3e-4,
                 seed: int = 0) -> list[float]:
        """AdamW with a cosine-decaying learning rate on your samples; returns the loss every step
        (scale-free, as in pretraining)."""
        x, y = self._tensor(inputs), self._tensor(targets)
        if len(x) != len(y):
            raise ValueError(f"{len(x)} inputs but {len(y)} targets")
        g = torch.Generator(device=self.device).manual_seed(seed)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.model.train()
        history = []
        for step in range(steps):
            for group in optimizer.param_groups:   # cosine decay to a tenth of lr, as in pretraining
                group["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(steps - 1, 1))))
            idx = torch.randint(0, len(x), (min(batch, len(x)),), device=self.device, generator=g)
            loss = per_sample_loss(self.model(x[idx]), y[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    def save(self, run_dir: str | Path) -> Path:
        """A folder `load_student` can read: config.json and student.pt."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(self.config, indent=2))
        torch.save(self.model.state_dict(), run_dir / "student.pt")
        return run_dir


def load_student(run_dir: str | Path, device: str | None = None) -> Pretrained:
    return Pretrained.load(run_dir, device)


def relative_error(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean over samples of ‖pred − target‖ / ‖target‖, the held-out metric."""
    p, t = prediction.reshape(len(prediction), -1), target.reshape(len(target), -1)
    return float(np.mean(np.linalg.norm(p - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-8)))
