"""Several models behind one `complete()`: each call goes to one of them, chosen by weight.

Mixing models gives the agents more varied rewrites than any one model does. In task.yaml,
`agents.llm` is either one model or a list of them, each with a `weight`.
"""

from __future__ import annotations

import threading

import numpy as np

from evoagent_no.llm.base import Reply


class Ensemble:
    def __init__(self, members: list, weights: list[float], seed: int = 0):
        if not members or len(members) != len(weights):
            raise ValueError("an ensemble needs one weight per model")
        total = float(sum(weights))
        if total <= 0 or any(w < 0 for w in weights):
            raise ValueError("ensemble weights must be non-negative and not all zero")
        self.members = members
        self.weights = [w / total for w in weights]
        self.rng = np.random.default_rng(seed)
        self._lock = threading.Lock()   # agents call from several threads at once

    def complete(self, prompt: str, *, system: str | None = None) -> Reply:
        with self._lock:
            member = self.members[self.rng.choice(len(self.members), p=self.weights)]
        return member.complete(prompt, system=system)
