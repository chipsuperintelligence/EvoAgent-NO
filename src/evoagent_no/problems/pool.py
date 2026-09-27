"""The question-maker's search within a run: which problems to pose each round.

Each round mixes three sources, as in the self-play paper: fresh problems from the generator,
mutations of problems that scored well, and replays of earlier ones (re-scored, because a
problem's score falls as the student masters it). A replay keeps the problem's physics and, if
the problem has a `seed` field, draws new random fields: the student meets the same kind of
problem again, not the same samples, which it would only memorize. The archive is a MAP-Elites grid over the
physics backend's `describe` cells, one elite per cell, so the curriculum cannot collapse onto
one kind of physics. Parents are picked by rank, not raw score, so a few outliers cannot take
over; scores are normalized by each round's mean and decay every round.

`fixed_prior=True` switches the pool off: every problem comes fresh. Fresh problems come in equal
shares from `mixture`, the generators the student trains on (by default just `generator`).
A generator that raises or makes malformed problems is covered by the fallback generator (the
task's original), and the count is kept in `malformed`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass
class Elite:
    problem: object
    score: float


class Pool:
    def __init__(self, generator, physics, rng: np.random.Generator, fixed_prior: bool = False,
                 fresh: float = 0.4, mutated: float = 0.4, decay: float = 0.8, fallback=None):
        self.generator = generator
        self.fallback = fallback or generator
        self.physics = physics
        self.rng = rng
        self.fixed_prior = fixed_prior
        self.fresh, self.mutated = fresh, mutated
        self.decay = decay
        self.mixture: list = []
        self.archive: dict[tuple, Elite] = {}
        self.malformed = 0

    def _checked(self, problems: list) -> list:
        good = []
        for p in problems:
            try:
                self.physics.check(p)
                good.append(p)
            except Exception:
                self.malformed += 1
        return good

    def _fresh(self, n: int) -> list:
        sources = self.mixture or [self.generator]
        problems = []
        for i, source in enumerate(sources):
            share = n // len(sources) + (i < n % len(sources))
            try:
                problems += self._checked(source.generate(self.rng, share))[:share]
            except Exception:
                self.malformed += share
        if len(problems) < n:
            problems += self._checked(self.fallback.generate(self.rng, n - len(problems)))
        return problems

    def _mutations(self, parents: list) -> list:
        children = []
        for parent in parents:
            try:
                child = self.generator.mutate(parent, self.rng)
                self.physics.check(child)
            except Exception:
                self.malformed += 1
                try:
                    child = self.fallback.mutate(parent, self.rng)
                    self.physics.check(child)
                except Exception:
                    child = parent
            children.append(child)
        return children

    def _reseed(self, problem):
        if dataclasses.is_dataclass(problem) and any(f.name == "seed" for f in dataclasses.fields(problem)):
            return dataclasses.replace(problem, seed=int(self.rng.integers(0, 2**31 - 1)))
        return problem

    def propose(self, n: int) -> tuple[list, list[str]]:
        """n problems and where each came from: 'fresh', 'mutated' or 'replay'."""
        if self.fixed_prior or not self.archive:
            problems = self._fresh(n)
            return problems, ["fresh"] * len(problems)
        elites = sorted(self.archive.values(), key=lambda e: -e.score)
        rank_weights = 1.0 / np.arange(1, len(elites) + 1)
        rank_weights /= rank_weights.sum()
        n_fresh = int(round(self.fresh * n))
        n_mut = int(round(self.mutated * n))
        n_replay = n - n_fresh - n_mut
        fresh = self._fresh(n_fresh)
        parents = [elites[i].problem for i in self.rng.choice(len(elites), size=n_mut, p=rank_weights)]
        mutated = self._mutations(parents)
        replay = [self._reseed(elites[i].problem) for i in self.rng.choice(len(elites), size=n_replay)]
        origins = ["fresh"] * len(fresh) + ["mutated"] * len(mutated) + ["replay"] * len(replay)
        return fresh + mutated + replay, origins

    def update(self, problems: list, scores: np.ndarray, accepted: np.ndarray) -> None:
        """Record this round's scores. Rejected problems are never archived."""
        if self.fixed_prior:
            return
        for elite in self.archive.values():
            elite.score *= self.decay
        mean = scores[accepted].mean() if accepted.any() else 0.0
        if mean <= 0:
            return
        for problem, score, ok in zip(problems, scores, accepted):
            if not ok:
                continue
            score = float(score / mean)
            cell = self.physics.describe(problem)
            elite = self.archive.get(cell)
            if elite is None or score > elite.score or elite.problem == problem:
                self.archive[cell] = Elite(problem, score)
