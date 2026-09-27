"""The generator database: a population of generator versions, organized for search.

Each island keeps a MAP-Elites grid of generators. A generator's cell comes from how its problems
behave: how many kinds of problem they cover, and how many the answer key accepts. Its score is
the learning progress its problems give the current student. The student keeps learning, so
scores go stale: they fade every slow step, and the strongest elites are re-measured.

Islands evolve separately and swap their best generator every few steps (a ring), which keeps
the search from settling on one idea too early.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ProbeResult:
    """What one probe of a generator on the current student measured."""
    score: float      # how much a short trial on its problems teaches the student; rejected count 0
    accepted: float   # fraction of its problems the answer key accepted
    coverage: float   # fraction of the backend's kinds of problem among them


@dataclass(eq=False)
class Entry:
    id: str
    path: Path
    program: object = field(repr=False)   # GeneratorProgram
    parent: str | None
    island: int
    round: int
    change: str = ""
    score: float | None = None
    accepted: float = 0.0
    coverage: float = 0.0
    measured_at: int = -1                  # slow step of the last probe

    def record(self, result: ProbeResult, step: int) -> None:
        self.score, self.accepted, self.coverage, self.measured_at = (
            result.score, result.accepted, result.coverage, step)

    @property
    def cell(self) -> tuple[int, int]:
        """(coverage quartile, acceptance: under 60%, under 90%, 90% or more)."""
        return min(int(self.coverage * 4), 3), int(np.digitize(self.accepted, (0.6, 0.9)))

    def summary(self) -> dict:
        return {"id": self.id, "parent": self.parent, "island": self.island, "round": self.round,
                "change": self.change, "score": self.score, "accepted": self.accepted,
                "coverage": self.coverage, "file": self.path.name}


class GeneratorDatabase:
    def __init__(self, islands: int = 2, decay: float = 0.8, exploit: float = 0.7, seed: int = 0):
        self.islands = islands
        self.decay = decay
        self.exploit = exploit
        self.rng = np.random.default_rng(seed)
        self.entries: dict[str, Entry] = {}
        self.grids: list[dict[tuple, str]] = [{} for _ in range(islands)]

    def add(self, entry: Entry, island: int | None = None) -> bool:
        """Place a measured entry in its island's grid if its cell is empty or it scores higher."""
        if entry.score is None:
            raise ValueError("measure an entry before adding it")
        self.entries[entry.id] = entry
        grid = self.grids[entry.island if island is None else island]
        holder = grid.get(entry.cell)
        if holder is None or holder == entry.id or entry.score > self.entries[holder].score:
            grid[entry.cell] = entry.id
            return True
        return False

    def elites(self, island: int | None = None) -> list[Entry]:
        grids = self.grids if island is None else [self.grids[island]]
        ids = dict.fromkeys(i for g in grids for i in g.values())
        return sorted((self.entries[i] for i in ids), key=lambda e: -(e.score or 0.0))

    def best(self) -> Entry:
        return self.elites()[0]

    def age(self) -> None:
        """Scores fade: a generator the student has learned from is worth less every step."""
        for entry in self.elites():
            if entry.score is not None:
                entry.score *= self.decay

    def regrid(self) -> None:
        """Re-place elites after re-measuring, since their cells can move."""
        placed = [(i, self.entries[e]) for i, g in enumerate(self.grids) for e in g.values()]
        self.grids = [{} for _ in range(self.islands)]
        for island, entry in placed:
            self.add(entry, island)

    def sample_parent(self, island: int) -> Entry:
        """Mostly the island's strong generators (by rank), sometimes any of its elites."""
        elites = self.elites(island) or self.elites()
        if self.rng.random() < self.exploit:
            weights = 1.0 / np.arange(1, len(elites) + 1)
            return elites[self.rng.choice(len(elites), p=weights / weights.sum())]
        return elites[self.rng.integers(len(elites))]

    def inspirations(self, parent: Entry, k: int) -> list[Entry]:
        """Up to k other generators to show the agent: the strongest, plus one at random."""
        others = [e for e in self.elites() if e.id != parent.id]
        if k <= 0 or not others:
            return []
        top = others[:max(k - 1, 1)]
        chosen = {e.id for e in top}
        rest = [e for e in others if e.id not in chosen]
        if rest and len(top) < k:
            top.append(rest[self.rng.integers(len(rest))])
        return top

    def migrate(self) -> None:
        """Each island's best generator also joins the next island (a ring)."""
        if self.islands < 2:
            return
        bests = [self.elites(i)[0] if self.grids[i] else None for i in range(self.islands)]
        for i, entry in enumerate(bests):
            if entry is not None:
                self.add(entry, (i + 1) % self.islands)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "islands": [{"elites": [self.entries[i].id for i in g.values()]} for g in self.grids],
            "entries": [e.summary() for e in self.entries.values()],
        }, indent=2))
