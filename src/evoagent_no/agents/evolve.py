"""The slow loop: LLM agents evolve the task's generator, and the student decides what survives.

Every `agents.every` rounds of the fast loop, one evolution step runs:

    1. Age and re-measure. Scores in the generator database fade, and the strongest elites
       are probed again on the current student, which has moved on since they were measured.
    2. Prompt. For each of `candidates` agents: pick an island, a parent from it (mostly strong,
       sometimes any elite), and `inspirations` other generators. The prompt holds the physics
       (context.md), the parent's code, the inspirations' EVOLVE-BLOCKs, the report of what the
       student is learning from (training signals only, never held-out scores), and earlier
       attempts.
    3. Call the LLMs, `parallel` at a time. Each reply is SEARCH/REPLACE edits to the parent's
       EVOLVE-BLOCK, or a whole new generator.py.
    4. Build, validate, measure. The edits are applied to the parent's block and spliced into
       its file; the result runs in a separate process with a time limit; then it is probed on
       the student and placed in its island's MAP-Elites grid if it earns a cell.
    5. Migrate every `migrate_every` steps, then hand the best generator to the fast loop.

One step makes exactly `candidates` LLM calls. Every attempt, and the database after every step,
is written to the run folder.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from evoagent_no.agents.database import Entry, GeneratorDatabase, ProbeResult
from evoagent_no.agents.diff import apply_edits, parse_edits
from evoagent_no.problems.generator import GeneratorProgram, extract_code, splice, split, validate

SYSTEM = (
    "You improve the problem generator of an EvoAgent-NO task. A neural operator, the student, is "
    "pretrained only on the problems the generator poses. Change the code inside the EVOLVE-BLOCK "
    "so that the student learns more from the problems it gets."
)

EDIT_FORMAT = {
    "diff": """\
Start with one line, `Change: <what you changed and why>`. Then give your change as one or more
edits to the EVOLVE-BLOCK of the generator to improve, in exactly this form:

<<<<<<< SEARCH
lines copied exactly from the EVOLVE-BLOCK
=======
the lines that replace them
>>>>>>> REPLACE

Each SEARCH must match exactly one place. For a large change, you may instead give the complete
new generator.py in one ```python block, keeping both EVOLVE-BLOCK markers.""",
    "full": """\
Start with one line, `Change: <what you changed and why>`. Then give the complete new generator.py
in one ```python block, keeping both EVOLVE-BLOCK markers.""",
}

RULES = """\
- Keep `generate_problems(rng, n)` returning a list of exactly n problems, and
  `mutate_problem(problem, rng)` returning one. Use only `rng` for randomness.
- Use the imports already in the file (numpy, math, the physics module). No files, network,
  subprocesses or global state that changes between calls.
- Every problem must stay inside the hard limits; problems outside them are dropped.
- Aim for problems the student is learning from now: the report shows which kinds score well,
  which score poorly and which are never posed."""


@dataclass
class Attempt:
    round: int
    candidate: int
    island: int
    parent: str
    file: str
    model: str | None
    kind: str | None          # "diff" or "full"
    change: str
    score: float | None
    accepted: float | None
    coverage: float | None
    placed: bool              # earned a cell in its island's grid
    champion: bool            # became the generator the fast loop uses
    reason: str
    cost_usd: float | None
    seconds: float
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None


def _change_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith("change:"):
            return line.split(":", 1)[1].strip()[:200]
    return ""


class Trigger:
    """When an evolution step is worth its LLM calls.

    "every":   at every check (every `agents.every` rounds).
    "plateau": only when the student has stopped improving on the current curriculum: its training
               loss over the last `every` rounds improved by less than `min_improvement` against
               the `every` rounds before, both windows after the last evolution step. The first
               check always evolves.
    """

    def __init__(self, mode: str = "plateau", every: int = 20, min_improvement: float = 0.05):
        if mode not in ("plateau", "every"):
            raise ValueError(f"agents.trigger must be 'plateau' or 'every', got {mode!r}")
        self.mode, self.every, self.min_improvement = mode, every, min_improvement
        self.last_evolved: int | None = None

    def check(self, round_index: int, losses: list[float]) -> tuple[bool, str]:
        """(evolve now?, why). `losses` holds the training loss of every round before this one."""
        if self.mode == "every" or self.last_evolved is None:
            return True, "scheduled" if self.mode == "every" else "first check"
        n = self.every
        if round_index - self.last_evolved < 2 * n:
            return False, "not enough rounds since the last evolution step to judge"
        recent = [x for x in losses[round_index - n:round_index] if x == x]
        before = [x for x in losses[round_index - 2 * n:round_index - n] if x == x]
        if not recent or not before:
            return True, "no loss to compare"
        improvement = 1.0 - (sum(recent) / len(recent)) / max(sum(before) / len(before), 1e-12)
        if improvement < self.min_improvement:
            return True, f"student plateaued ({improvement:+.0%} loss improvement over {n} rounds)"
        return False, f"student still improving ({improvement:+.0%} loss over {n} rounds)"

    def evolved(self, round_index: int) -> None:
        self.last_evolved = round_index


class GeneratorEvolver:
    def __init__(self, task, llm, out_dir: Path, settings: dict, log=print, seed: int = 0):
        self.task, self.llm, self.log = task, llm, log
        self.candidates = int(settings.get("candidates", 2))
        self.parallel = max(1, int(settings.get("parallel", self.candidates)))
        self.rescore = int(settings.get("rescore", 8))
        self.n_inspirations = int(settings.get("inspirations", 2))
        self.migrate_every = int(settings.get("migrate_every", 3))
        self.edits = settings.get("edits", "diff")
        if self.edits not in EDIT_FORMAT:
            raise ValueError(f"agents.edits must be 'diff' or 'full', got {self.edits!r}")
        self.db = GeneratorDatabase(islands=int(settings.get("islands", 2)), seed=seed)
        self.dir = out_dir / "generators"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.attempts: list[Attempt] = []
        self.steps = 0
        self.champions = 0
        self._log_file = out_dir / "agents.jsonl"
        seed_path = self.dir / "v000.py"
        seed_path.write_text(task.generator_path.read_text())
        self.seed_entry = Entry(id="v000", path=seed_path, program=GeneratorProgram.load(seed_path),
                                parent=None, island=0, round=0, change="the task's own generator")
        self.champion: Entry = self.seed_entry

    # ---- checkpoints ------------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Everything needed to continue: entries by file (programs are reloaded), grids, history."""
        return {
            "steps": self.steps, "champions": self.champions, "champion": self.champion.id,
            "db_rng": self.db.rng.bit_generator.state,
            "entries": [{**e.summary(), "path": str(e.path), "measured_at": e.measured_at}
                        for e in self.db.entries.values()],
            "grids": [[[list(cell), i] for cell, i in g.items()] for g in self.db.grids],
            "attempts": [asdict(a) for a in self.attempts],
        }

    def load_state_dict(self, state: dict) -> None:
        self.steps, self.champions = state["steps"], state["champions"]
        self.db.rng.bit_generator.state = state["db_rng"]
        self.db.entries = {}
        for d in state["entries"]:
            entry = Entry(id=d["id"], path=Path(d["path"]), program=GeneratorProgram.load(d["path"]),
                          parent=d["parent"], island=d["island"], round=d["round"], change=d["change"],
                          score=d["score"], accepted=d["accepted"], coverage=d["coverage"],
                          measured_at=d["measured_at"])
            self.db.entries[entry.id] = entry
        self.db.grids = [{tuple(cell): i for cell, i in g} for g in state["grids"]]
        self.attempts = [Attempt(**a) for a in state["attempts"]]
        self.seed_entry = self.db.entries.get("v000", self.seed_entry)
        self.champion = self.db.entries.get(state["champion"], self.seed_entry)

    def population(self) -> list[GeneratorProgram]:
        """The generators the student trains on, in equal shares: the task's own, always, plus every
        elite that beat it in the latest trial. A rewrite adds to what the student sees; it never
        takes the original away, so a misleading trial costs at most its share."""
        seed = self.seed_entry
        better = [e for e in self.db.elites() if e is not seed and e.score is not None
                  and seed.score is not None and e.score > seed.score]
        return [seed.program] + [e.program for e in better]

    # ---- prompts ----------------------------------------------------------------------------

    def prompt(self, parent: Entry, inspirations: list[Entry], report: str) -> str:
        def describe(e: Entry) -> str:
            return (f"score {e.score:.4f}, covers {e.coverage:.0%} of kinds of problem, "
                    f"{e.accepted:.0%} accepted" if e.score is not None else "not measured yet")

        others = "\n\n".join(
            f"## {e.id} ({describe(e)})" + (f"\nChange: {e.change}" if e.change else "")
            + f"\n```python\n{split(e.program.source)[1].strip()}\n```"
            for e in inspirations) or "(none yet)"
        earlier = "\n".join(
            f"- {a.file}: {'kept' if a.placed else 'dropped'}"
            f"{' (new best)' if a.champion else ''}, score {'-' if a.score is None else f'{a.score:.4f}'}"
            f"{': ' + a.change if a.change else ''}{'' if a.placed else f' [{a.reason}]'}"
            for a in self.attempts[-8:]) or "- none yet"
        # Parts that never change within a run come first, so providers can serve them from their
        # prompt cache; the parts that change every step come after.
        return (
            f"# The physics\n\n{self.task.context}\n\n"
            f"# How to reply\n\n{EDIT_FORMAT[self.edits]}\n\n# Rules\n\n{RULES}\n\n"
            f"# The generator to improve: {parent.id} ({describe(parent)})\n\n"
            f"```python\n{parent.program.source}\n```\n\n"
            f"# Other generators, for ideas (EVOLVE-BLOCKs only)\n\n{others}\n\n"
            f"# What the student is learning from\n\n{report}\n\n"
            f"# Earlier attempts\n\n{earlier}\n"
        )

    def build(self, parent: Entry, reply: str) -> tuple[str, str]:
        """The new generator.py from a reply, and whether it came from edits or a full rewrite."""
        block = split(parent.program.source)[1]
        edits = parse_edits(reply)
        if edits:
            return splice(parent.program.source, apply_edits(block, edits)), "diff"
        code = extract_code(reply)
        try:
            new_block = split(code)[1]
        except ValueError:
            raise ValueError("the reply has neither SEARCH/REPLACE edits nor a file with EVOLVE-BLOCK markers") from None
        return splice(parent.program.source, new_block), "full"

    # ---- one evolution step -----------------------------------------------------------------

    def step(self, round_index: int, report: str, probe: Callable[[GeneratorProgram], ProbeResult]) -> GeneratorProgram:
        self.steps += 1
        if self.steps == 1:
            self.seed_entry.record(probe(self.seed_entry.program), self.steps)
            self.db.add(self.seed_entry)
        else:
            self.db.age()
            for entry in dict.fromkeys(self.db.elites()[:self.rescore] + [self.seed_entry]):
                entry.record(probe(entry.program), self.steps)   # the seed too: it is the yardstick
            self.db.regrid()

        jobs = []
        for c in range(1, self.candidates + 1):
            island = (self.steps + c) % self.db.islands
            parent = self.db.sample_parent(island)
            jobs.append((c, island, parent, self.prompt(parent, self.db.inspirations(parent, self.n_inspirations), report)))

        def call(job):
            start = time.monotonic()
            try:
                return self.llm.complete(job[3], system=SYSTEM), None, time.monotonic() - start
            except Exception as e:   # a failed call is one dropped attempt, never a stopped run
                return None, e, time.monotonic() - start

        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            replies = list(pool.map(call, jobs))

        for (c, island, parent, _), (reply, error, seconds) in zip(jobs, replies):
            path = self.dir / f"r{round_index:04d}-c{c}.py"
            attempt = Attempt(round_index, c, island, parent.id, path.name, getattr(reply, "model", None), None,
                              _change_line(reply.text) if reply else "", None, None, None, False, False, "",
                              getattr(reply, "cost_usd", None), seconds, getattr(reply, "input_tokens", None),
                              getattr(reply, "cached_tokens", None), getattr(reply, "output_tokens", None))
            if reply is not None:   # the raw reply, for anyone checking what the agent said
                path.with_suffix(".reply.md").write_text(reply.text)
            try:
                if error is not None:
                    raise error
                source, attempt.kind = self.build(parent, reply.text)
                twin = next((e.id for e in self.db.entries.values() if e.program.source == source), None)
                if twin is not None:   # nothing new: skip the validation and the GPU probe
                    raise ValueError(f"same code as {twin}")
                path.write_text(source)
                validate(path, self.task.settings["physics"], self.task.root)
                program = GeneratorProgram.load(path)
                result = probe(program)
                attempt.score, attempt.accepted, attempt.coverage = result.score, result.accepted, result.coverage
                entry = Entry(id=path.stem, path=path, program=program, parent=parent.id, island=island,
                              round=round_index, change=attempt.change)
                entry.record(result, self.steps)
                attempt.placed = self.db.add(entry)
                attempt.reason = "earned a cell" if attempt.placed else "a better generator holds its cell"
            except Exception as e:
                attempt.reason = f"{type(e).__name__}: {e}"[:300]
            self.attempts.append(attempt)

        if self.migrate_every and self.steps % self.migrate_every == 0:
            self.db.migrate()
        best = self.db.best()
        if best is not self.champion:
            self.champions += 1
            (self.dir / f"v{self.champions:03d}.py").write_text(best.program.source)
            for a in self.attempts[-len(jobs):]:
                a.champion = a.file == best.path.name
        self.champion = best
        self.db.save(self.dir / "database.json")
        with self._log_file.open("a") as f:
            for a in self.attempts[-len(jobs):]:
                f.write(json.dumps(asdict(a)) + "\n")
                score = "-" if a.score is None else f"{a.score:.4f}"
                self.log(f"  agent r{a.round} c{a.candidate} island {a.island} from {a.parent}: "
                         f"{'NEW BEST' if a.champion else 'kept' if a.placed else 'dropped'} "
                         f"score {score} ({a.kind or '-'}; {a.reason}; {a.seconds:.0f} s)")
        self.log(f"  generators: {len(self.db.elites())} elites on {self.db.islands} islands; "
                 f"best {best.id} score {best.score:.4f}")
        return best.program
