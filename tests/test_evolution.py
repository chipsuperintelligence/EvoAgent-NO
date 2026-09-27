"""The evolutionary slow loop: edits, the generator database, LLM ensembles, one full step."""

import json
import threading
import time
from pathlib import Path

import pytest

from evoagent_no.agents.database import Entry, GeneratorDatabase, ProbeResult
from evoagent_no.agents.diff import apply_edits, parse_edits
from evoagent_no.agents.evolve import GeneratorEvolver, Trigger
from evoagent_no.llm import Ensemble, from_settings
from evoagent_no.llm.base import Reply
from evoagent_no.problems.generator import GeneratorProgram, splice

SEARCH = '"diffusion":      (0.8,'
GOOD_EDIT = f'Change: diffusion on more often\n<<<<<<< SEARCH\n{SEARCH}\n=======\n"diffusion":      (0.9,\n>>>>>>> REPLACE\n'


def reply(text, model="fake"):
    return Reply(text=text, model=model, input_tokens=0, output_tokens=0, cost_usd=None, seconds=0.0)


class ScriptedLLM:
    def __init__(self, texts):
        self.texts, self.prompts = list(texts), []

    def complete(self, prompt, *, system=None):
        self.prompts.append(prompt)
        return reply(self.texts.pop(0))


def probe(program):
    """A stand-in for the student: the edited generator is better."""
    return ProbeResult(score=1.0 if "(0.9," in program.source else 0.5, accepted=1.0, coverage=0.5)


# ---- edits -------------------------------------------------------------------------------------

def test_edits_apply_in_order_and_must_match_once():
    edits = parse_edits("x\n<<<<<<< SEARCH\na = 1\n=======\na = 2\n>>>>>>> REPLACE\n"
                        "<<<<<<< SEARCH\nb = 1\n=======\nb = 3\n>>>>>>> REPLACE")
    assert edits == [("a = 1", "a = 2"), ("b = 1", "b = 3")]
    assert apply_edits("a = 1\nb = 1\n", edits) == "a = 2\nb = 3\n"
    with pytest.raises(ValueError, match="not found"):
        apply_edits("c = 1\n", edits)
    with pytest.raises(ValueError, match="appears 2 times"):
        apply_edits("a = 1\na = 1\n", [("a = 1", "a = 2")])


# ---- database ----------------------------------------------------------------------------------

def entry(name, score, coverage=0.5, accepted=1.0, island=0):
    e = Entry(id=name, path=Path(f"{name}.py"), program=None, parent=None, island=island, round=0)
    e.record(ProbeResult(score, accepted, coverage), 1)
    return e


def test_database_keeps_one_elite_per_cell_per_island():
    db = GeneratorDatabase(islands=2, seed=0)
    assert db.add(entry("a", 0.5))
    assert db.add(entry("b", 0.9))                           # same cell, better: replaces a
    assert not db.add(entry("c", 0.1))                       # same cell, worse
    assert db.add(entry("d", 0.2, coverage=0.1))             # a different cell
    assert db.add(entry("e", 0.3, island=1))
    assert [e.id for e in db.elites(0)] == ["b", "d"]
    assert db.best().id == "b"


def test_scores_fade_and_regrid_moves_remeasured_elites():
    db = GeneratorDatabase(islands=1, decay=0.5)
    e = entry("a", 0.8)
    db.add(e)
    db.age()
    assert e.score == pytest.approx(0.4)
    e.record(ProbeResult(0.4, 0.3, 0.9), 2)                  # re-measured: now a different cell
    db.regrid()
    assert list(db.grids[0]) == [e.cell] == [(3, 0)]


def test_migration_is_a_ring_of_island_bests():
    db = GeneratorDatabase(islands=3)
    db.add(entry("a", 0.9, island=0))
    db.add(entry("b", 0.5, island=1, coverage=0.9))
    db.migrate()
    assert {e.id for e in db.elites(1)} == {"a", "b"}        # a moved 0 → 1
    assert {e.id for e in db.elites(2)} == {"b"}             # b moved 1 → 2


def test_parents_come_from_their_island_and_inspirations_exclude_them():
    db = GeneratorDatabase(islands=2, seed=1)
    for i, cov in enumerate((0.1, 0.3, 0.6, 0.9)):
        db.add(entry(f"g{i}", 0.1 * (i + 1), coverage=cov))
    db.add(entry("other", 0.05, island=1))
    assert db.sample_parent(1).id == "other"
    parent = db.sample_parent(0)
    ideas = db.inspirations(parent, 2)
    assert parent.id.startswith("g") and len(ideas) == 2 and parent.id not in {e.id for e in ideas}


# ---- LLM ensembles -----------------------------------------------------------------------------

def test_ensemble_picks_models_by_weight():
    a, b = ScriptedLLM(["from a"] * 5), ScriptedLLM(["from b"] * 5)
    ensemble = Ensemble([a, b], [1.0, 0.0])
    assert all(ensemble.complete("hi").text == "from a" for _ in range(5))
    built = from_settings([{"backend": "claude-cli", "weight": 2}, {"backend": "claude-cli", "weight": 1}])
    assert isinstance(built, Ensemble) and built.weights == pytest.approx([2 / 3, 1 / 3])


# ---- one evolution step ------------------------------------------------------------------------

def settings(**changes):
    return {"candidates": 4, "parallel": 1, "islands": 2, "migrate_every": 3, "rescore": 8,
            "inspirations": 2, "edits": "diff", **changes}


def test_a_step_keeps_the_better_edit_and_drops_the_rest(small_task, tmp_path):
    source = GeneratorProgram.load(small_task.generator_path).source
    broken = "```python\n" + splice(source, "def generate_problems(rng, n):\n    return [None] * n\n"
                                            "def mutate_problem(p, rng):\n    return p\n") + "```"
    llm = ScriptedLLM([
        GOOD_EDIT,                                                             # better: new best
        "```python\ndef generate_problems(rng, n):\n    return []\n```",      # no markers
        broken,                                                                # fails validation
        "<<<<<<< SEARCH\nnot in the file\n=======\nx\n>>>>>>> REPLACE",        # edit does not apply
    ])
    evolver = GeneratorEvolver(small_task, llm, tmp_path, settings(), log=lambda *_: None)

    best = evolver.step(20, "REPORT TEXT", probe)

    assert "(0.9," in best.source and len(llm.prompts) == 4                    # one call per candidate
    good, markers, invalid, missing = evolver.attempts
    assert good.placed and good.champion and good.kind == "diff" and good.change == "diffusion on more often"
    assert "EVOLVE-BLOCK markers" in markers.reason
    assert "expected a pde1d Problem" in invalid.reason
    assert "not found" in missing.reason
    assert "(0.9," in (tmp_path / "generators" / "v001.py").read_text()
    assert len((tmp_path / "agents.jsonl").read_text().splitlines()) == 4
    saved = json.loads((tmp_path / "generators" / "database.json").read_text())
    assert {e["id"] for e in saved["entries"]} >= {"v000", "r0020-c1"}
    prompt = llm.prompts[0]
    assert "REPORT TEXT" in prompt and "<<<<<<< SEARCH" in prompt and "v000" in prompt


def test_the_next_step_remeasures_elites_and_shows_the_others(small_task, tmp_path):
    def broader(program):   # the edited generator covers more kinds, so both keep a cell
        wider = "(0.9," in program.source
        return ProbeResult(score=1.0 if wider else 0.5, accepted=1.0, coverage=0.9 if wider else 0.5)

    llm = ScriptedLLM([GOOD_EDIT, GOOD_EDIT])
    measured = []
    evolver = GeneratorEvolver(small_task, llm, tmp_path, settings(candidates=1), log=lambda *_: None)
    evolver.step(20, "r", broader)
    assert {e.id for e in evolver.db.elites()} == {"v000", "r0020-c1"}
    evolver.step(40, "r", lambda g: measured.append(g.path.name) or broader(g))
    assert measured[:2] == ["r0020-c1.py", "v000.py"]                         # elites re-measured, best first
    second = llm.prompts[1]
    assert "Other generators, for ideas" in second and ("## v000" in second or "## r0020-c1" in second)


def test_llm_calls_run_in_parallel(small_task, tmp_path):
    lock, active, peak = threading.Lock(), [0], [0]

    class SlowLLM:
        def complete(self, prompt, *, system=None):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.3)
            with lock:
                active[0] -= 1
            return reply(GOOD_EDIT)

    evolver = GeneratorEvolver(small_task, SlowLLM(), tmp_path, settings(candidates=2, parallel=2),
                               log=lambda *_: None)
    evolver.step(20, "r", probe)
    assert peak[0] == 2


# ---- spending fewer LLM calls ------------------------------------------------------------------

def test_the_trigger_waits_for_a_plateau():
    t = Trigger("plateau", every=10, min_improvement=0.05)
    assert t.check(10, [1.0] * 10) == (True, "first check")
    t.evolved(10)
    assert t.check(20, [1.0] * 20)[0] is False                              # needs two windows after it
    improving = [1.0] * 10 + [0.8] * 10 + [0.6] * 10
    assert t.check(30, improving)[0] is False
    flat = [1.0] * 10 + [0.8] * 10 + [0.79] * 10
    go, why = t.check(30, flat)
    assert go and "plateaued" in why
    assert Trigger("every", every=10).check(40, [])[0] is True


def test_stable_prompt_parts_come_first(small_task, tmp_path):
    evolver = GeneratorEvolver(small_task, ScriptedLLM([]), tmp_path, settings(), log=lambda *_: None)
    evolver.seed_entry.record(ProbeResult(0.5, 1.0, 0.5), 1)
    prompt = evolver.prompt(evolver.seed_entry, [], "REPORT")
    order = [prompt.index(h) for h in ("# The physics", "# How to reply", "# Rules",
                                       "# The generator to improve", "# What the student", "# Earlier attempts")]
    assert order == sorted(order)


def test_a_reply_that_changes_nothing_is_not_measured(small_task, tmp_path):
    source = GeneratorProgram.load(small_task.generator_path).source
    measured = []
    llm = ScriptedLLM([f"Change: none\n```python\n{source}```"])
    evolver = GeneratorEvolver(small_task, llm, tmp_path, settings(candidates=1), log=lambda *_: None)
    evolver.step(20, "r", lambda g: measured.append(g.path.name) or probe(g))
    assert measured == ["v000.py"]                                          # only the seed
    assert "same code as v000" in evolver.attempts[0].reason


def test_a_broken_edit_format_is_rejected_with_a_clear_reason():
    broken = "<<<<<<< SEARCH\na = 1\n=======\na = 2\n=======\na = 3\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="malformed"):
        parse_edits(broken)


def test_raw_replies_are_kept(small_task, tmp_path):
    evolver = GeneratorEvolver(small_task, ScriptedLLM([GOOD_EDIT]), tmp_path, settings(candidates=1),
                               log=lambda *_: None)
    evolver.step(20, "r", probe)
    assert (tmp_path / "generators" / "r0020-c1.reply.md").read_text() == GOOD_EDIT


def test_edits_ignore_copied_block_markers():
    block = "\nx = 1\ny = 2\n"
    reply = ("<<<<<<< SEARCH\n# EVOLVE-BLOCK-START\nx = 1\ny = 2\n# EVOLVE-BLOCK-END\n=======\n"
             "# EVOLVE-BLOCK-START\nx = 3\ny = 2\n# EVOLVE-BLOCK-END\n>>>>>>> REPLACE\n")
    assert apply_edits(block, parse_edits(reply)) == "\nx = 3\ny = 2\n"


def test_a_stray_separator_before_replace_is_dropped():
    edits = parse_edits("<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n=======\n>>>>>>> REPLACE\n")
    assert edits == [("x = 1", "x = 2")]
