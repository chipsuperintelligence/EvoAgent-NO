"""Generator programs (EVOLVE-BLOCK, validation in a separate process) and the pool that uses them."""

import numpy as np
import pytest

from evoagent_no.physics import pde1d
from evoagent_no.problems.generator import GeneratorProgram, extract_code, splice, split, validate
from evoagent_no.problems.pool import Pool

BLOCK_FILE = "head\n# EVOLVE-BLOCK-START\nold = 1\n# EVOLVE-BLOCK-END\ntail\n"


def test_splice_replaces_only_the_block():
    before, block, after = split(BLOCK_FILE)
    assert block.strip() == "old = 1"
    assert splice(BLOCK_FILE, "new = 2") == "head\n# EVOLVE-BLOCK-START\nnew = 2\n# EVOLVE-BLOCK-END\ntail\n"
    with pytest.raises(ValueError, match="exactly one"):
        split("no markers here")


def test_extract_code_takes_the_last_python_block():
    reply = "I changed X.\n```python\nfirst\n```\nthen\n```python\nsecond\n```\n"
    assert extract_code(reply).strip() == "second"
    with pytest.raises(ValueError, match="no ```python"):
        extract_code("just prose")


def test_template_generator_validates(task_dir):
    validate(task_dir / "generator.py", "evoagent_no.physics.pde1d", task_dir)


def test_broken_and_hanging_generators_are_rejected(task_dir):
    source = (task_dir / "generator.py").read_text()
    broken = task_dir / "broken.py"
    broken.write_text(splice(source, "def generate_problems(rng, n):\n    return [1] * n\n"
                                     "def mutate_problem(p, rng):\n    return p\n"))
    with pytest.raises(ValueError, match="expected a pde1d Problem"):
        validate(broken, "evoagent_no.physics.pde1d", task_dir)
    hanging = task_dir / "hanging.py"
    hanging.write_text(splice(source, "import time\ndef generate_problems(rng, n):\n    time.sleep(60)\n"
                                      "def mutate_problem(p, rng):\n    return p\n"))
    with pytest.raises(ValueError, match="did not finish"):
        validate(hanging, "evoagent_no.physics.pde1d", task_dir, timeout_s=3)


def test_pool_mixes_sources_and_keeps_the_best_per_cell(task_dir):
    generator = GeneratorProgram.load(task_dir / "generator.py")
    pool = Pool(generator, pde1d, np.random.default_rng(0))
    problems, origins = pool.propose(20)
    assert set(origins) == {"fresh"} and len(problems) == 20
    scores = np.linspace(0.1, 2.0, 20)
    pool.update(problems, scores, np.ones(20, dtype=bool))
    best = {}
    for p, s in zip(problems, scores):
        cell = pde1d.describe(p)
        best[cell] = max(best.get(cell, 0.0), s / scores.mean())
    assert {cell: e.score for cell, e in pool.archive.items()} == pytest.approx(best)

    problems, origins = pool.propose(20)
    assert (origins.count("fresh"), origins.count("mutated"), origins.count("replay")) == (8, 8, 4)


def test_fixed_prior_never_adapts(task_dir):
    generator = GeneratorProgram.load(task_dir / "generator.py")
    pool = Pool(generator, pde1d, np.random.default_rng(0), fixed_prior=True)
    problems, _ = pool.propose(10)
    pool.update(problems, np.ones(10), np.ones(10, dtype=bool))
    assert pool.archive == {} and set(pool.propose(10)[1]) == {"fresh"}


def test_a_bad_generator_falls_back_to_the_original(task_dir):
    original = GeneratorProgram.load(task_dir / "generator.py")

    class Bad:
        def generate(self, rng, n):
            raise RuntimeError("boom")

        def mutate(self, problem, rng):
            return "not a problem"

    pool = Pool(Bad(), pde1d, np.random.default_rng(0), fallback=original)
    problems, _ = pool.propose(10)
    assert len(problems) == 10 and pool.malformed == 10
    for p in problems:
        pde1d.check(p)


def test_mutation_chains_stay_inside_the_limits(task_dir):
    generator = GeneratorProgram.load(task_dir / "generator.py")
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = generator.generate(rng, 1)[0]
        for _ in range(60):
            p = generator.mutate(p, rng)
            pde1d.check(p)


def test_fresh_problems_come_from_every_generator_in_the_mixture(task_dir):
    generator = GeneratorProgram.load(task_dir / "generator.py")

    class OnlyHeat:
        def generate(self, rng, n):
            return [pde1d.Problem(coef=(0.1,) + (0.0,) * 8, ic_amp=1.0, ic_slope=2.0, ic_mean=0.0,
                                  ic_squash=False, t0_frames=0, seed=i) for i in range(n)]

    pool = Pool(generator, pde1d, np.random.default_rng(0), fixed_prior=True)
    pool.mixture = [generator, OnlyHeat()]
    problems, _ = pool.propose(9)
    heat = [p for p in problems if p.coef == (0.1,) + (0.0,) * 8]
    assert len(problems) == 9 and len(heat) == 4                      # 5 + 4: equal shares
