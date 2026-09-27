"""One run on a task folder: pretrain a student on problems the generator poses.

Each round (the fast loop, no LLM in it):
    1. the generator poses problems
    2. the physics backend solves them and rejects the ones it cannot solve reliably
    3. the student trains on the answers
Every `agents.every` rounds, with agents on (the default), LLM agents may rewrite the generator
(the slow loop, `agents.evolve`). A rewrite that a short trial shows the student learns more from
than from the task's own generator joins it: the student then trains on both, in equal shares.
Every `eval_every` rounds the student is tested on held-out physics.

With self-play on (optional), a pool decides which problems to pose within the round: fresh ones
from the generator, mutations of problems the student is learning from fastest, replays; problems
far harder than the generator's typical one are set aside as out of reach.

With both off, the generator stays as written: the baseline. Every variant trains the student for
the same number of steps per round.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evoagent_no import evals
from evoagent_no.agents.database import ProbeResult
from evoagent_no.agents.evolve import GeneratorEvolver, Trigger
from evoagent_no.llm import from_settings
from evoagent_no.problems.generator import GeneratorProgram
from evoagent_no.problems.pool import Elite, Pool
from evoagent_no.score.progress import learning_progress
from evoagent_no.student.learner import Learner


def _cell_name(task, cell) -> str:
    return task.physics.cell_name(cell) if hasattr(task.physics, "cell_name") else str(cell)


def training_report(task, records: list[dict], losses: list[float]) -> str:
    """What the agents see: training signals from recent rounds. Never held-out scores."""
    posed = len(records)
    accepted = [r for r in records if r["accepted"]]
    by_cell = defaultdict(list)
    for r in records:
        by_cell[r["cell"]].append(r)
    rows = []
    for cell, rs in by_cell.items():
        ok = [r for r in rs if r["accepted"]]
        far = sum(not r.get("reach", True) for r in ok)
        rows.append((np.mean([r["score"] for r in ok]) if ok else 0.0, cell, len(rs), len(ok),
                     np.mean([r["loss"] for r in ok]) if ok else float("nan"), far))
    rows.sort(key=lambda row: -row[0])
    line = lambda row: (f"  {_cell_name(task, row[1]):40s} posed {row[2]:5d}  accepted {row[3] / row[2]:5.0%}"
                        f"  out of reach {row[5] / max(row[3], 1):5.0%}"
                        f"  score {row[0]:.4f}  student loss {row[4]:.4f}")
    by_origin = defaultdict(list)
    for r in accepted:
        by_origin[r["origin"]].append(r["score"])
    cells = getattr(task.physics, "CELL_NAMES", None)
    never = []
    if cells:
        seen = set(by_cell)
        never = [_cell_name(task, c) for c in itertools.product(*(range(len(n)) for n in cells)) if c not in seen]
    return "\n".join([
        f"Last {posed} problems: {len(accepted) / max(posed, 1):.0%} accepted by the answer key; "
        f"{sum(not r.get('reach', True) for r in accepted) / max(len(accepted), 1):.0%} of those were "
        "out of reach (far harder than the round's typical problem, so not trained on).",
        f"Student training loss: {losses[0]:.4f} → {losses[-1]:.4f} over these rounds." if losses else "",
        "Mean score by source: " + ", ".join(f"{o} {np.mean(v):.4f}" for o, v in sorted(by_origin.items())),
        "Highest-scoring kinds of problem (score = learning progress, higher is better):",
        *map(line, rows[:8]),
        "Lowest-scoring kinds:",
        *map(line, rows[-5:]),
        "Kinds never posed: " + (", ".join(never) if never else "none"),
    ])


def label(agents: bool, selfplay: bool) -> str:
    """A run's variant in words, also its default folder name: agents, selfplay, selfplay+agents, fixed."""
    return "+".join(["selfplay"] * selfplay + ["agents"] * agents) or "fixed"


def resolve(task, agents: bool | None = None, selfplay: bool | None = None) -> tuple[bool, bool]:
    """Whether agents and self-play are on: the argument if given, else task.yaml."""
    return (task.section("agents")["enabled"] if agents is None else agents,
            task.section("selfplay")["enabled"] if selfplay is None else selfplay)


def run(task, out_dir: Path, seed: int = 0, device: str = "cuda", log=print, suite=None, llm=None,
        agents: bool | None = None, selfplay: bool | None = None, resume: bool = False, **overrides) -> dict:
    """Train one student. `agents` and `selfplay` default to task.yaml. `overrides` replace
    `training` or `selfplay` settings (rounds, problems_per_round, …); `suite` lets several runs
    share one held-out set, which is slow to build. With `resume`, a run folder that has a
    checkpoint continues from it (and `rounds` can extend a finished run)."""
    out_dir = Path(out_dir)
    checkpoint_file = out_dir / "checkpoint.pt"
    state = None
    if resume:
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"no checkpoint in {out_dir} to resume from")
        saved = json.loads((out_dir / "config.json").read_text())
        seed, selfplay = saved["seed"], saved["selfplay"]
        agents = saved["agents"] if agents is None else agents
        overrides = {**saved["settings"], **{k: v for k, v in overrides.items() if v is not None}}
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)   # our own file
    s = {**task.section("training"), **task.section("selfplay"),
         **{k: v for k, v in overrides.items() if v is not None}}
    s.pop("enabled", None)
    a = task.section("agents")
    agents_on, selfplay_on = resolve(task, agents, selfplay)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "task": task.name, "task_root": str(task.root), "variant": label(agents_on, selfplay_on),
        "seed": seed, "agents": agents_on, "selfplay": selfplay_on, "settings": s,
        "student": task.section("student"), "io": task.physics.IO, "agents_settings": a}, indent=2))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    learner = Learner.for_task(task, device)
    learner.plan(s["rounds"] * s["steps_per_round"])   # the learning-rate schedule spans the whole run
    trivial = evals.trivial_prediction(task.physics.IO)
    original = GeneratorProgram.load(task.generator_path)
    pool = Pool(original, task.physics, rng, fixed_prior=not selfplay_on,
                fresh=s["pool"]["fresh"], mutated=s["pool"]["mutated"], fallback=original)
    if suite is None:
        suite = task.heldout.build(count=s["eval_count"], device=device)
    evolver = None
    if agents_on:
        llm = llm or from_settings(a["llm"], seed=seed)
        evolver = GeneratorEvolver(task, llm, out_dir, a, log=log, seed=seed)
        trigger = Trigger(a.get("trigger", "plateau"), a["every"], a.get("min_improvement", 0.05))
    metrics = (out_dir / "metrics.jsonl").open("a" if state else "w")
    records: list[dict] = []
    recent_losses: list[float] = []
    loss_history: list[float] = []   # every round's training loss, for the agents' trigger
    floor = None
    first_round, elapsed, already_evaluated = 0, 0.0, False
    if state:
        learner.load_state_dict(state["learner"] if "learner" in state else   # older checkpoints
                                {k: state[k] for k in ("model", "optimizer", "snapshots")})
        generator.set_state(state["torch_generator"])
        rng.bit_generator.state = state["numpy_rng"]
        pool.archive = {cell: Elite(problem, score) for cell, problem, score in state["pool"]["archive"]}
        pool.malformed = state["pool"]["malformed"]
        pool.generator = GeneratorProgram.load(state["pool"]["generator"])
        if evolver and state["evolver"]:
            evolver.load_state_dict(state["evolver"])
            pool.mixture = evolver.population()
            trigger.last_evolved = state.get("trigger_last_evolved")
        loss_history = state.get("loss_history", [])
        records, recent_losses, floor = state["records"], state["recent_losses"], state["floor"]
        first_round, elapsed = state["next_round"], state["elapsed"]
        already_evaluated = state.get("evaluated", False)   # a finished run, now extended
        log(f"resuming {out_dir} at round {first_round}")

    def save_checkpoint(next_round: int, evaluated: bool = False) -> None:
        torch.save({
            "next_round": next_round, "evaluated": evaluated, "elapsed": time.monotonic() - start,
            "learner": learner.state_dict(), "torch_generator": generator.get_state(),
            "numpy_rng": rng.bit_generator.state,
            "pool": {"archive": [(cell, e.problem, e.score) for cell, e in pool.archive.items()],
                     "malformed": pool.malformed, "generator": str(pool.generator.path)},
            "evolver": evolver.state_dict() if evolver else None,
            "trigger_last_evolved": trigger.last_evolved if evolver else None, "loss_history": loss_history,
            "records": records, "recent_losses": recent_losses, "floor": floor,
        }, checkpoint_file)

    def evaluate(round_index: int) -> dict:
        scores = evals.evaluate(learner.predict, suite, trivial)
        metrics.write(json.dumps({"round": round_index, "eval": scores}) + "\n")
        metrics.flush()
        families = [k for k in scores if "/" not in k and k != "mean"]
        log(f"round {round_index:4d}  held-out nRMSE {scores['mean']:.4f}  "
            + "  ".join(f"{f} {scores[f]:.3f}" for f in families))
        return scores

    cells = getattr(task.physics, "CELL_NAMES", None)
    n_cells = int(np.prod([len(n) for n in cells])) if cells else None

    def probe(program: GeneratorProgram, round_index: int) -> ProbeResult:
        """A generator measured on the current student by a short trial: the student trains on
        half of the generator's problems for one round's worth of steps, and is then put back as
        it was. The score is how much error the trial removed on the other half, per posed
        problem (rejected ones count 0), in units of the round's median training loss. Problems
        out of reach (far harder than the generator's own typical problem) are left out of the
        trial, as they are left out of training. Near zero for problems that are trivial or
        already mastered, negative when the trial made things worse. The same budget and random
        stream for every generator, so scores compare. Also returns acceptance and how many kinds
        of problem the generator covers.

        Measured, not extrapolated: learning progress along the student's past trajectory cannot
        see physics the student has never trained on, which is exactly what a new generator is
        for."""
        k = a["probe_problems"]
        probe_pool = Pool(program, task.physics, np.random.default_rng([seed, round_index]),
                          fixed_prior=True, fallback=program)
        problems = probe_pool._checked(program.generate(probe_pool.rng, k))
        if not problems:
            return ProbeResult(0.0, 0.0, 0.0)
        kinds = len({task.physics.describe(p) for p in problems})
        coverage = min(kinds / (n_cells or len(problems)), 1.0)
        samples = task.physics.solve(problems, device)
        accepted = float(samples.accepted.float().sum().item() / k)
        if s.get("too_hard"):
            with torch.no_grad():
                loss = learner.per_problem_loss(samples.inputs, samples.targets, samples.problem, len(problems))
            present = torch.zeros_like(loss, dtype=torch.bool)
            present[samples.problem] = True
            in_reach = (loss <= s["too_hard"] * loss[present].median())[samples.problem]
        else:
            in_reach = torch.ones_like(samples.problem, dtype=torch.bool)
        is_test = samples.problem % 2 == 1
        train, test = in_reach & ~is_test, in_reach & is_test
        if not train.any() or not test.any():
            return ProbeResult(0.0, accepted, coverage)
        before, after = learner.trial(
            (samples.inputs[train], samples.targets[train]),
            (samples.inputs[test], samples.targets[test], samples.problem[test]),
            len(problems), s["steps_per_round"], s["batch"],
            torch.Generator(device=device).manual_seed(seed * 100003 + round_index))
        removed = (before - after).sum() / max(len(problems) // 2, 1)   # untested problems add 0
        unit = floor or (recent_losses[-1] if recent_losses else 1.0)   # a typical loss, for readable scores
        return ProbeResult(float(removed.item()) / unit, accepted, coverage)

    start = time.monotonic() - elapsed
    for e in range(first_round, s["rounds"]):
        if e % s["eval_every"] == 0 and not (already_evaluated and e == first_round):
            evaluate(e)
        problems, origins = pool.propose(s["problems_per_round"])
        samples = task.physics.solve(problems, device)
        accepted = samples.accepted
        kept = accepted.nonzero().squeeze(1)
        learner.snapshot(e)

        scores = torch.zeros(len(problems), device=device)
        losses = torch.zeros(len(problems), device=device)
        if selfplay_on and len(kept):
            scores, losses = learning_progress(learner.model, learner.optimizer, learner.lookback(e),
                                               samples.inputs, samples.targets, samples.problem, len(problems),
                                               normalize=s["score"])
            floor = losses[kept].median().item()
            if s.get("too_hard"):
                # Out of reach for now: a loss more than `too_hard` times the median of this round's
                # fresh problems (what the generator currently offers; the pool's own problems may
                # be long mastered). A few such problems would dominate the gradient and pull the
                # student away from what it can learn; they are not trained on and the pool does
                # not keep them.
                fresh = accepted.clone()
                fresh[[i for i, o in enumerate(origins) if o != "fresh"]] = False
                typical = losses[fresh].median().item() if fresh.any() else floor
                reach = losses <= s["too_hard"] * typical
                in_reach = reach[samples.problem]
                samples = type(samples)(samples.inputs[in_reach], samples.targets[in_reach],
                                        samples.problem[in_reach], accepted & reach)

        if evolver and e > 0 and e % a["every"] == 0:
            go, why = trigger.check(e, loss_history)
            if go:
                report = training_report(task, records, recent_losses)
                log(f"round {e:4d}  agents: evolving the generator ({why})")
                pool.generator = evolver.step(e, report, lambda g: probe(g, e))   # mutations use the best
                pool.mixture = evolver.population()
                trigger.evolved(e)
                records, recent_losses = [], []
            else:
                log(f"round {e:4d}  agents: no LLM calls, {why}")
                metrics.write(json.dumps({"round": e, "agents": "skipped", "why": why}) + "\n")

        loss = learner.train(samples.inputs, samples.targets, s["steps_per_round"], s["batch"], generator) \
            if len(samples.problem) else float("nan")

        accepted_np, scores_np, losses_np = accepted.cpu().numpy(), scores.float().cpu().numpy(), losses.float().cpu().numpy()
        reach_np = samples.accepted.cpu().numpy()   # accepted and within reach
        pool.update(problems, scores_np, reach_np)
        recent_losses.append(loss)
        loss_history.append(loss)
        if not evolver:   # nobody reads older records; keep checkpoints small
            records = records[-a["every"] * s["problems_per_round"]:]
        for p, o, ok, near, sc, ls in zip(problems, origins, accepted_np, reach_np, scores_np, losses_np):
            records.append({"cell": task.physics.describe(p), "origin": o, "accepted": bool(ok),
                            "reach": bool(near), "score": float(sc), "loss": float(ls)})
        by_origin = defaultdict(list)
        for o, ok, sc, ls in zip(origins, accepted_np, scores_np, losses_np):
            if ok:
                by_origin[o].append((sc, ls))
        metrics.write(json.dumps({
            "round": e, "train_loss": loss, "accepted": float(accepted_np.mean()),
            "out_of_reach": float((accepted_np & ~reach_np).mean()),
            "malformed_total": pool.malformed, "archive_cells": len(pool.archive),
            "generator": pool.generator.path.name,
            "mixture": [g.path.name for g in pool.mixture] or [pool.generator.path.name],
            "score_by_origin": {o: float(np.mean([v[0] for v in vs])) for o, vs in by_origin.items()},
            "loss_by_origin": {o: float(np.mean([v[1] for v in vs])) for o, vs in by_origin.items()},
            "seconds": time.monotonic() - start,
        }) + "\n")
        if (e + 1) % s.get("checkpoint_every", 10) == 0:
            save_checkpoint(e + 1)

    final = evaluate(s["rounds"])
    save_checkpoint(s["rounds"], evaluated=True)
    torch.save(learner.student_state(), out_dir / "student.pt")
    metrics.close()
    log(f"done in {time.monotonic() - start:.0f} s → {out_dir}")
    return final
