"""The `evoagent-no` command.

    evoagent-no new pde1d my-task          start a task folder from a template
    evoagent-no check my-task              is the task wired up? (seconds, tiny)
    evoagent-no run my-task                LLM agents evolve the generator while the student trains
    evoagent-no run my-task --no-agents    the baseline: the generator as written
    evoagent-no run my-task --selfplay     also pick each round's problems by self-play
    evoagent-no quick my-task              agents vs baseline, small: a few minutes on one GPU
    evoagent-no compare runs/my-task/*     held-out scores side by side
    evoagent-no report runs/my-task/agents-s0        what happened in one run
    evoagent-no run my-task --resume       continue a stopped run (or extend one with --rounds)
    evoagent-no predict RUN --inputs x.npy --out y.npy
    evoagent-no finetune RUN --data mine.npz        adapt the pretrained student to your data
    evoagent-no llm check                  one round trip to the agents' LLM
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from evoagent_no.llm import BACKENDS, DEFAULT_BACKEND, LLMError, make_llm
from evoagent_no.llm.claude_cli import DEFAULT_EFFORT, DEFAULT_MODEL, EFFORTS


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---- tasks -------------------------------------------------------------------------------------

def _new(args: argparse.Namespace) -> int:
    from evoagent_no.task import TEMPLATES, load_task, templates

    if args.template is None:
        print("templates:")
        for name in templates():
            description = load_task(TEMPLATES / name).settings["description"].strip()
            print(f"  {name:12s} {description}")
        print("usage: evoagent-no new <template> <folder>   (or copy examples/transport1d for your own physics)")
        return 0
    if args.folder is None:
        print("error: give a folder to create: evoagent-no new <template> <folder>", file=sys.stderr)
        return 2
    if args.template not in templates():
        print(f"error: no template {args.template!r}; available: {', '.join(templates())}", file=sys.stderr)
        return 2
    target = Path(args.folder)
    if target.exists():
        print(f"error: {target} already exists", file=sys.stderr)
        return 2
    shutil.copytree(TEMPLATES / args.template, target, ignore=shutil.ignore_patterns("__pycache__"))
    config = target / "task.yaml"   # the task is named after its folder, and so are its runs
    config.write_text(re.sub(r"(?m)^name: .*$", f"name: {target.name}", config.read_text(), count=1))
    print(f"created {target}/ from the {args.template} template:")
    for f in sorted(target.iterdir()):
        print(f"  {f.name}")
    print(f"next: evoagent-no check {target}")
    return 0


def _check(args: argparse.Namespace) -> int:
    import numpy as np

    from evoagent_no.problems.generator import GeneratorProgram, validate
    from evoagent_no.student.learner import Learner
    from evoagent_no.task import load_task

    device = args.device or _default_device()
    task = load_task(args.task)
    print(f"task      {task.name}: {task.settings['description'].strip()}")
    print(f"physics   {task.settings['physics']}\nheld-out  {task.settings['heldout']}")
    validate(task.generator_path, task.settings["physics"], task.root)
    generator = GeneratorProgram.load(task.generator_path)
    print(f"generator {task.generator_path.name}: runs in a separate process and makes well-formed problems")
    problems = generator.generate(np.random.default_rng(0), 16)
    start = time.monotonic()
    samples = task.physics.solve(problems, device)
    cells = {task.physics.describe(p) for p in problems}
    print(f"answer    16 problems solved in {time.monotonic() - start:.1f} s on {device}; "
          f"{int(samples.accepted.sum())} accepted; {len(cells)} kinds of problem")
    print(f"samples   {len(samples.problem)}: inputs {tuple(samples.inputs.shape[1:])} → "
          f"targets {tuple(samples.targets.shape[1:])}")
    learner = Learner.for_task(task, device)
    out = learner.predict(samples.inputs[:2])
    size = sum(p.numel() for p in learner.model.parameters())
    print(f"student   {task.physics.IO['dims']}D FNO with {size / 1e6:.2f} M parameters; "
          f"prediction {tuple(out.shape[1:])}")
    agents = task.section("agents")
    print(f"agents    {'on' if agents['enabled'] else 'off'} ({agents['llm']['backend']}, "
          f"{agents['llm']['model']}, effort {agents['llm']['effort']})")
    print(f"ready. next: evoagent-no quick {args.task}   (agents vs baseline, a few minutes on one GPU)")
    return 0


def _run(args: argparse.Namespace) -> int:
    from evoagent_no.selfplay import label, resolve, run
    from evoagent_no.task import load_task

    task = load_task(args.task)
    out = Path(args.out or f"runs/{task.name}/{label(*resolve(task, args.agents, args.selfplay))}-s{args.seed}")
    if (out / "checkpoint.pt").is_file() and not args.resume:
        print(f"error: {out} already has a run; add --resume to continue it, or choose another --out",
              file=sys.stderr)
        return 2
    run(task, out, seed=args.seed, device=args.device or _default_device(),
        agents=args.agents, selfplay=args.selfplay, resume=args.resume, rounds=args.rounds, problems_per_round=args.problems_per_round)
    return 0


def _report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    config = json.loads((run_dir / "config.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    evals = [r for r in rows if "eval" in r]
    rounds = [r for r in rows if "train_loss" in r]
    s = config["settings"]
    print(f"run       {run_dir}\ntask      {config['task']} ({config['task_root']})")
    print(f"variant   {config['variant']}, seed {config['seed']}, "
          f"{len(rounds)}/{s['rounds']} rounds × {s['problems_per_round']} problems")
    if rounds:
        losses = [r["train_loss"] for r in rounds]
        print(f"training  loss {losses[0]:.4f} → {losses[-1]:.4f}; "
              f"{sum(r['accepted'] for r in rounds) / len(rounds):.0%} of problems accepted; "
              f"{rounds[-1]['malformed_total']} malformed; {rounds[-1]['seconds']:.0f} s")
    if evals:
        families = [k for k in evals[-1]["eval"] if "/" not in k and k != "mean"]
        print("held-out  nRMSE by round (lower is better)")
        print(f"  {'round':>5}  {'mean':>8}  " + "  ".join(f"{f:>12}" for f in families))
        for r in evals:
            print(f"  {r['round']:>5}  {r['eval']['mean']:8.4f}  " + "  ".join(f"{r['eval'][f]:12.4f}" for f in families))
        if "mean/trivial" in evals[-1]["eval"]:
            print(f"  trivial {evals[-1]['eval']['mean/trivial']:.4f} (copy input / predict zero)")
    agents_file = run_dir / "agents.jsonl"
    if agents_file.is_file():
        attempts = [json.loads(line) for line in agents_file.open()]
        kept = [a for a in attempts if a["placed"]]
        best = [a for a in attempts if a["champion"]]
        cost = sum(a["cost_usd"] or 0 for a in attempts)
        skipped = [r for r in rows if r.get("agents") == "skipped"]
        tokens_in = sum(a.get("input_tokens") or 0 for a in attempts)
        cached = sum(a.get("cached_tokens") or 0 for a in attempts)
        tokens_out = sum(a.get("output_tokens") or 0 for a in attempts)
        print(f"agents    {len(attempts)} LLM calls, {len(kept)} kept, {len(best)} became the best generator; "
              f"{sum(a['seconds'] for a in attempts):.0f} s of calls"
              + (f", ${cost:.2f} at API prices" if cost else ""))
        print(f"          {len(skipped)} checks skipped (student still improving); tokens {tokens_in} in "
              f"({cached} from cache), {tokens_out} out")
        for a in attempts:
            mark = "BEST" if a["champion"] else "kept" if a["placed"] else "drop"
            score = "-" if a["score"] is None else f"{a['score']:.3f}"
            print(f"  {mark}  {a['file']:16s} score {score:>8}  {a['change'][:90] or a['reason'][:90]}")
    if (run_dir / "checkpoint.pt").is_file():
        print(f"resume    evoagent-no run {config['task_root']} --out {run_dir} --resume [--rounds N]")
    if (run_dir / "student.pt").is_file():
        print(f"student   {run_dir / 'student.pt'}: evoagent-no predict / finetune {run_dir}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    import numpy as np

    from evoagent_no.student.pretrained import load_student

    student = load_student(args.run, args.device)
    out = student.predict(np.load(args.inputs))
    np.save(args.out, out)
    print(f"{args.out}: {out.shape[0]} predictions of shape {out.shape[1:]}")
    return 0


def _finetune(args: argparse.Namespace) -> int:
    import numpy as np

    from evoagent_no.student.pretrained import load_student, relative_error

    data = np.load(args.data)
    if "inputs" not in data or "targets" not in data:
        print(f"error: {args.data} needs arrays named `inputs` and `targets`", file=sys.stderr)
        return 2
    x, y = data["inputs"], data["targets"]
    student = load_student(args.run, args.device)
    held = max(1, len(x) // 5) if len(x) >= 5 else 0       # a fifth held back to show the effect
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x))
    test, train = order[:held], order[held:]
    if held:
        print(f"before    relative error {relative_error(student.predict(x[test]), y[test]):.4f} on {held} held-back samples")
    history = student.finetune(x[train], y[train], steps=args.steps, lr=args.lr, batch=args.batch)
    print(f"training  loss {history[0]:.4f} → {history[-1]:.4f} over {args.steps} steps on {len(train)} samples")
    if held:
        print(f"after     relative error {relative_error(student.predict(x[test]), y[test]):.4f}")
    out = student.save(args.out or f"{str(args.run).rstrip('/')}-finetuned")
    print(f"saved     {out}")
    return 0


def _quick(args: argparse.Namespace) -> int:
    from evoagent_no.selfplay import label, run
    from evoagent_no.task import load_task

    start = time.monotonic()
    task = load_task(args.task)
    device = args.device or _default_device()
    suite = task.heldout.build(count=args.eval_count, device=device)
    print(f"held-out suite: {len(suite)} families × {args.eval_count} ({time.monotonic() - start:.0f} s)")
    out = Path(args.out or f"runs/{task.name}-quick")
    small = dict(rounds=args.rounds, problems_per_round=args.problems_per_round,
                 eval_every=max(args.rounds // 4, 1), eval_count=args.eval_count)
    variants = [(args.agents, args.selfplay), (False, False)]   # what you asked for, and the baseline
    for agents, selfplay in variants:
        name = label(agents, selfplay)
        print(f"\n{name}")
        run(task, out / f"{name}-s{args.seed}", seed=args.seed, device=device, suite=suite,
            agents=agents, selfplay=selfplay, **small)
    print()
    _compare(argparse.Namespace(runs=[str(out / f"{label(*v)}-s{args.seed}") for v in variants]))
    print(f"\ntotal {time.monotonic() - start:.0f} s")
    return 0


def _compare(args: argparse.Namespace) -> int:
    rows, by_variant, columns = [], defaultdict(list), None
    for path in map(Path, args.runs):
        if not (path / "metrics.jsonl").is_file():
            continue
        config = json.loads((path / "config.json").read_text())
        evals = [r for r in map(json.loads, (path / "metrics.jsonl").open()) if "eval" in r]
        if not evals:
            continue
        last = evals[-1]
        columns = columns or ["mean"] + [k for k in last["eval"] if "/" not in k and k != "mean"]
        by_variant[config["variant"]].append(last["eval"])
        rows.append((path.name, last["round"], last["eval"]))
    if not rows:
        print("no finished runs found")
        return 1
    fmt = lambda name, rounds, ev: f"{name:30s} {rounds:>6} " + " ".join(f"{ev[c]:10.4f}" for c in columns)
    print(f"{'run':30s} {'rounds':>6s} " + " ".join(f"{c:>10s}" for c in columns))
    for name, rounds, ev in rows:
        print(fmt(name, rounds, ev))
    print()
    for variant, scores in sorted(by_variant.items()):
        mean = {c: sum(s[c] for s in scores) / len(scores) for c in columns}
        print(fmt(f"mean {variant} ({len(scores)} runs)", "", mean))
    if all(c + "/trivial" in rows[0][2] for c in columns):
        floor = {c: rows[0][2][c + "/trivial"] for c in columns}
        print(fmt("trivial (copy input / zero)", "", floor))
    return 0


# ---- llm ---------------------------------------------------------------------------------------

def _add_llm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=sorted(BACKENDS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT, choices=(*EFFORTS, "none"),
                        help="reasoning effort; none leaves it to the model")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds")
    parser.add_argument("--budget", type=float, default=1.0, help="USD cap per call")


def _llm(args: argparse.Namespace):
    return make_llm(args.backend, model=args.model, effort=None if args.effort == "none" else args.effort,
                    timeout_s=args.timeout, max_budget_usd=args.budget)


def _llm_check(args: argparse.Namespace) -> int:
    reply = _llm(args).complete("Reply with exactly the word: pong")
    ok = reply.text.strip().strip(".").lower() == "pong"
    cost = "n/a" if reply.cost_usd is None else f"${reply.cost_usd:.4f}"
    print(f"backend  {args.backend}\nmodel    {reply.model} (effort {args.effort})")
    print(f"tokens   {reply.input_tokens} in ({reply.cached_tokens} from cache), {reply.output_tokens} out\n"
          f"cost     {cost}")
    print(f"time     {reply.seconds:.1f} s\nreply    {reply.text.strip()!r}")
    print("OK" if ok else "Unexpected reply")
    return 0 if ok else 1


def _llm_ask(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt != "-" else sys.stdin.read()
    print(_llm(args).complete(prompt, system=args.system).text)
    return 0


# ---- parser ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evoagent-no",
                                     description="LLM agents + neural operators + self-play.")
    commands = parser.add_subparsers(dest="command", required=True)

    new = commands.add_parser("new", help="start a task folder from a template")
    new.add_argument("template", nargs="?", help="pde1d or elliptic2d; omit to list templates")
    new.add_argument("folder", nargs="?")
    new.set_defaults(func=_new)

    check = commands.add_parser("check", help="check that a task is wired up (seconds, tiny)")
    check.add_argument("task")
    check.add_argument("--device", default=None)
    check.set_defaults(func=_check)

    run = commands.add_parser("run", help="pretrain a student on a task")
    run.add_argument("task")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--rounds", type=int, default=None, help="default: task.yaml")
    run.add_argument("--problems-per-round", dest="problems_per_round", type=int, default=None)
    run.add_argument("--agents", action=argparse.BooleanOptionalAction, default=None,
                     help="let LLM agents rewrite the generator (default: task.yaml, on); "
                          "--no-agents is the baseline")
    run.add_argument("--selfplay", action=argparse.BooleanOptionalAction, default=None,
                     help="pick each round's problems by self-play (default: task.yaml, off)")
    run.add_argument("--out", default=None, help="default: runs/<task>/<variant>-s<seed>")
    run.add_argument("--resume", action="store_true", help="continue from the run folder's checkpoint")
    run.add_argument("--device", default=None)
    run.set_defaults(func=_run)

    report = commands.add_parser("report", help="what happened in one run")
    report.add_argument("run")
    report.set_defaults(func=_report)

    predict = commands.add_parser("predict", help="predict with a pretrained student")
    predict.add_argument("run", help="a run folder (config.json + student.pt)")
    predict.add_argument("--inputs", required=True, help=".npy array (samples, channels, *space)")
    predict.add_argument("--out", required=True, help=".npy file to write")
    predict.add_argument("--device", default=None)
    predict.set_defaults(func=_predict)

    finetune = commands.add_parser("finetune", help="adapt a pretrained student to your own data")
    finetune.add_argument("run", help="a run folder (config.json + student.pt)")
    finetune.add_argument("--data", required=True, help=".npz with arrays `inputs` and `targets`")
    finetune.add_argument("--steps", type=int, default=500)
    finetune.add_argument("--lr", type=float, default=3e-4)
    finetune.add_argument("--batch", type=int, default=32)
    finetune.add_argument("--out", default=None, help="default: <run>-finetuned")
    finetune.add_argument("--device", default=None)
    finetune.set_defaults(func=_finetune)

    quick = commands.add_parser("quick", help="agents vs baseline, small and fast")
    quick.add_argument("task")
    quick.add_argument("--seed", type=int, default=0)
    quick.add_argument("--rounds", type=int, default=40)
    quick.add_argument("--problems-per-round", dest="problems_per_round", type=int, default=128)
    quick.add_argument("--eval-count", dest="eval_count", type=int, default=32)
    quick.add_argument("--agents", action=argparse.BooleanOptionalAction, default=True)
    quick.add_argument("--selfplay", action=argparse.BooleanOptionalAction, default=False)
    quick.add_argument("--out", default=None)
    quick.add_argument("--device", default=None)
    quick.set_defaults(func=_quick)

    compare = commands.add_parser("compare", help="compare finished runs")
    compare.add_argument("runs", nargs="+")
    compare.set_defaults(func=_compare)

    llm = commands.add_parser("llm", help="the LLM backend the agents use").add_subparsers(
        dest="llm_command", required=True)
    llm_check = llm.add_parser("check", help="one round trip to confirm the backend works")
    _add_llm_options(llm_check)
    llm_check.set_defaults(func=_llm_check)
    ask = llm.add_parser("ask", help="send one prompt and print the reply")
    ask.add_argument("prompt", help='the prompt, or "-" to read it from stdin')
    ask.add_argument("--system", default=None, help="system prompt")
    _add_llm_options(ask)
    ask.set_defaults(func=_llm_ask)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (LLMError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
