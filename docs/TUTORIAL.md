# Your own problem in EvoAgent-NO

This walks through setting up a new problem domain, from an empty folder to a run with LLM agents
rewriting the problem generator. The worked example is
[`examples/transport1d`](../examples/transport1d), a complete task with its own physics;
[`examples/heat3d`](../examples/heat3d) is the same pattern in 3D.

## What you provide

A task is a program that poses problems, the physics that answers them, and a few settings. LLM
agents evolve the program; what decides which versions survive is not the program's output
itself, but how much a neural operator learns from the problems it generates.

| In an EvoAgent-NO task | What it is |
|---|---|
| `generator.py`: `generate_problems()` and `mutate_problem()` in an EVOLVE-BLOCK | The code the LLM rewrites |
| Built in: a short trial of the student on the generator's problems | What decides which versions survive |
| `physics.py`: `solve()` | The exact computation behind the score |
| `task.yaml` | Models, run length, search settings |
| `context.md` | What the LLM is told about the domain |
| `heldout.py` | Tests the student never trains on and the LLM never sees |

## 1. Start a folder

```bash
uv run evoagent-no new                          # lists templates
uv run evoagent-no new pde1d tasks/my-task      # built-in physics, ready to run
cp -r examples/transport1d tasks/my-task         # or start from the own-physics example
```

## 2. The answer key: `physics.py`

A physics backend is a module with five names. From the example:

```python
IO = {"dims": 1, "in_channels": 3, "out_channels": 1, "norm_groups": [[0], [1], [2]],
      "residual": 0, "scale": (0, "std"), "periodic": True}

@dataclass(frozen=True)
class Problem: ...                        # one problem, plain data

def check(problem): ...                   # raise ValueError if malformed
def describe(problem) -> tuple: ...       # which kind of problem: a MAP-Elites cell
def solve(problems, device) -> Samples: ...
```

`solve` returns `Samples(inputs, targets, problem, accepted)`: inputs `(M, in_channels, *space)`
and targets `(M, out_channels, *space)` on a regular grid in 1D, 2D or 3D, the index of the problem each sample came
from, and one accepted flag per problem. Reject every problem you cannot answer reliably: solve
it twice at different resolutions and drop it if the answers disagree. A wrong answer is worse
than no answer, because the student trains on it.

`IO` tells the student how to read the samples, so no model code is needed:

| Key | Meaning | transport1d | elliptic2d |
|---|---|---|---|
| `dims` | 1, 2 or 3 spatial dimensions | 1 | 2 |
| `in_channels`, `out_channels` | fields in and out | 3, 1 | 2, 1 |
| `norm_groups` | input channels normalized together | each alone | each alone |
| `residual` | the output is a change from this input channel | 0 (u0) | none |
| `scale` | the output is in units of this channel's std or RMS | u0's std | f's RMS |
| `periodic` | space wraps around | yes | no |

`describe` sorts problems into a few kinds (2–40 cells). The pool keeps the best problem of
each kind, so the curriculum cannot collapse onto one. It lives in `physics.py`, not in the
generator, so the LLM cannot redefine what counts as coverage.

## 3. Held-out tests: `heldout.py`

`build(count, device) -> {family: (inputs, targets)}`, in the same layout as `Samples`. Two rules:

- **References come from a different method than your solver:** exact solutions, manufactured
  solutions, or an independent code. Otherwise the test measures agreement with your solver's
  errors.
- **Families the generator does not target.** They measure transfer, and they are never shown to
  the LLM.

## 4. The question-maker: `generator.py`

```python
from task_physics import Problem      # this task's physics, whichever backend task.yaml names

# EVOLVE-BLOCK-START
def generate_problems(rng, n): ...    # n fresh problems; use only rng for randomness
def mutate_problem(problem, rng): ... # one nearby problem
# EVOLVE-BLOCK-END
```

Write a reasonable first version by hand: it is also the baseline (`--no-agents`) and the
fallback when a rewrite misbehaves. It does not need to be good; the agents improve it. The agents change only the code between the markers.

## 5. What the LLM is told: `context.md`

The equation, what each `Problem` field means, how problems are scored, and the hard limits from
`check`. Do not describe the held-out families.

## 6. Run it

```bash
uv run evoagent-no check tasks/my-task              # seconds: validates the generator, solves 16 problems
uv run evoagent-no quick tasks/my-task              # agents vs baseline, small
uv run evoagent-no run tasks/my-task                # full run; agents rewrite the generator when needed
uv run evoagent-no run tasks/my-task --no-agents    # baseline: the generator as written, same compute
uv run evoagent-no run tasks/my-task --selfplay     # optional: also pick each round's problems by self-play
uv run evoagent-no compare runs/my-task/*/
```

A run folder holds `config.json`, `metrics.jsonl` (per round: loss, acceptance, scores by source;
every `eval_every` rounds: held-out scores), `checkpoint.pt` (every `checkpoint_every` rounds),
`student.pt`, and with agents on, `agents.jsonl` and `generators/` (every rewrite, each new best as
`vNNN.py`, and the generator database). `evoagent-no report <run>` summarizes it, and
`evoagent-no run <task> --out <run> --resume` continues it.

## 7. Use the pretrained student

The student is the product. Load it from the run folder, no task folder needed:

```python
import evoagent_no as evo
student = evo.load_student("runs/my-task/agents-s0")
student.io                                   # the layout it expects
y = student.predict(x)                       # numpy (samples, in_channels, *space) → (samples, out_channels, *space)
student.finetune(my_x, my_y, steps=500)      # adapt it to your own data
student.save("runs/my-task/finetuned")
```

From the command line: `evoagent-no finetune <run> --data mine.npz` (arrays `inputs` and
`targets`; it reports the error before and after on a fifth of the samples it holds back) and
`evoagent-no predict <run> --inputs x.npy --out y.npy`.

## The LLM

| Backend | Setup | task.yaml |
|---|---|---|
| `claude-cli` (default) | `claude auth login` once; calls count against your Claude plan | `model: claude-sonnet-5`, `effort: medium` (any model the CLI accepts; `effort: null` for models without effort levels) |
| `openai` | `uv sync --extra openai`, `OPENAI_API_KEY` | `backend: openai`, `model: <name>`; `base_url` for compatible endpoints |

`uv run evoagent-no llm check` confirms the backend answers. To mix models, make `llm` a list:

```yaml
agents:
  llm:
    - {backend: claude-cli, model: claude-sonnet-5, effort: medium, weight: 0.7}
    - {backend: openai, model: <name>, weight: 0.3}
```

The LLM only edits the generator's code. Each call is a single completion with no tools, so
there are no permission prompts and nothing to approve; runs are unattended.

## How the agents search

Every `agents.every` rounds the run checks whether to evolve. With `trigger: plateau` (the
default) it evolves only when the student has stopped improving: its training loss improved by less
than `min_improvement` over the last `every` rounds, against the `every` rounds before, both after
the last evolution step (the first check always evolves). Otherwise it logs why it skipped and makes
no LLM calls. An evolution step:

1. Scores in the generator database fade (`decay`), and the `rescore` strongest generators are
   measured again on the current student.
2. `candidates` agents each get a parent from an island (mostly a strong one) and `inspirations`
   other generators, plus the report of what the student is learning from and earlier attempts.
3. The LLM calls run `parallel` at a time. Replies are SEARCH/REPLACE edits to the parent's
   EVOLVE-BLOCK (`edits: diff`) or a whole file.
4. Each result runs in a separate process first, then is measured on the student with a short
   trial (train on half of its problems for one round, see how much error that removed on the
   other half, put the student back) and placed in its island's MAP-Elites grid, keyed by how many kinds of problem it covers and how many the
   answer key accepts.
5. Every `migrate_every` steps each island's best generator joins the next island. The best
   generator overall becomes the one the fast loop samples from.

The run folder keeps every attempt (`agents.jsonl`, `generators/r<round>-c<n>.py`), each new best
generator (`generators/vNNN.py`) and the database (`generators/database.json`).

## Things to know

- **Cost and load.** The fast loop keeps one GPU busy for the whole run. With agents on, each
  evolution step makes `agents.candidates` LLM calls, each capped by `budget_usd`: with the
  defaults, at most 18 calls in a 200-round run, fewer when the plateau trigger skips steps. To
  spend less: `effort: low`, fewer `candidates`, a larger `every`, or a weighted list that sends
  most calls to a cheaper setting. Prompts put their fixed parts first for prompt caching, and a
  reply that reproduces an existing generator is dropped before any GPU time is spent on it.
  Measuring generators takes GPU time: `rescore` + `candidates` probes of `probe_problems`
  problems per step.
- **Rewritten code runs on your machine.** Every rewrite is first run in a separate process with
  a time limit and must produce well-formed problems; then it is loaded into the run. That
  catches broken and hanging code, but it is not a security sandbox. Use models you trust.
- **What a good generator looks like is decided by the student.** A rewrite is kept only if a
  short trial on its problems removes more of the current student's error than a trial on the
  current generator's. `evoagent-no report <run>` lists every attempt and its trial score.
- **Self-play is optional.** With `--selfplay` (or `selfplay.enabled: true`) a pool picks each
  round's problems: fresh ones, mutations of problems the student is learning from fastest, and
  replays, and problems with a loss more than `selfplay.too_hard` (4) times the median of the
  round's fresh problems are set aside. It helps most when the generator is broad and rough and
  agents are off; with a narrow generator and no agents it can hurt.
