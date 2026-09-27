#!/usr/bin/env bash
# M0: the pde1d template three ways (the generator as written, agents, self-play), 3 seeds, 400
# rounds each. One run at a time; a few hours of full GPU use and up to 38 LLM calls per agent run.
# Results in runs/pde1d/; then:
#   uv run evoagent-no compare runs/pde1d/*/
set -u
cd "$(dirname "$0")/.."
[ -d tasks/pde1d ] || uv run evoagent-no new pde1d tasks/pde1d
for seed in 0 1 2; do
  for variant in "--no-agents" "--agents" "--no-agents --selfplay"; do
    name=$(echo "$variant" | tr -d ' -')
    uv run evoagent-no run tasks/pde1d $variant --seed "$seed" --rounds 400 \
      > "runs/pde1d-$name-s$seed.log" 2>&1
  done
done
echo "M0 experiment finished"
