"""The question-maker: the task's generator program and the pool that searches with it.

`generator` loads a task's generator.py and handles its EVOLVE-BLOCK; `pool` decides each round
which problems to pose (fresh, mutated, replayed) and keeps a MAP-Elites archive of good ones.
"""
