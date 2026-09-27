"""The slow loop: LLM agents that evolve the task's generator.

`evolve.GeneratorEvolver` runs one evolution step every `agents.every` rounds: re-measure the
strongest generators on the current student, prompt `candidates` agents with a parent and
inspirations from the `database.GeneratorDatabase` (MAP-Elites grids on islands), apply their
SEARCH/REPLACE edits (`diff`), validate and measure the results, and hand the best generator to
the fast loop. The fast loop never calls an LLM.
"""
