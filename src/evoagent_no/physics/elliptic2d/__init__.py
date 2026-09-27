"""elliptic2d: steady diffusion −∇·(a∇u) = f on the unit square, u = 0 on the boundary.

An input → output operator in 2D: the student sees (log a, f) and predicts u. See
`evoagent_no.physics` for what a backend provides.
"""

from evoagent_no.physics.elliptic2d.problem import (CELL_NAMES, GRID, IO, KMAX, LIMITS, MAX_CONTRAST, Problem,
                                                     cell_name, check, describe)
from evoagent_no.physics.elliptic2d.solver import solve

__all__ = ["CELL_NAMES", "GRID", "IO", "KMAX", "LIMITS", "MAX_CONTRAST", "Problem", "cell_name", "check",
           "describe", "solve"]
