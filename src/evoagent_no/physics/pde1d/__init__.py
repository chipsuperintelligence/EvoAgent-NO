"""pde1d: 1D PDEs on a periodic interval, solved exactly on the GPU.

The student sees HISTORY frames and predicts the next one (time stepping), without being told
the PDE. See `evoagent_no.physics` for what a backend provides.
"""

from evoagent_no.physics.pde1d.problem import (CELL_NAMES, COEF_LIMITS, FIELD_LIMITS, FRAME_DT, FRAMES, GRID,
                                                HISTORY, IO, KMAX, MAX_T0_FRAMES, TERMS, Problem, cell_name,
                                                check, describe)
from evoagent_no.physics.pde1d.solver import solve

__all__ = ["CELL_NAMES", "COEF_LIMITS", "FIELD_LIMITS", "FRAME_DT", "FRAMES", "GRID", "HISTORY", "IO", "KMAX",
           "MAX_T0_FRAMES", "TERMS", "Problem", "cell_name", "check", "describe", "solve"]
