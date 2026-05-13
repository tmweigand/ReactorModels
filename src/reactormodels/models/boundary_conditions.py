"""boundary_conditions.py"""

from enum import Enum, auto


class InletBC(Enum):
    """Inlet boundary condition specification."""

    DIRICHLET = auto()  # C(0, t) = C_in  (simple concentration pin)
    DANCKWERTS = auto()  # v*C_in = v*C(0) - DL*(dC/dx)|_0  (flux-conserving)
