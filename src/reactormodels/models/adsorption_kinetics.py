"""adsorption_kinetics.py"""

from enum import Enum, auto


class AdsorptionKinetics(Enum):
    """Set the form of the adsorption kinetics."""

    LOCAL_EQUILIBRIUM = auto()
    LINEAR_DRIVING_FORCE = auto()
    SECOND_ORDER = auto()
