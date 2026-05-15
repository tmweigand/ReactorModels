"""analytic_models.py"""

import numpy as np
from scipy.special import erfc, erfcx


def ogata_banks(
    x: np.ndarray,
    time: float,
    velocity: float,
    diffusion: float,
    inlet_concentration: float = 1.0,
):
    """Ogata Banks solution for 1D advection-diffusion with step input."""
    Pe_local = velocity * x / diffusion
    arg1 = (x - velocity * time) / (2 * np.sqrt(diffusion * time))
    arg2 = (x + velocity * time) / (2 * np.sqrt(diffusion * time))
    exponent = Pe_local - arg2**2
    term2 = np.where(
        exponent > 500,
        0.0,
        erfcx(arg2) * np.exp(exponent),
    )
    return inlet_concentration * 0.5 * (erfc(arg1) + term2)
