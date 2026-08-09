"""fixtures.py

Helpers for building standard test problems (demos, benchmarks, tests).
"""

from __future__ import annotations

import numpy as np

from .properties import Breakthrough
from .properties import Column
from .properties import Media
from .properties import Water
from .properties import Chemical

__all__ = ["make_breakthrough"]


def make_breakthrough(
    *,
    length: float = 5.0,
    diameter: float = 0.1,
    porosity: float = 0.5,
    superficial_velocity: float = 1.0,
    diffusion: float = 0.01,
    inlet_concentration: float = 1.0,
    initial_concentration: float = 0.0,
    t_end: float = 1.0,
) -> Breakthrough:
    """Build a simple 1D advection-diffusion breakthrough problem."""
    column = Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        media=Media(particle_density=1.0),
        water=Water(name="clean-water"),
    )
    chemical = Chemical(name="tracer", diffusion=diffusion)

    return Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=inlet_concentration,
        initial_concentration=initial_concentration,
        superficial_velocity=superficial_velocity,
        time=np.array([t_end]),
    )
