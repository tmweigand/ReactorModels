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
    length: float = 1.0,
    diameter: float = 0.1,
    porosity: float = 0.5,
    bulk_density: float = 1.0,
    superficial_velocity: float = 1.0,
    axial_diffusion: float = 0.01,
    inlet_concentration: np.ndarray = np.array([1.0]),
    initial_concentration: float = 0.0,
    time: np.ndarray = np.array([1.0]),
    water: Water | None = None,
) -> Breakthrough:
    """Build a simple 1D advection-diffusion breakthrough problem."""
    if water is not None:
        _water = water
    else:
        _water = Water(name="water")

    column = Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        bulk_density=bulk_density,
        media=Media(particle_density=1.0, particle_radius=1.0),
        water=_water,
    )
    chemical = Chemical(name="tracer", axial_diffusion=axial_diffusion)

    return Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=inlet_concentration,
        initial_concentration=initial_concentration,
        superficial_velocity=superficial_velocity,
        time=time,
    )
