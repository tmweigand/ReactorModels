import numpy as np
import pytest

import reactormodels


@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks(diffusion):
    """Collocation solution must match the Ogata-Banks solution."""
    interstitial_velocity = 1.0
    column_length = 5.0
    column_diameter = 0.2
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0

    media = reactormodels.Media(
        particle_porosity=0.3,
        particle_density=1.2,
        mean_diameter=0.001,
        sphericity=0.9,
    )

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
        diameter=column_diameter,
        media=media,
    )

    t_eval = np.array([1.0, 2.0, 3.0])

    superficial_velocity = interstitial_velocity * porosity
    flow_rate = superficial_velocity * column.cross_section_area()

    breakthrough = reactormodels.Breakthrough(
        column=column,
        compound="Test compound",
        feed_concentrations=inlet_concentration,
        effluent_concentrations=np.zeros_like(t_eval),
        flow_rate=flow_rate,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=30,
    )

    model = reactormodels.models.AdvectionDiffusion(
        column=column,
        breakthrough=breakthrough,
        diffusion=diffusion,
        initial_concentration=initial_concentration,
        numerics=numerics,
    )


def test_multi_element_ogata_banks():
    """High-Pe case that fails with one element should pass with multiple elements."""
    interstitial_velocity = 1.0
    diffusion = 0.01
    column_length = 5.0
    column_diameter = 0.2
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0
    t_eval = np.array([2.0, 4.0])

    media = reactormodels.Media(
        particle_porosity=0.3,
        particle_density=1.2,
        mean_diameter=0.001,
        sphericity=0.9,
    )

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
        diameter=column_diameter,
        media=media,
    )

    superficial_velocity = interstitial_velocity * porosity
    flow_rate = superficial_velocity * column.cross_section_area()

    breakthrough = reactormodels.Breakthrough(
        column=column,
        compound="Test compound",
        feed_concentrations=inlet_concentration,
        effluent_concentrations=np.zeros_like(t_eval),
        flow_rate=flow_rate,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=20,
    )

    model = reactormodels.models.AdvectionDiffusion(
        column=column,
        breakthrough=breakthrough,
        diffusion=diffusion,
        initial_concentration=initial_concentration,
        numerics=numerics,
    )
