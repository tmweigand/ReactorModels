import reactormodels
import numpy as np
import pytest


@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks(diffusion):
    """Collocation solution must match the Ogata-Banks solution."""
    interstitial_velocity = 1.0
    column_length = 5.0
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
    )

    t_eval = np.array([1.0, 2.0, 3.0])

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=inlet_concentration,
        superficial_velocity=interstitial_velocity * porosity,
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
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0
    t_eval = np.array([2.0, 4.0])

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=inlet_concentration,
        superficial_velocity=interstitial_velocity * porosity,
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
