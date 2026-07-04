import reactormodels
import numpy as np
import pytest


@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks(diffusion):
    """
    Collocation solution must match Ogata-Banks analytical solution
    for 1D advection-diffusion with step inlet BC.
    """
    velocity = 1  # m/s
    column_length = 5.0  # m
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0
    diameter = 1
    t_eval = np.array([1.0, 2.0, 3.0])

    column = reactormodels.Column(
        length=column_length, porosity=porosity, diameter=diameter
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=inlet_concentration,
        superficial_velocity=velocity * porosity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=30
    )

    model = reactormodels.models.AdvectionDiffusion(
        column=column,
        velocity=velocity,
        diffusion=diffusion,
        inlet_concentration=inlet_concentration,
        initial_concentration=initial_concentration,
        numerics=numerics,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)
    OgataBanks = reactormodels.models.OgataBanks(breakthrough, diffusion)
    for i, t in enumerate(t_eval):
        # Only compare interior of domain, away from outlet BC influence
        mask = x < 0.8 * column_length
        C_analytical = OgataBanks.spatial_profile(
            x[mask],
            t,
        )
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(
            C_analytical, abs=1e-2
        ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_multi_element_ogata_banks():
    """High-Pe case that fails with single element should pass with multi-element."""

    velocity = 1
    diffusion = 0.01
    column_length = 5.0
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0
    t_eval = np.array([2.0, 4.0])
    diameter = 1

    column = reactormodels.Column(
        length=column_length, porosity=porosity, diameter=diameter
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=inlet_concentration,
        superficial_velocity=velocity * porosity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=20,
    )
    model = reactormodels.models.AdvectionDiffusion(
        column=column,
        velocity=velocity,
        diffusion=diffusion,
        inlet_concentration=inlet_concentration,
        initial_concentration=initial_concentration,
        numerics=numerics,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    OgataBanks = reactormodels.models.OgataBanks(breakthrough, diffusion)
    for i, t in enumerate(t_eval):
        mask = x < 0.8 * column_length
        assert C[i, mask] == pytest.approx(
            OgataBanks.spatial_profile(x[mask], t),
            abs=1e-2,
        )
