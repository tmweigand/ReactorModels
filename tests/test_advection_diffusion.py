import reactormodels
import numpy as np
import pytest


@pytest.mark.parametrize("diffusion", [0.01, 0.1])
def test_ogata_banks(diffusion):
    """
    Collocation solution must match Ogata-Banks analytical solution
    for 1D advection-diffusion with step inlet BC.
    """
    velocity = 1.0  # m/s
    column_length = 5.0  # m
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0

    column = reactormodels.Column(length=column_length, porosity=porosity)

    t_eval = np.array([1.0, 2.0, 3.0])

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

    for i, t in enumerate(t_eval):
        # Only compare interior of domain, away from outlet BC influence
        mask = x < 0.8 * column_length
        C_analytical = reactormodels.models.OgataBanks(
            x[mask], t, velocity, diffusion, inlet_concentration
        ).concentration_profile()
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(
            C_analytical, abs=1e-2
        ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_multi_element_ogata_banks():
    """High-Pe case that fails with single element should pass with multi-element."""

    velocity = 1.0
    diffusion = 0.01
    column_length = 5.0
    porosity = 0.5
    inlet_concentration = 1.0
    initial_concentration = 0.0
    t_eval = np.array([2.0, 4.0])

    column = reactormodels.Column(length=column_length, porosity=porosity)

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

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * column_length
        assert C[i, mask] == pytest.approx(
            reactormodels.models.OgataBanks(
                x[mask], t, velocity, diffusion, inlet_concentration
            ).concentration_profile(),
            abs=1e-2,
        )
