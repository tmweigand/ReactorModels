import reactormodels
import numpy as np
import pytest


@pytest.mark.parametrize("axial_diffusion", [0.01, 0.1])
def test_ogata_banks(axial_diffusion):
    """
    Collocation solution must match Ogata-Banks analytical solution
    for 1D advection-diffusion with step inlet BC.
    """
    superficial_velocity = 1  # m/s
    column_length = 5.0  # m
    porosity = 0.5
    column_diameter = 1
    t_eval = np.array([1.0, 2.0, 3.0])

    breakthrough = reactormodels.fixtures.make_breakthrough(
        length=column_length,
        diameter=column_diameter,
        porosity=porosity,
        superficial_velocity=superficial_velocity,
        axial_diffusion=axial_diffusion,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        n_interior_points=5, n_elements=20, domain_length=column_length
    )

    model = reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthrough,
        numerics=numerics,
    )

    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=breakthrough, diffusion=axial_diffusion
    )

    for i, t in enumerate(t_eval):
        # Only compare interior of domain, away from outlet BC influence
        mask = x < 0.8 * column_length
        C_analytical = ogata_banks.spatial_profile(
            x[mask],
            t,
        )
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(
            C_analytical, abs=1e-2
        ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_multi_element_ogata_banks():
    """High-Pe case that fails with single element should pass with multi-element."""

    superficial_velocity = 1
    axial_diffusion = 0.01
    column_length = 5.0
    porosity = 0.5
    t_eval = np.array([2.0, 4.0])
    diameter = 1

    breakthrough = reactormodels.fixtures.make_breakthrough(
        length=column_length,
        diameter=diameter,
        porosity=porosity,
        superficial_velocity=superficial_velocity,
        axial_diffusion=axial_diffusion,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column_length,
        n_interior_points=5,
        n_elements=20,
    )

    model = reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthrough,
        numerics=numerics,
    )

    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=breakthrough, diffusion=axial_diffusion
    )

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * column_length
        C_analytical = ogata_banks.spatial_profile(
            x[mask],
            t,
        )
        C_numerical = C[i, mask]

        assert C_numerical == pytest.approx(
            C_analytical, abs=1e-2
        ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"
