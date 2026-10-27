import reactormodels
import numpy as np
import pytest


def _base_model(axial_diffusion):
    superficial_velocity = 1  # m/s
    column_length = 5.0  # m
    porosity = 0.5
    column_diameter = 1
    t_eval = np.array([1.0, 2.0, 3.0])

    column = reactormodels.Column(
        length=column_length,
        diameter=column_diameter,
        porosity=porosity,
        media=reactormodels.Media(),
        water=reactormodels.Water(),
    )

    breakthroughs = []
    for i in range(len(axial_diffusion)):
        breakthrough = reactormodels.Breakthrough(
            feed_concentrations=1,
            column=column,
            superficial_velocity=superficial_velocity,
            chemical=reactormodels.Chemical(axial_diffusion=axial_diffusion[i]),
            time=t_eval,
        )
        breakthroughs.append(breakthrough)

    numerics = reactormodels.numerics.NumericsConfig(
        n_interior_points=5, n_elements=20, domain_length=column_length
    )

    return reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthroughs,
        numerics=numerics,
    )


@pytest.mark.parametrize(
    "axial_diffusion", [np.linspace(0.01, 0.05, 5), np.linspace(0.1, 0.5, 5)]
)
def test_ogata_banks(axial_diffusion):
    """
    Collocation solution must match Ogata-Banks analytical solution
    for 1D advection-diffusion with step inlet BC.
    """
    model = _base_model(axial_diffusion)
    x, C = model.solve()

    for j in range(len(axial_diffusion)):
        ogata_banks = reactormodels.models.OgataBanks(
            breakthrough=model.breakthroughs[0], diffusion=axial_diffusion[j]
        )
        for i, t in enumerate(model.t_eval):
            # Only compare interior of domain, away from outlet BC influence
            mask = x < 0.8 * model.column.length
            C_analytical = ogata_banks.spatial_profile(
                x[mask],
                t,
            )
            C_numerical = C[i, j, mask]

            assert C_numerical == pytest.approx(
                C_analytical, abs=1e-2
            ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_multi_element_ogata_banks():
    """High-Pe case that fails with single element should pass with multi-element."""

    model = _base_model(axial_diffusion=np.linspace(0.01, 0.05, 5))
    x, C = model.solve()

    for j in range(len(model.axial_diffusion)):
        ogata_banks = reactormodels.models.OgataBanks(
            breakthrough=model.breakthroughs[0], diffusion=model.axial_diffusion[j]
        )
        for i, t in enumerate(model.t_eval):
            mask = x < 0.8 * model.column.length
            C_analytical = ogata_banks.spatial_profile(
                x[mask],
                t,
            )
            C_numerical = C[i, j, mask]

            assert C_numerical == pytest.approx(
                C_analytical, abs=1e-2
            ), f"Failed at t={t}: max error = {np.abs(C_numerical - C_analytical).max():.2e}"


def test_parameter_check():
    """Set diffusion to None and ensure error."""
    superficial_velocity = 1  # m/sn
    column_length = 5.0  # m
    porosity = 0.5
    column_diameter = 1
    t_eval = np.array([1.0, 2.0, 3.0])

    breakthrough = reactormodels.fixtures.make_breakthrough(
        length=column_length,
        diameter=column_diameter,
        porosity=porosity,
        superficial_velocity=superficial_velocity,
        axial_diffusion=None,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        n_interior_points=5, n_elements=20, domain_length=column_length
    )

    with pytest.raises(ValueError):
        reactormodels.models.AdvectionDiffusion(
            breakthrough=breakthrough,
            numerics=numerics,
        )
