import reactormodels
import numpy as np
import pytest


def _build_solved_model():
    bulk_density = 1.0
    K = 0.5
    porosity = 0.3
    velocity = 1.0
    column_length = 5.0
    inlet_concentration = 1.0
    initial_concentration = 0.0
    diameter = 1
    axial_diffusion = 0.1

    R = 1.0 + (bulk_density * K) / porosity
    v_eff = velocity / (porosity * R)
    t_eval = np.linspace(0.1 * column_length / v_eff, 1.5 * column_length / v_eff, 8)

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
        media=reactormodels.Media(),
        water=reactormodels.Water(),
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical(axial_diffusion=axial_diffusion),
        feed_concentrations=inlet_concentration,
        initial_concentration=initial_concentration,
        superficial_velocity=velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_elements=5, n_interior_points=10, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthrough,
        isotherm=reactormodels.models.LinearIsotherm(K=K),
        numerics=numerics,
        kinetics=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
    )

    x, C, q = model.solve()
    return model, C, q


@pytest.fixture(scope="module")
def mass_balance():
    model, C, q = _build_solved_model()
    return reactormodels.postprocess.MassBalance(
        model=model,
        liquid_concentration=C,
        sorbent_mass_fraction=q,
    )


def test_mass_balance_holds_at_every_time(mass_balance):
    """Core physical check: in - out - stored ~= 0, at every time point."""
    assert mass_balance.is_balanced(rel_tol=0.05).all(), mass_balance.summary()


def test_arrays_are_shaped_like_time(mass_balance):
    """Every derived quantity should be one value per time step, no more no less."""
    n_t = mass_balance.time.shape[0]
    for name in (
        "mass_in",
        "mass_out",
        "mass_fluid",
        "mass_adsorbed",
        "mass_stored",
        "error",
        "relative_error",
    ):
        arr = getattr(mass_balance, name)
        assert arr.shape == (n_t,), f"{name} has shape {arr.shape}, expected ({n_t},)"


def test_mass_in_is_monotonically_increasing(mass_balance):
    """Constant feed concentration -> cumulative mass in only ever grows."""
    assert np.all(np.diff(mass_balance.mass_in) >= 0)


def test_mass_out_never_exceeds_mass_in(mass_balance):
    """Can't have exited more mass than has entered the column."""
    assert np.all(mass_balance.mass_out <= mass_balance.mass_in + 1e-8)


def test_stored_mass_is_nonnegative(mass_balance):
    """Fluid-phase and solid-phase mass are physical quantities, can't be negative."""
    assert np.all(mass_balance.mass_fluid >= -1e-8)
    assert np.all(mass_balance.mass_adsorbed >= -1e-8)
