import numpy as np
import pytest

import reactormodels

SECONDS_PER_DAY = 24 * 60 * 60


def _make_particle(
    Ds: float = 5e-9,
    C_in: float = 1,
    time: np.ndarray | None = None,
    state_var: str = "C",
) -> reactormodels.models.PSDMSolid | reactormodels.models.PSDM:
    """Create a PSDM model for testing."""

    # Particle properties
    particle_porosity = 0.5
    particle_density = 600  # g/mL
    particle_diameter = 0.07  # cm
    pore_diffusion = 5e-6  # cm²/s
    k_film = 0.1  # cm/s

    # Column properties
    axial_diffusion = 0
    K = 100  # (mg/g) * (L/mg)
    initial_concentration = 0
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/mL
    flow_rate = 40  # cm³/s

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
    )

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        diameter=diameter,
        bulk_density=bulk_density,
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=Ds,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=C_in,
        initial_concentration=initial_concentration,
        flow_rate=flow_rate,
        time=time,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )

    if state_var == "C":
        state = reactormodels.models.PSDM
    else:
        state = reactormodels.models.PSDMSolid

    return state(
        isotherm=isotherm,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=k_film,
    )


def test_surface_concentration_increases():
    """Surface concentration should increase monotonically with time."""

    time = np.linspace(1e-10, 3600, 50)
    model = _make_particle(time=time)

    _, _, _, Cp = model.solve()

    # Cp shape: (time, column position, particle radius)
    surface_concentration = Cp[:, 0, -1]

    assert np.all(np.diff(surface_concentration) >= -1e-8)


def test_solve_reaches_equilibrium():
    """Long-time solution should approach adsorption equilibrium."""

    time = np.linspace(1e-10, 125 * 1440 * 60, 50)
    model = _make_particle(time=time)

    Cb = 1.0
    _, r, C, Cp = model.solve()

    # Bulk concentration should approach the feed concentration.
    np.testing.assert_allclose(C[-1, :-1], Cb, rtol=1e-2)

    # Calculate radial average of the final sorbed concentration.
    Cp_final = Cp[-1, 1:, :]
    q_final = model.isotherm.q(Cp_final)

    q_avg = np.trapz(q_final * r**2, r, axis=1) / np.trapz(r**2, r)

    q_target = model.isotherm.q(np.array([Cb]))[0]
    relative_error = abs(q_avg[0] - q_target) / q_target

    assert relative_error < 0.05, f"q_avg={q_avg[0]:.4f}, q_target={q_target:.4f}"


def test_surface_diffusion_delays_breakthrough():
    """Increasing surface diffusion should delay breakthrough."""

    time = np.linspace(1e-10, 25 * SECONDS_PER_DAY, 50)

    model_no_surface_diffusion = _make_particle(
        Ds=1e-15,
        time=time,
    )
    model_with_surface_diffusion = _make_particle(
        Ds=1e-9,
        time=time,
    )

    _, _, C_no_diffusion, _ = model_no_surface_diffusion.solve()
    _, _, C_with_diffusion, _ = model_with_surface_diffusion.solve()

    outlet_no_diffusion = C_no_diffusion[-1, -1]
    outlet_with_diffusion = C_with_diffusion[-1, -1]

    assert outlet_with_diffusion <= outlet_no_diffusion + 1e-8


def test_algebraic_vars():
    """Algebraic variable indices should be valid and unique."""

    model = _make_particle(time=np.linspace(1e-10, 175 * SECONDS_PER_DAY, 200))

    algebraic_vars = model._algebraic_vars_idx()

    assert isinstance(algebraic_vars, list)
    assert all(isinstance(index, int) for index in algebraic_vars)

    expected_count = 1 + 2 * (model.N_column - 1)
    assert len(algebraic_vars) == expected_count

    # Inlet boundary condition.
    assert algebraic_vars[0] == 0

    # Indices should be unique and within the full state vector.
    assert len(algebraic_vars) == len(set(algebraic_vars))

    state_size = model.N_column + (model.N_column - 1) * model.N_particle
    assert all(0 <= index < state_size for index in algebraic_vars)

    # First particle: center and edge.
    first_particle = model.N_column
    assert first_particle in algebraic_vars
    assert first_particle + model.N_particle - 1 in algebraic_vars

    # Last particle: center and edge.
    last_particle = model.N_column + ((model.N_column - 2) * model.N_particle)
    assert last_particle in algebraic_vars
    assert last_particle + model.N_particle - 1 in algebraic_vars


def test_parameter_check():
    """Missing required parameters should raise ValueError."""

    with pytest.raises(ValueError):
        _make_particle(Ds=None)


def test_get_sorbed_mass_fraction():
    """Sorbed mass fraction should have the same shape as pore concentration."""

    time = np.linspace(1e-10, 3600, 50)
    model = _make_particle(time=time)

    _, _, _, Cp = model.solve()
    q = model.get_sorbed_mass_fraction(Cp)

    assert q.shape == Cp.shape


def test_mass_balance():
    """Mass should be conserved throughout the PSDM simulation."""

    time = np.linspace(1e-10, 100 * SECONDS_PER_DAY, 50)
    model = _make_particle(time=time)

    _, _, C, Cp = model.solve()

    mass_balance = reactormodels.postprocess.MassBalance(
        model=model,
        liquid_concentration=C,
        sorbent_mass_fraction=model.get_sorbed_mass_fraction(Cp),
        pore_concentration=Cp,
    )

    assert mass_balance.is_balanced(rel_tol=1e-3).all(), mass_balance.summary()


def test_different_states_are_equal():
    """q version and C version should be equal."""
    time = np.linspace(1e-10, 100 * SECONDS_PER_DAY, 50)
    model = _make_particle(time=time)
    q_model = _make_particle(time=time, state_var="q")

    _, _, C, Cp = model.solve()
    _, _, C, q = q_model.solve()
    Cp_q = q_model.get_pore_concentration(q)

    np.testing.assert_allclose(Cp, Cp_q, atol=1e-5)
