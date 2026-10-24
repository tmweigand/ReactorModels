"""test_batch_psdm.py"""

import numpy as np
import pytest
import reactormodels

SECONDS_PER_DAY = 24 * 60 * 60


def _make_batch(Ds=5e-9, C_in=1, time=np.linspace(1e-10, 1000000, 200)):
    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07 * 2  # cm
    pore_diffusion = 5e-6  # cm2/s
    k_film = 0.1  # cm/s
    K = 100  # (mg/g) * (L/mg)
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bed_density = 399.8  # g/L
    sorbent_dosage = 0.001

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
        bed_density=bed_density,
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        pore_diffusion=pore_diffusion,
        surface_diffusion=Ds,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=C_in,
        time=time,
        flow_rate=0,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=30,
        n_elements=1,
        add_inlet=True,
    )

    return reactormodels.models.BatchPSDM(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        k_film=k_film,
        sorbent_dosage=sorbent_dosage,
    )


def test_surface_concentration_increases():
    """Surface concentration should increase monotonically with time."""

    time = np.linspace(1e-10, 3600, 50)
    model = _make_batch(time=time)

    r, C, Cp = model.solve()

    # Cp shape: (time, particle node)
    surface_concentration = Cp[:, -1]

    assert np.all(np.diff(surface_concentration) >= -1e-8)


def test_solve_reaches_equilibrium():
    """Long-time solution should approach adsorption equilibrium."""

    time = np.linspace(1e-10, 1000 * SECONDS_PER_DAY, 50)
    model = _make_batch(time=time)

    Cb = 1.0
    r, C, Cp = model.solve()

    # Bulk concentration approaches the particle concentration.
    np.testing.assert_allclose(C[-1], Cp[-1, 0], rtol=1e-4)

    # Solid phase loading reaches equilibrium with bulk.
    q_final = model.iso.q(Cp[-1, -1])
    q_expected = model.iso.q(C[-1])
    np.testing.assert_allclose(q_final, q_expected, rtol=1e-4)


def test_surface_diffusion_delays_breakthrough():
    """Increasing surface diffusion should delay breakthrough."""

    time = np.linspace(1e-10, 5 * SECONDS_PER_DAY, 50)

    model_no_surface_diffusion = _make_batch(
        Ds=1e-15,
        time=time,
    )
    model_with_surface_diffusion = _make_batch(
        Ds=1e-9,
        time=time,
    )

    _, C_no_diffusion, _ = model_no_surface_diffusion.solve()
    _, C_with_diffusion, _ = model_with_surface_diffusion.solve()

    outlet_no_diffusion = C_no_diffusion[-1]
    outlet_with_diffusion = C_with_diffusion[-1]

    assert outlet_with_diffusion <= outlet_no_diffusion + 1e-8


def test_algebraic_vars():
    """Algebraic variable indices should be valid and unique."""

    model = _make_batch(time=np.linspace(1e-10, 15 * SECONDS_PER_DAY, 200))

    algebraic_vars = model._algebraic_vars_idx()

    assert isinstance(algebraic_vars, list)
    assert all(isinstance(index, int) for index in algebraic_vars)

    expected_count = 2
    assert len(algebraic_vars) == expected_count

    # Surface boundary condition.
    assert algebraic_vars[0] == 1
    # Center boundary condition
    assert algebraic_vars[1] == model.N

    # Indices should be unique and within the full state vector.
    assert len(algebraic_vars) == len(set(algebraic_vars))

    state_size = model.N + 1
    assert all(0 <= index < state_size for index in algebraic_vars)


def test_parameter_check():
    """Missing required parameters should raise ValueError."""

    with pytest.raises(ValueError):
        _make_batch(Ds=None)


def test_get_solid_phase_loading():
    """Solid phase loading should have the same shape as pore concentration."""

    time = np.linspace(1e-10, 3600, 50)
    model = _make_batch(time=time)

    _, _, Cp = model.solve()
    q = model.get_solid_phase_loading(Cp)

    assert q.shape == Cp.shape


def test_mass_balance():
    """Total mass per liquid volume is conserved in the batch PSDM."""
    time = np.linspace(1e-10, 10 * SECONDS_PER_DAY, 50)
    model = _make_batch(time=time)
    _, C, Cp = model.solve()  # C: (time,), Cp: (time, N)

    q = np.array([model.iso.q(r) for r in Cp])  # (time, N), q at each node

    cp_avg = model.get_radial_average(Cp[:, np.newaxis, :])[:, 0]
    q_avg = model.get_radial_average(q[:, np.newaxis, :])[:, 0]

    eps_p = model.media.particle_porosity
    rho_p = model.media.particle_density
    ratio = model.sorbent_dosage / rho_p  # V_p / V_l

    total = C + ratio * (eps_p * cp_avg + rho_p * q_avg)

    # initial total from the initial state, not from the solved output
    cp0 = model.breakthrough.initial_concentration
    total0 = model.bulk_initial_concentration + ratio * (
        eps_p * cp0 + rho_p * model.iso.q(np.array([cp0]))[0]
    )

    np.testing.assert_allclose(total, total0, rtol=1e-3)
