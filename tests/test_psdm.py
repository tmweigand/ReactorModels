import reactormodels
import numpy as np
import pytest


def _make_particle(Ds=5e-9):
    # particle
    particle_porosity = 0.5
    particle_density = 0.6  # g/mL
    particle_diameter = 0.07  # cm
    pore_diffusion = 5e-6  # cm2/s
    k_film = 0.1  # cm/s

    # column
    axial_diffusion = 0
    K = 10000  # (mg/g) * (L/mg)
    initial_concentration = 0
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 0.3998  # g/mL
    feed_concentrations = 1  # mg/L
    flow_rate = 4  # cm3/s
    t_eval = np.linspace(1e-10, 175 * 1440 * 60, 200)  # s

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        particle_porosity=particle_porosity,
        bulk_density=bulk_density,
        particle_density=particle_density,
        diameter=diameter,
        particle_diameter=particle_diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=t_eval,
    )
    column_numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
        resolution=reactormodels.models.DomainResolution.COLUMN,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
        resolution=reactormodels.models.DomainResolution.PARTICLE,
    )
    return reactormodels.models.DomainCoupling(
        isotherm=isotherm,
        column=column,
        breakthrough=breakthrough,
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=Ds,
        initial_concentration=initial_concentration,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_film=k_film,
    )


def test_initial_state_zero():
    p = _make_particle()
    y0, ydot0 = p._initial_conditions(
        C_in=p.inlet_concentration,
        C_init=p.initial_concentration,
        Cp_init=p.initial_concentration,
    )

    C, Cp = p._split(y0)

    np.testing.assert_allclose(ydot0, 0)
    np.testing.assert_allclose(C[1:], 0)
    np.testing.assert_allclose(Cp[:, :], 0)
    assert C[0] == 1
    assert y0.shape == (p._n_vars(),)
    assert len(y0) == p._n_vars()


def test_initial_state_nonzero():
    p = _make_particle()
    y0, ydot0 = p._initial_conditions(C_in=5, C_init=1, Cp_init=2)

    C, Cp = p._split(y0)

    np.testing.assert_allclose(ydot0, 0)
    np.testing.assert_allclose(C[1:], 1)
    np.testing.assert_allclose(Cp[:, :], 2)
    assert C[0] == 5
    assert y0.shape == (p._n_vars(),)
    assert len(y0) == p._n_vars()


def test_surface_concentration_increases():
    p = _make_particle()
    t_eval = np.linspace(1e-10, 3600, 50)
    z, r, C, Cp = p.solve(
        t_span=(0, t_eval[-1]),
        t_eval=t_eval,
    )

    Cp_surface = Cp[:, :, -1]  # if shape is (time,z,r)

    inlet_surface = Cp_surface[:, 0]

    assert np.all(np.diff(inlet_surface) >= -1e-8)


def test_solve_reaches_equilibrium():
    """
    After a long time, average loading should approach q*(Cb).
    """
    p = _make_particle()
    Cb = 1.0
    t_eval = np.linspace(1e-10, 175 * 1440 * 60, 50)  # very long time

    z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    print("bulk inlet final:", C[-1, 0])
    print("particle inlet final:", Cp[-1, -1, :])

    np.testing.assert_allclose(C[-1, -1], Cb, rtol=1e-2)

    Cp_final = Cp[-1, 1:, :]  # final time: (Nz, Nr)

    q_final = p.iso.q(Cp_final)

    q_avg = np.trapz(
        q_final * r**2,
        r,
        axis=1,
    ) / np.trapz(r**2, r)

    q_target = p.iso.q(np.array([Cb]))[0]

    rel_err = abs(q_avg[0] - q_target) / q_target

    assert rel_err < 0.05, f"q_avg={q_avg[0]:.4f}, q_target={q_target:.4f}"


def test_surface_diffusion_accelerates_uptake():
    """
    Surface diffusion should increase adsorption, delaying breakthrough.
    """
    t_eval = np.linspace(1e-10, 20 * 1440 * 60, 50)

    p0 = _make_particle(Ds=1e-10)

    ps = _make_particle(Ds=1e-9)

    _, _, C0, _ = p0.solve(
        t_span=(0, t_eval[-1]),
        t_eval=t_eval,
    )

    _, _, Cs, _ = ps.solve(
        t_span=(0, t_eval[-1]),
        t_eval=t_eval,
    )

    # Outlet concentration
    Cout0 = C0[:, -1]
    Couts = Cs[:, -1]

    # Surface diffusion should delay breakthrough
    assert np.all(Couts <= Cout0 + 1e-8)


def test_average_loading_nonnegative():
    p = _make_particle()
    t_eval = np.linspace(1e-10, 20 * 1440 * 60, 50)

    z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    for k in range(C.shape[1]):
        assert np.all(C[:, k]) >= -1e-10
    for j in range(Cp.shape[1]):
        assert np.all(Cp[:, j]) >= -1e-10


@pytest.mark.skip
def test_mass_balance():
    """Mass entering particle equals mass leaving bulk."""
    p = _make_particle()
    t_eval = np.linspace(1e-10, 20 * 1440 * 60, 50)

    z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    A = p.column.cross_section_area()

    Mf = p.column.porosity * A * np.trapz(C[-1], z)

    q = p.iso.q(Cp[-1])

    q_avg = np.trapz(
        q * r**2,
        r,
        axis=1,
    ) / np.trapz(r**2, r)

    Ms = (1 - p.column.porosity) * A * p.column.bulk_density * np.trapz(q_avg, z)

    Cp_avg = np.trapz(
        Cp[-1] * r**2,
        r,
        axis=1,
    ) / np.trapz(r**2, r)

    Mp = (1 - p.column.porosity) * p.column.particle_porosity * A * np.trapz(Cp_avg, z)

    Mtotal = Mf + Mp + Ms

    Cin = p.inlet_concentration

    Min = p.breakthrough.flow_rate * Cin * t_eval[-1]
    Mout = p.breakthrough.flow_rate * np.trapz(C[:, -1], t_eval)
    balance = Min - Mout

    print("Fluid      :", Mf)
    print("Particle fluid:", Mp)
    print("Adsorbed   :", Ms)
    print("Stored     :", Mtotal)
    print("In-Out     :", balance)

    np.testing.assert_allclose(
        Mtotal,
        balance,
        rtol=1e-2,
    )
