import reactormodels
import numpy as np
import pytest


def _make_particle(Ds=5e-9, C_in=1):
    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/mL
    particle_diameter = 0.07  # cm
    pore_diffusion = 5e-6  # cm2/s
    k_film = 0.1  # cm/s

    # column
    axial_diffusion = 0
    K = 100  # (mg/g) * (L/mg)
    initial_concentration = 0
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/mL
    flow_rate = 40  # cm3/s
    t_eval = np.linspace(1e-10, 175 * 1440 * 60, 200)  # s

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

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical(),
        feed_concentrations=C_in,
        flow_rate=flow_rate,
        time=t_eval,
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
    return reactormodels.models.DomainCoupling(
        isotherm=isotherm,
        breakthrough=breakthrough,
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=Ds,
        initial_concentration=initial_concentration,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
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
    p = _make_particle(C_in=5)
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
    t_eval = np.linspace(1e-10, 125 * 1440 * 60, 50)  # very long time

    z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    print("bulk final:", C[-1, :-1])
    # print("particle inlet final:", Cp[-1, -1, :])

    np.testing.assert_allclose(C[-1, :-1], Cb, rtol=1e-2)

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
    t_eval = np.linspace(1e-10, 25 * 1440 * 60, 50)

    p0 = _make_particle(Ds=1e-15)

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
    Cout0 = C0[-1, -1]
    Couts = Cs[-1, -1]

    # Surface diffusion should delay breakthrough
    assert np.all(Couts <= Cout0 + 1e-8)


def test_mass_balance():
    """Mass entering particle equals mass leaving bulk."""
    p = _make_particle()
    t_eval = np.linspace(1e-10, 100 * 1440 * 60, 50)

    z, r, C, Cp = p.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    A = p.column.cross_section_area()

    # print("Fluid      :", Mf)
    # print("Particle fluid:", Mp)
    # print("Adsorbed   :", Ms)
    # print("Stored     :", Mtotal)
    # print("In-Out     :", balance)

    errors = []

    for k, t in enumerate(t_eval):

        # Bulk fluid inventory
        Mf = p.column.porosity * A * np.trapz(C[k, :] / 1000, z)

        # Adsorbed phase
        q = p.iso.q(Cp[k, :, :])

        q_avg = np.trapz(q * r**2, r, axis=1) / np.trapz(r**2, r)

        Ms = A * p.column.bulk_density * np.trapz(q_avg, z) / 1000

        # Pore fluid
        Cp_avg = np.trapz(Cp[k, :, :] * r**2, r, axis=1) / np.trapz(r**2, r)

        Mp = (
            (1 - p.column.porosity)
            * p.column.media.particle_porosity
            * A
            * np.trapz(Cp_avg, z)
            / 1000
        )

        Mstored = Mf + Mp + Ms

        Min = p.breakthrough.flow_rate * p.inlet_concentration * t / 1000

        Mout = p.breakthrough.flow_rate * np.trapz(
            C[: k + 1, -1] / 1000, t_eval[: k + 1]
        )

        balance = Min - Mout

        err = (Mstored - balance) / balance if balance > 0 else 0.0
        errors.append(err)

        print(f"Day {t/86400:6.1f}: error = {100*err:7.3f}%")

    np.testing.assert_allclose(
        Mstored,
        balance,
        rtol=1e-2,
    )


def test_algebraic_vars():
    p = _make_particle()
    alg_vars = p._algebraic_vars_idx()

    # Basic type checks
    assert isinstance(alg_vars, list)
    assert all(isinstance(v, int) for v in alg_vars)

    # Expected count: 1 inlet + 2 per column node (center + edge)
    expected_len = 1 + 2 * (p.N_column - 1)
    assert len(alg_vars) == expected_len

    # Inlet BC is always index 0
    assert alg_vars[0] == 0

    # No duplicate indices
    assert len(alg_vars) == len(set(alg_vars))

    # All indices within bounds of the full y array
    y_len = p.N_column + (p.N_column - 1) * p.N_particle
    assert all(0 <= v < y_len for v in alg_vars)

    # Spot-check the particle center/edge indices for column node 0
    assert p.N_column in alg_vars  # first particle center
    assert p.N_column + (p.N_particle - 1) in alg_vars  # first particle edge

    # Spot-check the last column node
    i_last = p.N_column - 2
    assert p.N_column + i_last * p.N_particle in alg_vars
    assert p.N_column + i_last * p.N_particle + (p.N_particle - 1) in alg_vars
