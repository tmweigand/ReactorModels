import reactormodels

import numpy as np
import pytest


def _base_model(mode, k_ldf=0.1, n_col=30):
    """Shared setup for all adsorption tests."""
    column_length = 5.0
    diameter = 1
    porosity = 0.5
    bulk_density = 500.0
    superficial_velocity = 0.5
    axial_diffusion = 0.1
    K = 0.5
    dummy_time = [0, 1, 2]

    breakthrough = reactormodels.fixtures.make_breakthrough(
        length=column_length,
        diameter=diameter,
        porosity=porosity,
        bulk_density=bulk_density,
        superficial_velocity=superficial_velocity,
        axial_diffusion=axial_diffusion,
        time=dummy_time,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column_length, n_interior_points=n_col, add_inlet=True
    )

    return (
        reactormodels.models.AdvectionDiffusionAdsorption(
            breakthrough=breakthrough,
            isotherm=reactormodels.models.LinearIsotherm(K=K),
            numerics=numerics,
            mode=mode,
            k_ldf=k_ldf,
            inlet_bc=reactormodels.models.DirichletBC,
        ),
        axial_diffusion,
        column_length,
        porosity,
        bulk_density,
        K,
    )


def test_local_equilibrium_vs_ogata_banks():
    """Linear isotherm + local equilibrium = retarded Ogata-Banks."""
    model, D, column_length, porosity, rho_b, K = _base_model(
        reactormodels.models.adsorption_kinetics.AdsorptionKinetics.LOCAL_EQUILIBRIUM
    )

    R = 1.0 + (rho_b * K) / porosity

    t_mid = 0.5 * column_length / (model.breakthrough.interstitial_velocity / R)
    t_eval = np.array([0.25 * t_mid, t_mid, 2.0 * t_mid])

    model.breakthrough.time = t_eval

    x, C, q = model.solve()

    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=model.breakthrough, diffusion=D, retardation=R
    )

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * column_length
        C_analytical = ogata_banks.spatial_profile(x[mask], t)
        assert C[i, mask] == pytest.approx(C_analytical, abs=1e-2)


def test_ldf_converges_to_equilibrium_at_high_kldf():
    """At very high k_ldf, LDF solution should match local equilibrium."""
    eq_model, *params = _base_model(
        reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM
    )
    ldf_model, *_ = _base_model(
        reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE, k_ldf=1000.0
    )

    D, L, eps, rho_b, K = params
    R = 1.0 + (rho_b * K) / eps
    v_eff = eq_model.breakthrough.interstitial_velocity / (eps * R)
    t_eval = np.array([0.5 * L / v_eff, L / v_eff])

    eq_model.breakthrough.time = t_eval
    ldf_model.breakthrough.time = t_eval

    _, C_eq, q_eq = eq_model.solve()
    _, C_ldf, q_ldf = ldf_model.solve()

    # High k_ldf → LDF ≈ equilibrium
    assert C_ldf == pytest.approx(C_eq, abs=1e-2)

    assert q_ldf == pytest.approx(q_eq, abs=1e-2)


def test_ldf_q_tracks_equilibrium():
    """q should approach q*(C) over time."""
    model, D, L, eps, rho_b, K = _base_model(
        reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE, k_ldf=0.5
    )
    R = 1.0 + (rho_b * K) / eps
    v_eff = model.breakthrough.interstitial_velocity / (eps * R)
    t_long = np.array([5.0 * L / v_eff])  # run long enough for q to equilibrate

    model.breakthrough.time = t_long

    _, C, q = model.solve()

    q_eq = model.iso.q(C[0])
    # q should be close to q*(C) at long times
    assert q[0] == pytest.approx(q_eq, rel=0.05)
