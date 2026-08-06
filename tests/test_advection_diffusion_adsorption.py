import reactormodels

import numpy as np
import pytest


def _base_model(mode, k_ldf=0.1, n_col=30):
    """Shared setup for all adsorption tests."""
    superficial_velocity = 0.5
    diffusion = 0.1
    column_length = 5.0
    inlet_concentration = 1.0
    initial_concentration = 0.0
    porosity = 0.5
    bulk_density = 500.0
    K = 0.5
    diameter = 1
    time = [0, 1, 2]

    media = reactormodels.Media(
        particle_porosity=0.3,
        particle_density=bulk_density / (1 - porosity),
    )
    water = reactormodels.Water(water_matrix="tested_water")
    chemical = reactormodels.Chemical(
        compound="Test compound",
    )
    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
        media=media,
        water=water,
        chemical=chemical,
    )

    flow_rate = superficial_velocity * column.cross_section_area()

    breakthrough = reactormodels.Breakthrough(
        column=column,
        superficial_velocity=superficial_velocity,
        feed_concentrations=inlet_concentration,
        flow_rate=flow_rate,
        effluent_concentrations=np.zeros(len(time)),
        compound="Test compound",
        time=time,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=n_col, add_inlet=True
    )

    return (
        reactormodels.models.AdvectionDiffusionAdsorption(
            column=column,
            breakthrough=breakthrough,
            initial_concentration=initial_concentration,
            diffusion=diffusion,
            isotherm=reactormodels.models.LinearIsotherm(K=K),
            numerics=numerics,
            mode=mode,
            k_ldf=k_ldf,
            inlet_bc=reactormodels.models.DirichletBC,
        ),
        diffusion,
        column_length,
        porosity,
        bulk_density,
        K,
    )


def test_local_equilibrium_vs_ogata_banks():
    """Linear isotherm + local equilibrium = retarded Ogata-Banks."""
    model, D, column_length, eps, rho_b, K = _base_model(
        reactormodels.models.adsorption_kinetics.AdsorptionKinetics.LOCAL_EQUILIBRIUM
    )

    R = 1.0 + (rho_b * K) / eps

    t_mid = 0.5 * column_length / (model.breakthrough.interstitial_velocity() / R)
    t_eval = np.array([0.25 * t_mid, t_mid, 2.0 * t_mid])

    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

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
    v_eff = eq_model.breakthrough.interstitial_velocity() / (eps * R)
    t_eval = np.array([0.5 * L / v_eff, L / v_eff])

    x, C_eq, q_eq = eq_model.solve((0, t_eval[-1]), t_eval)
    x, C_ldf, q_ldf = ldf_model.solve((0, t_eval[-1]), t_eval)

    # High k_ldf → LDF ≈ equilibrium
    assert C_ldf == pytest.approx(C_eq, abs=1e-2)


def test_ldf_q_tracks_equilibrium():
    """Q should approach q*(C) over time."""
    model, D, L, eps, rho_b, K = _base_model(
        reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE, k_ldf=0.5
    )
    R = 1.0 + (rho_b * K) / eps
    v_eff = model.breakthrough.interstitial_velocity() / (eps * R)
    t_long = 5.0 * L / v_eff  # run long enough for q to equilibrate

    x, C, q = model.solve(t_span=(0, t_long), t_eval=np.array([t_long]))

    q_eq = model.iso.q(C[0])
    # q should be close to q*(C) at long times
    assert q[0] == pytest.approx(q_eq, rel=0.05)
