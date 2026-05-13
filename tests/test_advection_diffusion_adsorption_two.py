import reactormodels

import numpy as np
import pytest


def _base_model(mode, k_ldf=0.1, n_col=30):
    """Shared setup for all adsorption tests."""
    velocity = 1.0
    dispersion = 0.1
    L = 5.0
    porosity = 0.4
    bulk_density = 500.0
    K = 0.5

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=n_col, add_inlet=True
    )
    return (
        reactormodels.models.AdvectionDiffusionAdsorption1D_two(
            column_length=L,
            velocity=velocity,
            dispersion=dispersion,
            isotherm=reactormodels.models.LinearIsotherm(K=K),
            bulk_density=bulk_density,
            porosity=porosity,
            oc=oc,
            mode=mode,
            k_ldf=k_ldf,
            inlet_bc=reactormodels.models.InletBC.DIRICHLET,
        ),
        velocity,
        dispersion,
        L,
        porosity,
        bulk_density,
        K,
    )


def test_local_equilibrium_vs_ogata_banks():
    """Linear isotherm + local equilibrium = retarded Ogata-Banks."""
    model, v, D, L, eps, rho_b, K = _base_model(
        reactormodels.models.adsorption_kinetics.AdsorptionKinetics.LOCAL_EQUILIBRIUM
    )

    R = 1.0 + (rho_b * K) / eps
    v_eff = v / (eps * R)
    D_eff = D / R

    t_mid = 0.5 * L / v_eff
    t_eval = np.array([0.25 * t_mid, t_mid, 2.0 * t_mid])

    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=1.0)

    for i, t in enumerate(t_eval):
        mask = x < 0.8 * L
        C_analytical = reactormodels.models.ogata_banks(
            x[mask], t, v_eff, D_eff, C0=1.0
        )
        assert C[i, mask] == pytest.approx(C_analytical, abs=1e-2)


def test_ldf_converges_to_equilibrium_at_high_kldf():
    """At very high k_ldf, LDF solution should match local equilibrium."""
    eq_model, *params = _base_model(
        reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM
    )
    ldf_model, *_ = _base_model(
        reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE, k_ldf=1000.0
    )

    v, D, L, eps, rho_b, K = params
    R = 1.0 + (rho_b * K) / eps
    v_eff = v / (eps * R)
    t_eval = np.array([0.5 * L / v_eff, L / v_eff])

    x, C_eq, q_eq = eq_model.solve((0, t_eval[-1]), t_eval)
    x, C_ldf, q_ldf = ldf_model.solve((0, t_eval[-1]), t_eval)

    # High k_ldf → LDF ≈ equilibrium
    assert C_ldf == pytest.approx(C_eq, abs=1e-2)


def test_ldf_q_tracks_equilibrium():
    """q should approach q*(C) over time."""
    model, v, D, L, eps, rho_b, K = _base_model(
        reactormodels.models.AdsorptionKinetics.LINEAR_DRIVING_FORCE, k_ldf=0.5
    )
    R = 1.0 + (rho_b * K) / eps
    v_eff = v / (eps * R)
    t_long = 5.0 * L / v_eff  # run long enough for q to equilibrate

    x, C, q = model.solve(t_span=(0, t_long), t_eval=np.array([t_long]), C_in=1.0)

    q_eq = model.iso.q(C[0])
    # q should be close to q*(C) at long times
    assert q[0] == pytest.approx(q_eq, rel=0.05)
