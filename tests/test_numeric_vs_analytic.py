import reactormodels

import numpy as np
import pytest


def _base_model(mode, n_col=30):
    """Shared setup for all adsorption tests."""
    superficial_velocity = 0.5
    diffusion = 1e-10
    column_length = 5.0
    inlet_concentration = 1.0
    initial_concentration = 0.0
    porosity = 0.5
    bulk_density = 500.0
    K = 1e10
    q_m = 1000
    k_ldf = 0.1
    diameter = 1
    time = np.linspace(0, 5, 6)

    column = reactormodels.Column(
        length=column_length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        superficial_velocity=superficial_velocity,
        feed_concentrations=inlet_concentration,
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
            isotherm=reactormodels.models.LangmuirIsotherm(K=K, q_m=q_m),
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
        q_m,
    )


def test_bohart_adams():
    """Numerical solution matches analytical solution."""
    model, *params = _base_model(
        reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM
    )

    D, L, eps, rho_b, K, q_m = params
    R = 1.0 + (rho_b / eps) * K * q_m / (
        1 + K * model.breakthrough.mean_feed_concentration()
    )
    v_eff = model.breakthrough.interstitial_velocity() / (eps * R)
    t_eval = np.linspace(0, 200, 10)
    # np.array([0.5 * L / v_eff, L / v_eff])

    x, C, q = model.solve((0, t_eval[-1]), t_eval)

    bohart_adams = reactormodels.models.BohartAdams(model.breakthrough, 0.1, q_m)

    for i, t in enumerate(t_eval):
        x = L
        C_analytical = bohart_adams.breakthrough_profile(time=t_eval, x=x)
        assert C[i] == pytest.approx(C_analytical, abs=1e-2)
