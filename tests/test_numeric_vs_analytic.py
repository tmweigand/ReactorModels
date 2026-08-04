import reactormodels

import numpy as np
import pytest


def test_thomas():
    """Numerical solution matches analytical solution."""
    t_eval = np.linspace(1e-10, 10, 200)
    length = 1 / np.pi
    diameter = 2
    porosity = 0.4
    bulk_density = 0.36
    particle_density = 0.6
    feed_concentrations = 1
    flow_rate = 1
    q_m = 20
    k = 1
    K = 5000
    initial_concentration = 0
    diffusion = 1e-20

    isotherm = reactormodels.models.LangmuirIsotherm(K=K, q_m=q_m)

    media = reactormodels.Media(
        particle_porosity=0.3,
        particle_density=particle_density,
    )

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
        media=media,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=5, n_elements=20, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        column=column,
        breakthrough=breakthrough,
        diffusion=diffusion,
        initial_concentration=initial_concentration,
        isotherm=isotherm,
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.SECOND_ORDER,
        k_ldf=k,
    )
    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    thomas = reactormodels.models.ThomasLangmuir(
        breakthrough=breakthrough, langmuir_constant=K, sorbent_capacity=q_m, k_Th=k
    )

    C_thomas = thomas.breakthrough_profile(time=t_eval, x=length)
    outlet_idx = np.argmin(np.abs(x - length))
    C_numerical = C[:, outlet_idx]
    assert C_numerical == pytest.approx(C_thomas, abs=1e-2)
