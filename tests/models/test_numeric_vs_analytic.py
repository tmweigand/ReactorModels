import reactormodels

import numpy as np
import pytest


def test_thomas():
    """Numerical solution matches analytical solution."""
    t_eval = np.linspace(1e-10, 10, 200)
    length = 1 / np.pi
    diameter = 2
    porosity = 0.4
    bed_density = 0.36
    particle_density = 0.6
    feed_concentrations = 1
    flow_rate = 1
    q_m = 20
    rate_constant = 1
    K = 5000
    initial_concentration = 0
    axial_diffusion = 1e-20

    isotherm = reactormodels.models.LangmuirIsotherm(K=K, q_m=q_m)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        diameter=diameter,
        media=reactormodels.Media(
            particle_density=particle_density, bed_density=bed_density
        ),
        water=reactormodels.Water(),
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical(axial_diffusion=axial_diffusion),
        feed_concentrations=feed_concentrations,
        initial_concentration=initial_concentration,
        flow_rate=flow_rate,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_interior_points=5, n_elements=10, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        kinetics=reactormodels.models.SecondOrder(rate_constant),
    )
    x, C, _ = model.solve()

    thomas = reactormodels.models.ThomasLangmuir(
        breakthrough=breakthrough,
        langmuir_constant=K,
        sorbent_capacity=q_m,
        k_Th=rate_constant,
    )

    C_thomas = thomas.breakthrough_profile(time=t_eval, x=length)
    outlet_idx = np.argmin(np.abs(x - length))
    C_numerical = C[:, outlet_idx]
    assert C_numerical == pytest.approx(C_thomas, abs=1e-2)
