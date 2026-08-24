"""test_intraparticle_transport.py"""

import numpy as np
import reactormodels


def _make_particle(Ds=5e-9, C_in=1):
    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/mL
    particle_diameter = 0.07  # cm
    pore_diffusion = 5e-6  # cm2/s
    k_film = 0.1  # cm/s

    # column
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

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )
    return reactormodels.models.IntraparticleTransport(
        isotherm=isotherm,
        breakthrough=breakthrough,
        pore_diffusion=pore_diffusion,
        surface_diffusion=Ds,
        initial_concentration=initial_concentration,
        numerics=particle_numerics,
        k_film=k_film,
    )


# def test_intraparticle():
#     """test transport within particle"""

#     p = _make_particle()
#     t_eval = np.linspace(1e-10, 3600, 5)
#     z, c, q = p.solve(
#         t_span=(0, t_eval[-1]),
#         t_eval=t_eval,
#     )
#     print(c)
#     np.testing.assert_approx_equal(
#         np.array([0.0, 0.00394456, 0.0175, 0.03105544, 0.035]), c[0]
#     )
