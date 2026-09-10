import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import copy


def run_demo():
    """Plot breakthrough profile for pore and surface diffusion model."""
    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.007 * 2  # cm
    k_film = 0.1  # cm/s

    # column
    axial_diffusion = 0  # cm2/s
    K = [75, 150]  # (mg/g) * (L/mg)
    k_Th = [0.0002, 0.0001]
    qe = [75, 150]
    length = 1.5  # cm
    diameter = 0.318  # cm
    porosity = 0.38
    bulk_density = 399.8  # g/L
    flow_rate = 0.1  # cm3/s
    BVs = np.linspace(1, 150000, 25)

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

    pfoa = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
        pore_diffusion=5e-6,
        surface_diffusion=3e-9,
        name="PFOA",
    )

    pfos = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
        pore_diffusion=5.1e-6,
        surface_diffusion=6e-9,
        name="PFOS",
    )

    chemicals = [pfoa, pfos]

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

    breakthroughs = []
    model_outs = []

    model_names = ["PSDM", "Thomas"]

    for i, chemical in enumerate(chemicals):
        breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=1,
            flow_rate=flow_rate,
            bed_volumes=BVs,
        )

        isotherm = reactormodels.models.LinearIsotherm(K=K[i])
        psdm = reactormodels.models.PSDM(
            breakthrough=breakthrough,
            isotherm=isotherm,
            column_numerics=column_numerics,
            particle_numerics=particle_numerics,
            k_film=k_film,
        )
        z, r, C, Cp = psdm.solve()

        psdm_out = C[:, -1] / breakthrough.mean_feed_concentration()

        psdm_breakthrough = copy.copy(breakthrough)
        psdm_breakthrough.effluent_concentrations = psdm_out

        breakthroughs.append(psdm_breakthrough)

        thomas = reactormodels.models.ThomasRectangular(breakthrough, k_Th[i], qe[i])
        thomas_out = thomas.breakthrough_profile(breakthrough.time, length)

        model_outs.append(
            [
                psdm_out,
                thomas_out,
            ]
        )

    plot = reactormodels.postprocess.Plotting(breakthroughs)

    plot.plot_breakthrough_and_model(
        model_names=model_names,
        model_outs=model_outs,
        save_path=f"data_out/plotting/multiple_models.png",
    )


if __name__ == "__main__":
    run_demo()
