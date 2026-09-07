import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo():
    """Plot breakthrough profile for pore and surface diffusion model."""
    input_file = "examples/NOM1_data.txt"

    data = np.array(np.genfromtxt(input_file, delimiter="\t"))

    bed_volumes = np.array(data[3:, 0])
    effluent_concentrations = np.array(data[3:, 1:6])
    feed_concentrations = np.array(data[1:3, 1:6])
    species = np.genfromtxt(input_file, delimiter="\t", dtype=str)[0, 1:6]

    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.007 * 2  # cm
    pore_diffusion = 5e-6  # cm2/s
    surface_diffusion = 5e-10  # cm2/s
    k_film = 0.1  # cm/s

    # column
    axial_diffusion = 0  # cm2/s
    K = [100, 200, 300, 400, 500]  # (mg/g) * (L/mg)
    k_Th = [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
    qe = [20, 50, 100, 120, 150]
    length = 1.5  # cm
    diameter = 0.318  # cm
    porosity = 0.38
    bulk_density = 399.8  # g/L
    flow_rate = 0.1  # cm3/s

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

    chemical = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
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

    breakthroughs = []
    model_outs = []

    model_names = ["PSDM", "Thomas"]

    for i, name in enumerate(species):
        breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=feed_concentrations[:, i],
            flow_rate=flow_rate,
            bed_volumes=bed_volumes,
            effluent_concentrations=effluent_concentrations[:, i],
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

        breakthroughs.append(breakthrough)

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
        names=species,
        save_path=f"data_out/plotting/multiple_models.png",
    )


if __name__ == "__main__":
    run_demo()
