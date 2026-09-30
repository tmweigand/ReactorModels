import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/psdm_multi_demo.png",
):
    """Plot breakthrough profile for pore and surface diffusion model."""
    n_species = 1
    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07 * 2  # cm
    pore_diffusion = np.full(n_species, 5e-6)  # cm2/s
    surface_diffusion = np.full(n_species, 5e-9)  # cm2/s
    k_film = np.full(n_species, 0.1)  # cm/s

    # column
    axial_diffusion = np.zeros(n_species)  # cm2/s
    n = np.linspace(1.1, 1.1, n_species)
    K = np.linspace(100, 100, n_species)  # np.linspace(100, 150, n_species)
    q_m = 6000
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/L
    feed_concentrations = [1]  # mg/L
    flow_rate = 40  # cm3/s
    ads_data = np.loadtxt("examples/6_species_curves.txt", skiprows=4)
    time = np.array(ads_data[:, 0])
    t_eval = time * 60  # s

    isotherm = reactormodels.models.CompetitiveFreundlichIsotherm(K, n)

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

    breakthroughs = []
    for i in range(n_species):
        chemical = reactormodels.Chemical(
            axial_diffusion=axial_diffusion[i],
            pore_diffusion=pore_diffusion[i],
            surface_diffusion=surface_diffusion[i],
        )
        breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=feed_concentrations,
            flow_rate=flow_rate,
            time=t_eval,
            initial_mass_fraction=1e-8,
        )
        breakthroughs.append(breakthrough)

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

    model = reactormodels.models.PSDM(
        breakthrough=breakthroughs,
        isotherm=isotherm,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=k_film,
    )
    z, r, C, Cp, q = model.solve()

    species = [f"Species {i + 1}" for i in range(n_species)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label in enumerate(species):
        ax.plot(time / 1440, C[:, i, -1], linestyle="-", label=label)
        ax.plot(time / 1440, ads_data[:, i + 3], linestyle="--", label=f"{label} (ads)")

    ax.set_title("PSDM Breakthrough Profile")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("C / C_in")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


if __name__ == "__main__":
    run_demo()
