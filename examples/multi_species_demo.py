import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/multi_species_demo.png",
):
    """Plot the multi-species advection-diffusion-adsorption solution."""
    superficial_velocity = 1.0  # m/s
    domain_length = 5.0  # m
    porosity = 0.4
    bed_density = 500.0  # kg/m^3
    diameter = 0.1

    n_species = 2

    # Freundlich params
    n = np.linspace(1.2, 1, n_species)
    K_f = np.linspace(0.5, 2, n_species)

    # Langmuir params
    C_in = 1
    K_l = np.linspace(0.01, 0.05, n_species)
    q_m = 50
    axial_diffusion = np.linspace(0.2, 0.1, n_species)  # m^2/s
    isotherm = reactormodels.models.CompetitiveFreundlichIsotherm(K_f, n)

    t_eval = np.linspace(1e-10, 7500, 200)

    media = reactormodels.Media(bed_density=bed_density)
    water = reactormodels.Water()

    column = reactormodels.Column(
        media=media,
        water=water,
        length=domain_length,
        porosity=porosity,
        diameter=diameter,
    )

    breakthroughs = []
    for i in range(n_species):
        chemical = reactormodels.Chemical(axial_diffusion=axial_diffusion[i])
        breakthrough = reactormodels.Breakthrough(
            chemical=chemical,
            column=column,
            feed_concentrations=C_in,
            superficial_velocity=superficial_velocity,
            time=t_eval,
            initial_mass_fraction=1e-10,
        )
        breakthroughs.append(breakthrough)

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=domain_length, n_interior_points=8, n_elements=6, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthroughs, isotherm=isotherm, column_numerics=column_numerics
    )
    x, C, q = model.solve()

    species = [f"Species {i + 1}" for i in range(n_species)]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label in enumerate(species):
        ax.plot(t_eval, C[:, i, -1], linestyle="-", label=label)

    ax.set_title("Advection-Diffusion-Adsorption: Multi-Species")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("C / C_in")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10, ncols=1)
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
