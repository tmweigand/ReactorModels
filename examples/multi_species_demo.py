import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/multi_species_demo.png",
):
    """Plot the numerical collocation solution against the Ogata-Banks solution."""
    superficial_velocity = 1.0  # m/s
    domain_length = 5.0  # m
    porosity = 0.4
    bulk_density = 500.0  # kg/m^3
    diameter = 0.1

    # multi-species params
    K = [0.3, 0.5]
    C_in = [[1], [1]]
    diffusion = [0.2, 0.1]  # m^2/s
    n = [1.2, 1.1]

    isotherm = reactormodels.models.FreundlichIsotherm(K, n)

    t_eval = np.linspace(1e-10, 2000, 200)

    media = reactormodels.Media()
    water = reactormodels.Water()
    chemical = reactormodels.Chemical(diffusion=diffusion)

    column = reactormodels.Column(
        media=media,
        water=water,
        length=domain_length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        chemical=chemical,
        column=column,
        feed_concentrations=C_in,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=8, n_elements=6, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        # k_ldf=0.1,
    )
    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    fig, ax = plt.subplots(figsize=(8, 5))
    species = ["1", "2"]
    for i, label in enumerate(species):
        R = 1.0 + (bulk_density * K[i]) / porosity
        ogata_breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=C_in[i],
            superficial_velocity=superficial_velocity,
            time=t_eval,
        )
        ogata_banks = reactormodels.models.OgataBanks(
            breakthrough=ogata_breakthrough, diffusion=diffusion[i], retardation=R
        )
        C_ogata = ogata_banks.breakthrough_profile(time=t_eval, x=domain_length)
        ax.plot(t_eval, C_ogata, linestyle="-", label=label)
        ax.plot(t_eval, C[:, i, -1], linestyle="--", label=label)

    ax.set_title("Advection-Diffusion-Adsorption: Multi-Species")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("C / C_in")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=len(species))
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
