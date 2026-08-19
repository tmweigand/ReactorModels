import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/intraparticle_profile_demo.png",
):
    """Plot the intraparticle concentration profile."""

    length = 1
    diameter = 2
    porosity = 0.4
    bulk_density = 500
    superficial_velocity = 1

    feed_concentrations = 1
    particle_porosity = 0.5
    particle_density = 0.6
    initial_concentration = 0
    particle_diameter = 2.0
    pore_diffusion = 0.1
    surface_diffusion = 0.01
    K = 1
    t_eval = np.array([2, 4, 6, 8, 10])

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

    chemical = reactormodels.Chemical(
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=feed_concentrations,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=5,
        n_elements=10,
        add_inlet=True,
    )

    model = reactormodels.models.IntraparticleTransport(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
    )
    x, C, q = model.solve()

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(t_eval):
        mask = x < 0.99 * (particle_diameter / 2)
        C_numerical = C[i, mask]

        ax.plot(x[mask], C_numerical, marker="o", linestyle="-", label=f"t={t:g}")

    ax.set_title("Intraparticle Concentration Profile")
    ax.set_xlabel("r")
    ax.set_ylabel("C / C_in")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=1)
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
