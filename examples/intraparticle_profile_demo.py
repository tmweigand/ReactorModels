import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/intraparticle_profile_demo.png",
):
    """Plot the intraparticle concentration profile."""

    length = 6
    diameter = 2
    porosity = 0.4
    bulk_density = 500
    superficial_velocity = 1

    feed_concentrations = 1
    particle_porosity = 0.5
    particle_density = 0.6
    initial_concentration = 0
    particle_radius = 1.0
    pore_diffusion = 0.1
    surface_diffusion = 0
    K = 0
    t_eval = np.array([2, 4, 6, 8, 10])

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        particle_porosity=particle_porosity,
        bulk_density=bulk_density,
        particle_density=particle_density,
        diameter=diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=5, n_elements=30, add_inlet=True
    )

    model = reactormodels.models.IntraparticleTransport(
        column=column,
        breakthrough=breakthrough,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
        initial_concentration=initial_concentration,
        isotherm=isotherm,
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
    )
    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(t_eval):
        mask = x < 0.99 * particle_radius
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
