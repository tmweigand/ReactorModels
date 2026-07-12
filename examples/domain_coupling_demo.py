import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/domain_coupling_demo.png",
):
    """Plot spatial profile solving particle and column PDEs."""

    # particle
    particle_porosity = 0.5
    particle_density = 0.6
    particle_diameter = 2
    pore_diffusion = 0.1
    surface_diffusion = 0.01
    k_film = 0.01

    # column
    axial_diffusion = 0.01
    K = 1
    initial_concentration = 0
    length = 6
    diameter = 2
    porosity = 0.4
    bulk_density = 500
    feed_concentrations = 1
    superficial_velocity = 1
    t_eval = np.array([10])

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        particle_porosity=particle_porosity,
        bulk_density=bulk_density,
        particle_density=particle_density,
        diameter=diameter,
        particle_diameter=particle_diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
        resolution=reactormodels.models.DomainResolution.COLUMN,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        column=column,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
        resolution=reactormodels.models.DomainResolution.PARTICLE,
    )

    model = reactormodels.models.DomainCoupling(
        column=column,
        breakthrough=breakthrough,
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
        initial_concentration=initial_concentration,
        isotherm=isotherm,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_film=k_film,
    )
    z, r, C, Cp = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Choose several axial locations
    indices = [
        0,
        len(z) // 4,
        len(z) // 2,
        3 * len(z) // 4,
        len(z) - 1,
    ]

    for i in indices:
        ax.plot(
            r,
            Cp[0, i, :],
            marker="o",
            label=f"z={z[i]:.2f}",
        )

    ax.set_title("Particle Concentration Profile at t = 1")
    ax.set_xlabel("Particle radius, r")
    ax.set_ylabel("Pore concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()

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
