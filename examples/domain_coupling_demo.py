import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/domain_coupling_demo.png",
):
    """Plot spatial profile solving particle and column PDEs."""

    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07  # cm
    pore_diffusion = 5e-6  # cm2/s
    surface_diffusion = 5e-9  # cm2/s
    k_film = 0.1  # cm/s

    # column
    axial_diffusion = 0  # cm2/s
    K = 100  # (mg/g) * (L/mg)
    initial_concentration = 0
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/L
    feed_concentrations = 1  # mg/L
    flow_rate = 40  # cm3/s
    t_eval = np.array([50 * 1440 * 60])

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
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=t_eval,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=3,
        n_elements=8,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=8,
    )

    model = reactormodels.models.DomainCoupling(
        breakthrough=breakthrough,
        isotherm=isotherm,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=k_film,
    )
    z, r, C, Cp = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Choose several axial locations
    indices = [
        1,
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

    ax.set_title(f"Particle Concentration Profile at t = {t_eval}")
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
