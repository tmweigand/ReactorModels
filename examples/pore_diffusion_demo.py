import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/pore_diffusion_demo.png",
):
    """Plot breakthrough profile for pore diffusion model."""

    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07 * 2  # cm
    pore_diffusion = 2.5e-5  # cm2/s
    surface_diffusion = 0  # cm2/s
    k_film = 0.1  # cm/s

    # column
    axial_diffusion = 0  # cm2/s
    K = 0  # (mg/g) * (L/mg)
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bed_density = 399.8  # g/L
    feed_concentrations = 1  # mg/L
    flow_rate = 40  # cm3/s
    t_eval = np.linspace(
        1e-10, length * np.pi * (diameter / 2) ** 2 / flow_rate * porosity * 5, 200
    )  # s

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
        bed_density=bed_density,
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
        add_inlet=True,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )

    model = reactormodels.models.PSDM(
        breakthrough=breakthrough,
        isotherm=isotherm,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=k_film,
    )
    z, r, C, q = model.solve()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t_eval, C[:, -1], linestyle="-")

    ax.set_title("PDM Breakthrough Profile")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("C / C_in")
    ax.grid(True, alpha=0.3)
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
