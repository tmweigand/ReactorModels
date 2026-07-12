import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/psdm_breakthrough_demo.png",
):
    """Plot breakthrough profile for pore and surface diffusion model."""

    # particle
    particle_porosity = 0.5
    particle_density = 0.6
    particle_diameter = 2
    pore_diffusion = 0.001
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
    t_eval = np.linspace(1e-10, 10, 200)

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
    outlet_idx = np.argmin(np.abs(z - length))
    C_numerical = C[:, outlet_idx]

    ax.plot(t_eval, C_numerical, linestyle="-", label="Numeric")

    ax.set_title("PSDM Breakthrough Profile")
    ax.set_xlabel("Time")
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
