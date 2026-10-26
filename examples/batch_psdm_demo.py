import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/batch_psdm_demo.png",
):
    """Plot the intraparticle concentration profile."""

    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07 * 2  # cm
    pore_diffusion = 5e-6  # cm2/s
    surface_diffusion = 5e-9  # cm2/s
    k_film = 0.1  # cm/s
    K = 1000  # (mg/g) * (L/mg)
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bed_density = 399.8  # g/L
    feed_concentrations = 1  # mg/L
    sorbent_dosage = 0.001
    t_eval = np.linspace(1e-10, 1000000, 200)

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
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=feed_concentrations,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=30,
        n_elements=1,
        add_inlet=True,
    )

    model = reactormodels.models.BatchPSDM(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        k_film=k_film,
        sorbent_dosage=sorbent_dosage,
    )
    x, C, Cp = model.solve()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # intraparticle profiles
    t_space = np.array([50000, 100000, 250000, 1000000])
    mask = x < 0.99 * (particle_diameter / 2)

    for t in t_space:
        idx = np.abs(t_eval - t).argmin()
        ax1.plot(
            x[mask],
            Cp[idx, mask],
            marker="o",
            linestyle="-",
            label=f"t={t_eval[idx]/60/1440:.2f} days",
        )
    ax1.set_title("Intraparticle Concentration Profile")
    ax1.set_xlabel("r")
    ax1.set_ylabel("C / C_in")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, ncols=1)

    # bulk time profile
    ax2.plot(t_eval / 60 / 1440, C)
    ax2.set_title("Batch PSDM Kinetic Test")
    ax2.set_xlabel("time (days)")
    ax2.set_ylabel("C / C_in")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2)


if __name__ == "__main__":
    run_demo()
