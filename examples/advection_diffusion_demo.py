import reactormodels

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False, save_path: str | Path = "data_out/advection_diffusion_demo.png"
):
    """Plot the numerical collocation solution against the Ogata-Banks solution."""
    superficial_velocity = 1.0  # m/s
    domain_length = 5.0  # m
    porosity = 0.5
    t_eval = np.array([[0.1], [0.5], [1.0]])
    diameter = 0.1
    diffusions = np.linspace(0.1, 1, 3)  # m^2/s
    C_in = 1

    column = reactormodels.Column(
        diameter=diameter,
        length=domain_length,
        porosity=porosity,
        media=reactormodels.Media(),
        water=reactormodels.Water(),
    )

    breakthroughs = []
    for i in range(len(diffusions)):
        breakthrough = reactormodels.Breakthrough(
            column=column,
            feed_concentrations=C_in,
            superficial_velocity=superficial_velocity,
            time=t_eval[i],
            chemical=reactormodels.Chemical(axial_diffusion=diffusions[i]),
        )
        breakthroughs.append(breakthrough)

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_interior_points=5, n_elements=20, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthroughs,
        numerics=numerics,
    )
    x, C = model.solve()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    mask = x < 0.8 * domain_length

    for j, (ax, breakthrough, diffusion) in enumerate(
        zip(axes, breakthroughs, diffusions)
    ):
        ogata_banks = reactormodels.models.OgataBanks(
            breakthrough=breakthrough,
            diffusion=diffusion,
        )

        for i, t in enumerate(model.t_eval):
            C_analytical = ogata_banks.spatial_profile(
                x=x[mask],
                time=t,
            )
            C_numerical = C[i, j, mask]

            max_error = np.abs(C_numerical - C_analytical).max()

            print(f"Species {j + 1}, t={t:g} s, max error={max_error:.2e}")

            ax.plot(
                x[mask],
                C_numerical,
                marker="o",
                linestyle="-",
                label=f"numerical t={t:g}",
            )
            ax.plot(
                x[mask],
                C_analytical,
                linestyle="--",
                label=f"analytical t={t:g}",
            )

        ax.set_title(f"Species {j + 1}")
        ax.set_ylabel("C / C_in")
        ax.set_xlabel("x [m]")
        ax.legend(fontsize=8, ncols=1)

    fig.suptitle("Advection-Diffusion: Numerical vs Analytical Solutions")
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
