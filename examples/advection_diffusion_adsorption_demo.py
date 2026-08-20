import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/advection_diffusion_adsorption_demo.png",
):
    """Plot the numerical collocation solution against the Ogata-Banks solution."""
    superficial_velocity = 1.0  # m/s
    diffusion = 0.01  # m^2/s
    domain_length = 5.0  # m
    porosity = 0.4
    bulk_density = 500.0  # kg/m^3
    diameter = 0.1
    K = 0.5
    C_in = 1.0

    isotherm = reactormodels.models.LinearIsotherm(K=K)
    R = 1.0 + (bulk_density * K) / porosity  # retardation factor

    t_eval = np.array([100.0, 200.0, 500.0, 1000.0])

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
        column=column, n_interior_points=5, n_elements=20, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorptionSolid(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.LOCAL_EQUILIBRIUM,
    )
    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=breakthrough, diffusion=diffusion, retardation=R
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(t_eval):
        mask = x < 0.8 * domain_length
        C_analytical = ogata_banks.spatial_profile(time=t, x=x[mask])
        C_numerical = C[i, mask]
        max_error = np.abs(C_numerical - C_analytical).max()

        print(f"t={t:g} s, max error={max_error:.2e}")

        ax.plot(
            x[mask], C_numerical, marker="o", linestyle="-", label=f"numerical t={t:g}"
        )
        ax.plot(x[mask], C_analytical, linestyle="--", label=f"analytical t={t:g}")

    ax.set_title("Advection-Diffusion: Numerical vs Analytical Solutions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("C / C_in")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
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
