import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/spatial_numeric_demo.png",
):
    """Plot the numerical collocation solution against the Bohart-Adams solution."""

    k_ldf = 0.002
    diffusion = 0.01
    K = 0.5
    initial_concentration = 0
    q_m = 5
    t_eval = np.array([5000.0])
    length = 6
    diameter = 2
    porosity = 0.4
    bulk_density = 500
    feed_concentrations = 1
    superficial_velocity = 1

    isotherm = reactormodels.models.LangmuirIsotherm(K=K, q_m=q_m)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        superficial_velocity=superficial_velocity,
        time=t_eval,
    )

    numerics = reactormodels.numerics.NumericsConfig(
        column=column, n_interior_points=5, n_elements=20, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        column=column,
        breakthrough=breakthrough,
        diffusion=diffusion,
        initial_concentration=initial_concentration,
        isotherm=isotherm,
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.SECOND_ORDER,
        k_ldf=k_ldf,
    )
    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough, k_BA=k_ldf, sorbent_capacity=q_m
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(t_eval):
        mask = x < 0.99 * length
        C_analytical = bohart_adams.spatial_profile(time=t, x=x[mask])
        C_numerical = C[i, mask]
        max_error = np.abs(C_numerical - C_analytical).max()

        print(f"t={t:g} s, max error={max_error:.2e}")

        ax.plot(
            x[mask], C_numerical, marker="o", linestyle="-", label=f"numerical t={t:g}"
        )
        ax.plot(x[mask], C_analytical, linestyle="--", label=f"analytical t={t:g}")

    ax.set_title("Bohart-Adams: Numerical vs Analytical Solutions")
    ax.set_xlabel("x")
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
