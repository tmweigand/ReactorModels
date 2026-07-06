import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/numeric_models_demo.png",
):
    """Plot the numerical collocation solution against the Bohart-Adams solution."""

    k = 0.1
    diffusion = 1e-10  # m^2/s
    K = 1e10
    initial_concentration = 0
    q_m = 20
    time = np.linspace(1e-10, 10, 200)
    length = 1 / np.pi
    diameter = 2
    porosity = 0.4
    bulk_density = 0.36
    particle_density = 0.6
    diffusion = 0.1
    feed_concentrations = 1
    flow_rate = 1

    isotherm = reactormodels.models.LangmuirIsotherm(K=K, q_m=q_m)

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        bulk_density=bulk_density,
        diameter=diameter,
        particle_density=particle_density,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=time,
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
        k_ldf=k,
    )
    x, C, q = model.solve(t_span=(time[0], time[-1]), t_eval=time)

    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough, k_BA=k, sorbent_capacity=q_m
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    C_analytical = bohart_adams.breakthrough_profile(time=time, x=length)
    C_numerical = C
    max_error = np.abs(C_numerical - C_analytical).max()

    print(f"max error={max_error:.2e}")

    ax.plot(time, C_numerical, marker="o", linestyle="-", label="numerical")
    ax.plot(time, C_analytical, linestyle="--", label="analytical")

    ax.set_title("Bohart-Adams: Numerical vs Analytical Solutions")
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
