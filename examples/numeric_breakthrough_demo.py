import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/numeric_breakthrough_demo.png",
):
    """Plot the numerical collocation solution against the Bohart-Adams solution."""

    t_eval = np.linspace(1e-10, 20000, 200)
    length = 5
    diameter = 0.1
    porosity = 0.4
    bed_density = 500
    feed_concentrations = 1
    superficial_velocity = 1
    q_m = 6
    rate_constant = 5e-4
    K = 1e4
    axial_diffusion = 0.01

    isotherm = reactormodels.models.LangmuirIsotherm(K, q_m)

    column = reactormodels.Column(
        diameter=diameter,
        length=length,
        porosity=porosity,
        media=reactormodels.Media(),
        water=reactormodels.Water(),
        bed_density=bed_density,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        feed_concentrations=feed_concentrations,
        superficial_velocity=superficial_velocity,
        time=t_eval,
        chemical=reactormodels.Chemical(axial_diffusion=axial_diffusion),
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_interior_points=8, n_elements=6, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        kinetics=reactormodels.models.SecondOrder(rate_constant),
        inlet_bc=reactormodels.models.DirichletBC,
    )
    x, C, q = model.solve()

    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough, k_BA=rate_constant, sorbent_capacity=q_m
    )

    thomas = reactormodels.models.ThomasLangmuir(
        breakthrough=breakthrough,
        langmuir_constant=K,
        sorbent_capacity=q_m,
        k_Th=rate_constant,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    C_analytical = bohart_adams.breakthrough_profile(time=t_eval, x=length)
    C_thomas = thomas.breakthrough_profile(time=t_eval, x=length)
    C_numerical = C[:, -1]
    max_error = np.abs(C_numerical - C_thomas).max()
    max_rel_error = (np.abs(C_numerical - C_thomas) / C_numerical).max()

    print(f"max error={max_error:.2e}")
    print(f"max rel error={max_rel_error:.2e}")

    ax.plot(t_eval, C_numerical, linestyle="-", label="Numerical")
    ax.plot(t_eval, C_analytical, linestyle="--", label="Bohart-Adams")
    ax.plot(t_eval, C_thomas, linestyle=":", label="Thomas")

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
