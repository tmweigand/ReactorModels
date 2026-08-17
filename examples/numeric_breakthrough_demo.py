import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/numeric_breakthrough_demo.png",
):
    """Plot the numerical collocation solution against the Bohart-Adams solution."""

    t_eval = np.linspace(1e-10, 10, 200)
    length = 1 / np.pi
    diameter = 2
    porosity = 0.4
    bulk_density = 0.36
    particle_density = 0.6
    feed_concentrations = 1
    flow_rate = 1
    q_m = 10
    k = 10
    K = 5000
    initial_concentration = 0
    diffusion = 1e-20

    isotherm = reactormodels.models.LangmuirIsotherm(K=K, q_m=q_m)

    column = reactormodels.Column(
        diameter=diameter,
        length=length,
        porosity=porosity,
        bulk_density=bulk_density,
        media=reactormodels.Media(particle_density=particle_density),
        water=reactormodels.Water(),
    )

    # column = reactormodels.Column(
    #     length=length,
    #     porosity=porosity,
    #     bulk_density=bulk_density,
    #     diameter=diameter,
    #     particle_density=particle_density,
    # )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        initial_concentration=initial_concentration,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=t_eval,
        chemical=reactormodels.Chemical(diffusion=diffusion),
    )

    numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length, n_interior_points=5, n_elements=20, add_inlet=True
    )

    model = reactormodels.models.AdvectionDiffusionAdsorption(
        breakthrough=breakthrough,
        isotherm=isotherm,
        numerics=numerics,
        mode=reactormodels.models.AdsorptionKinetics.SECOND_ORDER,
        k_ldf=k,
    )
    x, C, q = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval)

    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough, k_BA=k, sorbent_capacity=q_m
    )

    thomas = reactormodels.models.ThomasLangmuir(
        breakthrough=breakthrough, langmuir_constant=K, sorbent_capacity=q_m, k_Th=k
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    C_analytical = bohart_adams.breakthrough_profile(time=t_eval, x=length)
    C_thomas = thomas.breakthrough_profile(time=t_eval, x=length)
    outlet_idx = np.argmin(np.abs(x - length))
    C_numerical = C[:, outlet_idx]
    max_error = np.abs(C_numerical - C_analytical).max()

    print(f"max error={max_error:.2e}")

    ax.plot(t_eval, C_numerical, linestyle="-", label="Numerical")
    ax.plot(t_eval, C_analytical, linestyle="--", label="Bohart-Adams")
    ax.plot(t_eval, C_thomas, linestyle=":", label="Thomas")
    # ax.plot(t_eval, q[:, 1], linestyle="-.", label="solid")

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
