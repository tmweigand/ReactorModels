import reactormodels

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True, save_path: str | Path = "data_out/advection_diffusion_demo.png"
):
    """Plot the numerical collocation solution against the Ogata-Banks solution."""
    velocity = 1.0  # m/s
    diffusion = 0.5  # m^2/s
    domain_length = 5.0  # m
    porosity = 0.5
    C_in = 1.0
    t_eval = np.array([1.0, 2.0, 3.0])

    numerics = reactormodels.numerics.NumericsConfig(
        n_interior_points=10, n_elements=3, add_inlet=True
    )
    column = reactormodels.Column(length=domain_length, porosity=porosity)

    model = reactormodels.models.AdvectionDiffusion(
        column=column,
        inlet_concentration=C_in,
        velocity=velocity,
        diffusion=diffusion,
        numerics=numerics,
    )
    x, C = model.solve(t_span=(0, t_eval[-1]), t_eval=t_eval, C_in=C_in)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(t_eval):
        mask = x < 0.8 * domain_length
        C_analytical = reactormodels.models.ogata_banks(
            x[mask], t, velocity, diffusion, C_in
        )
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
