"""Convergence and work-precision demo versus the Ogata-Banks analytical solution."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import reactormodels
from reactormodels.numerics import NumericsConfig, convergence_order
from reactormodels.fixtures import make_breakthrough


def _run_case(n_interior_points: int, n_elements: int, t_end: float) -> dict:
    """Solve one case and compare against the Ogata-Banks analytical solution."""
    breakthrough = make_breakthrough(time=np.array([t_end]))
    numerics = NumericsConfig(
        domain_length=breakthrough.column.length,
        n_interior_points=n_interior_points,
        n_elements=n_elements,
    )
    model = reactormodels.models.AdvectionDiffusion(
        breakthrough=breakthrough, numerics=numerics
    )

    start = perf_counter()
    x, concentration_history = model.solve()
    cpu_time_seconds = perf_counter() - start
    concentration_numerical = concentration_history[-1]

    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=breakthrough, diffusion=breakthrough.chemical.axial_diffusion
    )

    error = concentration_numerical - ogata_banks.spatial_profile(x=x, time=t_end)

    n_nodes = len(x)
    return {
        "n_interior_points": n_interior_points,
        "n_elements": n_elements,
        "n_nodes": n_nodes,
        "effective_dx": breakthrough.column.length / (n_nodes - 1),
        "cpu_time_seconds": cpu_time_seconds,
        "l2_error": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
        "x": x,
        "concentration": concentration_numerical,
    }


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/ogata_banks_convergence_demo.png",
):
    """Sweep numerical settings and plot error/work-precision vs Ogata-Banks."""
    t_end = 1.0
    interior_points_sweep = (1, 3, 5, 7, 10)
    elements_sweep = (5, 10, 20, 40)

    cmap = plt.get_cmap("tab10")
    colors = {value: cmap(i) for i, value in enumerate(interior_points_sweep)}

    results = [
        _run_case(n_interior_points, n_elements, t_end)
        for n_interior_points in interior_points_sweep
        for n_elements in elements_sweep
    ]

    # attach observed convergence orders, grouped by n_interior_points
    for n_interior_points in interior_points_sweep:
        group = sorted(
            (r for r in results if r["n_interior_points"] == n_interior_points),
            key=lambda r: r["n_elements"],
        )
        h = np.array([r["effective_dx"] for r in group])
        for key in ("l2_error", "max_error"):
            orders = convergence_order(h, np.array([r[key] for r in group]))
            for r, order in zip(group, orders):
                r[f"order_{key.split('_')[0]}"] = order

    breakthrough = make_breakthrough(time=t_end)
    x_analytic = np.linspace(0.0, breakthrough.column.length, 500)
    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=breakthrough, diffusion=breakthrough.chemical.axial_diffusion
    )
    concentration_analytic = ogata_banks.spatial_profile(x=x_analytic, time=t_end)

    fig, (ax_analytic, ax_error, ax_work_precision) = plt.subplots(
        1, 3, figsize=(16, 5)
    )

    best_result = min(results, key=lambda r: r["max_error"])
    worst_result = max(results, key=lambda r: r["max_error"])

    ax_analytic.plot(
        x_analytic,
        concentration_analytic,
        color="#1f1f1f",
        linewidth=1.5,
        label="Ogata-Banks (analytical)",
        zorder=3,
    )
    ax_analytic.plot(
        best_result["x"],
        best_result["concentration"],
        marker=".",
        color="#4daf4a",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Best: n_int={best_result['n_interior_points']}, "
            f"n_elem={best_result['n_elements']} "
            f"(max err={best_result['max_error']:.2e})"
        ),
        zorder=2,
    )
    ax_analytic.plot(
        worst_result["x"],
        worst_result["concentration"],
        marker=".",
        color="#c83737",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Worst: n_int={worst_result['n_interior_points']}, "
            f"n_elem={worst_result['n_elements']} "
            f"(max err={worst_result['max_error']:.2e})"
        ),
        zorder=2,
    )
    ax_analytic.set(
        title="Ogata-Banks Analytical Profile vs Best/Worst Numerical",
        xlabel="x [m]",
        ylabel="C/C0 at t = 1 s",
    )
    ax_analytic.grid(alpha=0.3)
    ax_analytic.legend(fontsize=7)

    for n_interior_points in interior_points_sweep:
        group = sorted(
            (r for r in results if r["n_interior_points"] == n_interior_points),
            key=lambda r: r["n_elements"],
        )
        color = colors.get(n_interior_points, "#4c78a8")

        ax_error.loglog(
            [r["n_nodes"] for r in group],
            [r["max_error"] for r in group],
            marker=".",
            color=color,
            label=f"n_int={n_interior_points}",
        )

        ax_work_precision.loglog(
            [r["cpu_time_seconds"] for r in group],
            [r["max_error"] for r in group],
            marker=".",
            color=color,
            label=f"n_int={n_interior_points}",
        )

    ax_error.set(
        title="Max Error vs Number of Nodes",
        xlabel="Number of nodes",
        ylabel="Max error",
    )
    ax_error.grid(which="both", alpha=0.3)
    ax_error.legend(fontsize=8)

    ax_work_precision.set(
        title="Work-Precision (CPU Time vs Max Error)",
        xlabel="CPU time [s]",
        ylabel="Max error",
    )
    ax_work_precision.grid(which="both", alpha=0.3)
    ax_work_precision.legend(fontsize=8)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, results


if __name__ == "__main__":
    run_demo()
