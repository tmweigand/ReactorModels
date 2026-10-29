import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import time
import csv


def run_demo(
    show: bool = True,
    save_path: str | Path = "data_out/benchmarking_demo.png",
):
    """Time the multi-species advection-diffusion-adsorption solution."""
    superficial_velocity = 1.0  # m/s
    domain_length = 5.0  # m
    porosity = 0.4
    bed_density = 500.0  # kg/m^3
    diameter = 0.1

    results = []

    for n_species in [5, 10, 15]:

        print(f"\nRunning {n_species} species...")

        # Multi-species params
        C_in = 1

        n = np.linspace(1.2, 1, n_species)
        K = np.linspace(0.5, 2, n_species)
        axial_diffusion = np.linspace(0.1, 0.11, n_species)

        isotherm = reactormodels.models.CompetitiveFreundlichIsotherm(
            K,
            n,
        )

        t_eval = np.linspace(1e-10, 7500, 200)

        media = reactormodels.Media()
        water = reactormodels.Water()

        column = reactormodels.Column(
            media=media,
            water=water,
            length=domain_length,
            porosity=porosity,
            diameter=diameter,
            bed_density=bed_density,
        )

        breakthroughs = []

        for i in range(n_species):
            chemical = reactormodels.Chemical(
                axial_diffusion=axial_diffusion[i],
            )

            breakthrough = reactormodels.Breakthrough(
                chemical=chemical,
                column=column,
                feed_concentrations=C_in,
                superficial_velocity=superficial_velocity,
                time=t_eval,
            )

            breakthroughs.append(breakthrough)

        numerics = reactormodels.numerics.NumericsConfig(
            domain_length=domain_length,
            n_interior_points=8,
            n_elements=6,
            add_inlet=True,
        )

        model = reactormodels.models.AdvectionDiffusionAdsorption(
            breakthrough=breakthroughs,
            isotherm=isotherm,
            numerics=numerics,
        )

        # Time the complete solve
        start = time.perf_counter()

        _, _, _ = model.solve()

        runtime = time.perf_counter() - start

        results.append(
            {
                "n_species": n_species,
                "runtime_s": runtime,
                "residual_evaluations": model._residual_calls,
                "jacobian_evaluations": model._jacobian_calls,
                "residual_time_s": model._residual_time,
                "jacobian_time_s": model._jacobian_time,
                "dC_dq_time_s": model._dC_dq_time,
                "residual_ms": model._residual_time / model._residual_calls * 1000,
                "jacobian_ms": model._jacobian_time / model._jacobian_calls * 1000,
                "dC_dq_ms": model._dC_dq_time / model._dC_dq_calls * 1000,
            }
        )

        print(
            f"  Runtime: {runtime:.2f} s | "
            f"Residuals: {model._residual_calls:,} | "
            f"Jacobians: {model._jacobian_calls:,}"
        )
    print("\nBenchmark results:")
    print(
        f"{'Species':>8} "
        f"{'Runtime (s)':>12} "
        f"{'Residuals':>12} "
        f"{'Jacobians':>12} "
        f"{'Residual (s)':>14} "
        f"{'Jacobian (s)':>14} "
        f"{'dC/dq (s)':>12}"
    )
    print("-" * 100)

    for result in results:
        print(
            f"{result['n_species']:>8} "
            f"{result['runtime_s']:>12.2f} "
            f"{result['residual_evaluations']:>12,} "
            f"{result['jacobian_evaluations']:>12,} "
            f"{result['residual_time_s']:>14.2f} "
            f"{result['jacobian_time_s']:>14.2f} "
            f"{result['dC_dq_time_s']:>12.2f}"
        )
    print(
        f"{'Species':>8} "
        f"{'Runtime':>10} "
        f"{'Jac calls':>10} "
        f"{'Jac (s)':>10} "
        f"{'Jac ms/call':>12} "
        f"{'dC/dq (s)':>12} "
        f"{'dC/dq ms':>10}"
    )
    print("-" * 88)

    for result in results:
        print(
            f"{result['n_species']:>8} "
            f"{result['runtime_s']:>10.2f} "
            f"{result['jacobian_evaluations']:>10,} "
            f"{result['jacobian_time_s']:>10.2f} "
            f"{result['jacobian_ms']:>12.3f} "
            f"{result['dC_dq_time_s']:>12.2f} "
            f"{result['dC_dq_ms']:>10.3f}"
        )

    with open("vectorized_jacobian_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run_demo()
