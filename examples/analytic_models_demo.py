from reactormodels.models import AnalyticModels

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True, save_path: str | Path = "data_out/analytic_models_demo.png"
):
    """Plot breakthrough curves from analytical solutions."""
    time = np.linspace(0, 200, 200)
    bed_volumes_treated = np.linspace(0, 200, 200)

    yoon_nelson = AnalyticModels.yoon_nelson(
        time=time,
        tau=100,
        k_YN=0.06,
    )

    clark = AnalyticModels.clark(
        time=time,
        r=0.05,
        A=500,
        n=2.5,
    )

    bohart_adams = AnalyticModels.bohart_adams(
        sorbent_loading=1.0,
        k_BA=0.002,
        sorbent_capacity=1000,
        bed_length=1.0,
        velocity=0.1,
        time=time,
        inlet_concentration=100,
    )

    thomas_rectangular = AnalyticModels.thomas_rectangular(
        sorbent_mass=1.0,
        k_Th=0.002,
        sorbent_capacity=5000,
        bed_volume=1.0,
        bed_volumes_treated=bed_volumes_treated,
        inlet_concentration=100,
    )

    thomas_langmuir = np.array(
        [
            AnalyticModels.thomas_langmuir(
                langmuir_constant=0.05,
                apparent_density=0.6,
                inlet_concentration=100,
                sorbent_capacity=1000,
                k_Th=0.002,
                bed_length=1.0,
                bed_void_fraction=0.40,
                interstitial_velocity=0.10,
                time=t,
            )
            for t in time
        ]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(time, yoon_nelson, label="Yoon-Nelson")
    plt.plot(time, clark, label="Clark")
    plt.plot(time, bohart_adams, label="Bohart-Adams")
    plt.plot(bed_volumes_treated, thomas_rectangular, label="Thomas (Rectangular)")
    plt.plot(time, thomas_langmuir, label="Thomas (Langmuir)")

    plt.xlabel("Time")
    plt.ylabel("C/C₀")
    plt.title("Adsorption Breakthrough Models")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return plt


if __name__ == "__main__":
    run_demo()
