import reactormodels

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = True, save_path: str | Path = "data_out/analytic_models_demo.png"
):
    """Plot breakthrough curves from analytical solutions."""
    time = np.linspace(0, 200, 200)
    bed_volumes_treated = np.linspace(0, 200, 200)

    ogata_banks = reactormodels.models.OgataBanks(
        time=time, x=1, interstitial_velocity=0.1, diffusion=0.01, inlet_concentration=1
    ).concentration_profile()

    yoon_nelson = reactormodels.models.YoonNelson(
        time=time,
        t_50=100,
        k_YN=0.06,
    ).concentration_profile()

    clark = reactormodels.models.Clark(
        time=time,
        r=0.05,
        A=500,
        n=2.5,
    ).concentration_profile()

    bohart_adams = reactormodels.models.BohartAdams(
        sorbent_loading=1.0,
        k_BA=0.002,
        sorbent_capacity=1000,
        x=1.0,
        velocity=0.1,
        time=time,
        inlet_concentration=100,
    ).concentration_profile()

    thomas_rectangular = reactormodels.models.ThomasRectangular(
        sorbent_mass=1.0,
        k_Th=0.002,
        sorbent_capacity=5000,
        bed_volume=1.0,
        bed_volumes_treated=bed_volumes_treated,
        inlet_concentration=100,
    ).concentration_profile()

    thomas_langmuir = np.array(
        [
            reactormodels.models.ThomasLangmuir(
                langmuir_constant=0.05,
                apparent_density=0.6,
                inlet_concentration=100,
                sorbent_capacity=1000,
                k_Th=0.002,
                x=1.0,
                bed_void_fraction=0.40,
                interstitial_velocity=0.10,
                time=t,
            ).concentration_profile()
            for t in time
        ]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(time, ogata_banks, label="Ogata-Banks")
    plt.plot(time, yoon_nelson, label="Yoon-Nelson")
    plt.plot(time, clark, label="Clark")
    plt.plot(time, bohart_adams, label="Bohart-Adams")
    plt.plot(bed_volumes_treated, thomas_rectangular, label="Thomas (Rectangular)")
    plt.plot(time, thomas_langmuir, label="Thomas (Langmuir)")

    plt.xlabel("Time")
    plt.ylabel("C/C₀")
    plt.title("Breakthrough Models")
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
