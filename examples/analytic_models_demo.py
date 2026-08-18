import reactormodels

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False, save_path: str | Path = "data_out/analytic_models_demo.png"
):
    """Plot breakthrough curves from analytical solutions."""
    time = np.linspace(1e-10, 10, 200)
    length = 1 / np.pi
    diameter = 2
    porosity = 0.4
    bulk_density = 0.36
    particle_density = 0.6
    diffusion = 0.1
    feed_concentrations = 1
    flow_rate = 1

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        bulk_density=bulk_density,
        media=reactormodels.Media(particle_density=particle_density),
        water=reactormodels.Water(),
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        flow_rate=flow_rate,
        feed_concentrations=feed_concentrations,
        time=time,
        chemical=reactormodels.Chemical(axial_diffusion=diffusion),
    )

    ogata_banks = reactormodels.models.OgataBanks(
        breakthrough=breakthrough, diffusion=diffusion
    )

    yoon_nelson = reactormodels.models.YoonNelson(
        t_50=5,
        k_YN=5,
    )

    clark = reactormodels.models.Clark(
        r=5,
        A=10000,
        n=2.5,
    )

    k_BA = 3
    sorbent_capacity = 20
    bohart_adams = reactormodels.models.BohartAdams(
        breakthrough=breakthrough,
        k_BA=k_BA,
        sorbent_capacity=sorbent_capacity,
    )

    thomas_rectangular = reactormodels.models.ThomasRectangular(
        breakthrough=breakthrough,
        k_Th=k_BA * length / breakthrough.superficial_velocity,
        sorbent_capacity=15,
    )

    thomas_langmuir = reactormodels.models.ThomasLangmuir(
        breakthrough=breakthrough,
        langmuir_constant=0.5,
        sorbent_capacity=20,
        k_Th=1,
    )

    ob_breakthrough = ogata_banks.breakthrough_profile(time=time, x=length)
    yn_breakthrough = yoon_nelson.breakthrough_profile(time=time)
    c_breakthrough = clark.breakthrough_profile(time=time)
    ba_breakthrough = bohart_adams.breakthrough_profile(time=time, x=length)
    tr_breakthrough = thomas_rectangular.breakthrough_profile(x=length, time=time)
    tl_breakthrough = thomas_langmuir.breakthrough_profile(time=time, x=length)

    plt.figure(figsize=(8, 5))
    plt.plot(time, ob_breakthrough, label="Ogata-Banks")
    plt.plot(time, yn_breakthrough, label="Yoon-Nelson")
    plt.plot(time, c_breakthrough, label="Clark")
    plt.plot(time, ba_breakthrough, label="Bohart-Adams")
    plt.plot(
        breakthrough.time_to_bed_volumes(),
        tr_breakthrough,
        label="Thomas (Rectangular)",
    )
    plt.plot(time, tl_breakthrough, label="Thomas (Langmuir)")

    plt.xlabel("Time or Bed Volumes")
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
