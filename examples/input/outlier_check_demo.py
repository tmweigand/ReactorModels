from reactormodels.Input import identify_curve_outliers
from reactormodels.numerics.helpers import compute_rmse
import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/input/outlier_check_demo.png",
):
    """Plot breakthrough profile for data w/ outliers."""

    input_file = "examples/input/NOM5_outliers.txt"

    data = np.array(np.genfromtxt(input_file, delimiter="\t"))

    time = np.array(data[3:, 0])
    effluent_concentrations = np.array(data[3:, 1:])
    feed_concentrations = np.array(data[1:3, 1:])

    species = np.genfromtxt(input_file, delimiter="\t", dtype=str)[0, 1:]

    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07 * 2  # cm
    pore_diffusion = 5e-6  # cm2/s
    surface_diffusion = 5e-10  # cm2/s

    # column
    axial_diffusion = 0  # cm2/s
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/L
    flow_rate = 40  # cm3/s

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
    )

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        diameter=diameter,
        bulk_density=bulk_density,
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=surface_diffusion,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=feed_concentrations,
        flow_rate=flow_rate,
        time=time,
        effluent_concentrations=effluent_concentrations,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    valid_data = breakthrough.valid_data()

    for i, name in enumerate(species):
        if not breakthrough.has_breakthrough()[i]:
            print(f"{name}: no significant breakthrough.")
            continue

        # NaN-filtered data used for outlier identification
        valid_time, valid_concentration = valid_data[i]

        outliers, _, removed = identify_curve_outliers(valid_time, valid_concentration)

        # Plot original data, including NaNs
        line = ax.plot(
            time,
            effluent_concentrations[:, i],
            label=name,
        )[0]

        # Plot identified outliers
        if np.any(outliers):
            ax.scatter(
                valid_time[outliers],
                valid_concentration[outliers],
                marker="x",
                s=60,
                color=line.get_color(),
                label=f"{name} outlier",
                zorder=3,
            )

            for result in removed:
                print(
                    f"{name}: removed time={result['time']:.0f}, "
                    f"value={result['value']:.5f}, "
                    f"predicted={result['predicted']:.5f}, "
                    f"absolute error={result['absolute_error']:.5f}"
                )

    ax.set_title("Experimental Data w/ outliers")
    ax.set_xlabel("BVs")
    ax.set_ylabel("C")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=1, bbox_to_anchor=(1, 1), loc="upper left")

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
