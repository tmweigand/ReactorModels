from reactormodels.Input import identify_curve_outliers
import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/input/experimental_outlier_demo.png",
):
    """Plot breakthrough profile for data w/ outliers."""

    input_file = "examples/input/experimental_outliers.txt"

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

    for i, name in enumerate(species):
        breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=feed_concentrations[:, i],
            flow_rate=flow_rate,
            time=time,
            effluent_concentrations=effluent_concentrations[:, i],
        )

        if not breakthrough.has_breakthrough(n_points=3):
            print(f"{name}: no significant breakthrough.")
            continue

        # clean data of NaNs
        valid_time, valid_concentration = breakthrough.valid_data()

        outliers, _, removed = identify_curve_outliers(
            valid_time,
            valid_concentration,
            absolute_tolerance=0.03,
            relative_tolerance=0.5,
            window_size=5,
            max_outliers=10,
        )

        # Normalize by mean feed concentration
        normalized_concentration = (
            valid_concentration / breakthrough.mean_feed_concentration()
        )

        # Create a new plot for this PFAS
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot non-outliers
        ax.plot(
            valid_time[~outliers],
            normalized_concentration[~outliers],
            label=name,
            marker="o",
            linestyle="None",
            markersize=7.5,
            zorder=2,
        )

        # Plot identified outliers
        if np.any(outliers):
            ax.scatter(
                valid_time[outliers],
                normalized_concentration[outliers],
                marker="o",
                s=60,
                color="tab:blue",
                zorder=2,
                alpha=0.5,
            )
            ax.scatter(
                valid_time[outliers],
                normalized_concentration[outliers],
                marker="x",
                s=60,
                color="tab:red",
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

        # Format this PFAS plot
        ax.set_title(f"{name}: Experimental Data w/ Outliers")
        ax.set_xlabel("BVs")
        ax.set_ylabel(r"$C/C_0$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncols=1)

        fig.tight_layout()

        # Save one file for each PFAS
        save_path = Path(save_path)

        # Add the PFAS name to the filename
        pfas_save_path = save_path.parent / f"{name}_outliers{save_path.suffix}"

        pfas_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            pfas_save_path,
            dpi=150,
        )

        print(f"Saved plot to {pfas_save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    run_demo()
