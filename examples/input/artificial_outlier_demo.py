from reactormodels.Input import identify_curve_outliers
import reactormodels

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_demo(
    show: bool = False,
    save_path: str | Path = "data_out/input/artificial_outlier_demo.png",
):
    """Plot breakthrough profile for PSDM data w/ outliers."""
    # particle
    particle_porosity = 0.5
    particle_density = 600  # g/L
    particle_diameter = 0.07 * 2  # cm
    pore_diffusion = 5e-6  # cm2/s
    surface_diffusion = 5e-10  # cm2/s
    k_film = 0.075  # cm/s
    K = [100, 1000]

    # column
    axial_diffusion = 0  # cm2/s
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/L
    flow_rate = 40  # cm3/s
    feed_concentrations = 1
    time = np.array(np.loadtxt("examples/ads_time.txt", skiprows=0))
    t_eval = time * 60  # s
    t_eval = t_eval[::5]

    species = ["PFOA", "PFOS"]

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

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )

    for i, name in enumerate(species):
        breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=feed_concentrations,
            flow_rate=flow_rate,
            time=t_eval,
        )

        isotherm = reactormodels.models.LinearIsotherm(K=K[i])

        model = reactormodels.models.PSDM(
            breakthrough=breakthrough,
            isotherm=isotherm,
            column_numerics=column_numerics,
            particle_numerics=particle_numerics,
            k_film=k_film,
        )
        z, r, C, Cp = model.solve()

        # add noise every third point
        rng = np.random.default_rng(0)
        c = np.maximum(C[:, -1], 0)
        c_outliers = c.copy()

        indices = np.arange(0, c.size, 3)

        # make proportional to concentration
        noise = rng.normal(0, 0.7 * c[indices], size=indices.size)
        c_outliers[indices] += noise

        # Add NaNs at selected points
        nan_indices = np.array([5, 10])

        c_outliers[nan_indices] = np.nan

        psdm_breakthrough = reactormodels.Breakthrough(
            column=column,
            chemical=chemical,
            feed_concentrations=feed_concentrations,
            flow_rate=flow_rate,
            time=t_eval,
            effluent_concentrations=c_outliers,
        )

        if not psdm_breakthrough.has_breakthrough(n_points=3):
            print(f"{name}: no significant breakthrough.")
            continue

        # clean data of NaNs
        valid_time, valid_concentration = psdm_breakthrough.valid_data()

        outliers, _, removed = identify_curve_outliers(
            valid_time,
            valid_concentration,
            absolute_tolerance=0.03,
            relative_tolerance=0.4,
            window_size=5,
            max_outliers=10,
            baseline_threshold=0.01,
        )

        # Normalize by mean feed concentration
        normalized_concentration = (
            valid_concentration / psdm_breakthrough.mean_feed_concentration()
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

        # Format this PFAS plot
        ax.set_title(f"PSDM Data w/ Outliers")
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
