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
    K = 100

    # column
    axial_diffusion = 0  # cm2/s
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/L
    flow_rate = 40  # cm3/s
    feed_concentrations = np.array([[1]])
    time = np.array(np.loadtxt("examples/ads_time.txt", skiprows=0))
    t_eval = time * 60  # s
    t_eval = t_eval[::5]

    isotherm = reactormodels.models.LinearIsotherm(K=K)

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
        time=t_eval,
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
    c = C[:, -1]
    c_outliers = c.copy()

    indices = np.arange(0, c.size, 3)

    # make proportional to concentration
    noise = rng.normal(0, 0.7 * c[indices], size=indices.size)
    c_outliers[indices] += noise

    fig, ax = plt.subplots(figsize=(8, 5))

    # apply outlier identification helper
    outliers, _, removed = identify_curve_outliers(
        t_eval,
        c_outliers,
        absolute_tolerance=0.02,
        relative_tolerance=0.4,
        window_size=5,
        max_outliers=10,
    )

    # Plot non-outliers
    ax.plot(
        t_eval[~outliers],
        c_outliers[~outliers],
        marker="o",
        label="Data",
        linestyle="None",
        markersize=7.5,
    )[0]

    # Plot identified outliers
    if np.any(outliers):
        ax.scatter(
            t_eval[outliers],
            c_outliers[outliers],
            marker="x",
            s=60,
            color="tab:red",
            label="Outlier",
            zorder=3,
        )
        ax.scatter(
            t_eval[outliers],
            c_outliers[outliers],
            marker="o",
            s=60,
            color="tab:blue",
            alpha=0.5,
            zorder=2,
        )

        for result in removed:
            print(
                f"removed time={result['time']:.0f}, "
                f"value={result['value']:.5f}, "
                f"predicted={result['predicted']:.5f}, "
                f"absolute error={result['absolute_error']:.5f}"
            )

        # Format this PFAS plot
        ax.set_title(f"PSDM Data w/ Outliers")
        ax.set_xlabel("BVs")
        ax.set_ylabel(r"$C/C_0$")
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


if __name__ == "__main__":
    run_demo()
