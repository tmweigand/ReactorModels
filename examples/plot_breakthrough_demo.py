import reactormodels
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run_demo(
    show: bool = True, save_path: str | Path = "data_out/plot_breakthrough_demo.png"
):
    """Plot normalized breakthrough curves."""
    feed_concentrations = [101, 103]
    compound = "PFOA"
    water_matrix = "Surface_Water"
    bed_volumes = [0, 100, 200, 250, 350, 400, 500, 600]
    time = np.array(bed_volumes) * np.pi / 8
    effluent_concentrations = [0, 0, 10, 15, 25, 80, 100, 100]
    flow_rate = 5

    breakthrough = reactormodels.Breakthrough(
        feed_concentrations=feed_concentrations,
        compound=compound,
        water_matrix=water_matrix,
        time=time,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
    )

    plt.figure()

    plt.plot(
        breakthrough.bed_volumes, breakthrough.normalize_concentration(), marker="o"
    )

    plt.xlabel("Bed Volumes")
    plt.ylabel("C/C₀")
    plt.title(f"{breakthrough.compound} - {breakthrough.water_matrix}")

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
