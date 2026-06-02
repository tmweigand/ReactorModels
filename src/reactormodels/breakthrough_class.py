"""Experimental breakthrough curve data model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class ExperimentalBreakthroughCurve:
    """Experimental breakthrough curve for one chemical."""

    def __init__(
        self,
        chemical_name,
        effluent_concentrations,
        feed_concentration,
        times=None,
        bed_volumes=None,
    ):
        self.chemical_name = chemical_name
        self.effluent_concentrations = np.asarray(
            effluent_concentrations,
            dtype=float,
        )
        self.feed_concentration = float(feed_concentration)

        self.times = None if times is None else np.asarray(times, dtype=float)
        self.bed_volumes = (
            None if bed_volumes is None else np.asarray(bed_volumes, dtype=float)
        )

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate breakthrough curve inputs."""

        if self.feed_concentration <= 0:
            raise ValueError("feed_concentration must be greater than zero.")

        if self.times is None and self.bed_volumes is None:
            raise ValueError("Either times or bed_volumes must be provided.")

        if self.times is not None:
            if self.times.shape != self.effluent_concentrations.shape:
                raise ValueError(
                    "times and effluent_concentrations must have the same shape."
                )

        if self.bed_volumes is not None:
            if self.bed_volumes.shape != self.effluent_concentrations.shape:
                raise ValueError(
                    "bed_volumes and effluent_concentrations must have the same shape."
                )

    @property
    def normalized_concentrations(self):
        """Return normalized effluent concentrations, C/C0."""

        return self.effluent_concentrations / self.feed_concentration

    def has_breakthrough(self, breakthrough_fraction=0.05):
        """Return True if C/C0 reaches the breakthrough fraction."""

        return bool(np.any(self.normalized_concentrations >= breakthrough_fraction))

    def breakthrough_index(self, breakthrough_fraction=0.05):
        """Return the first index where breakthrough occurs."""

        indices = np.where(self.normalized_concentrations >= breakthrough_fraction)[0]

        if indices.size == 0:
            return None

        return int(indices[0])

    def breakthrough_point(self, breakthrough_fraction=0.05):
        """Return the first breakthrough point."""

        index = self.breakthrough_index(breakthrough_fraction)

        if index is None:
            return None

        point = {
            "chemical_name": self.chemical_name,
            "effluent_concentration": self.effluent_concentrations[index],
            "normalized_concentration": self.normalized_concentrations[index],
        }

        if self.times is not None:
            point["time"] = self.times[index]

        if self.bed_volumes is not None:
            point["bed_volume"] = self.bed_volumes[index]

        return point

    def summary_line(self, breakthrough_fraction=0.05):
        """Return a one-line summary for the chemical."""

        point = self.breakthrough_point(breakthrough_fraction)

        if point is None:
            location = "none"
        elif "bed_volume" in point:
            location = f"bed_volume={point['bed_volume']:.6g}"
        else:
            location = f"time={point['time']:.6g}"

        return (
            f"{self.chemical_name}: "
            f"points={self.effluent_concentrations.size}, "
            f"feed_concentration={self.feed_concentration:.6g}, "
            f"max_normalized_concentration="
            f"{np.max(self.normalized_concentrations):.6g}, "
            f"breakthrough={self.has_breakthrough(breakthrough_fraction)}, "
            f"breakthrough_location={location}"
        )

    def plot(self, x_axis="bed_volume", save_path=None, show=True):
        """Plot C/C0 versus bed volume or time."""

        if x_axis == "bed_volume":
            if self.bed_volumes is None:
                raise ValueError("bed_volumes were not provided.")

            x_values = self.bed_volumes
            x_label = "Bed Volume"

        elif x_axis == "time":
            if self.times is None:
                raise ValueError("times were not provided.")

            x_values = self.times
            x_label = "Time"

        else:
            raise ValueError("x_axis must be 'bed_volume' or 'time'.")

        plt.figure()
        plt.plot(x_values, self.normalized_concentrations, marker="o")
        plt.xlabel(x_label)
        plt.ylabel("C/C0")
        plt.title(self.chemical_name)
        plt.grid(True)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()