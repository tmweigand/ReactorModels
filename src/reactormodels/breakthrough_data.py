"""breakthrough_data.py"""

import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class Breakthrough:
    """Experimental breakthrough parameters."""

    """water_matrix: label for water that tested."""
    """feed_concentrations: feed concentrations taken over duration of experiment."""

    def __init__(
        self,
        compound: str,
        water_matrix: str,
        feed_concentrations: np.ndarray,
        bed_volumes: np.ndarray,
        time: np.ndarray,
        effluent_concentrations: np.ndarray,
        flow_rate: float,
    ):
        assert compound is not None, "No compound provided"

        if bed_volumes is not None:
            assert np.all(np.isfinite(bed_volumes)), "Bed volume data contains NaN(s)"

        if time is not None:
            assert np.all(np.isfinite(time)), "Time data contains NaN(s)"

        if feed_concentrations is not None:
            assert np.all(
                np.isfinite(feed_concentrations)
            ), "Feed concentration data contains NaN(s)"

        if effluent_concentrations is not None:
            assert np.all(
                np.isfinite(effluent_concentrations)
            ), "Effluent concentration data contains NaN(s)"

        self.compound = compound
        self.water_matrix = water_matrix
        self.feed_concentrations = np.array(feed_concentrations)
        self.bed_volumes = np.array(bed_volumes)
        self.time = np.array(time)
        self.effluent_concentrations = np.array(effluent_concentrations)
        self.flow_rate = flow_rate

    def mean_feed_concentration(self) -> float:
        """Mean feed concentration during experiment."""
        assert (
            self.feed_concentrations is not None
        ), "Feed concentration data does not exist"
        return np.mean(self.feed_concentrations)

    def normalize_concentration(self) -> np.ndarray:
        """Normalize effluent concentration by mean feed concentration."""
        assert math.isfinite(
            self.mean_feed_concentration()
        ), "Mean feed concentration was not computed"
        assert (
            self.effluent_concentrations is not None
        ), "Effluent concentration data does not exist"
        return self.effluent_concentrations / self.mean_feed_concentration()

    def empty_bed_contact_time(self, column_volume: float) -> float:
        """Calculate empty bed contact time."""
        assert self.flow_rate is not None, "No flow rate provided"
        return column_volume / self.flow_rate

    def time_to_bed_volumes(self, column_volume: float) -> np.ndarray:
        """Convert time to bed volumes."""
        return self.time / self.empty_bed_contact_time(column_volume)

    def bed_volumes_to_time(self, column_volume: float) -> np.ndarray:
        """Convert bed volumes to time."""
        return self.bed_volumes * self.empty_bed_contact_time(column_volume)

    def breakthrough_threshold(self, column_volume: float, threshold: float) -> float:
        """Calculate bed volumes to specific breakthrough threshold."""
        assert (
            self.bed_volumes is not None
        ), "Bed volume data must be converted from time or does not exist."
        if self.bed_volumes is not None:
            bed_volumes = self.bed_volumes
        else:
            bed_volumes = self.time_to_bed_volumes(column_volume)
        df = pd.DataFrame(
            {
                "bed_volumes": bed_volumes,
                "normalized_concentrations": self.normalize_concentration(),
            }
        )
        above_threshold = (df["normalized_concentrations"] >= threshold).idxmax()

        bv_below = df.loc[above_threshold - 1, "bed_volumes"]
        normal_below = df.loc[above_threshold - 1, "normalized_concentrations"]

        bv_above = df.loc[above_threshold, "bed_volumes"]
        normal_above = df.loc[above_threshold, "normalized_concentrations"]

        return bv_below + (threshold - normal_below) * (bv_above - bv_below) / (
            normal_above - normal_below
        )

    def summary(self, column_volume: float, threshold: float) -> str:
        """Summarize breakthrough data."""
        n_c = self.normalize_concentration()
        ebct = self.empty_bed_contact_time(column_volume)
        bv_value = self.breakthrough_threshold(column_volume, threshold)
        summary = (
            f"Compound: {self.compound}\n"
            f"Water matrix: {self.water_matrix}\n"
            f"Mean feed concentration: {self.mean_feed_concentration():.3f}\n"
            f"Normalized concentrations: {np.array2string(n_c, precision=3)}\n"
            f"Flow rate: {self.flow_rate:.3f}\n"
            f"Bed volumes: {self.bed_volumes}\n"
            f"Time: {self.time}\n"
            f"Empty bed contact time: {ebct:.3f}\n"
            f"Bed volumes to {100*threshold:.0f}% breakthrough: {bv_value:.1f}"
        )
        return summary

    def plot_normalized_data(self, out_dir):
        """Generate and save plots of normalized data."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()

        plt.plot(self.bed_volumes, self.normalize_concentration(), marker="o")

        plt.xlabel("Bed Volumes")
        plt.ylabel("C/Co")
        plt.title(f"{self.compound} - {self.water_matrix}")

        plt.savefig(
            out_dir / f"{self.compound} - {self.water_matrix}.png", bbox_inches="tight"
        )
        plt.close()
