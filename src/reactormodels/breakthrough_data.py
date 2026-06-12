"""breakthrough_data.py"""

import numpy as np
import math


class Breakthrough:
    """Collect information from experimental breakthrough data.

    water_matrix: label for water that tested.
    feed_concentrations: feed concentrations taken over duration of experiment.
    """

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

        concentrations = self.normalize_concentration()

        mask = concentrations >= threshold

        if not np.any(mask):
            raise ValueError(
                f"Threshold {threshold} is never reached by the breakthrough curve."
            )

        above_idx = np.argmax(mask)
        if above_idx == 0:
            return bed_volumes[0]

        bv_above = bed_volumes[above_idx]
        c_above = concentrations[above_idx]

        bv_below = bed_volumes[above_idx - 1]
        c_below = concentrations[above_idx - 1]

        return bv_below + (threshold - c_below) * (bv_above - bv_below) / (
            c_above - c_below
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
