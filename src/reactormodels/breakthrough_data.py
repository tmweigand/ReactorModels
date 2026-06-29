"""breakthrough_data.py"""

import numpy as np


class Breakthrough:
    """Collect information from experimental breakthrough data.

    water_matrix: label for water that tested.
    feed_concentrations: feed concentrations taken over duration of experiment.
    """

    def __init__(
        self,
        compound: str,
        feed_concentrations: float | np.ndarray,
        effluent_concentrations: np.ndarray,
        flow_rate: float,
        bed_volumes: np.ndarray | None = None,
        time: np.ndarray | None = None,
        water_matrix: str | None = None,
    ):

        if time is None and bed_volumes is None:
            raise ValueError("Either time or bed_volumes must be provided.")

        if bed_volumes is not None:
            assert np.all(np.isfinite(bed_volumes)), "Bed volume data contains NaN"

        if time is not None:
            assert np.all(np.isfinite(time)), "Time data contains NaN"

        if feed_concentrations is not None:
            assert np.all(
                np.isfinite(feed_concentrations)
            ), "Feed concentration data contains NaN"

        if effluent_concentrations is not None:
            assert np.all(
                np.isfinite(effluent_concentrations)
            ), "Effluent concentration data contains NaN"

        self.compound = compound
        self.water_matrix = water_matrix
        self.feed_concentrations = np.array(feed_concentrations)
        self.bed_volumes = bed_volumes
        self.time = time
        self.effluent_concentrations = effluent_concentrations
        self.flow_rate = flow_rate

    def mean_feed_concentration(self) -> float:
        """Determine mean feed concentration"""
        return np.mean(self.feed_concentrations)

    def normalize_concentration(self) -> np.ndarray:
        """Normalize effluent concentration by mean feed concentration."""
        return self.effluent_concentrations / self.mean_feed_concentration()

    def empty_bed_contact_time(self, column_volume: float) -> float:
        """Calculate empty bed contact time."""
        return column_volume / self.flow_rate

    def time_to_bed_volumes(self, column_volume: float) -> None:
        """Convert time to bed volumes."""
        assert self.time is not None, "time is not provided"
        self.bed_volumes = self.time / self.empty_bed_contact_time(column_volume)

    def bed_volumes_to_time(self, column_volume: float) -> None:
        """Convert bed volumes to time."""
        assert self.bed_volumes is not None, "bed_volumes is not provided"
        self.time = self.bed_volumes * self.empty_bed_contact_time(column_volume)

    def has_breakthrough(self, breakthrough_fraction: float = 0.01) -> bool:
        """Return True if C/C0 reaches the breakthrough fraction."""
        return bool(np.any(self.normalize_concentration() >= breakthrough_fraction))

    def breakthrough_threshold(
        self, column_volume: float, threshold: float, return_index: bool = False
    ) -> float | tuple[float, int]:
        """Calculate bed volumes to specific breakthrough threshold."""
        if self.bed_volumes is None:
            self.time_to_bed_volumes(column_volume)
        assert self.bed_volumes is not None

        concentrations = self.normalize_concentration()

        mask = concentrations >= threshold

        if not np.any(mask):
            raise ValueError(
                f"Threshold {threshold} is never reached by the breakthrough curve."
            )

        above_idx = int(np.argmax(mask))
        if above_idx == 0:
            return self.bed_volumes[0]

        bv_above = self.bed_volumes[above_idx]
        c_above = concentrations[above_idx]

        bv_below = self.bed_volumes[above_idx - 1]
        c_below = concentrations[above_idx - 1]

        breakthrough_bv = float(
            bv_below
            + (threshold - c_below) * (bv_above - bv_below) / (c_above - c_below)
        )

        if return_index:
            return (breakthrough_bv, above_idx)

        return breakthrough_bv

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
