"""breakthrough_data.py"""

import numpy as np
from .column_data import Column


class Breakthrough:
    """Collect information from experimental breakthrough data.

    water_matrix: label for water that tested.
    feed_concentrations: feed concentrations taken over duration of experiment.
    """

    def __init__(
        self,
        column: Column,
        feed_concentrations: float | np.ndarray,
        flow_rate: float | None = None,
        effluent_concentrations: np.ndarray | None = None,
        compound: str | None = None,
        bed_volumes: np.ndarray | None = None,
        time: np.ndarray | None = None,
        water_matrix: str | None = None,
        superficial_velocity: float | None = None,
    ):
        self.column = column
        self.compound = compound
        self.water_matrix = water_matrix
        self.feed_concentrations = np.array(feed_concentrations)
        self.bed_volumes = bed_volumes
        self.time = time
        self.effluent_concentrations = effluent_concentrations
        self.flow_rate = flow_rate
        self.superficial_velocity = superficial_velocity

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

    def mean_feed_concentration(self) -> float:
        """Determine mean feed concentration"""
        return np.mean(self.feed_concentrations)

    def normalize_concentration(self) -> np.ndarray:
        """Normalize effluent concentration by mean feed concentration."""
        if self.effluent_concentrations is not None:
            return self.effluent_concentrations / self.mean_feed_concentration()
        else:
            raise ValueError("Must supply effluent concentration data.")

    def get_superficial_velocity(self) -> float:
        """Superficial velocity v = Q / A."""
        if self.flow_rate is not None:
            derived = self.flow_rate / self.column.cross_section_area()
            if self.superficial_velocity is not None:
                assert np.isclose(derived, self.superficial_velocity), (
                    f"Supplied superficial_velocity {self.superficial_velocity} "
                    f"inconsistent with flow_rate / cross_section_area = {derived:.4f}"
                )
            return derived
        elif self.superficial_velocity is not None:
            return self.superficial_velocity
        else:
            raise ValueError("Must supply either superficial_velocity or flow_rate.")

    def interstitial_velocity(self) -> float:
        """Interstitial velocity u = v / porosity."""
        if self.column.porosity is not None:
            return self.get_superficial_velocity() / self.column.porosity
        else:
            raise ValueError("Must supply porosity.")

    def peclet(self, diffusion: float) -> float:
        """Axial Peclet number Pe = v * L / (porosity * diffusion).

        Pe >> 1: advection dominated.
        Pe << 1: diffusion dominated.
        """
        if self.column.porosity is not None:
            v = self.get_superficial_velocity()
            return v * self.column.length / (self.column.porosity * diffusion)
        else:
            raise ValueError("Must supply porosity.")

    def empty_bed_contact_time(self) -> float:
        """Calculate empty bed contact time."""
        if self.flow_rate is not None:
            flow_derived = self.column.column_volume() / self.flow_rate
            velocity_derived = self.column.length / self.get_superficial_velocity()
            assert np.isclose(flow_derived, velocity_derived), (
                f"Flow-derived EBCT {flow_derived:.4f} inconsistent"
                f"with velocity-derived EBCT = {velocity_derived:.4f}"
            )
            return flow_derived
        else:
            raise ValueError(
                "Flow_rate or superficial_velocity needed to calculate EBCT."
            )

    def time_to_bed_volumes(self) -> np.ndarray:
        """Convert time to bed volumes."""
        if self.time is not None:
            derived = self.time / self.empty_bed_contact_time()
            if self.bed_volumes is not None:
                assert np.allclose(derived, self.bed_volumes), (
                    f"Supplied bed_volumes {self.bed_volumes} inconsistent "
                    f"with time / empty_bed_contact_time = {derived:.4f}"
                )
            return derived
        elif self.bed_volumes is not None:
            return self.bed_volumes
        else:
            raise ValueError("Must supply either bed_volumes or time.")

    def bed_volumes_to_time(self) -> np.ndarray:
        """Convert bed volumes to time."""
        if self.bed_volumes is not None:
            derived = self.bed_volumes * self.empty_bed_contact_time()
            if self.time is not None:
                assert np.allclose(derived, self.time), (
                    f"Supplied time {self.time} inconsistent "
                    f"with bed_volumes * empty_bed_contact_time = {derived:.4f}"
                )
            return derived
        elif self.time is not None:
            return self.time
        else:
            raise ValueError("Must supply either time or bed_volumes.")

    def has_breakthrough(self, breakthrough_fraction: float = 0.01) -> bool:
        """Return True if C/C0 reaches the breakthrough fraction."""
        return bool(np.any(self.normalize_concentration() >= breakthrough_fraction))

    def breakthrough_threshold(
        self, threshold: float, return_index: bool = False
    ) -> float | tuple[float, int]:
        """Calculate bed volumes to specific breakthrough threshold."""
        if self.bed_volumes is None:
            self.time_to_bed_volumes()
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

    def summary(self, threshold: float) -> str:
        """Summarize breakthrough data."""
        n_c = self.normalize_concentration()
        ebct = self.empty_bed_contact_time()
        bv_value = self.breakthrough_threshold(threshold)
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
