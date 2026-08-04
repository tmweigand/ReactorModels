"""breakthrough_data.py"""

import numpy as np

from .column_data import Column


class Breakthrough:
    """Collect information from experimental breakthrough data.

    water_matrix: label for water that was tested.
    feed_concentrations: feed concentrations over the experiment duration.
    """

    def __init__(
        self,
        feed_concentrations: float | np.ndarray,
        compound: str | None = None,
        flow_rate: float | None = None,
        effluent_concentrations: np.ndarray | None = None,
        bed_volumes: np.ndarray | None = None,
        time: np.ndarray | None = None,
        water_matrix: str | None = None,
        superficial_velocity: float | None = None,
        column: Column | None = None,
        **kwargs,
    ) -> None:
        if column is None:
            column = Column(**kwargs)
        elif kwargs:
            raise TypeError(
                "Do not provide Column keyword arguments when a column "
                "object is already supplied."
            )

        if time is None and bed_volumes is None:
            raise ValueError("Either time or bed_volumes must be provided.")

        if bed_volumes is not None:
            assert np.all(np.isfinite(bed_volumes)), "Bed volume data contains NaN"

        if time is not None:
            assert np.all(np.isfinite(time)), "Time data contains NaN"

        assert np.all(
            np.isfinite(feed_concentrations)
        ), "Feed concentration data contains NaN"

        if effluent_concentrations is not None:
            assert np.all(
                np.isfinite(effluent_concentrations)
            ), "Effluent concentration data contains NaN"

        self.column = column
        self.compound = compound

        self.water_matrix = (
            water_matrix
            if water_matrix is not None
            else (
                self.column.water.water_matrix
                if self.column.water is not None
                else None
            )
        )

        self.feed_concentrations = np.asarray(feed_concentrations)

        self.bed_volumes = None if bed_volumes is None else np.asarray(bed_volumes)

        self.time = None if time is None else np.asarray(time)

        self.effluent_concentrations = (
            None
            if effluent_concentrations is None
            else np.asarray(effluent_concentrations)
        )
        self.flow_rate = flow_rate

        self._superficial_velocity = superficial_velocity

    def mean_feed_concentration(self) -> float:
        """Determine mean feed concentration."""
        return float(np.mean(self.feed_concentrations))

    def normalize_concentration(self) -> np.ndarray:
        """Normalize effluent concentration by mean feed concentration."""
        if self.effluent_concentrations is None:
            raise ValueError("Must supply effluent concentration data.")

        return self.effluent_concentrations / self.mean_feed_concentration()

    @staticmethod
    def calculate_superficial_velocity(
        flow_rate: float,
        cross_section_area: float,
    ) -> float:
        """Calculate superficial velocity, v = Q / A."""
        assert flow_rate > 0, f"flow_rate must be positive, got {flow_rate}"

        assert cross_section_area > 0, (
            "cross_section_area must be positive, " f"got {cross_section_area}"
        )

        return flow_rate / cross_section_area

    @staticmethod
    def calculate_interstitial_velocity(
        superficial_velocity: float,
        porosity: float,
    ) -> float:
        """Calculate interstitial velocity, u = v / porosity."""
        assert superficial_velocity > 0, (
            "superficial_velocity must be positive, " f"got {superficial_velocity}"
        )

        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        return superficial_velocity / porosity

    def superficial_velocity(self) -> float:
        """Calculate or return superficial velocity."""
        if self._superficial_velocity is not None:
            return self._superficial_velocity

        if self.flow_rate is not None:
            return self.calculate_superficial_velocity(
                self.flow_rate,
                self.column.cross_section_area(),
            )

        raise ValueError("Must supply either flow_rate or superficial_velocity.")

    def interstitial_velocity(self) -> float:
        """Calculate interstitial velocity."""
        return self.calculate_interstitial_velocity(
            self.superficial_velocity(),
            self.column.porosity,
        )

    def peclet(self, diffusion: float) -> float:
        """Calculate the axial Peclet number."""
        assert diffusion > 0, f"diffusion must be positive, got {diffusion}"

        velocity = (
            self._superficial_velocity
            if self._superficial_velocity is not None
            else self.calculate_superficial_velocity(
                self.flow_rate,
                self.column.cross_section_area(),
            )
        )

        return velocity * self.column.length / (self.column.porosity * diffusion)

    def empty_bed_contact_time(
        self,
        column_volume: float | None = None,
    ) -> float:
        """Calculate empty-bed contact time."""
        if column_volume is None:
            column_volume = self.column.column_volume()

        velocity = (
            self._superficial_velocity
            if self._superficial_velocity is not None
            else self.calculate_superficial_velocity(
                self.flow_rate,
                self.column.cross_section_area(),
            )
        )

        if self.flow_rate is not None:
            flow_derived = column_volume / self.flow_rate

            velocity_derived = self.column.length / velocity

            assert np.isclose(
                flow_derived,
                velocity_derived,
            ), (
                f"Flow-derived EBCT {flow_derived:.4f} inconsistent "
                "with velocity-derived EBCT "
                f"= {velocity_derived:.4f}"
            )

            return flow_derived

        if self._superficial_velocity is not None:
            return self.column.length / velocity

        raise ValueError(
            "flow_rate or superficial_velocity is needed " "to calculate EBCT."
        )

    def bed_volumes_to_time(
        self,
        column_volume: float | None = None,
    ) -> np.ndarray:
        """Convert bed volumes to time."""
        if self.bed_volumes is not None:
            derived = self.bed_volumes * self.empty_bed_contact_time(column_volume)

            if self.time is not None:
                assert np.allclose(
                    derived,
                    self.time,
                ), (
                    "Supplied time is inconsistent with "
                    "bed_volumes * empty_bed_contact_time."
                )

            self.time = derived

            return derived

        if self.time is not None:
            return self.time

        raise ValueError("Must supply either time or bed_volumes.")

    def time_to_bed_volumes(
        self,
        column_volume: float | None = None,
    ) -> np.ndarray:
        """Convert time to bed volumes."""
        if self.time is not None:
            derived = self.time / self.empty_bed_contact_time(column_volume)

            if self.bed_volumes is not None:
                assert np.allclose(
                    derived,
                    self.bed_volumes,
                ), (
                    "Supplied bed_volumes are inconsistent with "
                    "time / empty_bed_contact_time."
                )

            return derived

        if self.bed_volumes is not None:
            return self.bed_volumes

        raise ValueError("Must supply either bed_volumes or time.")

    def has_breakthrough(
        self,
        breakthrough_fraction: float = 0.01,
    ) -> bool:
        """Return True if C/C0 reaches the breakthrough fraction."""
        return bool(np.any(self.normalize_concentration() >= breakthrough_fraction))

    def breakthrough_threshold(
        self,
        column_volume_or_threshold: float | None = None,
        threshold: float | None = None,
        return_index: bool = False,
    ) -> float | tuple[float, int]:
        """Calculate bed volumes to a breakthrough threshold.

        Supports both:
            breakthrough_threshold(threshold)
            breakthrough_threshold(column_volume, threshold)
        """
        if threshold is None:
            if column_volume_or_threshold is None:
                raise ValueError("A breakthrough threshold must be supplied.")

            threshold = column_volume_or_threshold
            column_volume = None

        else:
            column_volume = column_volume_or_threshold

        assert 0 <= threshold <= 1, (
            f"threshold must be between 0 and 1, " f"got {threshold}"
        )

        if self.bed_volumes is None:
            self.bed_volumes = self.time_to_bed_volumes(column_volume)

        concentrations = self.normalize_concentration()

        mask = concentrations >= threshold

        if not np.any(mask):
            raise ValueError(
                f"Threshold {threshold} is never reached " "by the breakthrough curve."
            )

        above_idx = int(np.argmax(mask))

        if above_idx == 0:
            breakthrough_bv = float(self.bed_volumes[0])

            if return_index:
                return (
                    breakthrough_bv,
                    above_idx,
                )

            return breakthrough_bv

        bv_above = self.bed_volumes[above_idx]

        c_above = concentrations[above_idx]

        bv_below = self.bed_volumes[above_idx - 1]

        c_below = concentrations[above_idx - 1]

        breakthrough_bv = float(
            bv_below
            + ((threshold - c_below) * (bv_above - bv_below) / (c_above - c_below))
        )

        if return_index:
            return (
                breakthrough_bv,
                above_idx,
            )

        return breakthrough_bv

    def summary(
        self,
        threshold: float,
    ) -> str:
        """Summarize breakthrough data."""
        normalized = self.normalize_concentration()

        ebct = self.empty_bed_contact_time()

        bv_value = self.breakthrough_threshold(threshold)

        flow_rate_text = (
            f"{self.flow_rate:.3f}" if self.flow_rate is not None else "Not supplied"
        )

        return (
            f"Compound: {self.compound}\n"
            f"Water matrix: {self.water_matrix}\n"
            "Mean feed concentration: "
            f"{self.mean_feed_concentration():.3f}\n"
            "Normalized concentrations: "
            f"{np.array2string(normalized, precision=3)}\n"
            f"Flow rate: {flow_rate_text}\n"
            f"Bed volumes: {self.bed_volumes}\n"
            f"Time: {self.time}\n"
            f"Empty bed contact time: {ebct:.3f}\n"
            f"Bed volumes to "
            f"{100 * threshold:.0f}% "
            f"breakthrough: {bv_value:.1f}"
        )
