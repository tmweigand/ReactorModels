"""breakthrough.py"""

import numpy as np

from .column import Column
from .chemical import Chemical


class Breakthrough:
    """Collect information from experimental breakthrough data.

    water_matrix: label for water that was tested.
    feed_concentrations: feed concentrations over the experiment duration.
    """

    def __init__(
        self,
        column: Column,
        chemical: Chemical,
        feed_concentrations: float | np.ndarray,
        initial_concentration: float = 0.0,
        initial_mass_fraction: float = 0.0,
        flow_rate: float | None = None,
        effluent_concentrations: np.ndarray | None = None,
        bed_volumes: np.ndarray | None = None,
        time: np.ndarray | None = None,
        superficial_velocity: float | None = None,
    ) -> None:

        if time is None and bed_volumes is None:
            raise ValueError("Either time or bed_volumes must be provided.")

        assert np.all(
            np.isfinite(feed_concentrations)
        ), "Feed concentration data contains NaN"

        if effluent_concentrations is not None:
            effluent_concentrations = np.asarray(
                effluent_concentrations,
                dtype=float,
            )

            self.effluent_concentrations = effluent_concentrations

            if time is not None and len(effluent_concentrations) != len(time):
                raise ValueError(
                    "The number of concentration measurements must match "
                    "the number of time points."
                )

        if feed_concentrations is not None:
            feed_concentrations = np.asarray(
                feed_concentrations,
                dtype=float,
            )

        self.column = column
        self.chemical = chemical
        self.feed_concentrations = feed_concentrations
        self.initial_concentration = initial_concentration
        self.initial_mass_fraction = initial_mass_fraction

        self._time: np.ndarray | None = None
        self._bed_volumes: np.ndarray | None = None

        self.bed_volumes = bed_volumes
        self.time = time
        self.flow_rate = flow_rate
        self._superficial_velocity = superficial_velocity
        self.initial_mass_fraction = initial_mass_fraction

    @property
    def time(self) -> np.ndarray | None:
        """Time data, derived from bed_volumes if not directly supplied."""
        if self._time is None and self._bed_volumes is not None:
            self._time = self._bed_volumes * self.empty_bed_contact_time()
        return self._time

    @time.setter
    def time(self, value: np.ndarray | None) -> None:
        if value is not None:
            value = np.asarray(value, dtype=float)
            assert np.all(np.isfinite(value)), "Time data contains NaN"
        self._time = value

    @property
    def bed_volumes(self) -> np.ndarray | None:
        """Bed volumes, derived from time if not directly supplied."""
        if self._bed_volumes is None and self._time is not None:
            self._bed_volumes = self._time / self.empty_bed_contact_time()
        return self._bed_volumes

    @bed_volumes.setter
    def bed_volumes(self, value: np.ndarray | None) -> None:
        if value is not None:
            value = np.asarray(value, dtype=float)
            assert np.all(np.isfinite(value)), "Bed volume data contains NaN"
        self._bed_volumes = value

    def valid_data(self):
        """Return time and effluent concentration values without NaNs."""
        if self.effluent_concentrations is None:
            raise ValueError("Must supply effluent concentration data.")
        mask = np.isfinite(self.time) & np.isfinite(self.effluent_concentrations)
        return (
            self.time[mask],
            self.effluent_concentrations[mask],
        )

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
        flow_rate: float, cross_section_area: float
    ) -> float:
        """Calculate superficial velocity, v = Q / A."""
        assert flow_rate > 0, f"flow_rate must be positive, got {flow_rate}"
        assert (
            cross_section_area > 0
        ), f"cross_section_area must be positive, got {cross_section_area}"
        return flow_rate / cross_section_area

    @staticmethod
    def calculate_interstitial_velocity(
        superficial_velocity: float, porosity: float
    ) -> float:
        """Calculate interstitial velocity, u = v / porosity."""
        assert (
            superficial_velocity > 0
        ), f"superficial_velocity must be positive, got {superficial_velocity}"
        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"
        return superficial_velocity / porosity

    @property
    def superficial_velocity(self) -> float:
        """Superficial velocity, either supplied directly or computed from flow_rate."""
        if self._superficial_velocity is None:
            if self.flow_rate is None:
                raise ValueError(
                    "Must supply either flow_rate or superficial_velocity."
                )
            self._superficial_velocity = self.calculate_superficial_velocity(
                self.flow_rate,
                self.column.cross_section_area(),
            )
        return self._superficial_velocity

    @property
    def interstitial_velocity(self) -> float:
        """Interstitial velocity, derived from superficial velocity and porosity."""
        return self.calculate_interstitial_velocity(
            self.superficial_velocity, self.column.porosity
        )

    def empty_bed_contact_time(
        self,
        column_volume: float | None = None,
    ) -> float:
        """Calculate empty-bed contact time."""
        if column_volume is None:
            column_volume = self.column.column_volume()

        velocity = self.superficial_velocity

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
        """Convert bed volumes to time (and cache the result on self.time)."""
        if self._bed_volumes is not None:
            derived = self._bed_volumes * self.empty_bed_contact_time(column_volume)

            if self._time is not None:
                assert np.allclose(
                    derived,
                    self._time,
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
        """Convert time to bed volumes (and cache the result on self.bed_volumes)."""
        if self._time is not None:
            derived = self._time / self.empty_bed_contact_time(column_volume)

            if self._bed_volumes is not None:
                assert np.allclose(
                    derived,
                    self._bed_volumes,
                ), (
                    "Supplied bed_volumes are inconsistent with "
                    "time / empty_bed_contact_time."
                )

            self.bed_volumes = derived
            return derived

        if self.bed_volumes is not None:
            return self.bed_volumes

        raise ValueError("Must supply either bed_volumes or time.")

    def has_breakthrough(
        self,
        breakthrough_fraction: float = 0.2,
        n_points: int | None = None,
    ) -> bool:
        """Return True if C/C0 exceeds breakthrough threshold at end or run."""
        normalized_c = self.normalize_concentration()
        if n_points is None:
            n_points = len(normalized_c)
        elif not isinstance(n_points, int):
            raise TypeError("n_points must be an integer or None")
        return bool(np.any(normalized_c[-n_points:] >= breakthrough_fraction))

    @staticmethod
    def identify_curve_outliers(
        x_values,
        y_values,
        n_mad,
        window_size,
        max_outliers,
        baseline_threshold,
    ):
        """Identify outliers using iterative local moving median fits.

        1. Select a local window around each candidate point.
        2. Remove the candidate point individually and fit a moving median
        trend to the remaining points.
        3. Calculate the scaled median absolute deviation (MAD) from the window.
        4. Calculate the candidates's deviation from the local median.
        5. A point is considered an outlier if the deviation exceeds the specified
        number of scaled MADs.
        6. Remove the point with the largest normalized deviation.
        7. Repeat using the reduced dataset.
        """
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        if len(x_values) != len(y_values):
            raise ValueError("time and values must have the same length.")

        if window_size < 3:
            raise ValueError("window_size must be at least 3.")

        if n_mad < 0:
            raise ValueError(" must be non-negative.")

        # Work with original indices so that the final outlier mask
        # corresponds to the original input arrays.
        remaining = list(range(len(x_values)))

        removed = []
        iteration_results = []

        for iteration in range(max_outliers):
            iteration_results = []

            half_window = window_size // 2

            # Evaluate every point as a potential outlier.
            for position, original_index in enumerate(remaining):

                # Determine the local window around the candidate.
                start = max(0, position - half_window)
                end = min(len(remaining), position + half_window + 1)

                window_indices = remaining[start:end]

                # Exclude the candidate from the local statistics.
                fit_indices = [
                    index for index in window_indices if index != original_index
                ]

                # Calculate the local median.
                predicted = np.median(y_values[fit_indices])

                # Calculate the local median absolute deviation.
                mad = np.median(np.abs(y_values[fit_indices] - predicted))

                # Scale MAD to be comparable to standard deviation.
                scaled_mad = 1.4826 * mad

                # Exclude tightly clustered values from outlier check
                minimum_mad = 0.05 * predicted
                effective_mad = max(scaled_mad, minimum_mad)

                # Calculate how far the candidate is from the local median.
                deviation = abs(y_values[original_index] - predicted)

                # Points below the detection threshold are not considered outliers.
                baseline_value = y_values[original_index] < baseline_threshold

                # Determine whether the candidate is an outlier.
                is_outlier = not baseline_value and deviation > n_mad * effective_mad

                # Calculate how many scaled MADs away the candidate is.
                violation_ratio = deviation / effective_mad

                iteration_results.append(
                    {
                        "index": original_index,
                        "time": x_values[original_index],
                        "value": y_values[original_index],
                        "predicted": predicted,
                        "mad": mad,
                        "scaled_mad": effective_mad,
                        "deviation": deviation,
                        "violation_ratio": violation_ratio,
                        "is_outlier": is_outlier,
                    }
                )

            # Keep only points that exceed the MAD threshold.
            candidates = [
                result for result in iteration_results if result["is_outlier"]
            ]

            # Stop if no points exceed the threshold.
            if not candidates:
                break

            # Remove the point with the greatest MAD violation.
            worst = max(
                candidates,
                key=lambda r: r["violation_ratio"],
            )

            removed.append(
                {
                    "iteration": iteration + 1,
                    "index": worst["index"],
                    "time": worst["time"],
                    "value": worst["value"],
                    "predicted": worst["predicted"],
                    "mad": worst["mad"],
                    "scaled_mad": worst["scaled_mad"],
                    "deviation": worst["deviation"],
                    "violation_ratio": worst["violation_ratio"],
                }
            )

            print(
                f"removed time={worst['time']:.0f}, "
                f"value={worst['value']:.5f}, "
                f"predicted={worst['predicted']:.5f}, "
                f"scaled MAD={worst['scaled_mad']:.5f}, "
                f"deviation={worst['deviation']:.5f}, "
                f"threshold={n_mad * worst['scaled_mad']:.5f}"
            )

            # Remove the point using its original index.
            remaining.remove(worst["index"])

        # Construct the final outlier mask using the original indices.
        outlier = np.zeros(
            len(y_values),
            dtype=bool,
        )

        for result in removed:
            outlier[result["index"]] = True

        return outlier, iteration_results, removed

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
            f"Compound: {self.chemical.name}\n"
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
