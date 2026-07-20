"""media_class.py"""

import numpy as np


class Media:
    """Adsorbent (media) particle parameters."""

    def __init__(
        self,
        particle_porosity: float,
        particle_density: float,
        mean_diameter: float,
        bed_density: float | None = None,
        sphericity: float | None = None,
        particle_radius: float | None = None,
    ):
        assert (
            0 < particle_porosity < 1
        ), f"particle_porosity must be in (0, 1), got {particle_porosity}"
        assert (
            particle_density > 0
        ), f"particle_density must be positive, got {particle_density}"
        assert mean_diameter > 0, f"mean_diameter must be positive, got {mean_diameter}"

        if bed_density is not None:
            assert bed_density > 0, f"bed_density must be positive, got {bed_density}"

        self.particle_porosity = particle_porosity
        self.particle_density = particle_density
        self.mean_diameter = mean_diameter
        self.bed_density = bed_density
        self.sphericity = sphericity
        self.particle_radius = particle_radius

    def get_bed_density(self, bed_porosity: float) -> float:
        """Bed density (rho_b) = (1 - bed_porosity) * particle_density."""
        assert (
            0 < bed_porosity < 1
        ), f"bed_porosity must be in (0, 1), got {bed_porosity}"

        derived = (1 - bed_porosity) * self.particle_density

        if self.bed_density is not None:
            assert np.isclose(derived, self.bed_density), (
                f"Supplied bed_density {self.bed_density} inconsistent "
                f"with (1 - bed_porosity) * particle_density = {derived:.4f}"
            )

        return derived

    def total_porosity(self, bed_porosity: float) -> float:
        """Total porosity = bed_porosity + (1 - bed_porosity) * particle_porosity."""
        assert (
            0 < bed_porosity < 1
        ), f"bed_porosity must be in (0, 1), got {bed_porosity}"

        return bed_porosity + (1 - bed_porosity) * self.particle_porosity
