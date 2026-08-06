"""media_class.py"""

import numpy as np


class Media:
    """Adsorbent (media) particle parameters."""

    def __init__(
        self,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        mean_diameter: float | None = None,
        bed_density: float | None = None,
        sphericity: float | None = None,
        particle_radius: float | None = None,
    ) -> None:

        self.particle_porosity = particle_porosity
        self.particle_density = particle_density
        self.mean_diameter = mean_diameter
        self.bed_density = bed_density
        self.sphericity = sphericity
        self.particle_radius = particle_radius

    def get_bed_density(self, bed_porosity: float) -> float:
        """Return bed density calculated from particle density and porosity."""
        assert (
            0 < bed_porosity < 1
        ), f"bed_porosity must be in (0, 1), got {bed_porosity}"

        if self.particle_density is None:
            raise ValueError("particle_density is required to calculate bed_density.")

        derived = (1.0 - bed_porosity) * self.particle_density

        if self.bed_density is not None:
            assert np.isclose(derived, self.bed_density), (
                f"Supplied bed_density {self.bed_density} is inconsistent "
                "with (1 - bed_porosity) * particle_density "
                f"= {derived:.4f}"
            )

        return derived
