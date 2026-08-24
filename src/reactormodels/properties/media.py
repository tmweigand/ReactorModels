"""media.py"""

import math


class Media:
    """Adsorbent (media) particle parameters."""

    def __init__(
        self,
        bed_density: float | None = None,
        sphericity: float | None = None,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        particle_radius: float | None = None,
        particle_diameter: float | None = None,
    ) -> None:
        self.bed_density = bed_density
        self.sphericity = sphericity
        self.particle_porosity = particle_porosity
        self.particle_density = particle_density
        self.particle_radius = particle_radius
        self.particle_diameter = particle_diameter

        # Radius vs diameter support
        if self.particle_radius is not None and self.particle_diameter is not None:
            if not math.isclose(
                self.particle_diameter,
                2 * self.particle_radius,
                rel_tol=1e-9,
            ):
                raise ValueError("particle_diameter must equal 2 * particle_radius")

        elif self.particle_radius is not None:
            self.particle_diameter = 2 * self.particle_radius

        elif self.particle_diameter is not None:
            self.particle_radius = 0.5 * self.particle_diameter

    def get_bed_density(self, porosity: float) -> float:
        """Return bed density calculated from particle density and porosity."""
        if self.bed_density is not None:
            return self.bed_density

        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        if self.particle_density is None:
            raise ValueError("particle_density is required to calculate bed_density.")

        self.bed_density = (1.0 - porosity) * self.particle_density
        return self.bed_density

    def get_particle_density(self, bulk_density: float, porosity: float) -> float:
        """Return the particle density with units of"""
        if self.particle_density is not None:
            return self.particle_density

        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        self.particle_density = bulk_density / (1.0 - porosity)
        return self.particle_density
