"""media.py"""

import math


class Media:
    """Adsorbent (media) particle parameters."""

    def __init__(
        self,
        sphericity: float | None = None,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        particle_radius: float | None = None,
        particle_diameter: float | None = None,
    ) -> None:
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
