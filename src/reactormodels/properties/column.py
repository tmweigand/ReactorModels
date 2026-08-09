"""column.py."""

import numpy as np

from .media import Media
from .water import Water


class Column:
    """Packed-bed column data and calculations."""

    def __init__(
        self,
        length: float,
        diameter: float,
        porosity: float,
        media: Media,
        water: Water,
        bulk_density: float | None = None,
        sorbent_mass: float | None = None,
    ) -> None:
        """Initialize packed-bed column properties."""
        assert length > 0, f"length must be positive, got {length}"
        assert diameter > 0, f"diameter must be positive, got {diameter}"
        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"
        if bulk_density is not None:
            assert (
                bulk_density > 0
            ), f"bulk_density must be positive, got {bulk_density}"
        if sorbent_mass is not None:
            assert (
                sorbent_mass > 0
            ), f"sorbent_mass must be positive, got {sorbent_mass}"

        self.length = length
        self.porosity = porosity
        self.diameter = diameter
        self.media = media
        self.bulk_density = bulk_density
        self.sorbent_mass = sorbent_mass
        self.water = water

    def cross_section_area(self) -> float:
        """Calculate the column cross-sectional area."""
        return float(0.25 * np.pi * self.diameter**2)

    def column_volume(self) -> float:
        """Calculate the total column volume."""
        return self.cross_section_area() * self.length

    def get_bulk_density(self) -> float:
        """Return or calculate the packed-bed bulk density."""
        if self.bulk_density is not None:
            return self.bulk_density

        if self.media.particle_density is None:
            raise ValueError(
                "Bulk density cannot be determined. "
                "Ensure media.particle_density is set."
            )

        self.bulk_density = (1.0 - self.porosity) * self.media.particle_density
        return self.bulk_density

    def get_particle_density(self) -> float:
        """Return or calculate the media particle density."""
        if self.media.particle_density is not None:
            return self.media.particle_density

        if self.bulk_density is None:
            raise ValueError(
                "Particle density cannot be determined."
                "Ensure media.particle_density is set."
            )

        self.media.particle_density = self.bulk_density / (1.0 - self.porosity)
        return self.media.particle_density

    def get_sorbent_mass(self) -> float:
        """Return or calculate the sorbent mass in the packed bed."""
        if self.sorbent_mass is not None:
            return self.sorbent_mass

        if self.bulk_density is None or self.media.particle_density is None:
            raise ValueError(
                "Sorbent mass cannot be determined. "
                "Ensure media.particle_density is set."
            )

        self.sorbent_mass = self.get_bulk_density() * self.column_volume()
        return self.sorbent_mass

    def get_total_porosity(self) -> float:
        """Calculate the total bed and intraparticle porosity."""
        if self.media.particle_porosity is None:
            raise ValueError(
                "Media.particle_porosity is required to compute total porosity."
            )

        return self.porosity + (1.0 - self.porosity) * self.media.particle_porosity
