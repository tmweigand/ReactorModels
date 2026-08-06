"""column_data.py."""

import numpy as np

from .chemical_class import Chemical
from .media_class import Media
from .water_class import Water


class Column:
    """Packed-bed column data and calculations."""

    def __init__(
        self,
        length: float,
        porosity: float,
        diameter: float,
        media: Media,
        water: Water,
        chemical: Chemical,
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
        self.chemical = chemical

    def cross_section_area(self) -> float:
        """Calculate the column cross-sectional area."""
        return float(0.25 * np.pi * self.diameter**2)

    def column_volume(self) -> float:
        """Calculate the total column volume."""
        return self.cross_section_area() * self.length

    def get_bulk_density(self) -> float:
        """Return or calculate the packed-bed bulk density."""
        particle_density = (
            self.media.particle_density if self.media is not None else None
        )

        if particle_density is not None:
            derived_bulk_density = (1.0 - self.porosity) * particle_density

            if self.bulk_density is not None:
                assert np.isclose(
                    derived_bulk_density,
                    self.bulk_density,
                ), (
                    f"Supplied bulk_density {self.bulk_density} is "
                    "inconsistent with "
                    "(1 - porosity) * media.particle_density "
                    f"= {derived_bulk_density:.4f}"
                )

            return derived_bulk_density

        if self.bulk_density is not None:
            return self.bulk_density

        raise ValueError(
            "Bulk density cannot be determined. Supply either "
            "bulk_density or media with particle_density."
        )

    def get_particle_density(self) -> float:
        """Return or calculate the media particle density."""
        particle_density = (
            self.media.particle_density if self.media is not None else None
        )

        if self.bulk_density is not None:
            derived_particle_density = self.bulk_density / (1.0 - self.porosity)

            if particle_density is not None:
                assert np.isclose(
                    derived_particle_density,
                    particle_density,
                ), (
                    f"Supplied particle_density {particle_density} is "
                    "inconsistent with "
                    "bulk_density / (1 - porosity) "
                    f"= {derived_particle_density:.4f}"
                )

            return derived_particle_density

        if particle_density is not None:
            return particle_density

        raise ValueError(
            "Particle density cannot be determined. Supply either "
            "bulk_density or media with particle_density."
        )

    def get_sorbent_mass(self) -> float:
        """Return or calculate the sorbent mass in the packed bed."""
        media_density_available = (
            self.media is not None and self.media.particle_density is not None
        )

        density_available = self.bulk_density is not None or media_density_available

        if density_available:
            derived_sorbent_mass = self.get_bulk_density() * self.column_volume()

            if self.sorbent_mass is not None:
                assert np.isclose(
                    derived_sorbent_mass,
                    self.sorbent_mass,
                ), (
                    f"Supplied sorbent_mass {self.sorbent_mass} is "
                    "inconsistent with bulk_density * column_volume "
                    f"= {derived_sorbent_mass:.4f}"
                )

            return derived_sorbent_mass

        if self.sorbent_mass is not None:
            return self.sorbent_mass

        raise ValueError(
            "Sorbent mass cannot be determined. Supply sorbent_mass, "
            "bulk_density, or media.particle_density."
        )

    def get_total_porosity(self) -> float:
        """Calculate the total bed and intraparticle porosity."""
        particle_porosity = self.porosity

        if self.media is not None and self.media.particle_porosity is not None:
            particle_porosity = self.media.particle_porosity

        return self.porosity + (1.0 - self.porosity) * particle_porosity
