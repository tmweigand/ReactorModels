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
        bed_density: float | None = None,
        sorbent_mass: float | None = None,
    ) -> None:
        """Initialize packed-bed column properties."""
        assert length > 0, f"length must be positive, got {length}"
        assert diameter > 0, f"diameter must be positive, got {diameter}"
        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"
        if bed_density is not None:
            assert bed_density > 0, f"bed_density must be positive, got {bed_density}"
        if sorbent_mass is not None:
            assert (
                sorbent_mass > 0
            ), f"sorbent_mass must be positive, got {sorbent_mass}"

        self.length = length
        self.porosity = porosity
        self.diameter = diameter
        self.media = media
        self.particle_density = media.particle_density
        self.bed_density = bed_density
        self.sorbent_mass = sorbent_mass
        self.water = water

    def cross_section_area(self) -> float:
        """Calculate the column cross-sectional area."""
        return float(0.25 * np.pi * self.diameter**2)

    def column_volume(self) -> float:
        """Calculate the total column volume."""
        return self.cross_section_area() * self.length

    def get_bed_density(self) -> float:
        """Return bed density calculated from particle density and porosity."""
        if self.particle_density is not None:
            if self.bed_density is not None:
                rho_b = (1.0 - self.porosity) * self.particle_density
                assert rho_b == self.bed_density, (
                    f"bed_density inconsistent with porosity and particle_density. "
                    f"Expected {rho_b}, got {self.bed_density}"
                )
        else:
            if self.bed_density is not None:
                rho_b = self.bed_density
                if self.sorbent_mass is not None:
                    assert rho_b == self.sorbent_mass / self.column_volume(), (
                        f"bed_density inconsistent with column_volume and "
                        f"sorbent_mass. Expected {rho_b}, got {self.bed_density}"
                    )
            elif self.sorbent_mass is not None:
                rho_b = self.sorbent_mass / self.column_volume()
            else:
                raise ValueError(
                    "particle_density or sorbent_mass is needed "
                    "to calculate bed density."
                )
        return rho_b

    def get_particle_density(self) -> float:
        """Return the particle density with units of"""
        if self.particle_density is not None:
            rho_p = self.particle_density
        elif self.bed_density is not None:
            rho_p = self.get_bed_density() / (1.0 - self.porosity)
        else:
            raise ValueError("particle_density of bed_density not provided.")
        return rho_p

    def get_sorbent_mass(self) -> float:
        """Return or calculate the sorbent mass in the packed bed."""
        if self.sorbent_mass is not None:
            m_s = self.sorbent_mass
            if self.bed_density is not None:
                assert m_s == self.bed_density * self.column_volume(), (
                    f"sorbent_mass inconsistent with bed_density. "
                    f"Expected {self.bed_density*self.column_volume()}, got {m_s}."
                )
        elif self.bed_density is not None:
            m_s = self.get_bed_density() * self.column_volume()
        else:
            raise ValueError(
                "particle_density cannot be determined. Ensure bed_density is set."
            )
        return m_s

    def get_total_porosity(self) -> float:
        """Calculate the total bed and intraparticle porosity."""
        if self.media.particle_porosity is None:
            raise ValueError(
                "Media.particle_porosity is required to compute total porosity."
            )
        return self.porosity + (1.0 - self.porosity) * self.media.particle_porosity
