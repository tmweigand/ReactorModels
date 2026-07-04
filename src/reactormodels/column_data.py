"""column_data.py"""

import numpy as np


class Column:
    """Packed bed column parameters."""

    def __init__(
        self,
        length: float,
        diameter: float,
        porosity: float | None = None,
        bulk_density: float | None = None,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        sorbent_mass: float | None = None,
    ):
        assert length > 0, f"length must be positive, got {length}"
        if porosity is not None:
            assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"
        if particle_porosity is not None:
            assert 0 < particle_porosity < 1

        self.length = length
        self.diameter = diameter
        self.porosity = porosity
        self.particle_porosity = particle_porosity
        self.particle_density = particle_density
        self.bulk_density = bulk_density
        self.sorbent_mass = sorbent_mass

    def cross_section_area(self) -> float:
        """Cross-sectional area of the column."""
        assert self.diameter is not None, "diameter required"
        return 0.25 * np.pi * self.diameter**2

    def column_volume(self) -> float:
        """Total column volume."""
        return self.cross_section_area() * self.length

    def get_bulk_density(self) -> float:
        """Bulk density rho_b = (1 - porosity) * particle_density."""
        if self.particle_density is not None and self.porosity is not None:
            derived = (1 - self.porosity) * self.particle_density
            if self.bulk_density is not None:
                assert np.isclose(derived, self.bulk_density), (
                    f"Supplied bulk_density {self.bulk_density} inconsistent "
                    f"with (1 - porosity) * particle_density = {derived:.4f}"
                )
            return derived
        elif self.bulk_density is not None:
            return self.bulk_density
        else:
            raise ValueError("Must supply either particle_density or bulk_density.")

    def get_particle_density(self) -> float:
        """Bulk density particle_density = rho_b / (1 - porosity)."""
        if self.porosity is not None:
            derived = self.get_bulk_density() / (1 - self.porosity)
            if self.particle_density is not None:
                assert np.isclose(derived, self.particle_density), (
                    f"Supplied particle_density {self.particle_density} inconsistent "
                    f"with rho_b / (1 - porosity) = {derived:.4f}"
                )
            return derived
        elif self.particle_density is not None:
            return self.particle_density
        else:
            raise ValueError("Must supply either bulk_density or particle_density.")

    def get_sorbent_mass(self) -> float:
        """Mass of sorbent in the bed bulk_density * column_volume."""
        if self.bulk_density is not None:
            derived = self.get_bulk_density() * self.column_volume()
            if self.sorbent_mass is not None:
                assert np.isclose(derived, self.sorbent_mass), (
                    f"Supplied sorbent_mass {self.sorbent_mass} inconsistent "
                    f"with bulk_density * column_volume = {derived:.4f}"
                )
            return derived
        elif self.sorbent_mass is not None:
            return self.sorbent_mass
        else:
            raise ValueError("Must supply either sorbent_mass or bulk_density.")

    def total_porosity(self) -> float:
        """Total porosity = porosity + (1 - porosity) * particle_porosity."""
        if self.particle_porosity is None or self.porosity is None:
            raise ValueError(
                "particle_porosity and porosity required to compute total_porosity."
            )
        return self.porosity + (1 - self.porosity) * self.particle_porosity
