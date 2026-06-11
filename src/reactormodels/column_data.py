"""column_data.py"""

import numpy as np


class Column:
    """Packed bed column parameters."""

    def __init__(
        self,
        length: float,
        porosity: float,
        bulk_density: float | None = None,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        diameter: float | None = None,
    ):
        assert length > 0, f"length must be positive, got {length}"
        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"
        if particle_porosity is not None:
            assert 0 < particle_porosity < 1

        self.length = length
        self.porosity = porosity
        self.particle_porosity = particle_porosity
        self.particle_density = particle_density
        self.bulk_density = bulk_density
        self.diameter = diameter

    def cross_section_area(self) -> float:
        """Cross-sectional area of the column."""
        assert self.diameter is not None, "diameter required"
        return 0.25 * np.pi * self.diameter**2

    def column_volume(self) -> float:
        """Total column volume."""
        return self.cross_section_area() * self.length

    def superficial_velocity(self, flow_rate: float) -> float:
        """Superficial velocity v = Q / A."""
        return flow_rate / self.cross_section_area()

    def interstitial_velocity(self, flow_rate: float) -> float:
        """Interstitial velocity u = Q / (A * porosity)."""
        return flow_rate / (self.cross_section_area() * self.porosity)

    def get_bulk_density(self) -> float:
        """Bulk density rho_b = (1 - porosity) * particle_density."""
        if self.particle_density is not None:
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

    def total_porosity(self) -> float:
        """Total porosity = porosity + (1 - porosity) * particle_porosity."""
        if self.particle_porosity is None:
            raise ValueError("particle_porosity required to compute total_porosity.")
        return self.porosity + (1 - self.porosity) * self.particle_porosity

    def peclet(self, flow_rate: float, diffusion: float) -> float:
        """Axial Peclet number Pe = v * L / (porosity * diffusion).

        Pe >> 1: advection dominated.
        Pe << 1: diffusion dominated.
        """
        v = self.superficial_velocity(flow_rate)
        return v * self.length / (self.porosity * diffusion)
