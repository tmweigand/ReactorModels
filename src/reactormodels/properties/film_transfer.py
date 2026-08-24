"""film_transfer.py"""

from .breakthrough import Breakthrough
from ..dimensionless_numbers import reynolds_number, schmidt_number, sherwood_number


class FilmTransfer:
    """Liquid-film mass transfer coefficient, k_film."""

    def __init__(
        self,
        breakthrough: Breakthrough,
        method: str = "gnielinski",
        k_film: float | None = None,
    ) -> None:
        self.breakthrough = breakthrough
        self.method = method
        self._k_film = k_film

        assert breakthrough.column.water.viscosity is not None
        self.viscosity = breakthrough.column.water.viscosity

        assert breakthrough.column.water.density is not None
        self.density = breakthrough.column.water.density

        assert breakthrough.chemical.axial_diffusion is not None
        self.diffusion_coefficient = breakthrough.chemical.axial_diffusion

    @property
    def reynolds(self) -> float:
        """Reynolds number for film transfer"""
        assert self.breakthrough.column.media.particle_diameter is not None

        return reynolds_number(
            self.density,
            self.breakthrough.interstitial_velocity,
            self.breakthrough.column.media.particle_diameter,
            self.viscosity,
            sphericity=self.breakthrough.column.media.sphericity or 1.0,
        )

    @property
    def schmidt(self) -> float:
        """Schmidt number for film transfer"""
        return schmidt_number(self.viscosity, self.density, self.diffusion_coefficient)

    @property
    def sherwood(self) -> float:
        """Sherwood number for film transfer"""
        return sherwood_number(
            self.method,
            self.reynolds,
            self.schmidt,
            bed_porosity=self.breakthrough.column.porosity,
        )

    @property
    def k_film(self) -> float:
        """Film mass transfer coefficient, computed unless explicitly supplied."""
        if self._k_film is not None:
            return self._k_film

        assert self.breakthrough.column.media.particle_diameter is not None

        self._k_film = (
            self.sherwood
            * self.diffusion_coefficient
            / self.breakthrough.column.media.particle_diameter
        )
        return self._k_film
