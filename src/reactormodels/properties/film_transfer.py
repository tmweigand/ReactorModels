"""film_transfer.py"""

from .breakthrough import Breakthrough
from ..dimensionless_numbers import reynolds_number, schmidt_number, sherwood_number


class FilmTransfer:
    """Liquid-film mass transfer coefficient, k_film."""

    def __init__(
        self,
        breakthrough: Breakthrough,
        viscosity: float,
        density: float,
        diffusion_coefficient: float,
        method: str = "gnielinski",
        k_film: float | None = None,
    ) -> None:
        self.breakthrough = breakthrough
        self.viscosity = viscosity
        self.density = density
        self.diffusion_coefficient = diffusion_coefficient
        self.method = method
        self._k_film = k_film

    @property
    def reynolds(self) -> float:
        return reynolds_number(
            self.density,
            self.breakthrough.interstitial_velocity,
            self.breakthrough.column.media.particle_diameter,
            self.viscosity,
            sphericity=self.breakthrough.column.media.sphericity or 1.0,
        )

    @property
    def schmidt(self) -> float:
        return schmidt_number(self.viscosity, self.density, self.diffusion_coefficient)

    @property
    def sherwood(self) -> float:
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
        d_p = self.breakthrough.column.media.particle_diameter
        self._k_film = self.sherwood * self.diffusion_coefficient / d_p
        return self._k_film
