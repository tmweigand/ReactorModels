"""film_transfer.py"""

from .breakthrough import Breakthrough
from ..dimensionless_numbers import (
    SHERWOOD_METHODS,
    SherwoodMethod,
    reynolds_number,
    schmidt_number,
    sherwood_number,
)


class FilmTransfer:
    """Liquid-film mass transfer coefficient, k_film.

    Provide either:
      * ``k_film``: a fixed value (m/s), or
      * ``method``: a Sherwood correlation name (see ``SHERWOOD_METHODS``)
        used to compute it.

    If neither is given, ``method`` defaults to ``DEFAULT_METHOD``.
    """

    DEFAULT_METHOD: SherwoodMethod = "gnielinski"

    def __init__(
        self,
        breakthrough: Breakthrough,
        method: SherwoodMethod | None = None,
        k_film: float | None = None,
    ) -> None:
        if k_film is not None and method is not None:
            raise ValueError("Provide either `k_film` or `method`, not both.")
        if k_film is not None and k_film <= 0:
            raise ValueError("`k_film` must be positive.")

        if k_film is None:
            method = (method or self.DEFAULT_METHOD).lower()  # type: ignore[assignment]
            if method not in SHERWOOD_METHODS:
                raise ValueError(
                    f"Unknown method {method!r}. "
                    f"Available: {', '.join(SHERWOOD_METHODS)}"
                )

        self.breakthrough = breakthrough
        self._k_film_override = k_film
        self.method = method  # None when k_film is supplied directly

    @property
    def viscosity(self) -> float:
        """Water dynamic viscosity."""
        viscosity = self.breakthrough.column.water.viscosity
        assert viscosity is not None, "Water viscosity is not set."
        return viscosity

    @property
    def density(self) -> float:
        """Water density."""
        density = self.breakthrough.column.water.density
        assert density is not None, "Water density is not set."
        return density

    @property
    def diffusion_coefficient(self) -> float:
        """Liquid diffusion coefficient of the chemical."""
        return self.breakthrough.chemical.liquid_diffusion_coefficient(self.viscosity)

    @property
    def particle_diameter(self) -> float:
        """Media particle diameter."""
        diameter = self.breakthrough.column.media.particle_diameter
        assert diameter is not None, "Media particle diameter is not set."
        return diameter

    @property
    def reynolds(self) -> float:
        """Reynolds number for film transfer."""
        return reynolds_number(
            self.density,
            self.breakthrough.interstitial_velocity,
            self.particle_diameter,
            self.viscosity,
            sphericity=self.breakthrough.column.media.sphericity or 1.0,
        )

    @property
    def schmidt(self) -> float:
        """Schmidt number for film transfer."""
        return schmidt_number(self.viscosity, self.density, self.diffusion_coefficient)

    @property
    def sherwood(self) -> float:
        """Sherwood number for film transfer."""
        if self.method is None:
            raise ValueError("No correlation in use: k_film was supplied directly.")

        return sherwood_number(
            self.method,
            self.reynolds,
            self.schmidt,
            bed_porosity=self.breakthrough.column.porosity,
        )

    @property
    def k_film(self) -> float:
        """Film mass transfer coefficient (m/s): the supplied value, or computed."""
        if self._k_film_override is not None:
            return self._k_film_override

        return self.sherwood * self.diffusion_coefficient / self.particle_diameter
