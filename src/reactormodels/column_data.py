"""column_data.py."""

import numpy as np

from .chemical_class import Chemical
from .media_class import Media
from .water_class import Water


class Column(Media, Water, Chemical):
    """Packed-bed column and associated parameter classes."""

    def __init__(
        self,
        length: float,
        porosity: float,
        diameter: float | None = None,
        bulk_density: float | None = None,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        mean_diameter: float | None = None,
        sorbent_mass: float | None = None,
        media: Media | None = None,
        water: Water | None = None,
        chemical: Chemical | None = None,
        **kwargs,
    ) -> None:
        """Initialize column, media, water, and chemical parameters."""
        if media is not None:
            kwargs.setdefault("particle_porosity", media.particle_porosity)
            kwargs.setdefault("particle_density", media.particle_density)
            kwargs.setdefault("mean_diameter", media.mean_diameter)
            kwargs.setdefault("bed_density", media.bed_density)
            kwargs.setdefault("sphericity", media.sphericity)
            kwargs.setdefault("particle_radius", media.particle_radius)
        else:
            if particle_porosity is not None:
                kwargs.setdefault("particle_porosity", particle_porosity)

            if particle_density is not None:
                kwargs.setdefault("particle_density", particle_density)

            if mean_diameter is not None:
                kwargs.setdefault("mean_diameter", mean_diameter)

            if bulk_density is not None:
                kwargs.setdefault("bed_density", bulk_density)

        if water is not None:
            kwargs.setdefault("water_matrix", water.water_matrix)
            kwargs.setdefault("density", water.density)
            kwargs.setdefault("viscosity", water.viscosity)
            kwargs.setdefault("temperature", water.temperature)

        if chemical is not None:
            kwargs.setdefault("compound", chemical.compound)
            kwargs.setdefault("molar_volume", chemical.molar_volume)
            kwargs.setdefault("molecular_weight", chemical.molecular_weight)
            kwargs.setdefault("chemical_density", chemical.chemical_density)
            kwargs.setdefault("solubility", chemical.solubility)
            kwargs.setdefault("vapor_pressure", chemical.vapor_pressure)
            kwargs.setdefault("boiling_point", chemical.boiling_point)
            kwargs.setdefault(
                "diffusion_parameter",
                chemical.diffusion_parameter,
            )

        super().__init__(**kwargs)

        assert length > 0, f"length must be positive, got {length}"
        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        if diameter is not None:
            assert diameter > 0, f"diameter must be positive, got {diameter}"

        if bulk_density is not None:
            assert (
                bulk_density > 0
            ), f"bulk_density must be positive, got {bulk_density}"

        if sorbent_mass is not None:
            assert (
                sorbent_mass > 0
            ), f"sorbent_mass must be positive, got {sorbent_mass}"

        self.length = length
        self.diameter = diameter
        self.porosity = porosity
        self.bulk_density = (
            bulk_density if bulk_density is not None else self.bed_density
        )
        self.sorbent_mass = sorbent_mass

        self.media = media
        self.water = water
        self.chemical = chemical

    def cross_section_area(self) -> float:
        """Calculate the column cross-sectional area."""
        assert (
            self.diameter is not None
        ), "diameter is required to calculate cross-sectional area"

        return 0.25 * np.pi * self.diameter**2

    def column_volume(self) -> float:
        """Calculate the total column volume."""
        return self.cross_section_area() * self.length

    @staticmethod
    def calculate_superficial_velocity(
        flow_rate: float,
        cross_section_area: float,
    ) -> float:
        """Calculate superficial velocity, v = Q / A."""
        assert flow_rate > 0, f"flow_rate must be positive, got {flow_rate}"
        assert cross_section_area > 0, (
            "cross_section_area must be positive, " f"got {cross_section_area}"
        )

        return flow_rate / cross_section_area

    @staticmethod
    def calculate_interstitial_velocity(
        superficial_velocity: float,
        porosity: float,
    ) -> float:
        """Calculate interstitial velocity, u = v / porosity."""
        assert superficial_velocity > 0, (
            "superficial_velocity must be positive, " f"got {superficial_velocity}"
        )
        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        return superficial_velocity / porosity

    def superficial_velocity(self, flow_rate: float) -> float:
        """Calculate superficial velocity using this column's area."""
        return self.calculate_superficial_velocity(
            flow_rate=flow_rate,
            cross_section_area=self.cross_section_area(),
        )

    def interstitial_velocity(self, flow_rate: float) -> float:
        """Calculate interstitial velocity using this column's porosity."""
        return self.calculate_interstitial_velocity(
            superficial_velocity=self.superficial_velocity(flow_rate),
            porosity=self.porosity,
        )

    def get_bulk_density(self) -> float:
        """Calculate or return bulk density."""
        if self.particle_density is not None:
            derived = (1 - self.porosity) * self.particle_density

            if self.bulk_density is not None:
                assert np.isclose(
                    derived,
                    self.bulk_density,
                ), (
                    f"Supplied bulk_density "
                    f"{self.bulk_density} inconsistent "
                    "with (1 - porosity) * particle_density "
                    f"= {derived:.4f}"
                )

            return derived

        if self.bulk_density is not None:
            return self.bulk_density

        raise ValueError("Must supply either particle_density or bulk_density.")

    def get_particle_density(self) -> float:
        """Calculate or return particle density."""
        if self.bulk_density is not None:
            derived = self.bulk_density / (1 - self.porosity)

            if self.particle_density is not None:
                assert np.isclose(
                    derived,
                    self.particle_density,
                ), (
                    f"Supplied particle_density "
                    f"{self.particle_density} inconsistent "
                    "with bulk_density / (1 - porosity) "
                    f"= {derived:.4f}"
                )

            return derived

        if self.particle_density is not None:
            return self.particle_density

        raise ValueError("Must supply either bulk_density or particle_density.")

    def get_sorbent_mass(self) -> float:
        """Calculate or return the sorbent mass."""
        if self.bulk_density is not None:
            derived = self.get_bulk_density() * self.column_volume()

            if self.sorbent_mass is not None:
                assert np.isclose(
                    derived,
                    self.sorbent_mass,
                ), (
                    f"Supplied sorbent_mass "
                    f"{self.sorbent_mass} inconsistent "
                    "with bulk_density * column_volume "
                    f"= {derived:.4f}"
                )

            return derived

        if self.sorbent_mass is not None:
            return self.sorbent_mass

        raise ValueError("Must supply either sorbent_mass or bulk_density.")

    def total_porosity(self) -> float:
        """Calculate total bed and particle porosity."""
        if self.particle_porosity is None:
            raise ValueError(
                "particle_porosity is required to calculate " "total_porosity."
            )

        return self.porosity + (1 - self.porosity) * self.particle_porosity
