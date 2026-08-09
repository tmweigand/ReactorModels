"""media.py"""


class Media:
    """Adsorbent (media) particle parameters."""

    def __init__(
        self,
        mean_diameter: float | None = None,
        bed_density: float | None = None,
        sphericity: float | None = None,
        particle_porosity: float | None = None,
        particle_density: float | None = None,
        particle_radius: float | None = None,
    ) -> None:

        self.mean_diameter = mean_diameter
        self.bed_density = bed_density
        self.sphericity = sphericity
        self.particle_porosity = particle_porosity
        self.particle_density = particle_density
        self.particle_radius = particle_radius

    def get_bed_density(self, porosity: float) -> float:
        """Return bed density calculated from particle density and porosity."""
        if self.bed_density is not None:
            return self.bed_density

        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        if self.particle_density is None:
            raise ValueError("particle_density is required to calculate bed_density.")

        self.bed_density = (1.0 - porosity) * self.particle_density
        return self.bed_density

    def get_particle_density(self, bulk_density: float, porosity: float) -> float:
        """Return the particle density with units of"""
        if self.particle_density is not None:
            return self.particle_density

        assert 0 < porosity < 1, f"porosity must be in (0, 1), got {porosity}"

        self.particle_density = bulk_density / (1.0 - porosity)
        return self.particle_density
