"""chemical.py"""


class Chemical:
    """Chemical properties."""

    def __init__(
        self,
        name: str = "default",
        axial_diffusion: float | None = None,
        pore_diffusion: float | None = None,
        surface_diffusion: float | None = None,
        liquid_diffusion: float | None = None,
        molar_volume: float | None = None,
        molecular_weight: float | None = None,
        density: float | None = None,
        solubility: float | None = None,
        vapor_pressure: float | None = None,
        boiling_point: float | None = None,
    ) -> None:

        if molar_volume is not None:
            assert (
                molar_volume > 0
            ), f"molar_volume must be positive, got {molar_volume}"

        if density is not None:
            assert density > 0, f"Chemical density must be positive, got {density}"

        if solubility is not None:
            assert solubility >= 0, f"Solubility must be non-negative, got {solubility}"

        if vapor_pressure is not None:
            assert (
                vapor_pressure >= 0
            ), f"vapor_pressure must be non-negative, got {vapor_pressure}"

        if axial_diffusion is not None:
            assert axial_diffusion >= 0, (
                "Axial diffusion must be positive, " f"got {axial_diffusion}"
            )

        if pore_diffusion is not None:
            assert pore_diffusion >= 0, (
                "Pore diffusion must be positive, " f"got {pore_diffusion}"
            )

        if surface_diffusion is not None:
            assert surface_diffusion >= 0, (
                "Surface diffusion must be positive, " f"got {surface_diffusion}"
            )

        if liquid_diffusion is not None:
            assert (
                liquid_diffusion > 0
            ), f"Liquid diffusion must be positive, got {liquid_diffusion}"

        self.name = name
        self.axial_diffusion = axial_diffusion
        self.pore_diffusion = pore_diffusion
        self.surface_diffusion = surface_diffusion
        self.liquid_diffusion = liquid_diffusion
        self.molecular_weight = molecular_weight
        self.molar_volume = molar_volume
        self.density = density
        self.chemical_density = density
        self.solubility = solubility
        self.vapor_pressure = vapor_pressure
        self.boiling_point = boiling_point

    def liquid_diffusion_coefficient(self, viscosity: float) -> float:
        """Return `liquid_diffusion`, computing and storing it if not provided.

        Correlation: Hayduk and Laudie (1974), AdDesignS manual appendix F eq. 7.
            DL: liquid diffusivity (cm2/s)
            Vb: molar volume at the normal boiling point (cm3/mol)
            uL: water viscosity (centipoise)


        Note: This relationship is for dilute, organic compounds.
        """
        if self.liquid_diffusion is None:
            if self.molar_volume is None:
                raise ValueError(
                    "molar_volume is required to calculate "
                    "the liquid diffusion coefficient."
                )
            assert viscosity > 0, f"viscosity must be positive, got {viscosity}"

            self.liquid_diffusion = float(
                13.26e-5 / (viscosity**1.14 * self.molar_volume**0.589)
            )

        return self.liquid_diffusion
