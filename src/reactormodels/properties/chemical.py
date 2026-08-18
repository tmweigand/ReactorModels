"""chemical.py"""

import numpy as np


class Chemical:
    """Chemical properties."""

    def __init__(
        self,
        name: str = "default",
        diffusion: float | None = None,
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

        if diffusion is not None:
            assert np.all(np.array(diffusion) > 0), (
                "diffusion_parameter must be positive, " f"got {diffusion}"
            )

        self.name = name
        self.diffusion = diffusion
        self.molecular_weight = molecular_weight
        self.molar_volume = molar_volume
        self.density = density
        self.chemical_density = density
        self.solubility = solubility
        self.vapor_pressure = vapor_pressure
        self.boiling_point = boiling_point

    def liquid_diffusion_coefficient(
        self,
        viscosity: float,
    ) -> float:
        """From a correlation given by Hayduk and Laudie (1974):

        ----reference----
        (AdDesignS manual appendix F eq.no.7)

        ----parameters----
        DL is the liquid diffusivity (cm2/s)
        Vb is the molar volume of the chemical at the normal boiling point (cm3/mol)
        uL is the water viscosity (centipoise)
        """
        assert viscosity > 0, f"viscosity must be positive, got {viscosity}"

        if self.molar_volume is None:
            raise ValueError(
                "molar_volume is required to calculate "
                "the liquid diffusion coefficient."
            )

        return 13.26e-5 / (
            np.power(viscosity, 1.14) * np.power(self.molar_volume, 0.589)
        )
