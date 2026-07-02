"""chemical_class.py"""

import numpy as np


class Chemical:
    """Chemical properties."""

    def __init__(
        self,
        name: str,
        molecular_weight: float,
        molar_volume: float,
        density: float | None = None,
        solubility: float | None = None,
        vapor_pressure: float | None = None,
        boiling_point: float | None = None,
        diffusion_parameter: float | None = None,
    ):
        assert (
            molecular_weight > 0
        ), f"molecular_weight must be positive, got {molecular_weight}"
        assert molar_volume > 0, f"molar_volume must be positive, got {molar_volume}"

        if density is not None:
            assert density > 0, f"density must be positive, got {density}"

        if solubility is not None:
            assert solubility >= 0, f"solubility must be non-negative, got {solubility}"

        if vapor_pressure is not None:
            assert (
                vapor_pressure >= 0
            ), f"vapor_pressure must be non-negative, got {vapor_pressure}"

        if diffusion_parameter is not None:
            assert (
                diffusion_parameter > 0
            ), f"diffusion_parameter must be positive, got {diffusion_parameter}"

        self.name = name
        self.molecular_weight = molecular_weight
        self.molar_volume = molar_volume
        self.density = density
        self.solubility = solubility
        self.vapor_pressure = vapor_pressure
        self.boiling_point = boiling_point
        self.diffusion_parameter = diffusion_parameter

    def liquid_diffusion_coefficient(self, viscosity: float) -> float:  # noqa: D417
        """Liquid-phase diffusion coefficient using the AdDesign equation.
        -----------

        Parameters
        ----------
        1. liquid diffusivity (cm2/s)
        2. molar volume of the chemical at the normal boiling point (cm3/mol)
        3. water viscosity (centipoise)

        """  # noqa: D205
        assert viscosity > 0, f"viscosity must be positive, got {viscosity}"

        return 13.26e-5 / (
            np.power(viscosity, 1.14) * np.power(self.molar_volume, 0.589)
        )
