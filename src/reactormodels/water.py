"""water.py"""


class Water:
    """Water properties."""

    def __init__(
        self,
        water_matrix: str,
        density: float | None = None,
        viscosity: float | None = None,
        temperature: float | None = None,
    ):

        if density is not None:
            assert density > 0, f"Water density must be positive, got {density}"

        if viscosity is not None:
            assert viscosity > 0, f"Water viscosity must be positive, got {viscosity}"

        self.water_matrix = water_matrix
        self.density = density
        self.viscosity = viscosity
        self.temperature = temperature
