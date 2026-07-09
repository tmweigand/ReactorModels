"""water_class.py"""


class Water:
    """Water properties."""

    def __init__(
        self,
        water_matrix: str,
        density: float | None = None,
        viscosity: float | None = None,
        temperature: float | None = None,
    ):

        self.water_matrix = water_matrix
        self.density = density
        self.viscosity = viscosity
        self.temperature = temperature
