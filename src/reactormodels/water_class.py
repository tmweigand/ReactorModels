"""water_class.py"""


class Water:
    """Water properties."""

    def __init__(
        self,
        name: str,
        density: float,
        viscosity: float,
        temperature: float,
    ):
        assert density > 0, f"density must be positive, got {density}"
        assert viscosity > 0, f"viscosity must be positive, got {viscosity}"

        self.name = name
        self.density = density
        self.viscosity = viscosity
        self.temperature = temperature
