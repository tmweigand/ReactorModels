"""column_data.py"""

import numpy as np


class Column:
    """Column data"""

    def __init__(
        self,
        length: float,
        porosity: float,
        bulk_density: float | None = None,
        diameter: float | None = None,
    ):
        self.length = length
        self.porosity = porosity
        self.bulk_density = bulk_density
        self.diameter = diameter

    def get_cross_section_area(self):
        """Calculate the cross-section area of column."""
        assert self.diameter is not None
        return 0.25 * np.pi * self.diameter**2

    def get_column_volume(self):
        """Calculate the volume of the column."""
        return self.get_cross_section_area() * self.length
