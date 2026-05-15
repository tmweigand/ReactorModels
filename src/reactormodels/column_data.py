"""column_data.py"""


class Column:
    """Column data"""

    def __init__(
        self, length: float, porosity: float, bulk_density: float | None = None
    ):
        self.length = length
        self.porosity = porosity
        self.bulk_density = bulk_density
