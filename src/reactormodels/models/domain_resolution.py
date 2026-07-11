"""domain_resolution.py"""

from enum import Enum, auto


class DomainResolution(Enum):
    """Set the form of the domain resolution."""

    COLUMN = auto()
    PARTICLE = auto()
