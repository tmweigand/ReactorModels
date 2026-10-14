"""parameter_estimation: model-agnostic parameter fitting built on lmfit."""

from .fitting import fit_parameters, fit_parameters_multistart, make_parameters
from .result import FitResult

__all__ = [
    "FitResult",
    "fit_parameters",
    "fit_parameters_multistart",
    "make_parameters",
]
