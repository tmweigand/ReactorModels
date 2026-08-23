"""helpers.py"""

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["convergence_order", "compute_rmse"]


def convergence_order(h: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Estimate observed order of convergence from grid spacing and error arrays.

    Both arrays must be ordered the same way (e.g. by decreasing h).
    Returns an array the same length as the inputs, with NaN for the first entry.
    """
    h = np.asarray(h, dtype=float)
    error = np.asarray(error, dtype=float)
    order = np.full(len(h), np.nan)
    order[1:] = np.log(error[:-1] / error[1:]) / np.log(h[:-1] / h[1:])
    return order


def compute_rmse(
    numerical: ArrayLike,
    reference: ArrayLike,
) -> float:
    """Compute the root mean square error between numerical and reference values."""
    numerical_array = np.asarray(numerical, dtype=float)
    reference_array = np.asarray(reference, dtype=float)

    if numerical_array.shape != reference_array.shape:
        raise ValueError(
            "numerical and reference must have the same shape, "
            f"got {numerical_array.shape} and {reference_array.shape}."
        )

    return float(np.sqrt(np.mean((numerical_array - reference_array) ** 2)))
