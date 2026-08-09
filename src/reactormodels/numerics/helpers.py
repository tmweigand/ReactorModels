"""helpers.py"""

import numpy as np

__all__ = ["convergence_order"]


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
