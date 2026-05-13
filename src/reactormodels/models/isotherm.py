"""isotherm.py"""

import numpy as np


class Isotherm:
    """Base class for adsorption isotherms."""

    def q(self, C: np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        raise NotImplementedError

    def dq_dC(self, C: np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        raise NotImplementedError

    def d2q_dC2(self, C: np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        raise NotImplementedError


class FreundlichIsotherm(Isotherm):
    """Freundlich isotherm: q* = K * C^(1/n)

    Parameters
    ----------
    K : float
        Freundlich capacity factor  [mg/g / (mg/L)^(1/n)]
    n : float
        Freundlich intensity factor (n >= 1 → favorable).

    """

    def __init__(self, K: float, n: float):
        self.K = K
        self.n = n

    def q(self, C: np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        return self.K * np.maximum(C, 0.0) ** (1.0 / self.n)

    def dq_dC(self, C: np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 1e-30)
        return (self.K / self.n) * C ** (1.0 / self.n - 1.0)

    def d2q_dC2(self, C: np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.maximum(np.asarray(C, dtype=float), 1e-30)
        return (self.K / self.n) * (1.0 / self.n - 1.0) * C ** (1.0 / self.n - 2.0)


class LinearIsotherm(Isotherm):
    """Linear isotherm: q* = K * C

    Parameters
    ----------
    K : float
        Henry constant  [mg/g / (mg/L)]

    """

    def __init__(self, K: float):
        self.K = K

    def q(self, C: np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        return self.K * np.asarray(C, dtype=float)

    def dq_dC(self, C: np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        return self.K * np.ones_like(np.asarray(C, dtype=float))

    def d2q_dC2(self, C: np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(C, dtype=float))
