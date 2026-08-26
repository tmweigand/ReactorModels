"""isotherm.py"""

import numpy as np
from scipy.optimize import curve_fit


class Isotherm:
    """Base class for adsorption isotherms."""

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        raise NotImplementedError

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        raise NotImplementedError

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        raise NotImplementedError

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration"""
        raise NotImplementedError

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        raise NotImplementedError

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        raise NotImplementedError


class LangmuirIsotherm(Isotherm):
    """Langmuir isotherm: q* = q_m * K * C / (1 + K * C)

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.

    """

    def __init__(self, K: float, q_m: float):
        self.K = K
        self.q_m = q_m

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        return (
            self.q_m * self.K * np.maximum(C, 0.0) / (1 + self.K * np.maximum(C, 0.0))
        )

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C = np.asarray(C, dtype=float)
        return self.q_m * self.K / (1 + self.K * np.maximum(C, 0.0)) ** 2

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.asarray(C, dtype=float)
        return -2 * self.q_m * self.K**2 / (1 + self.K * np.maximum(C, 0.0)) ** 3


class FreundlichIsotherm(Isotherm):
    """Freundlich isotherm: q* = K * C^(1/n)

    Parameters
    ----------
    K : float
        Freundlich capacity factor
    n : float
        Freundlich intensity factor.

    """

    def __init__(self, K: float, n: float):
        self.K = K
        self.n = n

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        return self.K * np.maximum(C, 0.0) ** (1.0 / self.n)

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 1e-30)
        return (self.K / self.n) * C ** (1.0 / self.n - 1.0)

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
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

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        return self.K * np.asarray(C, dtype=float)

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        return self.K * np.ones_like(np.asarray(C, dtype=float))

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(C, dtype=float))

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration."""
        return np.asarray(q, dtype=float) / self.K

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        return np.ones_like(np.asarray(q, dtype=float)) / self.K

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(q, dtype=float))


def fit_isotherm(
    isotherm_class: type[Isotherm],
    C: float | np.ndarray,
    q: float | np.ndarray,
    initial_guess: tuple[float, ...],
) -> Isotherm:
    """Fit an isotherm to equilibrium concentration data.

    Parameters
    ----------
    isotherm_class : type[Isotherm]
        Isotherm class to fit.
    C : float | np.ndarray
        Equilibrium fluid concentrations.
    q : float | np.ndarray
        Equilibrium sorbed concentrations.
    initial_guess : tuple[float, ...]
        Initial guess for the isotherm parameters.

    Isotherm
        Fitted isotherm instance.

    """
    C = np.asarray(C, dtype=float)
    q = np.asarray(q, dtype=float)

    if C.shape != q.shape:
        raise ValueError("C and q must have the same shape.")
    if np.any(C < 0) or np.any(q < 0):
        raise ValueError("C and q values must be nonnegative.")

    def model(C, *parameters):
        isotherm = isotherm_class(*parameters)
        return isotherm.q(C)

    bounds = (np.full(len(initial_guess), 1e-30), np.full(len(initial_guess), np.inf))

    popt, _ = curve_fit(
        model,
        C,
        q,
        p0=initial_guess,
        bounds=bounds,
    )

    return isotherm_class(*popt)
