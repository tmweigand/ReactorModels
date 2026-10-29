"""isotherm.py"""

import numpy as np


def _clip(x: float | np.ndarray, eps: float = 0):
    """Clip function to trim negative values."""
    return np.maximum(x, eps)


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
    """Langmuir isotherm:

    q* = q_m * K * C / (1 + K * C) or C* = q / (K (q_m - q))

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
        clip_C = _clip(np.asarray(C, dtype=float), 0.0)
        return self.q_m * self.K / (1 + self.K * clip_C) ** 2

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        clip_C = _clip(np.asarray(C, dtype=float), 0.0)
        return -2 * self.q_m * self.K**2 / (1 + self.K * clip_C) ** 3

    def multi_dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of q for multi-species structure."""
        return self.dq_dC(C)[np.newaxis, :, :]

    def multi_d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative for multi-species structure."""
        return self.d2q_dC2(C)[np.newaxis, np.newaxis, :, :]

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration"""
        q = np.asarray(q, dtype=float)
        return q / (self.K * (self.q_m - q))

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        q = np.asarray(q, dtype=float)
        return self.q_m / (self.K * (self.q_m - q) ** 2)

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        q = np.asarray(q, dtype=float)
        return 2 * self.q_m / (self.K * (self.q_m - q) ** 3)


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
        clip_C = _clip(np.asarray(C, dtype=float), 0.0)
        return self.K * clip_C ** (1.0 / self.n)

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        clip_C = _clip(np.asarray(np.abs(C), dtype=float), 1e-12)
        return (self.K / self.n) * clip_C ** (1.0 / self.n - 1.0)

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        clip_C = _clip(np.asarray(np.abs(C), dtype=float), 1e-12)
        return (self.K / self.n) * (1.0 / self.n - 1.0) * clip_C ** (1.0 / self.n - 2.0)

    def C(self, q):
        """Return liquid phase concentration"""
        clip_q = _clip(np.asarray(q, dtype=float), 0.0)
        return (clip_q / self.K) ** self.n

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        clip_q = _clip(np.asarray(q, dtype=float), 0.0)

        return self.n * (1 / self.K) ** self.n * clip_q ** (self.n - 1)

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        clip_q = _clip(np.asarray(q, dtype=float), 0.0)
        return self.n * (self.n - 1) * (1 / self.K) ** self.n * clip_q ** (self.n - 2)

    def multi_dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of C for multi-species structure."""
        return self.dC_dq(q)[np.newaxis, :, :]

    def multi_d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative for multi-species structure."""
        return self.d2C_dq2(q)[np.newaxis, np.newaxis, :, :]


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
        dq_dC = self.K * np.ones_like(np.asarray(C, dtype=float))
        return dq_dC[np.newaxis, :]

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        d2q_dC2 = np.zeros_like(np.asarray(C, dtype=float))
        return d2q_dC2[np.newaxis, np.newaxis, :]

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration."""
        return np.asarray(q, dtype=float) / self.K

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        dC_dq = np.ones_like(np.asarray(q, dtype=float)) / self.K
        return dC_dq[np.newaxis, :]

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        d2C_dq2 = np.zeros_like(np.asarray(q, dtype=float))
        return d2C_dq2[np.newaxis, np.newaxis, :]
