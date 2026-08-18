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
        """Calculate derivative of liquid phase concentration by sorbed mass concentration."""
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

    def __init__(self, K: float | np.ndarray, n: float | np.ndarray):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.n = np.atleast_1d(np.asarray(n, dtype=float))

        if self.K.shape != self.n.shape:
            raise ValueError("K and n must have the same shape.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if np.any(self.n <= 0):
            raise ValueError("n must be greater than zero.")

        self.n_species = len(self.K)

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 0.0)
        if self.n_species == 1:
            return self.K[0] * C ** (1.0 / self.n[0])

        return self.K[:, None] * C ** (1.0 / self.n[:, None])

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 1e-30)

        if self.n_species == 1:
            return self.K[0] / self.n[0] * C ** (1.0 / self.n[0] - 1.0)

        return self.K[:, None] / self.n[:, None] * C ** (1.0 / self.n[:, None] - 1.0)

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 1e-30)

        if self.n_species == 1:
            return (
                self.K[0]
                / self.n[0]
                * (1.0 / self.n[0] - 1.0)
                * C ** (1.0 / self.n[0] - 2.0)
            )

        return (
            self.K[:, None]
            / self.n[:, None]
            * (1.0 / self.n[:, None] - 1.0)
            * C ** (1.0 / self.n[:, None] - 2.0)
        )

    def C_coupled(self, q):
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if q.ndim == 1:
            Q = np.sum(q)
            S = np.sum(self.n * q)

            if Q == 0:
                return np.zeros_like(q)

            return q / Q * (S / (self.n * self.K)) ** self.n

        if q.ndim == 2:
            Q = np.sum(q, axis=0)
            S = np.sum(self.n[:, None] * q, axis=0)

            C = np.zeros_like(q)

            mask = Q > 0

            C[:, mask] = (
                q[:, mask]
                / Q[mask][None, :]
                * (S[mask][None, :] / (self.n[:, None] * self.K[:, None]))
                ** self.n[:, None]
            )

            return C

        raise ValueError("q must be 1D or 2D")

    def dC_dq_coupled(self, q):
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        # Single node: q = (n_species,)
        if q.ndim == 1:
            Q = np.sum(q)
            S = np.sum(self.n * q)

            J = np.zeros((self.n_species, self.n_species))

            if Q == 0 or S == 0:
                return J

            for i in range(self.n_species):
                A_i = (S / (self.n[i] * self.K[i])) ** self.n[i]

                for j in range(self.n_species):
                    delta_ij = 1.0 if i == j else 0.0

                    J[i, j] = A_i * (
                        (delta_ij * Q - q[i]) / Q**2
                        + q[i] * self.n[i] * self.n[j] / (Q * S)
                    )

            return J

        # Multiple spatial nodes:
        # q = (n_species, n_nodes)
        if q.ndim == 2:
            n_nodes = q.shape[1]
            J = np.zeros((self.n_species, self.n_species, n_nodes))

            for k in range(n_nodes):
                J[:, :, k] = self.dC_dq_coupled(q[:, k])

            return J

        raise ValueError("q must be 1D or 2D")

    def C(self, q: float | np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if self.n_species == 1:
            return (q / self.K[0]) ** self.n[0]

        if q.ndim == 1:
            return (q / self.K) ** self.n

        return (q / self.K[:, None]) ** self.n[:, None]

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if self.n_species == 1:
            return self.n[0] * self.K[0] ** (-self.n[0]) * q ** (self.n[0] - 1)

        if q.ndim == 1:
            return self.n * self.K ** (-self.n) * q ** (self.n - 1)

        return (
            self.n[:, None]
            * self.K[:, None] ** (-self.n[:, None])
            * q ** (self.n[:, None] - 1)
        )


class LinearIsotherm(Isotherm):
    """Linear isotherm: q* = K * C

    Parameters
    ----------
    K : float
        Henry constant  [mg/g / (mg/L)]

    """

    def __init__(self, K: float):
        self.K = np.asarray(K, dtype=float)

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration."""
        return self.K[:, None] * np.asarray(C, dtype=float)

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        return self.K[:, None] * np.ones_like(np.asarray(C, dtype=float))

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(C, dtype=float))

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration."""
        return np.asarray(q, dtype=float) / self.K[:, None]

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid phase concentration by sorbed mass concentration."""
        return np.ones_like(np.asarray(q, dtype=float)) / self.K[:, None]


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
