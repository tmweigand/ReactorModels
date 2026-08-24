"""isotherm.py"""

import numpy as np
from scipy.optimize import curve_fit


class Isotherm:
    """Base class for adsorption isotherms."""

    output = "q"

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

    output = "q"

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

    output = "q"

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

    def C(self, q):
        """Return liquid phase concentration"""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        return (q / self.K) ** self.n

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        return self.n * (1 / self.K) ** self.n * q ** (self.n - 1)

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)
        return self.n * (self.n - 1) * (1 / self.K) ** self.n * q ** (self.n - 2)


class LinearIsotherm(Isotherm):
    """Linear isotherm: q* = K * C

    Parameters
    ----------
    K : float
        Henry constant  [mg/g / (mg/L)]

    """

    output = "q"

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


class CompetitiveFreundlichIsotherm(Isotherm):
    """Competitive Freundlich isotherm:

    C_i = q_i / sum(q_j) * [sum(n_j*q_j) / (n_i*K_i)]^n_i

    Parameters
    ----------
    K : float
        Freundlich capacity factor
    n : float
        Freundlich intensity factor.

    """

    output = "C"

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

    def C(self, q: float | np.ndarray):
        """Return liquid phase concentration."""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        Q = np.sum(q)
        S = np.sum(self.n * q)

        # prevent C from blowing up
        if Q == 0:
            return np.zeros_like(q)

        return q / Q * (S / (self.n * self.K)) ** self.n

    def dC_dq(self, q: float | np.ndarray):
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        q_arr: np.ndarray = np.asarray(q, dtype=float)
        q_arr = np.maximum(q_arr, 0.0)

        Q = np.sum(q_arr)
        S = np.sum(self.n * q_arr)

        J = np.zeros((self.n_species, self.n_species))

        if Q == 0 or S == 0:
            return J

        for i in range(self.n_species):
            A_i = (S / (self.n[i] * self.K[i])) ** self.n[i]

            for j in range(self.n_species):
                delta_ij = 1.0 if i == j else 0.0

                J[i, j] = A_i * (
                    delta_ij / Q + q_arr[i] / Q * (self.n[i] * self.n[j] / S - 1 / Q)
                )

        return J

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        q_arr: np.ndarray = np.asarray(q, dtype=float)
        q_arr = np.maximum(q_arr, 0.0)

        Q = np.sum(q_arr)
        S = np.sum(self.n * q_arr)

        H = np.zeros((self.n_species, self.n_species, self.n_species))

        if Q == 0 or S == 0:
            return H

        C = self.C(q_arr)

        for i in range(self.n_species):
            for j in range(self.n_species):
                for k in range(self.n_species):
                    delta_ij = 1.0 if i == j else 0.0
                    delta_ik = 1.0 if i == k else 0.0

                    if q_arr[i] == 0:
                        d_ij_over_q = 0
                        d_ik_over_q = 0
                        d_ijk_over_q = 0
                    else:
                        d_ij_over_q = delta_ij / q_arr[i]
                        d_ik_over_q = delta_ik / q_arr[i]
                        d_ijk_over_q = delta_ij * delta_ik / q_arr[i] ** 2

                    L_ij = d_ij_over_q + self.n[i] * self.n[j] / S - 1.0 / Q

                    L_ik = d_ik_over_q + self.n[i] * self.n[k] / S - 1.0 / Q

                    H[i, j, k] = C[i] * (
                        L_ij * L_ik
                        - d_ijk_over_q
                        - self.n[i] * self.n[j] * self.n[k] / S**2
                        + 1.0 / Q**2
                    )

        return H


class CompetitiveLangmuirIsotherm(Isotherm):
    """Competitive Langmuir isotherm:

    q_i = q_m * K_i * C_i / (1 + sum(K_j*C_j))

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.

    """

    output = "q"

    def __init__(self, K: float | np.ndarray, q_m: float):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.q_m = q_m

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if self.q_m <= 0:
            raise ValueError("q_m must be greater than zero.")

        self.n_species = len(self.K)

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 0.0)

        D = 1 + np.sum(self.K * C)

        return self.q_m * self.K * C / D

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        D = 1 + np.sum(self.K * C_arr)

        J = np.zeros((self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                delta_ij = 1.0 if i == j else 0.0

                J[i, j] = (
                    self.q_m * self.K[i] * (delta_ij / D - C_arr[i] * self.K[j] / D**2)
                )

        return J

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        D = 1.0 + np.sum(self.K * C_arr)

        H = np.zeros((self.n_species, self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                for k in range(self.n_species):
                    delta_ij = 1.0 if i == j else 0.0
                    delta_ik = 1.0 if i == k else 0.0

                    H[i, j, k] = (
                        self.q_m
                        * self.K[i]
                        / D**2
                        * (
                            -delta_ij * self.K[k]
                            - delta_ik * self.K[j]
                            + 2.0 * C_arr[i] * self.K[j] * self.K[k] / D
                        )
                    )

        return H


class CompetitiveIonIsotherm(Isotherm):
    """Isotherm used by EPA's IX-ECM.

    Equation:
        C_i = (C_T * q_i) / sum(K_ij * q_j)

        K_ij = (q_i / C_i)^z_j * (C_j / q_j)^z_i

    Parameters
    ----------
    K : float
        Binary separation factor

    """

    output = "C"

    def __init__(
        self,
        K: float | np.ndarray,
        MW: float | np.ndarray,
        valence: float | np.ndarray,
        inlet_concentrations: float | np.ndarray,
        q_m: float,
        bulk_density: float,
        reference_concentration: float = 0,
        reference_mw: float = 35.45,  # chloride default
        reference_z: int = 1,
    ):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.MW = np.atleast_1d(np.asarray(MW, dtype=float))
        self.z = np.atleast_1d(np.asarray(valence, dtype=float))
        self.C_o = np.atleast_1d(np.asarray(inlet_concentrations, dtype=float))
        self.q_m = q_m
        self.C_Ao = reference_concentration
        self.MW_A = reference_mw
        self.z_A = reference_z
        self.rho_b = bulk_density

        if (
            self.K.shape != self.MW.shape
            or self.K.shape != self.z.shape
            or self.K.shape != self.C_o.shape
        ):
            raise ValueError("K, n, MW, and z must have the same shape.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if np.any(self.MW <= 0):
            raise ValueError("MW must be greater than zero.")

        if np.any(self.C_o <= 0):
            raise ValueError("Inlet concentrations must be greater than zero.")

        if np.any((self.z != 1) & (self.z != 2)):
            raise ValueError("Only monovalent and divalent ions are supported.")

        if self.z_A != 1:
            raise ValueError("Only monovalent reference ions are supported.")

        if self.q_m <= 0:
            raise ValueError("Capacity must be greater than zero.")

        self.n_species = len(self.K)

        self.mono_mask = self.z == 1
        self.di_mask = self.z == 2

        # calculate total charge equivalent concentration
        self.CT = np.sum(self.C_o * self.z / self.MW) + self.C_Ao * self.z_A / self.MW_A

    def _state_quantities(self, q):
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        q_scale = self.rho_b * self.z / self.MW
        q_eq = q * q_scale
        q_A = self.q_m - np.sum(q_eq)

        a = np.sum(q_eq[self.di_mask] / self.K[self.di_mask]) / q_A**2
        b = 1.0 + np.sum(q_eq[self.mono_mask] / self.K[self.mono_mask]) / q_A
        c = -self.CT

        C_A = 2.0 * c / (-b - np.sqrt(b**2 - 4.0 * a * c))

        return q_scale, q_eq, q_A, C_A

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration."""
        _, q_eq, q_A, C_A = self._state_quantities(q)
        return (C_A / q_A) ** self.z * q_eq / self.K

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        q_scale, q_eq, q_A, C_A = self._state_quantities(q)

        C = self.C(q)
        Z_c = np.sum(self.z * C)

        J = np.zeros((self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                delta_ij = 1.0 if i == j else 0.0

                if q_eq[i] == 0:
                    d_over_q = 0
                else:
                    d_over_q = delta_ij / q_eq[i]

                Cj_over_qj = (C_A / q_A) ** self.z[j] / self.K[j]

                J[i, j] = C[i] * (
                    d_over_q
                    + self.z[i] / q_A
                    - self.z[i] * (Cj_over_qj + Z_c / q_A) / (C_A + Z_c)
                )

        return J * q_scale

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        q_scale, q_eq, q_A, C_A = self._state_quantities(q)

        C = self.C(q)

        # Monovalent and divalent sums
        M = np.sum(q_eq[self.mono_mask] / self.K[self.mono_mask])
        N = np.sum(q_eq[self.di_mask] / self.K[self.di_mask])

        a = N / q_A**2
        b = 1.0 + M / q_A

        # First derivatives of M and N
        dM = np.zeros(self.n_species)
        dN = np.zeros(self.n_species)

        dM[self.mono_mask] = q_scale[self.mono_mask] / self.K[self.mono_mask]

        dN[self.di_mask] = q_scale[self.di_mask] / self.K[self.di_mask]

        # First derivatives of a and b
        da = dN / q_A**2 + 2.0 * N * q_scale / q_A**3

        db = dM / q_A + M * q_scale / q_A**2

        # First derivative of C_A
        denom = 2.0 * a * C_A + b

        dCA = -(da * C_A**2 + db * C_A) / denom

        # Second derivatives of a and b
        d2a = np.zeros((self.n_species, self.n_species))
        d2b = np.zeros((self.n_species, self.n_species))

        for j in range(self.n_species):
            for k in range(self.n_species):
                d2a[j, k] = (
                    6.0 * N * q_scale[j] * q_scale[k] / q_A**4
                    + 2.0 * dN[j] * q_scale[k] / q_A**3
                    + 2.0 * dN[k] * q_scale[j] / q_A**3
                )

                d2b[j, k] = (
                    dM[j] * q_scale[k] / q_A**2
                    + dM[k] * q_scale[j] / q_A**2
                    + 2.0 * M * q_scale[j] * q_scale[k] / q_A**3
                )

        # Second derivative of C_A
        d2CA = np.zeros((self.n_species, self.n_species))

        for j in range(self.n_species):
            for k in range(self.n_species):
                d2CA[j, k] = (
                    -(
                        2.0 * a * dCA[j] * dCA[k]
                        + (2.0 * da[j] * C_A + db[j]) * dCA[k]
                        + (2.0 * da[k] * C_A + db[k]) * dCA[j]
                        + d2a[j, k] * C_A**2
                        + d2b[j, k] * C_A
                    )
                    / denom
                )

        # Hessian of C
        H = np.zeros((self.n_species, self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                for k in range(self.n_species):

                    delta_ij = 1.0 if i == j else 0.0
                    delta_ik = 1.0 if i == k else 0.0

                    if q_eq[i] == 0:
                        d_ij_over_q = 0
                        d_ik_over_q = 0
                        d_ijk_over_q = 0
                    else:
                        d_ij_over_q = delta_ij / q_eq[i]
                        d_ik_over_q = delta_ik / q_eq[i]
                        d_ijk_over_q = delta_ij * delta_ik / q_eq[i] ** 2

                    # First logarithmic derivatives
                    L_ij = d_ij_over_q * q_scale[j] + self.z[i] * (
                        dCA[j] / C_A + q_scale[j] / q_A
                    )

                    L_ik = d_ik_over_q * q_scale[k] + self.z[i] * (
                        dCA[k] / C_A + q_scale[k] / q_A
                    )

                    # Second logarithmic derivative
                    L_ijk = -d_ijk_over_q * q_scale[j] * q_scale[k] + self.z[i] * (
                        d2CA[j, k] / C_A
                        - dCA[j] * dCA[k] / C_A**2
                        + q_scale[j] * q_scale[k] / q_A**2
                    )

                    H[i, j, k] = C[i] * (L_ij * L_ik + L_ijk)

        return H


class CompetitiveLangmuirFreundlichIsotherm(Isotherm):
    """Competitive Langmuir isotherm:

    q_i = q_m * K_i * C_i^n_i / (1 + sum(K_j*(C_j)^n_j))

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.
    n : float
        Freundlich intensity factor.

    """

    output = "q"

    def __init__(self, K: float | np.ndarray, n: float | np.ndarray, q_m: float):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.n = np.atleast_1d(np.asarray(n, dtype=float))
        self.q_m = q_m

        if self.K.shape != self.n.shape:
            raise ValueError("K and n must be the same shape.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if np.any(self.n <= 0):
            raise ValueError("n must be greater than zero.")

        if self.q_m <= 0:
            raise ValueError("q_m must be greater than zero.")

        self.n_species = len(self.K)

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 0.0)

        D = 1 + np.sum(self.K * C**self.n)

        return self.q_m * self.K * C**self.n / D

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        D = 1 + np.sum(self.K * C_arr**self.n)

        J = np.zeros((self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                delta_ij = 1.0 if i == j else 0.0

                J[i, j] = (
                    self.q_m
                    * self.K[i]
                    / D
                    * (
                        delta_ij * self.n[i] * C_arr[i] ** (self.n[i] - 1)
                        - C_arr[i] ** self.n[i]
                        * self.K[j]
                        * self.n[j]
                        * C_arr[j] ** (self.n[j] - 1)
                        / D
                    )
                )

        return J

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 0.0)

        D = 1.0 + np.sum(self.K * C**self.n)

        A = self.K * C**self.n
        B = self.K * self.n * C ** (self.n - 1)
        E = self.K * self.n * (self.n - 1) * C ** (self.n - 2)

        H = np.zeros((self.n_species, self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                for k in range(self.n_species):
                    delta_ij = 1.0 if i == j else 0.0
                    delta_ik = 1.0 if i == k else 0.0
                    delta_jk = 1.0 if j == k else 0.0

                    H[i, j, k] = self.q_m * (
                        delta_ij * delta_ik * E[i] / D
                        - delta_ij * B[i] * B[k] / D**2
                        - delta_ik * B[i] * B[j] / D**2
                        - A[i] * delta_jk * E[j] / D**2
                        + 2.0 * A[i] * B[j] * B[k] / D**3
                    )

        return H


class CompetitiveStoichiometricIsotherm(Isotherm):
    """Competitive Adsorption Isothermal Model:

    q_i = q_m * K_i * n_i * C_i^n_i / (1 + sum(K_j*(C_j)^n_j))

    n_iM_i + R_s <=> R_sM_i,n_i

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.
    n : float
        Adsorption stoichiometric factor.

    """

    output = "q"

    def __init__(self, K: float | np.ndarray, n: float | np.ndarray, q_m: float):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.n = np.atleast_1d(np.asarray(n, dtype=float))
        self.q_m = q_m

        if self.K.shape != self.n.shape:
            raise ValueError("K and n must be the same shape.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if np.any(self.n <= 0):
            raise ValueError("n must be greater than zero.")

        if self.q_m <= 0:
            raise ValueError("q_m must be greater than zero.")

        self.n_species = len(self.K)

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 0.0)

        D = 1 + np.sum(self.K * C**self.n)

        return self.q_m * self.K * self.n * C**self.n / D

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        D = 1 + np.sum(self.K * C_arr**self.n)

        J = np.zeros((self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                delta_ij = 1.0 if i == j else 0.0

                J[i, j] = (
                    self.q_m
                    * self.K[i]
                    * self.n[i]
                    / D
                    * (
                        delta_ij * self.n[i] * C_arr[i] ** (self.n[i] - 1)
                        - C_arr[i] ** self.n[i]
                        * self.K[j]
                        * self.n[j]
                        * C_arr[j] ** (self.n[j] - 1)
                        / D
                    )
                )

        return J

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 0.0)

        D = 1.0 + np.sum(self.K * C**self.n)

        A = self.K * C**self.n
        B = self.K * self.n * C ** (self.n - 1)
        E = self.K * self.n * (self.n - 1) * C ** (self.n - 2)

        H = np.zeros((self.n_species, self.n_species, self.n_species))

        for i in range(self.n_species):
            for j in range(self.n_species):
                for k in range(self.n_species):
                    delta_ij = 1.0 if i == j else 0.0
                    delta_ik = 1.0 if i == k else 0.0
                    delta_jk = 1.0 if j == k else 0.0

                    H[i, j, k] = (
                        self.n[i]
                        * self.q_m
                        * (
                            delta_ij * delta_ik * E[i] / D
                            - delta_ij * B[i] * B[k] / D**2
                            - delta_ik * B[i] * B[j] / D**2
                            - A[i] * delta_jk * E[j] / D**2
                            + 2.0 * A[i] * B[j] * B[k] / D**3
                        )
                    )

        return H


class MultiCapacityIsotherm(Isotherm):
    """Multi-Capacity Langmuir-type Isotherm:

    q_1 = q_m,2*K_1*C_1 / (1+K_1C_1+K_2C_2) + (q_m,1-q_m,2)K_1C_1 / (1+K_1C_1)
    q_2 = q_m,2*K_2*C_2 / (1+K_1C_1+K_2C_2)

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.

    """

    output = "q"

    def __init__(self, K: float | np.ndarray, q_m: float | np.ndarray):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.q_m = np.atleast_1d(np.asarray(q_m, dtype=float))

        if self.K.shape != (2,) or self.q_m.shape != (2,):
            raise ValueError("Only binary systems are supported.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if np.any(self.q_m <= 0):
            raise ValueError("q_m must be greater than zero.")

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        # Sort species by descending q_m
        order = np.argsort(-self.q_m)

        qm = self.q_m[order]
        K = self.K[order]
        C_sorted = C_arr[order]

        # Calculate q in sorted order
        q_sorted = np.zeros(2)

        D = 1 + np.sum(self.K * C)

        q_sorted[0] = qm[1] * K[0] * C_sorted[0] / D + (qm[0] - qm[1]) * K[
            0
        ] * C_sorted[0] / (1 + K[0] * C_sorted[0])

        q_sorted[1] = qm[1] * K[1] * C_sorted[1] / D

        # Return to original species ordering
        q = np.empty_like(q_sorted)
        q[order] = q_sorted

        return q

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        # Sort species by descending q_m
        order = np.argsort(-self.q_m)

        qm = self.q_m[order]
        K = self.K[order]
        C_sorted = C_arr[order]

        D = 1 + np.sum(self.K * C_arr)

        J_sorted = np.zeros((2, 2))

        J_sorted[0, 0] = (
            qm[1] * K[0] * (1 + K[1] * C_sorted[1]) / D**2
            + (qm[0] - qm[1]) * K[0] / (1 + K[0] * C_sorted[0]) ** 2
        )
        J_sorted[0, 1] = -qm[1] * K[0] * K[1] * C_sorted[0] / D**2
        J_sorted[1, 0] = -qm[1] * K[0] * K[1] * C_sorted[1] / D**2
        J_sorted[1, 1] = qm[1] * K[1] * (1 + K[0] * C_sorted[0]) / D**2

        J = np.empty_like(J_sorted)
        J = J_sorted[np.ix_(order, order)]

        return J

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        # Sort species by descending q_m
        order = np.argsort(-self.q_m)

        qm = self.q_m[order]
        K = self.K[order]
        C_sorted = C_arr[order]

        D = 1 + np.sum(K * C_sorted)
        E = 1 + K[0] * C_sorted[0]

        H_sorted = np.zeros((2, 2, 2))

        # Hessian of q_1
        H_sorted[0, 0, 0] = (
            -2 * qm[1] * K[0] ** 2 * (1 + K[1] * C_sorted[1]) / D**3
            - 2 * (qm[0] - qm[1]) * K[0] ** 2 / E**3
        )

        H_sorted[0, 0, 1] = (
            qm[1] * K[0] * K[1] * (K[0] * C_sorted[0] - K[1] * C_sorted[1] - 1) / D**3
        )

        H_sorted[0, 1, 0] = H_sorted[0, 0, 1]

        H_sorted[0, 1, 1] = 2 * qm[1] * K[0] * C_sorted[0] * K[1] ** 2 / D**3

        # Hessian of q_2
        H_sorted[1, 0, 0] = 2 * qm[1] * K[1] * C_sorted[1] * K[0] ** 2 / D**3

        H_sorted[1, 0, 1] = (
            qm[1] * K[0] * K[1] * (K[1] * C_sorted[1] - K[0] * C_sorted[0] - 1) / D**3
        )

        H_sorted[1, 1, 0] = H_sorted[1, 0, 1]

        H_sorted[1, 1, 1] = -2 * qm[1] * K[1] ** 2 * (1 + K[0] * C_sorted[0]) / D**3

        # Transform Hessian back to original species ordering
        H = H_sorted[np.ix_(order, order, order)]

        return H


class AdsorbateComplexIsotherm(Isotherm):
    """Adsorbate-complex Langmuir-type Isotherm:

    q_1 = q_m*K_1*C_1 (1 + (K/K_1)C_2) / (1+K_1C_1+K_2C_2 + 2*KC_1C_2)
    q_2 = q_m*K_2*C_2 (1 + (K/K_2)C_1) / (1+K_1C_1+K_2C_2 + 2*KC_1C_2)

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.

    """

    output = "q"

    def __init__(self, K: float | np.ndarray, q_m: float, K_x: float):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.q_m = q_m
        self.K_x = K_x

        if self.K.shape != (2,):
            raise ValueError("Only binary systems are supported.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if self.q_m <= 0:
            raise ValueError("q_m must be greater than zero.")

        if self.K_x <= 0:
            raise ValueError("K_x must be greater than zero.")

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        D = 1 + np.sum(self.K * C_arr) + 2 * self.K_x * np.prod(C_arr)

        q = np.zeros(2)
        q[0] = (
            self.q_m * self.K[0] * C_arr[0] * (1 + self.K_x * C_arr[1] / self.K[0]) / D
        )
        q[1] = (
            self.q_m * self.K[1] * C_arr[1] * (1 + self.K_x * C_arr[0] / self.K[1]) / D
        )

        return q

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        D = 1 + np.sum(self.K * C_arr) + 2 * self.K_x * np.prod(C_arr)

        J = np.zeros((2, 2))

        J[0, 0] = (
            self.q_m
            * self.K[0]
            * (1 + self.K_x * C_arr[1] / self.K[0])
            * (1 + self.K[1] * C_arr[1])
            / D**2
        )
        J[0, 1] = (
            self.q_m
            * C_arr[0]
            * (self.K_x * (1 - self.K[0] * C_arr[0]) - self.K[0] * self.K[1])
            / D**2
        )
        J[1, 0] = (
            self.q_m
            * C_arr[1]
            * (self.K_x * (1 - self.K[1] * C_arr[1]) - self.K[0] * self.K[1])
            / D**2
        )
        J[1, 1] = (
            self.q_m
            * self.K[1]
            * (1 + self.K_x * C_arr[0] / self.K[1])
            * (1 + self.K[0] * C_arr[0])
            / D**2
        )

        return J

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C_arr: np.ndarray = np.asarray(C, dtype=float)
        C_arr = np.maximum(C_arr, 0.0)

        C1, C2 = C_arr
        K1, K2 = self.K

        qm = self.q_m
        Kx = self.K_x

        # Denominator
        D = 1 + K1 * C1 + K2 * C2 + 2 * Kx * C1 * C2

        D_j = np.array(
            [
                K1 + 2 * Kx * C2,
                K2 + 2 * Kx * C1,
            ]
        )

        D_jk = np.array(
            [
                [0.0, 2 * Kx],
                [2 * Kx, 0.0],
            ]
        )

        # Numerators
        N = np.array(
            [
                qm * (K1 * C1 + Kx * C1 * C2),
                qm * (K2 * C2 + Kx * C1 * C2),
            ]
        )

        # First derivatives of numerators
        N_j = np.array(
            [
                [
                    qm * (K1 + Kx * C2),
                    qm * Kx * C1,
                ],
                [
                    qm * Kx * C2,
                    qm * (K2 + Kx * C1),
                ],
            ]
        )

        # Second derivatives of numerators
        N_jk = np.array(
            [
                [
                    [0.0, qm * Kx],
                    [qm * Kx, 0.0],
                ],
                [
                    [0.0, qm * Kx],
                    [qm * Kx, 0.0],
                ],
            ]
        )

        H = np.zeros((2, 2, 2))

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    H[i, j, k] = (
                        N_jk[i, j, k] / D
                        - (N_j[i, j] * D_j[k] + N_j[i, k] * D_j[j] + N[i] * D_jk[j, k])
                        / D**2
                        + 2 * N[i] * D_j[j] * D_j[k] / D**3
                    )

        return H


def flatten_parameters(parameters):
    """Flatten multi-species isotherm parameters for fitting."""
    flat = []

    for parameter in parameters:
        flat.extend(np.asarray(parameter).ravel())

    return np.asarray(flat, dtype=float)


def unflatten_parameters(flat, template):
    """Return fitting parameters to shape of initial guesses."""
    flat = np.asarray(flat, dtype=float)

    parameters = []
    i = 0

    for parameter in template:
        parameter = np.asarray(parameter)

        n = parameter.size

        values = flat[i : i + n]
        i += n

        if parameter.ndim == 0:
            parameters.append(values[0])
        else:
            parameters.append(values.reshape(parameter.shape))

    return tuple(parameters)


def fit_isotherm(
    isotherm_class: type[Isotherm],
    xdata: float | np.ndarray,
    ydata: float | np.ndarray,
    initial_guess: tuple[float, ...],
    fit_indices: tuple[int, ...],
    parameter_template: tuple[object, ...],
    output=None,
) -> Isotherm:
    """Fit an isotherm to equilibrium concentration data.

    Parameters
    ----------
    isotherm_class : type[Isotherm]
        Isotherm class to fit.
    xdata : float | np.ndarray
        Independent equilibrium concentration data.
    ydata : float | np.ndarray
        Dependent equilibrium concentration data.
    initial_guess : tuple[float, ...]
        Initial guess for fitting parameters.
    fit_indices : tuple[int, ...]
        Indices of the constructor parameters to be fitted.
    parameter_template : tuple[object, ...]
        Complete set of isotherm constructor parameters, in the same
        order as the isotherm constructor. Values corresponding to
        ``fit_indices`` are replaced during fitting.
    output : str, optional
        Variable being fitted. If ``"q"``, fit q as a function of C.
        If ``"C"``, fit C as a function of q. If None, use the
        default output specified by the isotherm class.

    Isotherm
        Fitted isotherm instance.

    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    if xdata.shape != ydata.shape:
        raise ValueError("C and q must have the same shape.")
    if np.any(xdata < 0) or np.any(ydata < 0):
        raise ValueError("C and q values must be nonnegative.")

    if output is None:
        output = isotherm_class.output

    if any(i < 0 or i >= len(parameter_template) for i in fit_indices):
        raise ValueError(
            "All fit_indices must correspond to parameters in " "parameter_template."
        )

    if len(initial_guess) != len(fit_indices):
        raise ValueError(
            "initial_guess must contain one value for each " "parameter in fit_indices."
        )

    template = tuple(
        np.asarray(parameter) if np.asarray(parameter).ndim > 0 else parameter
        for parameter in initial_guess
    )

    p0 = flatten_parameters(template)

    def model(x, *flat_parameters):
        fitted_parameters = unflatten_parameters(
            flat_parameters,
            template,
        )

        parameters = list(parameter_template)

        for index, value in zip(fit_indices, fitted_parameters):
            parameters[index] = value

        isotherm = isotherm_class(*parameters)

        function = isotherm.q if output == "q" else isotherm.C

        return np.array([function(x_i) for x_i in x]).ravel()

    bounds = (np.full(len(p0), 1e-30), np.full(len(p0), np.inf))

    popt, _ = curve_fit(
        model,
        xdata,
        ydata.ravel(),
        p0=p0,
        bounds=bounds,
    )

    fitted_parameters = unflatten_parameters(
        popt,
        template,
    )

    parameters = list(parameter_template)

    for index, value in zip(fit_indices, fitted_parameters):
        parameters[index] = value

    return isotherm_class(*parameters)
