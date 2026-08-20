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

    def C(self, q: float | np.ndarray):
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if self.n_species == 1:
            return self._C_single_species(q)

        else:
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

    def dC_dq(self, q: float | np.ndarray):
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if self.n_species == 1:
            return self._dC_dq_single_species(q)

        else:
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
                    J[:, :, k] = self.dC_dq(q[:, k])

                return J

            raise ValueError("q must be 1D or 2D")

    def _C_single_species(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        return (q / self.K[0]) ** self.n[0]

    def _dC_dq_single_species(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        return self.n[0] * self.K[0] ** (-self.n[0]) * q ** (self.n[0] - 1)


class CompetitiveIonIsotherm(Isotherm):
    """Isotherm used by EPA's IX-ECM.

    Equation:
        C_i = (C_T * q_i) / sum(beta_ij * q_j)

        K_ij = (q_i / C_i)^z_j * (C_j / q_j)^z_i

    Parameters
    ----------
    K : float
        Binary separation factor

    """

    def __init__(
        self,
        K: float | np.ndarray,
        MW: float | np.ndarray,
        valence: float | np.ndarray,
        inlet_concentrations: float | np.ndarray,
        capacity: float | np.ndarray,
        bulk_density: float,
        reference_concentration: float = 0,
        reference_mw: float = 35.45,  # chloride default
        reference_z: int = 1,
    ):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.MW = np.atleast_1d(np.asarray(MW, dtype=float))
        self.z = np.atleast_1d(np.asarray(valence, dtype=float))
        self.Co = np.atleast_1d(np.asarray(inlet_concentrations, dtype=float))
        self.q_m = capacity
        self.C_Ao = reference_concentration
        self.MW_A = reference_mw
        self.z_A = reference_z
        self.rho_b = bulk_density

        if (
            self.K.shape != self.MW.shape
            or self.K.shape != self.z.shape
            or self.K.shape != self.Co.shape
        ):
            raise ValueError("K, n, MW, and z must have the same shape.")

        if np.any(self.K <= 0):
            raise ValueError("K must be greater than zero.")

        if np.any(self.MW <= 0):
            raise ValueError("MW must be greater than zero.")

        if np.any(self.Co <= 0):
            raise ValueError("Inlet concentrations must be greater than zero.")

        if np.any((self.z != 1) & (self.z != 2)):
            raise ValueError("Only monovalent and divalent ions are supported.")

        if self.q_m <= 0:
            raise ValueError("Capacity must be greater than zero.")

        self.n_species = len(self.K)

        self.mono_mask = self.z == 1
        self.di_mask = self.z == 2

        # calculate total charge equivalent concentration
        self.CT = np.sum(self.Co * self.z / self.MW) + self.C_Ao * self.z_A / self.MW_A

    def C(self, q: float | np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if self.n_species == 1:
            q_eq = q * self.rho_b * self.z / self.MW

            q_A = self.q_m - q_eq

            if q_A <= 0:
                print("\nINVALID IEX-CM STATE")
                print("q =", q)
                print("q_eq =", q)
                print("q_m =", self.q_m)
                print("sum(q_eq) =", np.sum(q))
                print("q_A =", q_A)

            a = np.sum(q_eq[self.di_mask] / self.K[self.di_mask]) / q_A**2

            b = 1.0 + np.sum(q_eq[self.mono_mask] / self.K[self.mono_mask]) / q_A

            c = -self.CT

            C_A = 2.0 * c / (-b - np.sqrt(b**2 - 4.0 * a * c))

            return (C_A / q_A) ** self.z * q_eq / self.K

        if q.ndim == 1:
            return self._C_1d(q)

        if q.ndim == 2:
            return self._C_2d(q)

        raise ValueError("q must be 1D or 2D")

    def _C_1d(self, q):
        q_eq = q * self.rho_b * self.z / self.MW

        q_A = self.q_m - np.sum(q_eq)

        q_mono = q_eq[self.mono_mask]
        q_di = q_eq[self.di_mask]

        a = np.sum(q_di / self.K[self.di_mask]) / q_A**2

        b = 1.0 + np.sum(q_mono / self.K[self.mono_mask]) / q_A

        c = -self.CT

        D = np.sqrt(b**2 - 4.0 * a * c)

        C_A = 2.0 * c / (-b - D)

        return (C_A / q_A) ** self.z * q_eq / self.K

    def _C_2d(self, q):
        if q.shape[0] != self.n_species:
            raise ValueError(
                f"Expected q.shape[0] == {self.n_species}, " f"got {q.shape[0]}"
            )

        return np.column_stack([self._C_1d(q[:, j]) for j in range(q.shape[1])])

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        if q.ndim == 1:
            return self._dC_dq_1d(q)

        if q.ndim == 2:
            if q.shape[0] != self.n_species:
                raise ValueError(
                    f"Expected q.shape[0] == {self.n_species}, " f"got {q.shape[0]}"
                )

            return np.stack(
                [self._dC_dq_1d(q[:, j]) for j in range(q.shape[1])],
                axis=2,
            )

        raise ValueError("q must be 1D or 2D")

    def _dC_dq_1d(self, q):
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        # Conversion from mass-based q to charge-equivalent q
        q_scale = self.rho_b * self.z / self.MW
        q_eq = q * q_scale

        q_A = self.q_m - np.sum(q_eq)

        q_mono = q_eq[self.mono_mask]
        q_di = q_eq[self.di_mask]

        a = np.sum(q_di / self.K[self.di_mask]) / q_A**2

        b = 1.0 + np.sum(q_mono / self.K[self.mono_mask]) / q_A

        c = -self.CT

        D = np.sqrt(b**2 - 4.0 * a * c)

        C_A = 2.0 * c / (-b - D)

        # da/dq_eq
        da_dq = np.full(
            self.n_species,
            2.0 * a / q_A,
        )

        da_dq[self.di_mask] += 1.0 / (self.K[self.di_mask] * q_A**2)

        # db/dq_eq
        db_dq = np.full(
            self.n_species,
            (b - 1.0) / q_A,
        )

        db_dq[self.mono_mask] += 1.0 / (self.K[self.mono_mask] * q_A)

        # dD/dq_eq
        dD_dq = (b * db_dq - 2.0 * c * da_dq) / D

        # dC_A/dq_eq
        dCA_dq = C_A * (db_dq + dD_dq) / (-b - D)

        # R = C_A / q_A
        R = C_A / q_A

        # dR/dq_eq
        dR_dq = dCA_dq / q_A + C_A / q_A**2

        J_eq = np.zeros((self.n_species, self.n_species))

        for i in range(self.n_species):
            coef = R ** self.z[i] / self.K[i]

            # Direct derivative of q_i
            J_eq[i, i] += coef

            # Coupled derivative through R
            J_eq[i, :] += q_eq[i] / self.K[i] * self.z[i] * R ** (self.z[i] - 1) * dR_dq

        # Convert dC/dq_eq -> dC/dq
        return J_eq * q_scale[None, :]


class LinearIsotherm(Isotherm):
    """Linear isotherm: q* = K * C

    Parameters
    ----------
    K : float | np.ndarray
        Henry constant [mg/g / (mg/L)].
    """

    def __init__(self, K: float | np.ndarray, coupled: bool = False):
        self.K = np.atleast_1d(np.asarray(K, dtype=float))
        self.coupled = coupled
        self.n_species = len(self.K)

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration."""
        C = np.asarray(C, dtype=float)

        if self.n_species == 1:
            return self.K[0] * C

        if C.ndim == 1:
            return self.K * C

        if C.ndim == 2:
            return self.K[:, None] * C

        raise ValueError("C must be scalar, 1D, or 2D")

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate dq*/dC."""
        C = np.asarray(C, dtype=float)

        if self.n_species == 1:
            return np.full_like(C, self.K[0], dtype=float)

        if C.ndim == 1:
            return self.K.copy()

        if C.ndim == 2:
            return np.broadcast_to(
                self.K[:, None],
                C.shape,
            ).copy()

        raise ValueError("C must be scalar, 1D, or 2D")

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(C, dtype=float))

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration."""
        q = np.asarray(q, dtype=float)

        if self.n_species == 1:
            return q / self.K[0]

        if q.ndim == 1:
            return q / self.K

        if q.ndim == 2:
            return q / self.K[:, None]

        raise ValueError("q must be scalar, 1D, or 2D")

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate dC/dq."""
        q = np.asarray(q, dtype=float)

        if self.n_species == 1:
            return np.full_like(q, 1.0 / self.K[0], dtype=float)

        if q.ndim == 1:
            return 1.0 / self.K

        if q.ndim == 2:
            return np.broadcast_to(
                1.0 / self.K[:, None],
                q.shape,
            ).copy()

        raise ValueError("q must be scalar, 1D, or 2D")


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
