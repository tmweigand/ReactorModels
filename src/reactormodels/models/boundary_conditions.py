"""boundary_conditions.py"""

import numpy as np


class InletBC:
    """Base class for inlet boundary conditions.

    All inlet boundary condition classes should inherit from this class
    and implement the `residual` and `apply` methods.
    """

    def __init__(self, inlet_concentration, velocity=None, diffusion=None):
        self.inlet_concentration = np.asarray(inlet_concentration, dtype=float)

    def apply(self, gradient_concentration_0: None | float = None):
        """Return the value on the boundary"""
        raise NotImplementedError

    def residual(
        self,
        concentration_0: float | np.ndarray,
        gradient_concentration_0: float | np.ndarray | None = None,
    ):
        """Compute the residual at inlet boundary."""
        raise NotImplementedError

    def jacobian_row(self, A_row: np.ndarray) -> np.ndarray:
        """Return dF_inlet/dC, the Jacobian row for the inlet constraint."""
        raise NotImplementedError


class DirichletBC(InletBC):
    """Dirichlet inlet boundary condition.

    Enforces a fixed inlet concentration:
        C(0, t) = C_in
    """

    def __init__(self, inlet_concentration, velocity=None, diffusion=None):
        self.inlet_concentration = np.asarray(inlet_concentration, dtype=float)

    def apply(self, gradient_concentration_0: None | float = None):
        """Return the value on the boundary"""
        return self.inlet_concentration.copy()

    def residual(
        self, concentration_0: float, gradient_concentration_0: None | float = None
    ):
        """Compute the residual at inlet boundary."""
        return concentration_0 - self.inlet_concentration

    def jacobian_row(self, A_row: np.ndarray) -> np.ndarray:
        """dF/dC: residual = C[0] - C_in, so only the C[0] entry is non-zero."""
        row = np.zeros_like(A_row)
        row[0] = 1.0
        return row


class DanckwertsBC(InletBC):
    """Danckwerts (flux-conserving) inlet boundary condition.

    Governing equation:
        v*C_in = v*C(0) - DL*(dC/dx)|_0
    """

    def __init__(self, inlet_concentration, velocity, diffusion):
        self.inlet_concentration = np.asarray(inlet_concentration, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.diffusion = np.asarray(diffusion, dtype=float)

    def apply(self, gradient_concentration_0: None | float = None):
        """Return the value on the boundary"""
        return (
            self.inlet_concentration
            + (self.diffusion / self.velocity) * gradient_concentration_0
        )

    def residual(
        self, concentration_0: float, gradient_concentration_0: None | float = None
    ):
        """Compute the residual at inlet boundary."""
        if gradient_concentration_0 is None:
            raise ValueError(
                "gradient_concentration_0 must be provided " "for DanckwertsBC"
            )

        return (
            self.velocity * (self.inlet_concentration - concentration_0)
            + self.diffusion * gradient_concentration_0
        )

    def jacobian_row(self, A_row: np.ndarray) -> np.ndarray:
        """dF/dC: residual = v*(C_in - C[0]) + D*(A[0,:] @ C)."""
        row = self.diffusion * A_row
        row[0] -= self.velocity
        return row
