"""boundary_conditions.py"""

import numpy as np


class BoundaryConditions:
    """Base class for boundary conditions.

    All boundary condition classes should inherit from this class
    and implement the `residual` and `apply` methods.
    """

    def __init__(
        self, inlet_concentration=None, node: int = 0, velocity=None, diffusion=None
    ):
        raise NotImplementedError

    def apply(
        self,
        initial_concentrations: np.ndarray,
        first_derivative_row: np.ndarray,
    ):
        """Return the value on the boundary"""
        raise NotImplementedError

    def residual(
        self, concentration_0: float, gradient_concentration_0: None | float = None
    ):
        """Compute the residual at inlet boundary."""
        raise NotImplementedError

    def jacobian_row(self, first_derivative_row: np.ndarray) -> np.ndarray:
        """Return dF_inlet/dC, the Jacobian row for the inlet constraint."""
        raise NotImplementedError


class DirichletBC(BoundaryConditions):
    """Dirichlet inlet boundary condition.

    Enforces a fixed inlet concentration:
        C(0, t) = C_in
    """

    def __init__(
        self, inlet_concentration, node: int = 0, velocity=None, diffusion=None
    ):
        self.inlet_concentration = inlet_concentration
        self.node = node

    def apply(
        self,
        initial_concentrations: np.ndarray | None = None,
        first_derivative_row: np.ndarray | None = None,
    ):
        """Return the value on the boundary"""
        return self.inlet_concentration

    def residual(
        self,
        concentration_0: float,
        first_derivative_row: None | float = None,
    ):
        """Compute the residual at inlet boundary."""
        return concentration_0 - self.inlet_concentration

    def jacobian_row(self, first_derivative_row: np.ndarray) -> np.ndarray:
        """dF/dC: residual = C[0] - C_in, so only the C[0] entry is non-zero."""
        row = np.zeros_like(first_derivative_row)
        row[self.node] = 1.0
        return row


class DanckwertsBC(BoundaryConditions):
    """Danckwerts (flux-conserving) inlet boundary condition.

    Governing equation:
        v*C_in = v*C(0) - DL*(dC/dx)|_0
    """

    def __init__(self, inlet_concentration, velocity, diffusion, node: int = 0):
        self.inlet_concentration = inlet_concentration
        self.velocity = velocity
        self.diffusion = diffusion

    def apply(
        self,
        initial_concentrations: np.ndarray,
        first_derivative_row: np.ndarray,
    ):
        """Return the value on the boundary"""
        numerator = self.velocity * self.inlet_concentration + self.diffusion * (
            first_derivative_row[1:] @ initial_concentrations[1:]
        )
        denominator = self.velocity - self.diffusion * first_derivative_row[0]

        return numerator / denominator

    def residual(
        self, concentration_0: float, gradient_concentration_0: None | float = None
    ):
        """Compute the residual at inlet boundary."""
        if gradient_concentration_0 is None:
            raise ValueError(
                "gradient_concentration_0 must be provided for DanckwertsBC."
            )

        return (
            self.velocity * (self.inlet_concentration - concentration_0)
            + self.diffusion * gradient_concentration_0
        )

    def jacobian_row(self, first_derivative_row: np.ndarray) -> np.ndarray:
        """dF/dC: residual = v*(C_in - C[0]) + D*(A[0,:] @ C)."""
        row = self.diffusion * first_derivative_row
        row[0] -= self.velocity
        return row


class SymmetryBC(BoundaryConditions):
    """Zero gradient boundary condition at particle center.

    Enforces dC/dr = 0 at r=0
    """

    def __init__(self, node: int = 0, velocity=None, diffusion=None):
        self.node = node

    def apply(
        self,
        initial_concentrations: None | np.ndarray = None,
        first_derivative_row: None | np.ndarray = None,
    ):
        """Symmetry does not set a concentration value directly."""
        raise NotImplementedError("SymmetryBC has no direct value; use residual().")

    def residual(
        self,
        concentration_0: float | None = None,
        first_derivative_row: None | float = None,
    ):
        """Compute the residual: the gradient itself, which must equal zero."""
        if first_derivative_row is None:
            raise ValueError("first_derivative_row must be provided for SymmetryBC")
        return first_derivative_row

    def jacobian_row(self, first_derivative_row: np.ndarray) -> np.ndarray:
        """dF/dC: residual = first_derivative_row @ C, so the Jacobian row is first_derivative_row itself."""
        return first_derivative_row
