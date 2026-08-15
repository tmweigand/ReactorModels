"""Central numerics configuration for collocation and solver settings."""

import numpy as np

from .orthogonal_collocation import OrthogonalCollocation
from .time_integrator import TimeIntegrator


class NumericsConfig:
    """Numerical settings shared by transport models.

    This bundles collocation discretization options with IDA solver tolerances
    so users can pass one object into model constructors.  After construction
    the ready-to-use :class:`OrthogonalCollocation` instance is available as
    ``numerics.collocation``, and time integration is performed via
    ``numerics.integrate(...)``.
    """

    def __init__(
        self,
        domain_length: float,
        n_interior_points: int = 5,
        alpha: float = 0.0,
        beta: float = 0.0,
        add_inlet: bool = True,
        n_elements: int = 1,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_steps: int = 5000,
    ):
        self.n_interior_points = n_interior_points
        self.alpha = alpha
        self.beta = beta
        self.add_inlet = add_inlet
        self.n_elements = n_elements
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.domain_length = domain_length

        if self.n_interior_points < 1:
            raise ValueError("n_interior_points must be >= 1")
        if self.n_elements < 1:
            raise ValueError("n_elements must be >= 1")
        if self.rtol <= 0:
            raise ValueError("rtol must be > 0")
        if self.atol <= 0:
            raise ValueError("atol must be > 0")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        self.collocation = OrthogonalCollocation(
            domain_length=self.domain_length,
            n_interior_points=self.n_interior_points,
            alpha=self.alpha,
            beta=self.beta,
            add_inlet=self.add_inlet,
            n_elements=self.n_elements,
        )
        self.evaluate_gradient = self.collocation.evaluate_gradient
        self.evaluate_second_derivative = self.collocation.evaluate_second_derivative
        self.evaluate_radial_operator = self.collocation.evaluate_radial_operator

        self.time_integrator = TimeIntegrator(self.rtol, self.atol, self.max_steps)

    def integrate(
        self,
        residual,
        y0: np.ndarray,
        yp0: np.ndarray,
        t_span: tuple,
        t_eval: np.ndarray,
        algebraic_vars_idx: list,
        jacobian=None,
    ):
        """Integrate a DAE system using the SUNDIALS IDA solver."""
        return self.time_integrator.integrate_ida(
            residual=residual,
            jacobian=jacobian,
            y0=y0,
            yp0=yp0,
            t_span=t_span,
            t_eval=t_eval,
            algebraic_vars_idx=algebraic_vars_idx,
        )
