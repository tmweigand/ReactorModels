"""Central numerics configuration for collocation and solver settings."""

import numpy as np

from .orthogonal_collocation import OrthogonalCollocation
from .time_integrator import TimeIntegrator


def _check_time_settings(rtol: float, atol: float, max_steps: int) -> None:
    if rtol <= 0:
        raise ValueError("rtol must be > 0")
    if atol <= 0:
        raise ValueError("atol must be > 0")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")


class TimeSettings:
    """IDA solver tolerances and the ``integrate`` entry point."""

    def __init__(self, rtol: float = 1e-8, atol: float = 1e-10, max_steps: int = 5000):
        _check_time_settings(rtol, atol, max_steps)
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self._integrator = TimeIntegrator(rtol, atol, max_steps)

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
        return self._integrator.integrate_ida(
            residual=residual,
            jacobian=jacobian,
            y0=y0,
            yp0=yp0,
            t_span=t_span,
            t_eval=t_eval,
            algebraic_vars_idx=algebraic_vars_idx,
        )


class SpatialDiscretization:
    """Orthogonal collocation settings for a single spatial domain.

    After construction the ready-to-use :class:`OrthogonalCollocation` instance
    is available as ``collocation``.
    """

    def __init__(
        self,
        domain_length: float,
        n_interior_points: int = 5,
        alpha: float = 0.0,
        beta: float = 0.0,
        add_inlet: bool = True,
        n_elements: int = 1,
    ):
        self.n_interior_points = n_interior_points
        self.alpha = alpha
        self.beta = beta
        self.add_inlet = add_inlet
        self.n_elements = n_elements
        self.domain_length = domain_length

        if self.n_interior_points < 1:
            raise ValueError("n_interior_points must be >= 1")
        if self.n_elements < 1:
            raise ValueError("n_elements must be >= 1")

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
        self.spatial_integrate = self.collocation.integrate
        self.average = self.collocation.average


class NumericsConfig(SpatialDiscretization, TimeSettings):
    """Numerical settings shared by single-domain transport models.

    This bundles the collocation discretization with IDA solver tolerances so
    users can pass one object into model constructors.  Time integration is
    performed via ``numerics.integrate(...)``.
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
        # Time settings first so bad tolerances fail before collocation setup.
        TimeSettings.__init__(self, rtol, atol, max_steps)
        SpatialDiscretization.__init__(
            self,
            domain_length=domain_length,
            n_interior_points=n_interior_points,
            alpha=alpha,
            beta=beta,
            add_inlet=add_inlet,
            n_elements=n_elements,
        )


class PSDMNumerics(TimeSettings):
    """Numerical settings for models coupling a column and a particle domain.

    Holds a separate :class:`SpatialDiscretization` for the column
    (``numerics.column``) and for the particle (``numerics.particle``), and one
    set of IDA solver tolerances for the coupled system.
    """

    def __init__(
        self,
        column: SpatialDiscretization,
        particle: SpatialDiscretization,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_steps: int = 5000,
    ):
        super().__init__(rtol, atol, max_steps)
        self.column = column
        self.particle = particle
