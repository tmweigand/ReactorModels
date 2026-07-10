"""advection_diffusion.py"""

from typing import Type
import numpy as np

from ..column_data import Column
from ..breakthrough_data import Breakthrough
from ..numerics.config import NumericsConfig
from .boundary_conditions import InletBC, DirichletBC


class AdvectionDiffusion:
    """Advection-diffusion model.

    Governing equation:
        dC/dt + v * dC/dx - D * d2C/dx2 = 0

    """

    def __init__(
        self,
        column: Column,
        breakthrough: Breakthrough,
        initial_concentration: float,
        diffusion: float,
        numerics: NumericsConfig,
        inlet_bc: Type[InletBC] = DirichletBC,
    ):
        self.column = column
        self.velocity = breakthrough.interstitial_velocity()
        self.diffusion = diffusion
        self.inlet_concentration = breakthrough.mean_feed_concentration()
        self.initial_concentration = initial_concentration
        self.numerics = numerics
        self.inlet_bc = inlet_bc(
            self.inlet_concentration, self.velocity, self.diffusion
        )
        self.N = len(self.numerics.collocation.nodes)

    def _residual(self, t, C, Cdot, result):
        """IDA residual callback.  Writes into `result` in-place."""
        # Inlet boundary condition
        result[0] = self.inlet_bc.residual(
            C[0], self.numerics.collocation.evaluate_gradient(C, 0)
        )

        # Inter nodes and outlet
        rhs = -self.velocity * self.numerics.evaluate_gradient(
            C
        ) + self.diffusion * self.numerics.evaluate_second_derivative(C)
        result[1:] = Cdot[1:] - rhs[1:]

        return 0

    def _jacobian(self, t, C, Cdot, result, cj, jac):
        J = np.zeros((self.N, self.N))

        # Row 0: algebraic constraint, no Cdot term
        J[0, :] = self.inlet_bc.jacobian_row(
            self.numerics.collocation.first_derivative[0, :]
        )

        # Rows 1:: dF/dC = -pde_jac,  dF/dCdot = I  => full J row = -pde_jac + cj*I
        pde_jac = (
            -self.velocity * self.numerics.collocation.first_derivative
            + self.diffusion * self.numerics.collocation.second_derivative
        )  # (N,N)
        J[1:, :] = -pde_jac[1:, :]  # dF/dC contribution
        for i in range(1, self.N):
            J[i, i] += cj  # dF/dCdot: Cdot[i] appears only in row i

        jac[:, :] = J
        return 0

    def _initial_conditions(self):
        """Set the initial concentration and dcdt."""
        c = np.full(self.N, self.initial_concentration)
        c[0] = self.inlet_bc.apply(self.numerics.collocation.evaluate_gradient(c, 0))
        dcdt = np.zeros(self.N)
        return c, dcdt

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for this model.
        """
        return [0]

    def solve(self, t_span, t_eval):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        c, dcdt = self._initial_conditions()
        result = self.numerics.integrate(
            residual=self._residual,
            jacobian=self._jacobian,
            y0=c,
            yp0=dcdt,
            t_span=t_span,
            t_eval=t_eval,
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )
        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )
        C_history = result.values.y[1:]  # (n_times, N), skip the t=t_span[0] row
        return self.numerics.collocation.nodes, C_history
