"""advection_diffusion.py"""

from functools import partial

import numpy as np
from scikits.odes import dae

from ..column_data import Column
from ..numerics.orthogonal_collocation import OrthogonalCollocation
from .boundary_conditions import InletBC, DirichletBC, DanckwertsBC


class AdvectionDiffusion:
    """Advection-diffusion model.

    Governing equation:
        eps*dC/dt + v * dC/dx - D * d2C/dx2 = 0

    """

    def __init__(
        self,
        column: Column,
        inlet_concentration: float,
        velocity: float,
        diffusion: float,
        orthogonal_collocation: OrthogonalCollocation,
        inlet_bc: InletBC = DirichletBC,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        self.column = column
        self.v = velocity
        self.D = diffusion
        self.inlet_concentration = inlet_concentration
        self.inlet_bc = inlet_bc(self.inlet_concentration)
        self.oc = orthogonal_collocation
        self.rtol = rtol
        self.atol = atol
        self.N = len(self.oc.nodes)

        # Scaled differentiation matrices (physical units)
        self._A = self.oc.first_derivative / self.column.length
        self._B = self.oc.second_derivative / self.column.length**2

    def _residual(self, t, C, Cdot, result, C_in):
        """IDA residual callback.  Writes into `result` in-place."""

        # Inlet boundary condition
        result[0] = self.inlet_bc.residual(C[0], self.oc.gradient(C, 0))

        # Inter nodes and outlet
        pde_rhs = -self.v * (self._A @ C) + self.D * (self._B @ C)
        result[1:] = Cdot[1:] - pde_rhs[1:]

        return 0

    def _jacobian(self, t, C, Cdot, result, cj, jac, C_in):
        J = np.zeros((self.N, self.N))

        # Row 0: algebraic constraint, no Cdot term
        J[0, :] = self.inlet_bc.jacobian_row(self._A[0, :])

        # Rows 1:: dF/dC = -pde_jac,  dF/dCdot = I  => full J row = -pde_jac + cj*I
        pde_jac = -self.v * self._A + self.D * self._B  # (N,N)
        J[1:, :] = -pde_jac[1:, :]  # dF/dC contribution
        for i in range(1, self.N):
            J[i, i] += cj  # dF/dCdot: Cdot[i] appears only in row i

        jac[:, :] = J
        return 0

    def _initial_conditions(self, C_init: float, C_in: float):
        """Return (C0, Cdot0) consistent with the algebraic constraint.

        C[1:] = C_init everywhere.
        C[0]  = C_in  (Dirichlet) or recovered from Danckwerts formula.
        Cdot0 = 0 everywhere; IDA will correct Cdot via calc_initcond.
        """
        C0 = np.full(self.N, C_init)
        C0[0] = self.inlet_bc.apply(self.oc.gradient(C0, 0))
        Cdot0 = np.zeros(self.N)
        return C0, Cdot0

    def solve(self, t_span, t_eval, C_in: float = 1.0, C_init: float = 0.0):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval.

        Returns
        -------
        x_physical : np.ndarray, shape (N,)
        C_history  : np.ndarray, shape (n_times, N)
        """
        C0, Cdot0 = self._initial_conditions(C_init, C_in)

        residual = partial(self._residual, C_in=C_in)
        jacobian = partial(self._jacobian, C_in=C_in)

        solver = dae(
            "ida",
            residual,
            jacfn=jacobian,
            old_api=False,
            rtol=self.rtol,
            atol=self.atol,
            algebraic_vars_idx=[0],
            compute_initcond="yp0",
            linsolver="dense",
            max_steps=5000,
        )

        t_out = np.concatenate([[t_span[0]], t_eval])
        result = solver.solve(t_out, C0, Cdot0)

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        x_physical = self.oc.nodes * self.column.length
        C_history = result.values.y[1:]  # (n_times, N), skip the t=t_span[0] row
        return x_physical, C_history
