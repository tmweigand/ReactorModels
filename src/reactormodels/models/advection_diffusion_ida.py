"""advection_diffusion_ida.py"""

from functools import partial

import numpy as np
from scikits.odes import dae
from scipy.special import erfc, erfcx

from ..numerics.orthogonal_collocation import OrthogonalCollocation
from .boundary_conditions import InletBC


def ogata_banks(x, time, velocity, diffusion, C0=1.0):
    """Analytical solution for 1D advection-diffusion with step input."""
    Pe_local = velocity * x / diffusion
    arg1 = (x - velocity * time) / (2 * np.sqrt(diffusion * time))
    arg2 = (x + velocity * time) / (2 * np.sqrt(diffusion * time))
    exponent = Pe_local - arg2**2
    term2 = np.where(
        exponent > 500,
        0.0,
        erfcx(arg2) * np.exp(exponent),
    )
    return C0 * 0.5 * (erfc(arg1) + term2)


class AdvectionDiffusion1DIDA:
    """1-D advection-diffusion solved as a DAE with SUNDIALS IDA.

    The full N-vector is the IDA state.  Row 0 is always an algebraic
    constraint (no time derivative); rows 1: are differential.

    Inlet BC options
    ----------------
    DIRICHLET  : F[0] = C[0] - C_in = 0
    DANCKWERTS : F[0] = v*C_in - v*C[0] + D*(A[0,:]/L) @ C = 0

    Outlet BC (both modes)
    ----------------------
    Neumann zero-gradient is enforced implicitly: the last collocation
    node has no explicit BC row — it evolves freely under the PDE, which
    naturally satisfies dC/dx = 0 at the outlet for the collocation setup.
    """

    def __init__(
        self,
        domain_length: float,
        velocity: float,
        diffusion: float,
        orthogonal_collocation: OrthogonalCollocation,
        inlet_bc: InletBC = InletBC.DIRICHLET,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        self.L = domain_length
        self.v = velocity
        self.D = diffusion
        self.oc = orthogonal_collocation
        self.inlet_bc = inlet_bc
        self.rtol = rtol
        self.atol = atol

        N = len(self.oc.nodes)
        self.N = N

        self._A = self.oc.first_derivative / self.L  # (N,N)
        self._B = self.oc.second_derivative / self.L**2  # (N,N)

    def _residual(self, t, C, Cdot, result, C_in):
        """IDA residual callback.  Writes into `result` in-place."""

        if self.inlet_bc is InletBC.DIRICHLET:
            result[0] = C[0] - C_in
        else:
            result[0] = self.v * C_in - self.v * C[0] + self.D * (self._A[0, :] @ C)

        pde_rhs = -self.v * (self._A @ C) + self.D * (self._B @ C)
        result[1:] = Cdot[1:] - pde_rhs[1:]

        return 0

    def _jacobian(self, t, C, Cdot, result, cj, jac, C_in):
        J = np.zeros((self.N, self.N))

        # Row 0: algebraic constraint, no Cdot term
        if self.inlet_bc is InletBC.DIRICHLET:
            J[0, 0] = 1.0
        else:
            J[0, :] = self.D * self._A[0, :]
            J[0, 0] += -self.v

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

        if self.inlet_bc is InletBC.DIRICHLET:
            C0[0] = C_in
        else:
            C0[0] = (self.v * C_in + self.D * (self._A[0, 1:] @ C0[1:])) / (
                self.v - self.D * self._A[0, 0]
            )

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

        x_physical = self.oc.nodes * self.L
        C_history = result.values.y[1:]  # (n_times, N), skip the t=t_span[0] row
        return x_physical, C_history
