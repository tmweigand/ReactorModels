"""advection_diffusion_adsorption.py"""

from __future__ import annotations

from functools import partial

import numpy as np
from scikits.odes import dae
from typing import TYPE_CHECKING

from ..column_data import Column
from .isotherm import Isotherm
from .adsorption_kinetics import AdsorptionKinetics
from .boundary_conditions import InletBC, DirichletBC, DanckwertsBC

if TYPE_CHECKING:
    from ..numerics.orthogonal_collocation import OrthogonalCollocation

__all__ = ["AdvectionDiffusionAdsorption"]


class AdvectionDiffusionAdsorption:
    """1D advection-diffusion with adsorption in a packed bed.

    LOCAL_EQUILIBRIUM:
        R(C) * dC/dt + v/eps * dC/dx = D * d2C/dx2
        R(C) = 1 + (rho_b/eps_b) * dq*/dC
        state vector: [C_0, ..., C_N]
        Row 0 is an algebraic constraint (inlet BC); rows 1: are differential.

    LINEAR_DRIVING_FORCE:
        dC/dt = -(v/eps) * dC/dx + DL * d2C/dx2 - (rho_b/eps_b) * dq/dt
        dq/dt = k_ldf * (q*(C) - q)
        state vector: [C_0, ..., C_N, q_0, ..., q_N]
        Row 0 is an algebraic constraint (inlet BC); all other rows are differential.

    Inlet boundary conditions (inlet_bc):
        DIRICHLET  : C(0, t) = C_in
            Simple concentration pin. Does not conserve total flux when DL > 0.
        DANCKWERTS : v*C_in = (v/eps)*C(0) - DL*(dC/dx)|_0
            Flux-conserving BC. C[0] is algebraically constrained at each step.

    Outlet boundary condition (both modes):
        dC/dx(L) = 0      (Neumann outlet)

    """

    def __init__(
        self,
        column: Column,
        velocity: float,
        dispersion: float,
        inlet_concentration: float,
        isotherm: Isotherm,
        oc: OrthogonalCollocation,
        mode: AdsorptionKinetics = AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_ldf: float = 0.0,
        inlet_bc: InletBC = DanckwertsBC,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        self.column = column
        self.v = velocity
        self.DL = dispersion
        self.inlet_concentration = inlet_concentration
        self.iso = isotherm
        self.oc = oc
        self.mode = mode
        self.k_ldf = k_ldf
        self.inlet_bc = inlet_bc(self.inlet_concentration, velocity, dispersion)
        self.rtol = rtol
        self.atol = atol
        self.N = len(oc.nodes)

        if mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE and k_ldf <= 0:
            raise ValueError("k_ldf must be > 0 for LINEAR_DRIVING_FORCE mode")

        # Scaled differentiation matrices (physical units)
        self.A = oc.first_derivative / self.column.length
        self.B = oc.second_derivative / self.column.length**2

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            return 2 * self.N
        return self.N

    def _split(self, y: np.ndarray):
        """Return (C, q) where q is None for LOCAL_EQUILIBRIUM."""
        C = y[: self.N]
        q = (
            y[self.N :]
            if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE
            else None
        )
        return C, q

    def _residual(self, t, y, ydot, result, C_in):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        C, q = self._split(y)
        Cdot, qdot = self._split(ydot)

        # --- Row 0: algebraic inlet BC (no time-derivative term) ---
        if isinstance(self.inlet_bc, DirichletBC):
            result[0] = C[0] - C_in
        else:
            # Danckwerts: v*C_in = (v/eps)*C[0] - DL*(A[0,:] @ C)
            result[0] = (
                self.v * C_in
                - (self.v / self.column.porosity) * C[0]
                + self.DL * (self.A[0, :] @ C)
            )

        # --- PDE residuals for interior + outlet nodes (rows 1:N) ---
        adv = -(self.v / self.column.porosity) * (self.A @ C)
        diff = self.DL * (self.B @ C)

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            R = 1.0 + (
                self.column.bulk_density / self.column.porosity
            ) * self.iso.dq_dC(C)
            # F[i] = R*Cdot[i] - (adv + diff)[i] = 0
            result[1 : self.N] = R[1:] * Cdot[1:] - (adv + diff)[1:]

        else:  # LINEAR_DRIVING_FORCE
            dqdt_eq = self.k_ldf * (self.iso.q(C) - q)
            # C residuals (rows 1:N)
            result[1 : self.N] = (
                Cdot[1:]
                - (adv + diff)[1:]
                + (self.column.bulk_density / self.column.porosity) * qdot[1:]
            )
            # q residuals (rows N : 2N)
            result[self.N :] = qdot - dqdt_eq

        return 0

    def _jacobian(self, t, y, ydot, result, cj, jac, C_in):
        C, q = self._split(y)
        n = self._n_vars()
        J = np.zeros((n, n))

        # --- Row 0: algebraic inlet BC ---
        if isinstance(self.inlet_bc, DirichletBC):
            J[0, 0] = 1.0
        else:
            J[0, : self.N] = self.DL * self.A[0, :]
            J[0, 0] += -(self.v / self.column.porosity)

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            R = 1.0 + (
                self.column.bulk_density / self.column.porosity
            ) * self.iso.dq_dC(C)
            dRdC = (self.column.bulk_density / self.column.porosity) * self.iso.d2q_dC2(
                C
            )
            pde = -(self.v / self.column.porosity) * self.A + self.DL * self.B  # (N,N)

            for i in range(1, self.N):
                # dF/dC contributions
                J[i, : self.N] = -pde[i, :]
                J[i, i] += dRdC[i] * ydot[i]  # d(R*Cdot)/dC[i] extra term
                # dF/dCdot: R[i] * delta_{ij}
                J[i, i] += cj * R[i]

        else:  # LINEAR_DRIVING_FORCE
            pde = -(self.v / self.column.porosity) * self.A + self.DL * self.B  # (N,N)
            dqdC = self.iso.dq_dC(C)  # (N,)

            # C block rows (1:N)
            for i in range(1, self.N):
                J[i, : self.N] = -pde[i, :]  # dF_C / dC
                J[i, i] += cj  # dF_C / dCdot: +cj
                J[i, self.N + i] += cj * (
                    self.column.bulk_density / self.column.porosity
                )  # dF_C / dqdot

            # q block rows (N:2N)
            for i in range(self.N):
                J[self.N + i, i] = -self.k_ldf * dqdC[i]  # dF_q / dC
                J[self.N + i, self.N + i] = self.k_ldf + cj  # dF_q / dq + cj*I

        jac[:, :] = J
        return 0

    def _initial_conditions(self, C_init: float, C_in: float, q_init: float):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.N, C_init)

        # Satisfy inlet BC at t=0
        if isinstance(self.inlet_bc, DirichletBC):
            C0[0] = C_in
        else:
            C0[0] = (self.v * C_in + self.DL * (self.A[0, 1:] @ C0[1:])) / (
                self.v / self.column.porosity - self.DL * self.A[0, 0]
            )

        if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            q0 = np.full(self.N, q_init)
            q0[0] = self.iso.q(C_in)  # inlet node at equilibrium with feed
            y0 = np.concatenate([C0, q0])
        else:
            y0 = C0.copy()

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def solve(self, t_span, t_eval, C_in=1.0, C_init=0.0, q_init=0.0):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval.

        Returns
        -------
        x       : np.ndarray, shape (N,)
        C_out   : np.ndarray, shape (n_times, N)
        q_out   : np.ndarray, shape (n_times, N)
        """
        y0, ydot0 = self._initial_conditions(C_init, C_in, q_init)

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
        result = solver.solve(t_out, y0, ydot0)

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        x = self.oc.nodes * self.column.length

        # result.values.y has shape (n_out, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]  # (n_times, n_vars)

        C_out = y_out[:, : self.N]  # (n_times, N)

        if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            q_out = y_out[:, self.N :]  # (n_times, N)
        else:
            # Local equilibrium: recover q from C at each time step
            q_out = np.array([self.iso.q(C_out[i]) for i in range(len(t_eval))])

        return x, C_out, q_out
