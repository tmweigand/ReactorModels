"""advection_diffusion_adsorption.py"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from typing import TYPE_CHECKING


from .isotherm import Isotherm
from .adsorption_kinetics import AdsorptionKinetics
from .boundary_conditions import InletBC

if TYPE_CHECKING:
    from ..numerics.orthogonal_collocation import OrthogonalCollocation

__all__ = ["AdvectionDiffusionAdsorption1D_two"]


class AdvectionDiffusionAdsorption1D_two:
    """1D advection-diffusion with adsorption in a packed bed.

    LOCAL_EQUILIBRIUM:
        R(C) * dC/dt + v/eps * dC/dx = D * d2C/dx2
        R(C) = 1 + (rho_b/eps_b) * dq*/dC
        state vector: [C_0, ..., C_N]

    LINEAR_DRIVING_FORCE:
        dC/dt = -v * dC/dx + DL * d2C/dx2 - (rho_b/eps_b) * dq/dt
        dq/dt = k_ldf * (q*(C) - q)
        state vector: [C_0, ..., C_N, q_0, ..., q_N]

    Inlet boundary conditions (inlet_bc):
        DIRICHLET  : C(0, t) = C_in
            Simple concentration pin. Does not conserve total flux when DL > 0;
            introduces a spurious dispersive source at the inlet.
        DANCKWERTS : v*C_in = v*C(0) - DL*(dC/dx)|_0
            Flux-conserving BC. C[0] is an algebraic function of the interior
            nodes, recovered at each RHS evaluation. Satisfies exact mass balance.

    Outlet boundary condition (both modes):
        dC/dx(L) = 0      (Neumann outlet)

    Parameters
    ----------
    column_length : float       L [m]
    velocity      : float       interstitial velocity [m/s]
    dispersion    : float       axial dispersion DL [m^2/s]
    isotherm      : Isotherm
    bulk_density  : float       rho_b [kg/m^3]
    porosity      : float       eps_b [-]
    oc            : OrthogonalCollocation   (add_inlet=True)
    mode          : AdsorptionMode
    k_ldf         : float       [1/s]  required for LDF mode
    inlet_bc      : InletBC     DANCKWERTS (default) or DIRICHLET

    """

    def __init__(
        self,
        column_length: float,
        velocity: float,
        dispersion: float,
        isotherm: Isotherm,
        bulk_density: float,
        porosity: float,
        oc: OrthogonalCollocation,
        mode: AdsorptionKinetics = AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_ldf: float = 0.0,
        inlet_bc: InletBC = InletBC.DANCKWERTS,
    ):
        self.L = column_length
        self.v = velocity
        self.DL = dispersion
        self.iso = isotherm
        self.rho_b = bulk_density
        self.eps_b = porosity
        self.oc = oc
        self.mode = mode
        self.k_ldf = k_ldf
        self.inlet_bc = inlet_bc
        self.N = len(oc.nodes)

        if mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE and k_ldf <= 0:
            raise ValueError("k_ldf must be > 0 for LINEAR_DRIVING_FORCE mode")

        # Scaled differentiation matrices (physical units)
        self.A = oc.first_derivative / column_length
        self.B = oc.second_derivative / column_length**2

        # Danckwerts BC: v*C_in = v*C[0] - DL*(dC/dx)|_0
        # => C[0] = (v*C_in + DL*(A[0,1:]@C[1:])) / (v - DL*A[0,0])
        self._danckwerts_denom = self.v / self.eps_b - self.DL * self.A[0, 0]

    def _inlet_concentration(self, C_inner: np.ndarray, C_in: float) -> float:
        """Return C[0] consistent with the chosen inlet BC given C[1:] and C_in."""
        if self.inlet_bc is InletBC.DIRICHLET:
            return C_in
        return (
            self.v * C_in + self.DL * np.dot(self.A[0, 1:], C_inner)
        ) / self._danckwerts_denom

    def _unpack(self, y):
        """Break the solution vector into solution variables."""
        C_inner = y[: self.N - 1]
        q = (
            y[self.N - 1 :]
            if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE
            else None
        )
        return C_inner, q

    def _pack(self, C_inner, q=None):
        if q is not None:
            return np.concatenate([C_inner, q])
        return C_inner.copy()

    def initial_state(self, C0=0.0, q0=0.0):
        """Set initial conditions."""
        C_inner = np.full(self.N - 1, C0)
        if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            q = np.full(self.N, q0)
            return self._pack(C_inner, q)
        return self._pack(C_inner)

    def rhs(self, t, y, C_in):
        """Set the right hand side"""
        C_inner, q = self._unpack(y)

        # Reconstruct full concentration vector via Danckwerts BC
        C_0 = self._inlet_concentration(C_inner, C_in)
        C = np.concatenate([[C_0], C_inner])

        # adv = -self.v * (self.A @ C)
        adv = -(self.v / self.eps_b) * (self.A @ C)
        diff = self.DL * (self.B @ C)

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            R = 1.0 + (self.rho_b / self.eps_b) * self.iso.dq_dC(C)
            dCdt = (adv + diff) / R
            return self._pack(dCdt[1:])  # exclude inlet (algebraically constrained)

        else:  # LINEAR_DRIVING_FORCE
            dqdt = self.k_ldf * (self.iso.q(C) - q)
            dCdt = adv + diff - (self.rho_b / self.eps_b) * dqdt
            return self._pack(dCdt[1:], dqdt)  # C[0] excluded; q[:] all evolve freely

    def solve(self, t_span, t_eval, C_in=1.0, C_init=0.0, q_init=0.0):
        """Solve the governing equations."""
        y0 = self.initial_state(C0=C_init, q0=q_init)
        # For LDF, initialize q[0] at equilibrium with the feed
        if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            y0[self.N - 1] = self.iso.q(C_in)
        rhs = partial(self.rhs, C_in=C_in)

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method="BDF",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )

        x = self.oc.nodes * self.L
        C_inner_history = sol.y[: self.N - 1, :].T  # (n_times, N-1)

        # Reconstruct C[0] at each output time from Danckwerts BC
        C_0_history = np.array(
            [
                self._inlet_concentration(C_inner_history[i], C_in)
                for i in range(len(t_eval))
            ]
        )[:, np.newaxis]
        C_out = np.concatenate([C_0_history, C_inner_history], axis=1)  # (n_times, N)

        if self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            q_out = sol.y[self.N - 1 :, :].T  # (n_times, N)
            return x, C_out, q_out

        # For local equilibrium, compute q from C
        q_out = np.array([self.iso.q(C_out[i]) for i in range(len(t_eval))])
        return x, C_out, q_out
