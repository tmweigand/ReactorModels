"""advection_diffusion_adsorption.py"""

from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


from .isotherm import Isotherm
from ..numerics.orthogonal_collocation import OrthogonalCollocation


class AdvectionDiffusionAdsorption1D:
    """Solve the advection-diffusion-adsorption equation on [0, L]:

        R(C) * dC/dt + v/eps * dC/dx = D * d2C/dx2

    where R(C) = 1 + (rho_b/eps) * dq*/dC  is the retardation factor.

    Assumes local equilibrium (q = q*(C) instantaneously).

    Boundary conditions:
        C(0, t) = C_in   (Dirichlet inlet)
        dC/dx(L, t) = 0  (Neumann outlet)

    Parameters
    ----------
        domain_length       : float   L [m]
        velocity            : float   pore velocity v [m/s]
        diffusion           : float   dispersion coefficient D [m^2/s]
        isotherm            : Isotherm
        bulk_density        : float   rho_b [kg/m^3]
        porosity            : float   eps [-]
        orthogonal_collocation : OrthogonalCollocation

    """

    def __init__(
        self,
        domain_length: float,
        velocity: float,
        diffusion: float,
        isotherm: Isotherm,
        bulk_density: float,
        porosity: float,
        orthogonal_collocation: OrthogonalCollocation,
        upwind: bool = False,
    ):
        self.domain_length = domain_length
        self.velocity = velocity
        self.diffusion = diffusion
        self.isotherm = isotherm
        self.rho_b = bulk_density
        self.eps = porosity
        self.oc = orthogonal_collocation
        self.upwind = upwind
        if upwind:
            self._A_upwind = self._build_upwind(self.oc.nodes)

        # Precompute scaled linear parts
        self._A = self.oc.first_derivative / domain_length
        self._B = self.oc.second_derivative / domain_length**2

    def _retardation(self, C: np.ndarray) -> np.ndarray:
        """R(C) = 1 + (rho_b/eps) * dq*/dC  — vectorized over nodes."""
        return 1.0 + (self.rho_b / self.eps) * self.isotherm.dq_dC(C)

    def _build_upwind(self, nodes):
        """First-order upwind matrix on the collocation nodes.

        For left-to-right flow (v > 0): dC/dx_i ≈ (C_i - C_{i-1}) / (x_i - x_{i-1})
        """
        N = len(nodes)
        A_up = np.zeros((N, N))
        for i in range(1, N):
            dx = nodes[i] - nodes[i - 1]
            A_up[i, i] = 1.0 / dx
            A_up[i, i - 1] = -1.0 / dx
        A_up[0, 0] = 1.0  # inlet row (will be pinned anyway)
        return A_up

    def rhs(self, t: float, C: np.ndarray, C_in: float) -> np.ndarray:
        """Right hand side."""
        A = self._A_upwind / self.domain_length if self.upwind else self._A
        adv = -(self.velocity / self.eps) * (A @ C)
        diff = self.diffusion * (self._B @ C)
        R = self._retardation(C)

        dCdt = (adv + diff) / R

        dCdt[0] = 0.0  # pin inlet Dirichlet
        return dCdt

    def jacobian(self, C: np.ndarray) -> np.ndarray:
        """Analytic Jacobian d(rhs)/dC.

        For nonlinear isotherm R depends on C, so by quotient rule:
            d/dC [ (adv+diff)/R ] = (d(adv+diff)/dC * R - (adv+diff) * dR/dC) / R^2

        dR/dC = (rho_b/eps) * d2q*/dC2  (diagonal matrix)
        d(adv+diff)/dC = -v/eps * A + D * B  (the linear operator)
        """
        R = self._retardation(C)
        adv_diff = -(self.velocity / self.eps) * (self._A @ C) + self.diffusion * (
            self._B @ C
        )
        L_op = -(self.velocity / self.eps) * self._A + self.diffusion * self._B

        # dR/dC is diagonal: (rho_b/eps) * d2q/dC2
        d2q = self.isotherm.d2q_dC2(C)
        dR_dC = (self.rho_b / self.eps) * d2q  # shape (N,)

        # Quotient rule: J = (L @ C * R - adv_diff * dR_dC) / R^2
        # L_op contributes: diag(1/R) @ L_op
        # dR_dC contributes: -diag(adv_diff * dR_dC / R^2)
        J = (
            L_op * (1.0 / R)[:, None]
            - np.outer(adv_diff * dR_dC, np.ones(len(C))) / (R**2)[:, None]
        )

        J[0, :] = 0.0  # inlet pinned
        return J

    def jacobian_ivp(self, t: float, C: np.ndarray) -> np.ndarray:
        """Adapter with solve_ivp Jacobian signature jac(t, y)."""
        return self.jacobian(C)

    def solve(
        self,
        t_span,
        t_eval,
        C_in: float = 1.0,
        C_init: float = 0.0,
    ):
        """Solve the governing equation."""
        N = len(self.oc.nodes)
        y0 = np.full(N, C_init)
        y0[0] = C_in
        rhs = partial(self.rhs, C_in=C_in)

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method="BDF",
            t_eval=t_eval,
            jac=self.jacobian_ivp,
            rtol=1e-8,
            atol=1e-10,
        )

        x_physical = self.oc.nodes * self.domain_length
        return x_physical, sol.y.T  # (n_times, N)
