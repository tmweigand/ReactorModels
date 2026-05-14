"""advection_diffusion.py"""

from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfc, erfcx

from ..numerics.orthogonal_collocation import OrthogonalCollocation
from .advection_diffusion_adsorption_two import InletBC


def ogata_banks(x, time, velocity, diffusion, C0=1.0):
    """Analytical solution for 1D advection-diffusion with step input.

    Uses erfcx (scaled erfc) to avoid overflow at high Peclet numbers:
        exp(Pe) * erfc(arg2) = erfcx(arg2) * exp(Pe - arg2^2)
    which stays finite when exp(Pe) would overflow.
    """
    Pe_local = velocity * x / diffusion
    arg1 = (x - velocity * time) / (2 * np.sqrt(diffusion * time))
    arg2 = (x + velocity * time) / (2 * np.sqrt(diffusion * time))
    exponent = Pe_local - arg2**2
    term2 = np.where(
        exponent > 500,  # still overflows — shouldn't happen but guard it
        0.0,
        erfcx(arg2) * np.exp(exponent),
    )

    return C0 * 0.5 * (erfc(arg1) + term2)


class AdvectionDiffusion1D:
    """One-dimensional advection diffusion equation.

    Solves
        dC/dt + v*dC/dx = D*d2C/dx2 on [0, L]

    Inlet boundary conditions (inlet_bc):
        DIRICHLET  : C(0, t) = C_in  (default)
        DANCKWERTS : v*C_in = v*C(0) - D*(dC/dx)|_0  (flux-conserving)

    Outlet boundary condition:
        dC/dx(L, t) = 0  (Neumann, zero-gradient)
    """

    def __init__(
        self,
        domain_length,
        velocity,
        diffusion,
        orthogonal_collocation: OrthogonalCollocation,
        inlet_bc: InletBC = InletBC.DIRICHLET,
    ):
        self.domain_length = domain_length
        self.velocity = velocity
        self.diffusion = diffusion
        self.oc = orthogonal_collocation
        self.inlet_bc = inlet_bc
        # Danckwerts: C[0] = (v*C_in + D*A[0,1:]@C[1:]) / (v - D*A[0,0])
        self._danckwerts_denom = (
            velocity
            - diffusion * orthogonal_collocation.first_derivative[0, 0] / domain_length
        )

    def _inlet_concentration(self, C_inner: np.ndarray, C_in: float) -> float:
        """Return C[0] from the chosen inlet BC given C[1:] and C_in."""
        if self.inlet_bc is InletBC.DIRICHLET:
            return C_in

        # DANCKWERTS: v*C_in = v*C[0] - D*(dC/dx)|_0
        A0inner = self.oc.first_derivative[0, 1:] / self.domain_length
        return (
            self.velocity * C_in + self.diffusion * (A0inner @ C_inner)
        ) / self._danckwerts_denom

    def rhs(self, t, C, C_in):
        """ODE rhs.

        DIRICHLET : C is (N,) including inlet; C[0] row is zeroed.
        DANCKWERTS: C is (N-1,) interior+outlet; C[0] recovered each call.
        """
        if self.inlet_bc is InletBC.DIRICHLET:
            dCdt = -(self.velocity / self.domain_length) * (
                self.oc.first_derivative @ C
            ) + (self.diffusion / self.domain_length**2) * (
                self.oc.second_derivative @ C
            )
            dCdt[0] = 0.0
            return dCdt
        else:
            C0 = self._inlet_concentration(C, C_in)
            C_full = np.concatenate([[C0], C])
            dCdt = -(self.velocity / self.domain_length) * (
                self.oc.first_derivative @ C_full
            ) + (self.diffusion / self.domain_length**2) * (
                self.oc.second_derivative @ C_full
            )
            return dCdt[1:]  # exclude algebraically-constrained inlet

    def jacobian(self, C_in):
        """Analytic Jacobian of rhs w.r.t. the state vector.

        DIRICHLET : (N, N) with inlet row zeroed.
        DANCKWERTS: (N-1, N-1) reduced system with C[0] eliminated.
        """
        A = self.oc.first_derivative / self.domain_length
        B = self.oc.second_derivative / self.domain_length**2
        J_full = -self.velocity * A + self.diffusion * B

        if self.inlet_bc is InletBC.DIRICHLET:
            J_full[0, :] = 0.0
            return J_full

        # DANCKWERTS: C_full = E @ C_inner + f*C_in
        # E[0,:] = D*A[0,1:] / denom;  E[1:,:] = I
        N = len(self.oc.nodes)
        E = np.zeros((N, N - 1))
        E[0, :] = self.diffusion * A[0, 1:] / self._danckwerts_denom
        E[1:, :] = np.eye(N - 1)
        return (J_full @ E)[1:, :]  # reduced (N-1, N-1) Jacobian

    def solve(self, t_span, t_eval, C_in=1.0, C_init=0.0):
        """Solves the governing equation."""
        N = len(self.oc.nodes)
        J = self.jacobian(C_in)
        rhs = partial(self.rhs, C_in=C_in)

        if self.inlet_bc is InletBC.DIRICHLET:
            y0 = np.full(N, C_init)
            y0[0] = C_in

            sol = solve_ivp(
                rhs,
                t_span,
                y0,
                method="BDF",
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-10,
                jac=J,
            )
            x_physical = self.oc.nodes * self.domain_length
            return x_physical, sol.y.T  # (n_times, N)

        else:  # DANCKWERTS: state is C[1:]
            y0 = np.full(N - 1, C_init)

            sol = solve_ivp(
                rhs,
                t_span,
                y0,
                method="BDF",
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-10,
                jac=J,
            )
            x_physical = self.oc.nodes * self.domain_length
            C_inner_history = sol.y.T  # (n_times, N-1)
            # Recover C[0] at each output time
            C0_history = np.array(
                [
                    self._inlet_concentration(C_inner_history[i], C_in)
                    for i in range(len(t_eval))
                ]
            )[:, np.newaxis]
            C_out = np.concatenate(
                [C0_history, C_inner_history], axis=1
            )  # (n_times, N)
            return x_physical, C_out
