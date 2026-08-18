"""intraparticle_transport.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .boundary_conditions import InletBC, DirichletBC, SymmetryBC

__all__ = ["IntraparticleTransport"]


class IntraparticleTransport(NumericModel):
    """1D transport through pore and/or surface diffusion in a spherical particle.

    The governing equations are:

        Fluid phase: eps_p*dCp/dt - 1/(r^2)*d/dr[r^2*eps_p*D_p*dC_p/dr]
                    + rho_p*dq/dt  - 1/(r^2)*d/dr[r^2*rho_s*D_s*dq/dr] = 0

        Solid phase: dq_dt = dq*/dc*dc/dt

    """

    _param_names = (
        "pore_diffusion",
        "surface_diffusion",
        "iso",
        "numerics",
        "surface_bc",
        "center_bc",
    )

    def __init__(
        self,
        breakthrough: Breakthrough,
        isotherm: Isotherm,
        numerics: NumericsConfig,
        surface_bc: Type[InletBC] = DirichletBC,
        center_bc: Type[InletBC] = SymmetryBC,
    ):
        self.breakthrough = breakthrough
        self.media = breakthrough.column.media
        self.pore_diffusion = breakthrough.chemical.pore_diffusion
        self.surface_diffusion = breakthrough.chemical.surface_diffusion
        self.iso = isotherm
        self.numerics = numerics
        self.N = len(self.numerics.collocation.nodes)
        self.surface_bc = surface_bc(breakthrough.mean_feed_concentration(), node=-1)
        self.center_bc = center_bc(node=0)
        self.assert_parameters_set()

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        return self.N

    def _split(self, y: np.ndarray):
        """Return (C, q) where q is None for LOCAL_EQUILIBRIUM."""
        C = y[: self.N]
        q = None
        return C, q

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, q = self._split(y)
        dcdt, dqdt = self._split(ydot)

        # Center: symmetry
        result[0] = self.center_bc.residual(
            gradient_concentration_0=self.numerics.collocation.evaluate_gradient(c, 0)
        )

        # Surface: dirichlet
        result[-1] = self.surface_bc.residual(c[-1])

        # fluid phase - internal and outlet
        transport = (
            self.media.particle_porosity * dcdt[1 : self.N - 1]
            - self.media.particle_porosity
            * self.pore_diffusion
            * self.numerics.evaluate_radial_operator(c)[1 : self.N - 1]
        )
        dqdC = self.iso.dq_dC(c)
        lap_q = self.numerics.evaluate_radial_operator(self.iso.q(c))

        result[1 : self.N - 1] = (
            transport
            + self.media.particle_density * (dqdC * dcdt)[1 : self.N - 1]
            - self.media.particle_density
            * self.surface_diffusion
            * lap_q[1 : self.N - 1]
        )

    def _jacobian(self, t, y, ydot, result, cj, jac):
        C = y[: self.N]
        n = self._n_vars()
        J = np.zeros((n, n))

        J[0, :] = self.center_bc.jacobian_row(
            self.numerics.collocation.first_derivative[0, :]
        )

        J[-1, :] = self.surface_bc.jacobian_row(np.zeros(n))

        # derivative of transport
        d_transport = (
            -self.media.particle_porosity
            * self.pore_diffusion
            * self.numerics.collocation.radial_operator_matrix
        )

        J[1:-1, :] = d_transport[1:-1]

        dqdC = self.iso.dq_dC(C)

        J[1:-1, :] -= (
            self.media.particle_density
            * self.surface_diffusion
            * self.numerics.collocation.radial_operator_matrix[1:-1]
            @ np.diag(dqdC)
        )

        mass = self.media.particle_porosity + self.media.particle_density * dqdC

        J[1:-1, 1:-1] += cj * np.diag(mass[1:-1])

        jac[:, :] = J
        return 0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for this model.
        """
        return [0]

    def _initial_conditions(self, C_init: float, C_in: float, q_init: float):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.N, C_init)

        # Surface concentration
        C0[-1] = self.surface_bc.apply()

        y0 = C0.copy()
        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def solve(self, t_span, t_eval, C_in=1.0, C_init=0.0, q_init=0.0):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        y0, ydot0 = self._initial_conditions(C_init, C_in, q_init)

        result = self.numerics.integrate(
            residual=self._residual,
            jacobian=self._jacobian,
            y0=y0,
            yp0=ydot0,
            t_span=t_span,
            t_eval=t_eval,
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # result.values.y has shape (n_out, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]  # (n_times, n_vars)

        C_out = y_out[:, : self.N]  # (n_times, N)

        # Local equilibrium: recover q from C at each time step
        q_out = np.array([self.iso.q(C_out[i]) for i in range(len(t_eval))])

        return self.numerics.collocation.nodes, C_out, q_out
