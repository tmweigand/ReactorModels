"""advection_diffusion_adsorption.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .adsorption_kinetics import AdsorptionKinetics, LocalEquilibrium
from .boundary_conditions import InletBC, DanckwertsBC

__all__ = ["AdvectionDiffusionAdsorption"]


class AdvectionDiffusionAdsorption(NumericModel):
    """1D advection-diffusion with adsorption in a packed bed.

    The governing equations are:

        Fluid phase: eps*dC/dt + eps * v * dC/dx - eps* D * d2C/dx2 - rho_b * dq/dt = 0
        Solid phase: dq/dt = rate_constant *(q*(C) - q) for ldf
                     dq/dt = rate_constant * C *(q*(C) - q) for second order
                     where q* is provided isotherm.

    If local_equilibirium is assumed, the solid phase equation is ignored and:

        dq_dt = dq*/dc*dc/dt

    """

    _param_names = ("velocity", "axial_diffusion", "isotherm")

    def __init__(
        self,
        breakthrough: Breakthrough,
        isotherm: Isotherm,
        numerics: NumericsConfig,
        kinetics: AdsorptionKinetics | None = None,
        inlet_bc: Type[InletBC] = DanckwertsBC,
    ):
        # Physical parameters
        self.column = breakthrough.column
        self.breakthrough = breakthrough
        self.velocity = breakthrough.interstitial_velocity
        self.axial_diffusion = breakthrough.chemical.axial_diffusion
        self.isotherm = isotherm

        if kinetics is None:
            kinetics = LocalEquilibrium()

        kinetics.bind(
            column=breakthrough.column,
            breakthrough=breakthrough,
            numerics=numerics,
            isotherm=isotherm,
        )

        self.kinetics = kinetics

        # Initial conditions
        self.initial_concentration = breakthrough.initial_concentration

        # Boundary conditions
        self.inlet_concentration = breakthrough.mean_feed_concentration()
        self.inlet_bc = inlet_bc(
            self.inlet_concentration,
            node=0,
            velocity=self.velocity,
            diffusion=self.axial_diffusion,
        )

        # Numerics
        self.numerics = numerics

        # Discretization
        self.N = len(self.numerics.collocation.nodes)

        self.assert_parameters_set()

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, q = self.kinetics._split(y)
        dcdt, dqdt = self.kinetics._split(ydot)

        # fluid phase - inlet
        result[0] = self.inlet_bc.residual(
            c[0], self.numerics.collocation.evaluate_gradient(c, 0)
        )

        # fluid phase - internal and outlet
        transport = (
            self.column.porosity * dcdt[1:]
            + self.column.porosity
            * self.velocity
            * self.numerics.evaluate_gradient(c)[1:]
            - self.column.porosity
            * self.axial_diffusion
            * self.numerics.evaluate_second_derivative(c)[1:]
        )

        self.kinetics._residual_kinetics(result, c, q, dcdt, dqdt, transport)

        return 0

    def _jacobian(self, t, y, ydot, result, cj, jac):
        """Build jacobian of _residual."""
        C, q = self.kinetics._split(y)
        n = self.kinetics._n_vars()
        J = np.zeros((n, n))

        # Row 0: algebraic constraint
        J[0, : self.N] = self.inlet_bc.jacobian_row(
            self.numerics.collocation.first_derivative[0, :]
        )

        # derivative of transport dF/dc
        d_transport = (
            -self.column.porosity
            * self.velocity
            * self.numerics.collocation.first_derivative
            + self.column.porosity
            * self.axial_diffusion
            * self.numerics.collocation.second_derivative
        )
        J[1 : self.N, : self.N] = -d_transport[1:, :]

        self.kinetics._jacobian_kinetics(C, q, J, cj)

        jac[:, :] = J

        return 0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for this model.
        """
        return [0]

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.N, self.initial_concentration)
        C0[0] = self.inlet_bc.apply(self.numerics.collocation.evaluate_gradient(C0, 0))

        y0 = self.kinetics._y0(C0)
        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def solve(self):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        y0, ydot0 = self._initial_conditions()

        result = self.numerics.integrate(
            residual=self._residual,
            jacobian=self._jacobian,
            y0=y0,
            yp0=ydot0,
            t_span=[0, self.breakthrough.time.tolist()],
            t_eval=self.breakthrough.time,
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # result.values.y has shape (n_out, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]  # (n_times, n_vars)

        C_out, q_out = self.kinetics._solve_kinetics(y_out)
        return self.numerics.collocation.nodes, C_out, q_out
