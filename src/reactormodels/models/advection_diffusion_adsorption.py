"""advection_diffusion_adsorption.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .isotherm import Isotherm
from .adsorption_kinetics import AdsorptionKinetics
from .boundary_conditions import InletBC, DanckwertsBC

__all__ = ["AdvectionDiffusionAdsorption"]


class AdvectionDiffusionAdsorption:
    """1D advection-diffusion with adsorption in a packed bed.

    The governing equations are:

        Fluid phase: eps*dC/dt + eps * v * dC/dx - eps* D * d2C/dx2 - rho_b * dq/dt = 0
        Solid phase: dq/dt = k_l *(q*(C) - q) where q* is provided isotherm.

    If local_equilibirium is assumed, the solid phase equation is ignored and:

        dq_dt = dq*/dc*dc/dt

    """

    def __init__(
        self,
        breakthrough: Breakthrough,
        isotherm: Isotherm,
        numerics: NumericsConfig,
        mode: AdsorptionKinetics = AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_ldf: float = 0,
        inlet_bc: Type[InletBC] = DanckwertsBC,
    ):
        self.column = breakthrough.column
        self.breakthrough = breakthrough
        self.velocity = breakthrough.interstitial_velocity
        self.DL = np.atleast_1d(
            np.asarray(breakthrough.chemical.diffusion, dtype=float)
        )
        self.n_species = len(self.DL)
        self.inlet_concentration = np.atleast_1d(breakthrough.mean_feed_concentration())
        self.initial_concentration = breakthrough.initial_concentration
        self.iso = isotherm
        self.numerics = numerics
        self.mode = mode
        self.k_ldf = k_ldf
        self.inlet_bc = inlet_bc(self.inlet_concentration, self.velocity, self.DL)
        self.N = len(self.numerics.collocation.nodes)

        if len(self.inlet_concentration) != self.n_species:
            raise ValueError(
                "Number of inlet concentrations must match number of species."
            )

        if (
            mode
            in (
                AdsorptionKinetics.LINEAR_DRIVING_FORCE,
                AdsorptionKinetics.SECOND_ORDER,
            )
            and k_ldf <= 0
        ):
            raise ValueError(
                "k_ldf must be > 0 for LINEAR_DRIVING_FORCE or SECOND_ORDER mode"
            )

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        if (
            self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE
            or self.mode == AdsorptionKinetics.SECOND_ORDER
        ):
            return 2 * self.N
        return self.N * self.n_species

    def _split(self, y: np.ndarray):
        """Return (C, q) where q is None for LOCAL_EQUILIBRIUM."""
        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            y = y.reshape(self.n_species, self.N)

            C = np.empty_like(y)
            C[:, 0] = y[:, 0]

            q = y[:, 1:]
        else:
            q = y[self.N :]
            C = y[: self.N]
        return C, q

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, q = self._split(y)
        dcdt, dqdt = self._split(ydot)
        result = result.reshape(self.n_species, self.N)

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            c[:, 1:] = self.iso.C(q)
            dcdt[:, 1:] = self.iso.dC_dq(q) * dqdt

        # fluid phase - inlet
        gradient = self.numerics.collocation.evaluate_gradient(c)

        result[:, 0] = self.inlet_bc.residual(
            c[:, 0],
            gradient[:, 0],
        )

        # fluid phase - internal and outlet
        transport = (
            self.column.porosity * dcdt[:, 1:]
            + self.column.porosity
            * self.velocity
            * self.numerics.evaluate_gradient(c)[:, 1:]
            - self.column.porosity
            * self.DL[:, None]
            * self.numerics.evaluate_second_derivative(c)[:, 1:]
        )
        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            result[:, 1:] = transport + self.column.bulk_density * dqdt
        else:
            result[1 : self.N] = transport + self.column.bulk_density * dqdt[1:]

        # solid phase
        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            pass
        elif self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            result[self.N :] = dqdt - self.k_ldf * (self.iso.q(c) - q)
        else:
            result[self.N :] = dqdt - self.k_ldf * c * (self.iso.q(c) - q)

        return 0

    def _jacobian(self, t, y, ydot, result, cj, jac):
        """Build jacobian of _residual."""
        C, q = self._split(y)
        n = self._n_vars()
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
            * self.DL
            * self.numerics.collocation.second_derivative
        )
        J[1 : self.N, : self.N] = -d_transport[1:, :]

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            for i in range(1, self.N):
                J[i, i] += cj * (
                    self.column.porosity
                    + self.column.get_bulk_density() * self.iso.dq_dC(C[i])
                )

        elif self.mode == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
            for i in range(1, self.N):
                J[i, i] += cj * self.column.porosity
                J[i, self.N + i] += cj * self.column.get_bulk_density()

            for i in range(self.N):
                J[self.N + i, i] = -self.k_ldf * self.iso.dq_dC(C[i])
                J[self.N + i, self.N + i] = self.k_ldf + cj

        else:
            for i in range(1, self.N):
                J[i, i] += cj * self.column.porosity
                J[i, self.N + i] += cj * self.column.get_bulk_density()

            for i in range(self.N):
                J[self.N + i, i] = (
                    -self.k_ldf * (self.iso.q(C[i]) + C[i] * self.iso.dq_dC(C[i]))
                    + self.k_ldf * q[i]
                )
                J[self.N + i, self.N + i] = self.k_ldf * C[i] + cj

        jac[:, :] = J

        return 0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for this model.
        """
        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            return [i * self.N for i in range(self.n_species)]
        return [0]

    def _initial_conditions(
        self, C_init: float | np.ndarray, q_init: float | np.ndarray
    ):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C_init = np.asarray(C_init, dtype=float)

        if C_init.ndim == 0:
            C_init = np.full(self.n_species, C_init)

        C0 = np.broadcast_to(
            C_init[:, None],
            (self.n_species, self.N),
        ).copy()

        gradient0 = self.numerics.collocation.evaluate_gradient(C0, 0)
        C0[:, 0] = self.inlet_bc.apply(gradient0)

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            q0 = self.iso.q(C0[:, 1:])

            y0 = np.concatenate(
                [C0[:, 0:1], q0],
                axis=1,
            ).ravel()
        else:
            q_init = np.asarray(q_init, dtype=float)
            if q_init.ndim == 0:
                q_init = np.full(self.n_species, q_init)

            q0 = np.broadcast_to(
                q_init[:, None],
                (self.n_species, self.N),
            ).copy()

            q0[:, 0] = self.iso.q(
                self.inlet_concentration
            )  # inlet node at equilibrium with feed
            y0 = np.concatenate(
                [C0, q0],
                axis=1,
            ).ravel()

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def solve(self, t_span, t_eval, C_init=0.0, q_init=0.0):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        y0, ydot0 = self._initial_conditions(C_init, q_init)

        result = self.numerics.integrate(
            residual=self._residual,
            # jacobian=self._jacobian,
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

        if self.mode == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            # (n_times, n_species, N)
            y_out = y_out.reshape(
                len(y_out),
                self.n_species,
                self.N,
            )
            # State:
            # y[:, :, 0] = inlet C
            # y[:, :, 1:] = q
            C_out = np.empty_like(y_out)
            C_out[:, :, 0] = y_out[:, :, 0]
            C_out[:, :, 1:] = self.iso.C(y_out[:, :, 1:])
            q_out = y_out[:, :, 1:]
        else:
            # Existing single-species behavior
            C_out = y_out[:, : self.N]

            if self.mode in (
                AdsorptionKinetics.LINEAR_DRIVING_FORCE,
                AdsorptionKinetics.SECOND_ORDER,
            ):
                q_out = y_out[:, self.N :]
            else:
                q_out = np.array([self.iso.q(C_out[i]) for i in range(len(t_eval))])

        return (self.numerics.collocation.nodes, C_out, q_out)
