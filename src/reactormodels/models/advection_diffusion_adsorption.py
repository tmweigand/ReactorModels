"""advection_diffusion_adsorption.py"""

from __future__ import annotations
from typing import Type
import numpy as np
from collections.abc import Sequence

from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm
from .adsorption_kinetics import (
    AdsorptionKinetics,
    LocalEquilibrium,
    _is_implemented,
    LinearDrivingForce,
    SecondOrder,
)
from .boundary_conditions import BoundaryConditions, DirichletBC

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
        breakthrough: Breakthrough | Sequence[Breakthrough],
        isotherm: Isotherm | Sequence[Isotherm] | MultiSpeciesIsotherm,
        numerics: NumericsConfig,
        kinetics: AdsorptionKinetics = LocalEquilibrium(),
        inlet_bc: (
            Type[BoundaryConditions] | Sequence[Type[BoundaryConditions]]
        ) = DirichletBC,
    ):
        # Normalize breakthroughs to a list
        if isinstance(breakthrough, Breakthrough):
            self.breakthroughs = [breakthrough]
        else:
            self.breakthroughs = list(breakthrough)

        if not self.breakthroughs:
            raise ValueError("At least one breakthrough must be provided.")
        if not all(isinstance(b, Breakthrough) for b in self.breakthroughs):
            raise TypeError("breakthrough must be one or more Breakthroughs.")

        # number of species
        self.n_species = len(self.breakthroughs)

        # Physical parameters
        self.column = self.breakthroughs[0].column
        if not all(bt.column is self.column for bt in self.breakthroughs):
            raise ValueError("All breakthroughs must share the same Column object.")

        self.velocity = self.breakthroughs[0].interstitial_velocity
        if not all(
            np.isclose(bt.interstitial_velocity, self.velocity)
            for bt in self.breakthroughs
        ):
            raise ValueError(
                "All breakthroughs must have the same interstitial velocity."
            )

        # Isotherm: assert isotherm methods are compatible with kinetics
        # NOTE: update to check a list of isotherms
        self.isotherm = isotherm
        if isinstance(kinetics, (LinearDrivingForce, SecondOrder)):
            required_methods = ("q", "dq_dC", "d2q_dC2")
            if not all(
                _is_implemented(self.isotherm, method) for method in required_methods
            ):
                raise ValueError(
                    f"{type(kinetics).__name__} requires an isotherm "
                    "implementing q, dq_dC, and d2q_dC2."
                )

        # Species-specific parameters
        self.axial_diffusion = np.array(
            [bt.chemical.axial_diffusion for bt in self.breakthroughs]
        )
        self.initial_concentration = np.array(
            [bt.initial_concentration for bt in self.breakthroughs]
        )
        self.initial_solid_phase_loading = np.array(
            [bt.initial_solid_phase_loading for bt in self.breakthroughs]
        )
        self.inlet_concentration = np.array(
            [bt.mean_feed_concentration() for bt in self.breakthroughs]
        )

        # Normalize inlet BCs to a list
        if isinstance(inlet_bc, type):
            self.inlet_bcs = [inlet_bc] * len(self.breakthroughs)
        else:
            self.inlet_bcs = list(inlet_bc)

        if len(self.inlet_bcs) != len(self.breakthroughs):
            raise ValueError(
                "The number of inlet boundary conditions must match "
                "the number of breakthroughs."
            )
        self.inlet_bc = [
            bc(
                inlet_concentration,
                node=0,
                velocity=self.velocity,
                diffusion=diffusion,
            )
            for bc, inlet_concentration, diffusion in zip(
                self.inlet_bcs,
                self.inlet_concentration,
                self.axial_diffusion,
            )
        ]

        # Numerics
        self.numerics = numerics

        # Discretization
        self.axial_nodes = len(self.numerics.collocation.nodes)

        # Kinetics
        if type(kinetics) is LocalEquilibrium:
            kinetics = LocalEquilibrium.solution_variable(self.isotherm)
        kinetics.configure(
            self.breakthroughs,
            self.isotherm,
            self.axial_nodes,
            self.inlet_concentration,
            self.n_species,
        )
        self.kinetics = kinetics

        self.assert_parameters_set()

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, q = self.kinetics._split(y)
        dcdt, dqdt = self.kinetics._split(ydot)
        result = result.reshape(self.n_species, self.axial_nodes)

        # compute chain rule and adsorption components
        self.kinetics._residual_kinetics(result, c, q, dcdt, dqdt)

        # fluid phase - inlet
        G = self.numerics.collocation.evaluate_gradient(c)
        for i, bc in enumerate(self.inlet_bc):
            result[i, 0] = bc.residual(c[i, 0], G[i, 0])

        # fluid phase - internal and outlet
        result[:, 1:] += (
            self.column.porosity * dcdt[:, 1:]
            + self.column.porosity
            * self.velocity
            * self.numerics.evaluate_gradient(c)[:, 1:]
            - self.column.porosity
            * self.axial_diffusion[:, None]
            * self.numerics.evaluate_second_derivative(c)[:, 1:]
        )

        return 0

    def _jacobian(self, t, y, ydot, result, cj, jac):
        """Build jacobian of _residual."""

        # for i, bc in enumerate(self.inlet_bc):
        #     # Species inlet rows: algebraic constraint, no Cdot term
        #     inlet_node = i * self.axial_nodes
        #     outlet_node = inlet_node + self.axial_nodes
        #     species_nodes = slice(inlet_node, outlet_node)

        #     J[inlet_node, species_nodes] = bc.jacobian_row(
        #         self.numerics.collocation.first_derivative[0, :]
        #     )
        #     # Species interior and outlet rows: change in interior residual w.r.t to C
        #     interior_nodes = slice(inlet_node + 1, outlet_node)
        #     d_transport = (
        #         -self.column.porosity
        #         * self.velocity
        #         * self.numerics.collocation.first_derivative
        #         + self.column.porosity
        #         * self.axial_diffusion[i]
        #         * self.numerics.collocation.second_derivative
        #     )
        #     J[interior_nodes, species_nodes] = d_transport[1:, :]

        C, q = self.kinetics._split(y)
        dCdt, dqdt = self.kinetics._split(ydot)
        n = self.kinetics._n_vars()
        J = np.zeros((n, n))
        dCdq = np.zeros((self.n_species, self.n_species, self.axial_nodes))
        self.kinetics._jacobian_kinetics(C, q, dCdt, dqdt, dCdq, J, cj)

        for i, bc in enumerate(self.inlet_bc):
            # Species inlet rows: algebraic constraint, no Cdot term
            inlet_node = i * self.axial_nodes
            outlet_node = inlet_node + self.axial_nodes
            interior_nodes = slice(inlet_node + 1, outlet_node)

            # inlet state variable. bc_row is shape (axial_nodes,)
            bc_row = bc.jacobian_row(self.numerics.collocation.first_derivative[0, :])
            J[inlet_node, inlet_node] = bc_row[0]

            # change in transport w.r.t. q. shape (axial_nodes, axial_nodes)
            d_transport = (
                self.column.porosity
                * self.velocity
                * self.numerics.collocation.first_derivative
                - self.column.porosity
                * self.axial_diffusion[i]
                * self.numerics.collocation.second_derivative
            )
            J[interior_nodes, inlet_node] += d_transport[1:, 0]

            # Species interior and outlet rows: change in interior residual w.r.t to q
            for j in range(self.n_species):
                species_start = j * self.axial_nodes
                species_interior = slice(
                    species_start + 1, species_start + self.axial_nodes
                )
                # change in inlet C w.r.t. q
                J[inlet_node, species_interior] += bc_row[1:] * dCdq[i, j, 1:]

                # change in transport w.r.t. interior q
                J[interior_nodes, species_interior] += (
                    d_transport[1:, 1:] * dCdq[i, j, 1:]
                )

        jac[:, :] = J

        return 0

    def check_jacobian(self, t, y, ydot, cj=1.0, h=1e-6):
        """Compare analytical and numerical IDA Jacobians."""
        n = len(y)

        # Analytical Jacobian
        J_analytic = np.zeros((n, n))
        self._jacobian(t, y, ydot, None, cj, J_analytic)

        # Numerical Jacobian
        J_numerical = np.zeros((n, n))

        for j in range(n):
            step = h * max(1.0, abs(y[j]))

            y_plus = y.copy()
            y_minus = y.copy()
            ydot_plus = ydot.copy()
            ydot_minus = ydot.copy()

            y_plus[j] += step
            y_minus[j] -= step

            ydot_plus[j] += cj * step
            ydot_minus[j] -= cj * step

            F_plus = np.zeros(n)
            F_minus = np.zeros(n)

            self._residual(t, y_plus, ydot_plus, F_plus)
            self._residual(t, y_minus, ydot_minus, F_minus)

            J_numerical[:, j] = (F_plus - F_minus) / (2 * step)

        # Calculate errors
        abs_error = np.abs(J_analytic - J_numerical)
        scale = np.maximum(
            np.maximum(np.abs(J_analytic), np.abs(J_numerical)),
            1e-12,
        )
        rel_error = abs_error / scale

        # Sort entries by largest relative error
        indices = np.argsort(rel_error.ravel())[::-1]

        print("\nLargest Jacobian discrepancies:")
        print(
            f"{'Row':>5} {'Col':>5} "
            f"{'Analytical':>15} {'Numerical':>15} "
            f"{'Abs Error':>15} {'Rel Error':>15}"
        )

        for idx in indices[:20]:
            row, col = np.unravel_index(idx, (n, n))

            print(
                f"{row:5d} {col:5d} "
                f"{J_analytic[row, col]:15.6e} "
                f"{J_numerical[row, col]:15.6e} "
                f"{abs_error[row, col]:15.6e} "
                f"{rel_error[row, col]:15.6e}"
            )

        return J_analytic, J_numerical

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for each species.
        """
        return [i * self.axial_nodes for i in range(self.n_species)]

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C_init = np.asarray(self.initial_concentration, dtype=float)
        q_init = np.asarray(self.initial_solid_phase_loading, dtype=float)

        C0 = np.broadcast_to(
            C_init[:, None],
            (self.n_species, self.axial_nodes),
        ).copy()

        q0 = np.broadcast_to(
            q_init[:, None],
            (self.n_species, self.axial_nodes - 1),
        ).copy()

        first_derivative_matrix = self.numerics.collocation.first_derivative[0, :]
        for i, bc in enumerate(self.inlet_bc):
            C0[i, 0] = bc.apply(C0[i, :], first_derivative_matrix)
        C_inlet = C0[:, 0]
        y0 = np.column_stack((C_inlet, q0))
        # y0 = C0.ravel()
        y0 = y0.ravel()
        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def solve(self):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        y0, ydot0 = self._initial_conditions()

        # for h in [1e-4, 1e-5, 1e-6, 1e-7]:
        #     self.check_jacobian(t=0, y=y0, ydot=ydot0, cj=1.0, h=h)

        # Generate t_eval using unique time points from all breakthroughs
        self.t_eval = np.unique(np.concatenate([bt.time for bt in self.breakthroughs]))

        result = self.numerics.integrate(
            residual=self._residual,
            jacobian=self._jacobian,
            y0=y0,
            yp0=ydot0,
            t_span=[0, self.t_eval[-1]],
            t_eval=self.t_eval,
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )
        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # (n_times, n_species, axial_nodes), skip the t=t_span[0] row
        # C_out = result.values.y[1:]
        # C_out = C_out.reshape(len(C_out), self.n_species, self.axial_nodes)
        # q_out = np.empty_like(C_out)
        # for t in range(len(self.t_eval)):
        #     for i in range(self.axial_nodes):
        #         q_out[t, :, i] = self.isotherm.q(C_out[t, :, i])

        y_out = result.values.y[1:]
        y_out = y_out.reshape(len(y_out), self.n_species, self.axial_nodes)
        q_out = y_out[:, :, 1:]
        C_out = np.empty_like(y_out)
        C_out[:, :, 0] = y_out[:, :, 0]
        for t in range(len(self.t_eval)):
            for i in range(self.axial_nodes - 1):
                C_out[t, :, i + 1] = self.isotherm.C(q_out[t, :, i])
        return self.numerics.collocation.nodes, C_out, q_out
