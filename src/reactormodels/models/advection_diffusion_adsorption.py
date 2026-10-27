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
from .adsorption_kinetics import AdsorptionKinetics, LocalEquilibrium
from .boundary_conditions import BoundaryConditions, DanckwertsBC

__all__ = ["AdvectionDiffusionAdsorption"]


class AdvectionDiffusionAdsorption(NumericModel):
    """1D advection-diffusion with adsorption in a packed bed.

    The governing equations are:

    Fluid phase: eps * dC/dt + eps * v * dC/dx - eps * D * d2C/dx2 - rho_b * dq/dt = 0
    Solid phase: dq/dt = rate_constant *(q*(C) - q) for ldf
                 dq/dt = rate_constant * C *(q*(C) - q) for second order
                 where q* is calculated from the isotherm equation.

    If local_equilibirium is assumed, the solid phase equation is ignored and:

        dq_dt = dq*/dc * dc/dt

    """

    _param_names = ("velocity", "axial_diffusion", "isotherm")

    def __init__(
        self,
        breakthrough: Breakthrough | Sequence[Breakthrough],
        isotherm: Isotherm | Sequence[Isotherm] | MultiSpeciesIsotherm,
        numerics: NumericsConfig,
        kinetics: AdsorptionKinetics = LocalEquilibrium(),
<<<<<<< HEAD
        inlet_bc: Type[BoundaryConditions] = DanckwertsBC,
=======
        inlet_bc: Type[InletBC] | Sequence[Type[InletBC]] = DanckwertsBC,
>>>>>>> 84f9ea2 (start multi-species advection-diffusion-adsorption)
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

        # Isotherm
        self.isotherm = isotherm

        # Species-specific parameters
        self.axial_diffusion = np.array(
            [bt.chemical.axial_diffusion for bt in self.breakthroughs]
        )
        self.initial_concentration = np.array(
            [bt.initial_concentration for bt in self.breakthroughs]
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

<<<<<<< HEAD
        # Discretization (n_nodes = 1 + n_elements + n_points * n_elements)
        self.n_nodes = len(self.numerics.collocation.nodes)
=======
        # Discretization
        self.axial_nodes = len(self.numerics.collocation.nodes)
>>>>>>> 84f9ea2 (start multi-species advection-diffusion-adsorption)

        # Initialize AdsorptionKinetics class
        kinetics.configure(
            self.breakthroughs,
            self.isotherm,
            self.axial_nodes,
            self.inlet_concentration,
        )
        self.kinetics = kinetics

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

        # additional residual equations
        self.kinetics._residual_kinetics(result, c, q, dcdt, dqdt, transport)

        return 0

    def _jacobian(self, t, y, ydot, result, cj, jac):
        """Build jacobian of _residual."""
        C, q = self.kinetics._split(y)
        dcdt, _ = self.kinetics._split(ydot)

        # create n x n matrix where n is the number of variables/residual eqns
        n = self.kinetics._n_vars()
        J = np.zeros((n, n))

        # Row 0 and row n_nodes are algebraic constants for
        # liquid and solid phase concentrations, repsectively
        J[0, : self.n_nodes] = self.inlet_bc.jacobian_row(
            self.numerics.collocation.first_derivative[0]
        )

        # derivative of transport residual with respect to c
        d_transport = (
            self.column.porosity
            * self.velocity
            * self.numerics.collocation.first_derivative
            - self.column.porosity
            * self.axial_diffusion
            * self.numerics.collocation.second_derivative
        )
        # change in liquid phase residuals with respect to c
        J[1 : self.n_nodes, : self.n_nodes] = d_transport[1:, :]

        # additional jacobian rows and columns
        self.kinetics._jacobian_kinetics(C, dcdt, q, J, cj)

        jac[:, :] = J

        return 0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic."""
        # liquid phase inlet residual only
        return [0]

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        # set initial liquid phase concentration at all nodes
        C0 = np.full(self.n_nodes, self.initial_concentration)
        # replace inlet liquid phase concentration with boundary condition
        C0[0] = self.inlet_bc.apply(self.numerics.collocation.evaluate_gradient(C0, 0))

        y0 = self.kinetics._set_initial_conditions(C0)
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

        # result.values.y has shape (n_times, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]
        # extract/calculate liquid phase concentration and solid phase loading
        C_out, q_out = self.kinetics._parse_variables(y_out)
        return self.numerics.collocation.nodes, C_out, q_out
