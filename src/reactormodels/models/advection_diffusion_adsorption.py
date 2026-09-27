"""advection_diffusion_adsorption.py"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Type
import numpy as np

from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm
from .adsorption_kinetics import AdsorptionKinetics, LocalEquilibrium
from .boundary_conditions import InletBC, DanckwertsBC

__all__ = ["AdvectionDiffusionAdsorption"]


class AdvectionDiffusionAdsorption(NumericModel):
    """1D advection-diffusion with adsorption in a packed bed.

    The governing equations are:

        Fluid phase: eps*dC/dt + eps * v * dC/dx - eps* D * d2C/dx2 - rho_b * dq/dt = 0
        Solid phase: dq/dt = k_l *(q*(C) - q) where q* is provided isotherm.

    If local_equilibirium is assumed, the solid phase equation is ignored and:

        dq_dt = dq*/dc*dc/dt

    """

    _param_names = ("velocity", "axial_diffusion", "isotherm")

    def __init__(
        self,
        breakthrough: Breakthrough | Sequence[Breakthrough],
        isotherm: Isotherm | MultiSpeciesIsotherm,
        column_numerics: NumericsConfig,
        kinetics: AdsorptionKinetics = LocalEquilibrium(),
        inlet_bc: Type[InletBC] | Sequence[Type[InletBC]] = DanckwertsBC,
    ):
        # Normalize breakthroughs to a list
        if isinstance(breakthrough, Breakthrough):
            self.breakthroughs = [breakthrough]
        else:
            self.breakthroughs = list(breakthrough)

        if not self.breakthroughs:
            raise ValueError("At least one breakthrough must be provided.")

        # number of species
        self.n_species = len(self.breakthroughs)

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

        # Shared physical parameters
        self.column = self.breakthroughs[0].column
        self.velocity = self.breakthroughs[0].interstitial_velocity

        # Isotherm
        self.isotherm = isotherm

        # Species-specific parameters
        self.axial_diffusion = np.array(
            [bt.chemical.axial_diffusion for bt in self.breakthroughs]
        )

        self.initial_concentration = np.array(
            [bt.initial_concentration for bt in self.breakthroughs]
        )

        self.initial_mass_fraction = np.array(
            [bt.initial_mass_fraction for bt in self.breakthroughs]
        )

        self.inlet_concentration = np.array(
            [bt.mean_feed_concentration() for bt in self.breakthroughs]
        )

        # Boundary conditions
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
        self.column_numerics = column_numerics

        # Discretization
        self.axial_nodes = len(self.column_numerics.collocation.nodes)

        # Kinetics
        kinetics.configure(
            self.column,
            isotherm,
            self.inlet_concentration,
            self.n_species,
            self.inlet_bc,
            column_numerics,
        )
        self.kinetics = kinetics

        self.assert_parameters_set()

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, q, dcdt, dqdt = self.kinetics._state_variables(y, ydot)
        result = self.kinetics._reshape_result(result)

        # inlet bc
        self.kinetics._inlet_bc_loop(c, result)

        # fluid phase - internal and outlet
        transport = (
            self.column.porosity * dcdt[:, 1:]
            + self.column.porosity
            * self.velocity
            * self.column_numerics.evaluate_gradient(c)[:, 1:]
            - self.column.porosity
            * self.axial_diffusion[:, None]
            * self.column_numerics.evaluate_second_derivative(c)[:, 1:]
        )

        # liquid/solid phase residuals
        self.kinetics._residual_kinetics(result, c, q, dcdt, dqdt, transport)

        return 0

    def _jacobian(self, t, y, ydot, result, cj, jac):
        """Build jacobian of _residual."""
        c, q = self.kinetics._split(y)
        n = self.kinetics._n_vars()
        J = np.zeros((n, n))

        # first and second derivatives
        D1 = self.column_numerics.collocation.first_derivative
        D2 = self.column_numerics.collocation.second_derivative

        self.kinetics._jacobian_kinetics(
            c, q, J, cj, D1, D2, self.axial_diffusion, self.velocity
        )

        jac[:, :] = J

        return 0

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C_init = np.asarray(self.initial_concentration, dtype=float)
        q_init = np.asarray(self.initial_mass_fraction, dtype=float)

        C0 = np.broadcast_to(
            C_init[:, None],
            (self.n_species, self.axial_nodes),
        ).copy()

        gradient0 = self.column_numerics.collocation.evaluate_gradient(C0, 0)
        for i, bc in enumerate(self.inlet_bc):
            C0[i, 0] = bc.apply(gradient0[i])

        y0 = self.kinetics._set_initial_conditions(C0, q_init)

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def solve(self):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        y0, ydot0 = self._initial_conditions()

        result = self.column_numerics.integrate(
            residual=self._residual,
            jacobian=self._jacobian,
            y0=y0,
            yp0=ydot0,
            t_span=[0, self.breakthroughs[0].time.tolist()],
            t_eval=self.breakthroughs[0].time,
            algebraic_vars_idx=self.kinetics._algebraic_vars_idx(),
        )

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # result.values.y has shape (n_out, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]  # (n_times, n_vars)

        C_out, q_out = self.kinetics._parse_variables(y_out)

        return self.column_numerics.collocation.nodes, C_out, q_out
