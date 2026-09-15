"""advection_diffusion_adsorption.py"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Type
import numpy as np
import time

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
        breakthrough: Breakthrough | Sequence[Breakthrough],
        isotherm: Isotherm | MultiSpeciesIsotherm,
        numerics: NumericsConfig,
        kinetics: AdsorptionKinetics = LocalEquilibrium(),
        inlet_bc: Type[InletBC] = DanckwertsBC,
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
        self.numerics = numerics

        # Discretization
        self.n_nodes = len(self.numerics.collocation.nodes)

        # Kinetics
        kinetics.configure(
            breakthrough, isotherm, self.n_nodes, self.inlet_concentration
        )
        self.kinetics = kinetics

        self.assert_parameters_set()

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, q = self.kinetics._split(y)
        dcdt, dqdt = self.kinetics._split(ydot)

        try:
            c, q = self._split(y)
            dcdt, dqdt = self._split(ydot)

            if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
                result = result.reshape(self.n_species, self.N)
                c[:, 1:] = self.isotherm.C(q)

        self.kinetics._residual_kinetics(result, c, q, dcdt, dqdt, transport)

            # solid phase
            if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
                pass
            elif self.kinetics == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
                result[:, 1, :] = dqdt - self.k_ldf * (self.isotherm.q(c) - q)
            else:
                result[:, 1, :] = dqdt - self.k_ldf * c * (self.isotherm.q(c) - q)

            return 0
        finally:
            self._residual_time += time.perf_counter() - start
            self._residual_calls += 1

    def _jacobian(self, t, y, ydot, result, cj, jac):
        """Build jacobian of _residual."""
        C, q = self.kinetics._split(y)
        n = self.kinetics._n_vars()
        J = np.zeros((n, n))

        # Row 0: algebraic constraint
        J[0, : self.n_nodes] = self.inlet_bc.jacobian_row(
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
        J[1 : self.n_nodes, : self.n_nodes] = -d_transport[1:, :]

        self.kinetics._jacobian_kinetics(C, q, J, cj)

                # Inlet boundary conditions
                for i in range(self.n_species):
                    row = idx_C(i, 0)

                    bc_row = self.inlet_bc.jacobian_row(
                        D1[0, :],
                        species=i,
                    )
                    for k in range(self.N):
                        J[row, idx_C(i, k)] = bc_row[k]

                # Liquid-phase equations
                for i in range(self.n_species):
                    for k in range(1, self.N):
                        row = idx_C(i, k)

                        # dF_C / dC
                        for m in range(self.N):
                            J[row, idx_C(i, m)] += (
                                self.column.porosity * self.velocity * D1[k, m]
                                - self.column.porosity * self.DL[i] * D2[k, m]
                            )

                        # eps*dC/dt
                        J[row, idx_C(i, k)] += cj * self.column.porosity

                        # rho_b*dq/dt
                        J[row, idx_q(i, k)] += cj * self.column.media.bed_density

                # Solid-phase equations
                for i in range(self.n_species):
                    for k in range(self.N):
                        row = idx_q(i, k)
                        C = c[i, k]
                        q_i = q[i, k]

                        # LDF
                        # F_q = dq/dt - k_ldf * (q_eq(C) - q)
                        # dF/dC = -k * dq_eq/dC
                        # dF/dq = cj + k
                        if self.kinetics == AdsorptionKinetics.LINEAR_DRIVING_FORCE:
                            for k in range(self.N):
                                row = idx_q(i, k)
                                dq_dC = self.isotherm.K[i]

                                # dF_q / dC
                                J[row, idx_C(i, k)] = -self.k_ldf * dq_dC

                                # dF_q / dq
                                J[row, idx_q(i, k)] = self.k_ldf + cj

                        # SECOND ORDER
                        # F_q =
                        #   dq/dt - k*C*(q_eq(C) - q)
                        # dF/dC =
                        #   -k*(q_eq + C*dq_eq/dC) + k*q
                        # dF/dq =
                        #   k*C + cj
                        elif self.kinetics == AdsorptionKinetics.SECOND_ORDER:
                            for k in range(self.N):
                                row = idx_q(i, k)

                                C = c[i, k]
                                q_i = q[i, k]

                                q_eq = self.isotherm.K[i] * C
                                dq_dC = self.isotherm.K[i]

                                # dF_q / dC
                                J[row, idx_C(i, k)] = -self.k_ldf * (
                                    q_eq - q_i + C * dq_dC
                                )

                                # dF_q / dq
                                J[row, idx_q(i, k)] = self.k_ldf * C + cj

            jac[:, :] = J

            return 0
        finally:
            self._jacobian_time += time.perf_counter() - start_time
            self._jacobian_calls += 1

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic."""
        if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            return [i * self.N for i in range(self.n_species)]
        # LDF / SECOND_ORDER:
        # one algebraic variable per species: C_i at node 0
        return [2 * i * self.N for i in range(self.n_species)]

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.n_nodes, self.initial_concentration)
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
            t_span=[0, self.breakthroughs[0].time.tolist()],
            t_eval=self.breakthroughs[0].time,
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # result.values.y has shape (n_out, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]  # (n_times, n_vars)

        C_out, q_out = self.kinetics._parse_variables(y_out)
        return self.numerics.collocation.nodes, C_out, q_out
