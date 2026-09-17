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
from .multi_species_isotherm import MultiSpeciesIsotherm
from .adsorption_kinetics import AdsorptionKinetics
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

    _param_names = ("velocity", "axial_diffusion", "isotherm", "k_ldf")

    def __init__(
        self,
        breakthrough: Breakthrough | Sequence[Breakthrough],
        isotherm: Isotherm | MultiSpeciesIsotherm,
        numerics: NumericsConfig,
        kinetics: AdsorptionKinetics = AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_ldf: float = 0,
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
        self.isotherm = isotherm
        self.k_ldf = k_ldf
        self.kinetics = kinetics

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
        self.N = len(self.numerics.collocation.nodes)

        # Checks
        if (
            kinetics
            in (
                AdsorptionKinetics.LINEAR_DRIVING_FORCE,
                AdsorptionKinetics.SECOND_ORDER,
            )
            and k_ldf <= 0
        ):
            raise ValueError(
                "k_ldf must be > 0 for LINEAR_DRIVING_FORCE or SECOND_ORDER mode"
            )

        self.assert_parameters_set()

        self._residual_calls = 0
        self._jacobian_calls = 0
        self._residual_time = 0.0
        self._jacobian_time = 0.0
        self._dC_dq_calls = 0
        self._dC_dq_time = 0.0

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        if (
            self.kinetics == AdsorptionKinetics.LINEAR_DRIVING_FORCE
            or self.kinetics == AdsorptionKinetics.SECOND_ORDER
        ):
            return 2 * self.N * self.n_species
        return self.N * self.n_species

    def _split(self, y: np.ndarray):
        """Return (C, q) where q is None for LOCAL_EQUILIBRIUM."""
        if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            y = y.reshape(self.n_species, self.N)

            C = np.empty_like(y)
            C[:, 0] = y[:, 0]

            q = y[:, 1:]
        else:
            y = y.reshape(self.n_species, 2, self.N)

            C = y[:, 0, :]
            q = y[:, 1, :]
        return C, q

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        start = time.perf_counter()

        try:
            c, q = self._split(y)
            dcdt, dqdt = self._split(ydot)

            if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
                result = result.reshape(self.n_species, self.N)
                c[:, 1:] = self.isotherm.C(q)

                # inlet bc
                gradient = self.numerics.collocation.evaluate_gradient(c)

                for i, bc in enumerate(self.inlet_bc):
                    result[i, 0] = bc.residual(c[i, 0], gradient[i, 0])

                if self.n_species == 1:
                    dcdt[:, 1:] = self.isotherm.dC_dq(q) * dqdt
                else:
                    dC_dq = self.isotherm.dC_dq(q)

                    dcdt[:, 1:] = np.einsum(
                        "ijn,jn->in",
                        dC_dq,
                        dqdt,
                    )
            else:
                # non-local equilibrium shape
                result = result.reshape(self.n_species, 2, self.N)

                # inlet bc
                gradient = self.numerics.collocation.evaluate_gradient(c)

                for i, bc in enumerate(self.inlet_bc):
                    result[i, 0, 0] = bc.residual(c[i, 0], gradient[i, 0])

            # fluid phase - internal and outlet
            transport = (
                self.column.porosity * dcdt[:, 1:]
                + self.column.porosity
                * self.velocity
                * self.numerics.evaluate_gradient(c)[:, 1:]
                - self.column.porosity
                * self.axial_diffusion[:, None]
                * self.numerics.evaluate_second_derivative(c)[:, 1:]
            )

            if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
                result[:, 1:] = transport + self.column.media.bed_density * dqdt
            else:
                result[:, 0, 1:] = (
                    transport + self.column.media.bed_density * dqdt[:, 1:]
                )

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
        start_time = time.perf_counter()

        try:
            c, q = self._split(y)
            n = self._n_vars()
            J = np.zeros((n, n))

            # first and second derivatives
            D1 = self.numerics.collocation.first_derivative
            D2 = self.numerics.collocation.second_derivative

            # Local equilibrium
            if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
                # Inlet boundary conditions
                for i in range(self.n_species):
                    row = i * self.N
                    J[row, :] = 0.0

                    col_start = i * self.N
                    col_end = (i + 1) * self.N

                    J[row, col_start:col_end] = self.inlet_bc[i].jacobian_row(D1[0, :])

                # dC/dq
                dC_dq_start = time.perf_counter()
                dC_dq = self.isotherm.dC_dq(q)
                self._dC_dq_time += time.perf_counter() - dC_dq_start
                self._dC_dq_calls += 1

                # Interior equations
                interior_J = (
                    cj * self.column.porosity * dC_dq
                    + cj
                    * self.column.media.bed_density
                    * np.eye(self.n_species)[:, :, None]
                )

                for k in range(1, self.N):
                    indices = np.arange(self.n_species) * self.N + k

                    J[np.ix_(indices, indices)] += interior_J[:, :, k - 1]
            # LDF / SECOND ORDER
            else:

                def idx_C(i, k):
                    return i * 2 * self.N + k

                def idx_q(i, k):
                    return i * 2 * self.N + self.N + k

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
        C_init = np.asarray(self.initial_concentration, dtype=float)
        q_init = np.asarray(self.initial_mass_fraction, dtype=float)

        C0 = np.broadcast_to(
            C_init[:, None],
            (self.n_species, self.N),
        ).copy()

        gradient0 = self.numerics.collocation.evaluate_gradient(C0, 0)

        for i, bc in enumerate(self.inlet_bc):
            C0[i, 0] = bc.apply(gradient0[i])

        if q_init.ndim == 0:
            q_init = np.full(self.n_species, q_init)

        if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
            q0 = np.broadcast_to(
                q_init[:, None],
                (self.n_species, self.N - 1),
            ).copy()

            y0 = np.concatenate(
                [C0[:, 0:1], q0],
                axis=1,
            ).ravel()
        else:
            q0 = np.broadcast_to(
                q_init[:, None],
                (self.n_species, self.N),
            ).copy()

            y0 = np.stack([C0, q0], axis=1).ravel()

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

        if self.kinetics == AdsorptionKinetics.LOCAL_EQUILIBRIUM:
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

            for k in range(len(y_out)):
                C_out[k, :, 1:] = self.isotherm.C(y_out[k, :, 1:])

            q_out = y_out[:, :, 1:]

        else:
            y_out = y_out.reshape(
                len(y_out),
                self.n_species,
                2,
                self.N,
            )
            # Multi-species linear isotherm
            C_out = y_out[:, :, 0, :]
            q_out = y_out[:, :, 1, :]

        return self.numerics.collocation.nodes, C_out, q_out
