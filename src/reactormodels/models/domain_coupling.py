"""domain_coupling.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..properties.column import Column
from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .isotherm import Isotherm
from .boundary_conditions import InletBC, DirichletBC, SymmetryBC

__all__ = ["DomainCoupling"]


class DomainCoupling:
    """Solve conservation equations for column and particle domain simultaneously."""

    def __init__(
        self,
        breakthrough: Breakthrough,
        axial_diffusion: float,
        pore_diffusion: float,
        surface_diffusion: float,
        initial_concentration: float,
        isotherm: Isotherm,
        column_numerics: NumericsConfig,
        particle_numerics: NumericsConfig,
        k_film: float = 0,
        inlet_bc: Type[InletBC] = DirichletBC,
        center_bc: Type[InletBC] = SymmetryBC,
    ):
        self.breakthrough: Breakthrough = breakthrough
        self.column: Column = breakthrough.column
        self.velocity = breakthrough.interstitial_velocity
        self.DL = axial_diffusion
        self.Dp = pore_diffusion
        self.Ds = surface_diffusion
        self.inlet_concentration = breakthrough.mean_feed_concentration()
        self.initial_concentration = initial_concentration
        self.iso = isotherm
        self.column_numerics = column_numerics
        self.particle_numerics = particle_numerics
        self.k_film = k_film
        self.inlet_bc = inlet_bc(self.inlet_concentration)
        self.center_bc = center_bc(node=0)
        self.N_column = len(self.column_numerics.collocation.nodes)
        self.N_particle = len(self.particle_numerics.collocation.nodes)

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        return self.N_column + self.N_particle * self.N_column

    def _split(self, y: np.ndarray):
        """Return (C, Cp) where Cp is a 2D numpy array."""
        C = y[: self.N_column]
        Cp = y[self.N_column :].reshape(self.N_column, self.N_particle)
        return C, Cp

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, cp = self._split(y)
        dcdt, dcpdt = self._split(ydot)

        sink = np.zeros(self.N_column)
        result[:] = 0

        # bulk phase - inlet
        result[0] = self.inlet_bc.residual(c[0])

        # bulk phase - internal and outlet
        transport = (
            self.column.porosity * dcdt[1:]
            + self.column.porosity
            * self.velocity
            * self.column_numerics.evaluate_gradient(c)[1:]
            - self.column.porosity
            * self.DL
            * self.column_numerics.evaluate_second_derivative(c)[1:]
        )

        for i in range(self.N_column):
            cp_i = cp[i]
            dcpdt_i = dcpdt[i]

            offset = self.N_column + i * self.N_particle

            # center: symmetry
            result[offset] = self.center_bc.residual(
                gradient_concentration_0=self.particle_numerics.evaluate_gradient(
                    cp_i, 0
                )
            )

            # particle phase - internal
            Dp_term = (
                self.column.media.particle_porosity * dcpdt_i[1 : self.N_particle - 1]
                - self.column.media.particle_porosity
                * self.Dp
                * self.particle_numerics.evaluate_radial_operator(cp_i)[
                    1 : self.N_particle - 1
                ]
            )

            dqdCp = self.iso.dq_dC(cp_i)
            lap_q = self.particle_numerics.evaluate_radial_operator(self.iso.q(cp_i))

            Ds_term = (
                self.column.media.particle_density
                * (dqdCp * dcpdt_i)[1 : self.N_particle - 1]
                - self.column.media.particle_density
                * self.Ds
                * lap_q[1 : self.N_particle - 1]
            )

            intraparticle_transport = Dp_term + Ds_term
            result[offset + 1 : offset + self.N_particle - 1] = intraparticle_transport

            # boundary condition
            grad_cp = self.particle_numerics.evaluate_gradient(cp_i, -1)
            grad_q = self.particle_numerics.evaluate_gradient(self.iso.q(cp_i), -1)

            diffusive_flux = (
                self.column.media.particle_porosity * self.Dp * grad_cp
                + self.column.media.particle_density * self.Ds * grad_q
            )

            if i == 0:
                c_bulk = self.inlet_bc.apply()
            else:
                c_bulk = c[i]

            film_flux = self.k_film * (c_bulk - cp_i[-1])

            result[offset + self.N_particle - 1] = diffusive_flux - film_flux

            if i > 0:
                sink[i] = (
                    6
                    * film_flux
                    * (1 - self.column.porosity)
                    / self.column.media.particle_diameter
                )

        result[1 : self.N_column] = transport + sink[1:]

    def _jacobian(self, t, y, ydot, result, cj, jac):
        C, Cp = self._split(y)
        n = self._n_vars()
        J = np.zeros((n, n))

        J[0, : self.N_column] = self.inlet_bc.jacobian_row(
            self.column_numerics.collocation.first_derivative[0]
        )

        d_transport = (
            self.column.porosity
            * self.velocity
            * self.column_numerics.collocation.first_derivative
            - self.column.porosity
            * self.DL
            * self.column_numerics.collocation.second_derivative
        )

        J[1 : self.N_column, : self.N_column] = d_transport[1:, :]

        coef = (
            6
            * (1 - self.column.porosity)
            * self.k_film
            / self.column.media.particle_diameter
        )

        for i in range(self.N_column):
            offset = self.N_column + i * self.N_particle
            surface = offset + self.N_particle - 1

            cp_i = Cp[i]
            dqdCp = self.iso.dq_dC(cp_i)

            rows = slice(offset + 1, surface)
            cols = slice(offset, offset + self.N_particle)

            L = self.particle_numerics.collocation.radial_operator_matrix

            J[rows, cols] = -self.column.media.particle_porosity * self.Dp * L[
                1:-1, :
            ] - self.column.media.particle_density * self.Ds * L[1:-1, :] @ np.diag(
                dqdCp
            )

            mass = (
                self.column.media.particle_porosity
                + self.column.media.particle_density * dqdCp
            )

            for j in range(1, self.N_particle - 1):
                J[offset + j, offset + j] += cj * mass[j]

            J[offset, :] = 0
            J[offset, cols] = self.center_bc.jacobian_row(
                self.particle_numerics.collocation.first_derivative[0, :]
            )

            G = self.particle_numerics.collocation.first_derivative[-1, :]
            d2qdCp2 = self.iso.d2q_dC2(cp_i)
            grad_cp = G @ cp_i

            surface_diffusion_jac = (
                G @ np.diag(dqdCp) + np.outer(grad_cp, G) * d2qdCp2[-1]
            )

            J[surface, cols] = (
                self.column.media.particle_porosity * self.Dp * G
                + self.column.media.particle_density * self.Ds * surface_diffusion_jac
            )

            if i == 0:
                J[surface, surface] += self.k_film
            else:
                J[i, i] += cj * self.column.porosity
                J[i, i] += coef
                J[i, surface] -= coef

                J[surface, i] = -self.k_film
                J[surface, surface] += self.k_film

        jac[:, :] = J
        return 0

    def _initial_conditions(self, C_init: float, C_in: float, Cp_init: float):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.N_column, C_init)
        C0[0] = self.inlet_bc.apply()

        Cp0 = np.full((self.N_column, self.N_particle), Cp_init)

        y0 = np.concatenate(
            [
                C0,
                Cp0.ravel(),
            ]
        )

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def _algebraic_vars_idx(self) -> list[int]:
        """Return indices of algebraic (non-differential) equations.

        Currently: the inlet boundary condition, plus the particle-center
        and particle-edge boundary conditions for every column node.
        """
        i = np.arange(self.N_column - 1)

        var_idxs = [0]  # liquid_phase_inlet
        var_idxs.extend(
            (self.N_column + i * self.N_particle).tolist()
        )  # particle center
        var_idxs.extend(
            (self.N_column + i * self.N_particle + (self.N_particle - 1)).tolist()
        )  # particle edge

        return var_idxs

    def solve(self, t_span, t_eval, C_in=1.0, C_init=0.0, Cp_init=0.0):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        y0, ydot0 = self._initial_conditions(C_init, C_in, Cp_init)

        result = self.column_numerics.integrate(
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

        C_out = y_out[:, : self.N_column]  # (n_times, N)

        Cp_out = y_out[:, self.N_column :].reshape(
            len(t_eval), self.N_column, self.N_particle
        )

        return (
            self.column_numerics.collocation.nodes,
            self.particle_numerics.collocation.nodes,
            C_out,
            Cp_out,
        )
