"""domain_coupling.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..column_data import Column
from ..breakthrough_data import Breakthrough
from ..numerics.config import NumericsConfig
from .isotherm import Isotherm
from .adsorption_kinetics import AdsorptionKinetics
from .boundary_conditions import InletBC, DirichletBC

__all__ = ["DomainCoupling"]


class DomainCoupling:
    """Solve conservation equations for column and particle domain simultaneously."""

    def __init__(
        self,
        column: Column,
        breakthrough: Breakthrough,
        axial_diffusion: float,
        pore_diffusion: float,
        surface_diffusion: float,
        initial_concentration: float,
        isotherm: Isotherm,
        column_numerics: NumericsConfig,
        particle_numerics: NumericsConfig,
        mode: AdsorptionKinetics = AdsorptionKinetics.LOCAL_EQUILIBRIUM,
        k_film: float = 0,
        inlet_bc: Type[InletBC] = DirichletBC,
    ):
        self.column = column
        self.breakthrough = breakthrough
        self.velocity = breakthrough.interstitial_velocity()
        self.DL = axial_diffusion
        self.Dp = pore_diffusion
        self.Ds = surface_diffusion
        self.inlet_concentration = breakthrough.mean_feed_concentration()
        self.initial_concentration = initial_concentration
        self.iso = isotherm
        self.column_numerics = column_numerics
        self.particle_numerics = particle_numerics
        self.mode = mode
        self.k_film = k_film
        self.inlet_bc = inlet_bc(self.inlet_concentration)
        self.Nz = len(self.column_numerics.collocation.nodes)
        self.Nr = len(self.particle_numerics.collocation.nodes)

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        return self.Nz + self.Nr * self.Nz

    def _split(self, y: np.ndarray):
        """Return (C, Cp) where Cp is the pore liquid concentration."""
        C = y[: self.Nz]
        Cp = y[self.Nz :].reshape(self.Nz, self.Nr)
        return C, Cp

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, cp = self._split(y)
        dcdt, dcpdt = self._split(ydot)

        sink = np.zeros(self.Nz)

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

        for i in range(1, self.Nz):
            cp_i = cp[i]
            dcpdt_i = dcpdt[i]

            offset = self.Nz + i * self.Nr

            # center: symmetry
            result[offset] = self.particle_numerics.collocation.evaluate_gradient(
                cp_i, 0
            )

            # particle phase - internal
            Dp_term = (
                self.column.particle_porosity * dcpdt_i[1 : self.Nr - 1]
                - self.column.particle_porosity
                * self.Dp
                * self.particle_numerics.evaluate_radial_operator(cp_i)[1 : self.Nr - 1]
            )

            dqdCp = self.iso.dq_dC(cp_i)
            lap_q = self.particle_numerics.evaluate_radial_operator(self.iso.q(cp_i))

            Ds_term = (
                self.column.particle_density * (dqdCp * dcpdt_i)[1 : self.Nr - 1]
                - self.column.particle_density * self.Ds * lap_q[1 : self.Nr - 1]
            )

            intraparticle_transport = Dp_term + Ds_term

            result[offset + 1 : offset + self.Nr - 1] = intraparticle_transport

            # boundary condition
            grad_cp = self.particle_numerics.evaluate_gradient(cp_i, -1)
            grad_q = self.particle_numerics.evaluate_gradient(self.iso.q(cp_i), -1)

            diffusive_flux = (
                self.column.particle_porosity * self.Dp * grad_cp
                + self.column.particle_density * self.Ds * grad_q
            )

            film_flux = self.k_film * (c[i] - cp_i[-1])

            result[offset + self.Nr - 1] = diffusive_flux - film_flux

            sink[i] = (
                6
                * film_flux
                * (1 - self.column.porosity)
                / self.column.particle_diameter
            )

        result[1 : self.Nz] = transport + sink[1:]

    def _jacobian(self, t, y, ydot, result, cj, jac):
        C, Cp = self._split(y)
        n = self._n_vars()
        J = np.zeros((n, n))

        J[0, : self.Nz] = self.inlet_bc.jacobian_row(
            self.column_numerics.collocation.first_derivative[0]
        )

        # derivative of axial transport
        d_transport = (
            self.column.porosity
            * self.velocity
            * self.column_numerics.collocation.first_derivative
            - self.column.porosity
            * self.DL
            * self.column_numerics.collocation.second_derivative
        )

        J[1 : self.Nz, : self.Nz] = d_transport[1:, :]

        coef = (
            6 * (1 - self.column.porosity) / self.column.particle_diameter * self.k_film
        )

        for i in range(self.Nz):
            J[i, i] += cj * self.column.porosity
            J[i, i] += coef

            offset = self.Nz + i * self.Nr
            surface = offset + self.Nr - 1

            J[i, surface] -= coef

            cp_i = Cp[i]
            dqdCp = self.iso.dq_dC(cp_i)

            rows = slice(offset + 1, surface)
            cols = slice(offset, offset + self.Nr)

            L = self.particle_numerics.collocation.radial_operator_matrix

            J[rows, cols] = -self.column.particle_porosity * self.Dp * L[
                1:-1, :
            ] - self.column.particle_density * self.Ds * L[1:-1, :] @ np.diag(dqdCp)

            mass = self.column.particle_porosity + self.column.particle_density * dqdCp

            for j in range(1, self.Nr - 1):
                J[offset + j, offset + j] += cj * mass[j]

            J[offset, :] = 0

            J[offset, cols] = self.particle_numerics.collocation.first_derivative[0, :]

            J[surface, i] = -self.k_film

            G = self.particle_numerics.collocation.first_derivative[-1, :]
            d2qdCp2 = self.iso.d2q_dC2(cp_i)

            grad_cp = G @ cp_i

            surface_diffusion_jac = (
                G @ np.diag(dqdCp) + np.outer(grad_cp, G) * d2qdCp2[-1]
            )

            J[surface, cols] = (
                self.column.particle_porosity * self.Dp * G
                + self.column.particle_density * self.Ds * surface_diffusion_jac
            )
            J[surface, surface] += self.k_film

        jac[:, :] = J

        return 0

    def _initial_conditions(self, C_init: float, C_in: float, Cp_init: float):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.Nz, C_init)
        C0[0] = C_in

        Cp0 = np.full((self.Nz, self.Nr), Cp_init)

        y0 = np.concatenate(
            [
                C0,
                Cp0.ravel(),
            ]
        )

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for this model.
        """
        return [0]

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

        C_out = y_out[:, : self.Nz]  # (n_times, N)

        Cp_out = y_out[:, self.Nz :].reshape(len(t_eval), self.Nz, self.Nr)

        return (
            self.column_numerics.collocation.nodes,
            self.particle_numerics.collocation.nodes,
            C_out,
            Cp_out,
        )
