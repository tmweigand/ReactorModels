"""psdm.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..properties.column import Column
from ..properties.breakthrough import Breakthrough
from ..properties.film_transfer import FilmTransfer
from ..numerics.config import PSDMNumerics
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .boundary_conditions import BoundaryConditions, DirichletBC, SymmetryBC

__all__ = ["PSDM"]


class PSDM(NumericModel):
    """Solve conservation equations for column and particle domain simultaneously."""

    _param_names = (
        "velocity",
        "axial_diffusion",
        "pore_diffusion",
        "surface_diffusion",
        "isotherm",
        "k_film",
    )

    def __init__(
        self,
        breakthrough: Breakthrough,
        isotherm: Isotherm,
        numerics: PSDMNumerics,
        k_film: FilmTransfer,
        inlet_bc: Type[BoundaryConditions] = DirichletBC,
        center_bc: Type[BoundaryConditions] = SymmetryBC,
    ):
        # Physical parameters
        self.breakthrough: Breakthrough = breakthrough
        self.column: Column = breakthrough.column
        self.velocity = breakthrough.interstitial_velocity
        self.axial_diffusion = breakthrough.chemical.axial_diffusion

        if breakthrough.chemical.pore_diffusion is not None:
            self.pore_diffusion = breakthrough.chemical.pore_diffusion
        elif breakthrough.chemical.liquid_diffusion is not None:
            self.pore_diffusion = (
                breakthrough.chemical.liquid_diffusion
                / breakthrough.column.media.particle_tortuosity
            )

        self.surface_diffusion = breakthrough.chemical.surface_diffusion

        # Isotherm
        self.isotherm = isotherm

        # External properties
        self.axial_diffusion = breakthrough.chemical.axial_diffusion
        self.k_film = k_film.k_film if isinstance(k_film, FilmTransfer) else k_film

        # Boundary conditions
        self.inlet_bc = inlet_bc(breakthrough.mean_feed_concentration())
        self.center_bc = center_bc(node=0)

        # Numerics
        self.numerics = numerics
        self.column_numerics = numerics.column
        self.particle_numerics = numerics.particle

        # Discretization
        self.axial_nodes = len(self.column_numerics.collocation.nodes)
        self.radial_nodes = len(self.particle_numerics.collocation.nodes)

        self.assert_parameters_set()

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        return self.axial_nodes + self.radial_nodes * self.axial_nodes

    def _split(self, y: np.ndarray):
        """Return (C, Cp) where Cp is a 2D numpy array."""
        C = y[: self.axial_nodes]
        Cp = y[self.axial_nodes :].reshape(self.axial_nodes, self.radial_nodes)
        return C, Cp

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, cp = self._split(y)
        dcdt, dcpdt = self._split(ydot)

        # set interphase exchange terms to zero initially
        interphase_exchange = np.zeros(self.axial_nodes)

        # bulk phase - inlet
        result[0] = self.inlet_bc.residual(c[0])

        # bulk phase - internal and outlet
        transport = (
            self.column.porosity * dcdt[1:]
            + self.column.porosity
            * self.velocity
            * self.column_numerics.evaluate_gradient(c)[1:]
            - self.column.porosity
            * self.axial_diffusion
            * self.column_numerics.evaluate_second_derivative(c)[1:]
        )

        # particle domain loop
        for i in range(self.axial_nodes):
            cp_i = cp[i]
            dcpdt_i = dcpdt[i]

            # index of nodes at particle center
            center = self.axial_nodes + i * self.radial_nodes

            # center: symmetry (dcp/dr = 0)
            result[center] = self.center_bc.residual(
                gradient_concentration_0=self.particle_numerics.evaluate_gradient(
                    cp_i, 0
                )
            )

            # laplacian of q for surface diffusion term
            q_i = np.array([self.isotherm.q(c) for c in cp_i])
            lap_q = self.particle_numerics.evaluate_radial_operator(q_i)
            # laplacian of q for surface diffusion term
            lap_cp = self.particle_numerics.evaluate_radial_operator(cp_i)

            for r in range(1, self.radial_nodes - 1):
                # isotherm derivative for chain rule
                dqdCp = self.isotherm.dq_dC(cp_i[r])

                # particle phase - internal
                Dp_term = (
                    self.column.media.particle_porosity * dcpdt_i[r]
                    - self.column.media.particle_porosity
                    * self.pore_diffusion
                    * lap_cp[r]
                )
                Ds_term = (
                    self.column.media.particle_density * dqdCp * dcpdt_i[r]
                    - self.column.media.particle_density
                    * self.surface_diffusion
                    * lap_q[r]
                )
                result[center + r] = Dp_term + Ds_term

            # surface boundary condition - gradients at surface
            grad_cp = self.particle_numerics.evaluate_gradient(cp_i, -1)
            grad_q = self.particle_numerics.evaluate_gradient(q_i, -1)

            intraparticle_flux = (
                self.column.media.particle_porosity * self.pore_diffusion * grad_cp
                + self.column.media.particle_density * self.surface_diffusion * grad_q
            )
            surface_flux = self.k_film * (c[i] - cp_i[-1])

            result[center + self.radial_nodes - 1] = intraparticle_flux - surface_flux

            # compute interphase exchange terms
            interphase_exchange[i] = (
                6
                * surface_flux
                * (1 - self.column.porosity)
                / self.column.media.particle_diameter
            )
        # add interphase exchange term to internal bulk nodes
        result[1 : self.axial_nodes] = transport + interphase_exchange[1:]

    def _jacobian(self, t, y, ydot, result, cj, jac):
        _, Cp = self._split(y)
        _, dCpdt = self._split(ydot)
        n = self._n_vars()
        J = np.zeros((n, n))

        # bulk inlet boundary condition
        J[0, : self.axial_nodes] = self.inlet_bc.jacobian_row(
            self.column_numerics.collocation.first_derivative[0]
        )
        # derivative of transport
        d_transport = (
            self.column.porosity
            * self.velocity
            * self.column_numerics.collocation.first_derivative
            - self.column.porosity
            * self.axial_diffusion
            * self.column_numerics.collocation.second_derivative
        )
        # change in bulk residual w.r.t. bulk C
        J[1 : self.axial_nodes, : self.axial_nodes] = d_transport[1:, :]

        # coefficient for loss of mass from bulk phase
        film_coef = (
            6
            * (1 - self.column.porosity)
            * self.k_film
            / self.column.media.particle_diameter
        )

        for i in range(self.axial_nodes):
            # particle indexing
            center = self.axial_nodes + i * self.radial_nodes
            surface = center + self.radial_nodes - 1
            particle_nodes = slice(center, center + self.radial_nodes)

            for r in range(1, self.radial_nodes - 1):
                # change in particle residual w.r.t. Cp
                cp_r = Cp[i, r]
                dqdCp_r = self.isotherm.dq_dC(cp_r)
                d2qdCp2_r = self.isotherm.d2q_dC2(cp_r)

                L = self.particle_numerics.collocation.radial_operator_matrix

                for k in range(self.radial_nodes):
                    dqdCp_k = self.isotherm.dq_dC(Cp[i, k])
                    J[center + r, center + k] = (
                        -self.column.media.particle_porosity
                        * self.pore_diffusion
                        * L[r, k]
                        - self.column.media.particle_density
                        * self.surface_diffusion
                        * L[r, k]
                        * dqdCp_k
                    )
                J[center + r, center + r] += (
                    self.column.media.particle_density * d2qdCp2_r * dCpdt[i, r]
                )
                J[center + r, center + r] += cj * (
                    self.column.media.particle_porosity
                    + self.column.media.particle_density * dqdCp_r
                )

                # zero gradient at particle center
                J[center, particle_nodes] = self.center_bc.jacobian_row(
                    self.particle_numerics.collocation.first_derivative[0]
                )

                # change in surface residual w.r.t. Cp
                for k in range(self.radial_nodes):
                    dqdCp_k = self.isotherm.dq_dC(Cp[i, k])
                    G = self.particle_numerics.collocation.first_derivative[-1, :]

                    J[surface, center + k] = (
                        self.column.media.particle_porosity * self.pore_diffusion * G[k]
                        + self.column.media.particle_density
                        * self.surface_diffusion
                        * G[k]
                        * dqdCp_k
                    )
                J[surface, surface] += self.k_film

                # change in surface residual w.r.t. bulk C
                J[surface, i] = -self.k_film

        for i in range(1, self.axial_nodes):
            center = self.axial_nodes + i * self.radial_nodes
            surface = center + self.radial_nodes - 1

            # change in bulk residual w.r.t. bulk C
            J[i, i] += film_coef + cj * self.column.porosity
            # change in bulk residual w.r.t. surface Cp
            J[i, surface] -= film_coef

        jac[:, :] = J

        return 0

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C0 = np.full(self.axial_nodes, self.breakthrough.initial_concentration)
        C0[0] = self.inlet_bc.apply()

        Cp0 = np.full(
            (self.axial_nodes, self.radial_nodes),
            self.breakthrough.initial_concentration,
        )

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
        i = np.arange(self.axial_nodes - 1)

        var_idxs = [0]  # liquid_phase_inlet
        var_idxs.extend(
            (self.axial_nodes + i * self.radial_nodes).tolist()
        )  # particle center
        var_idxs.extend(
            (
                self.axial_nodes + i * self.radial_nodes + (self.radial_nodes - 1)
            ).tolist()
        )  # particle edge

        return var_idxs

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

        C_out = y_out[:, : self.axial_nodes]  # (n_times, axial_nodes)

        Cp_out = y_out[:, self.axial_nodes :].reshape(
            len(self.breakthrough.time), self.axial_nodes, self.radial_nodes
        )  # (n_times, axial_nodes, radial_nodes)

        return (
            self.column_numerics.collocation.nodes,
            self.particle_numerics.collocation.nodes,
            C_out,
            Cp_out,
        )

    def get_solid_phase_loading(
        self, pore_concentration: np.ndarray | float
    ) -> np.ndarray | float:
        """Calculate the solid phase loading q"""
        return self.isotherm.q(pore_concentration)

    def get_radial_average(self, data_in: np.ndarray) -> np.ndarray | float:
        """Return the volume-weighted average over a spherical particle."""
        assert self.column.media.particle_radius is not None

        if data_in.ndim != 3:
            raise ValueError(
                f"get_radial_average expects a 3D array, but got "
                f"an array with shape {data_in.shape}."
            )

        numerator = self.particle_numerics.collocation.integrate(
            data_in * self.particle_numerics.collocation.nodes**2, axis=2
        )
        return numerator / (self.column.media.particle_radius**3 / 3.0)
