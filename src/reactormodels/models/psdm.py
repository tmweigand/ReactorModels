"""psdm.py"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Type
import numpy as np

from ..properties.breakthrough import Breakthrough
from ..properties.film_transfer import FilmTransfer
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm
from .adsorption_kinetics import AdsorptionKinetics, LocalEquilibriumPSDM
from .boundary_conditions import InletBC, DirichletBC, SymmetryBC

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
        breakthrough: Breakthrough | Sequence[Breakthrough],
        isotherm: Isotherm | MultiSpeciesIsotherm,
        column_numerics: NumericsConfig,
        particle_numerics: NumericsConfig,
        kinetics: AdsorptionKinetics = LocalEquilibriumPSDM(),
        k_film: float | FilmTransfer | Sequence[float | FilmTransfer] = 0,
        inlet_bc: Type[InletBC] | Sequence[Type[InletBC]] = DirichletBC,
        center_bc: Type[InletBC] | Sequence[Type[InletBC]] = SymmetryBC,
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

        if isinstance(center_bc, type):
            self.center_bcs = [center_bc] * len(self.breakthroughs)
        else:
            self.center_bcs = list(center_bc)

        if len(self.inlet_bcs) != len(self.center_bcs) != len(self.breakthroughs):
            raise ValueError(
                "The number of inlet and center boundary conditions "
                "must match the number of breakthroughs."
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
        self.pore_diffusion = np.array(
            [bt.chemical.pore_diffusion for bt in self.breakthroughs]
        )
        self.surface_diffusion = np.array(
            [bt.chemical.surface_diffusion for bt in self.breakthroughs]
        )
        self.initial_concentration = np.array(
            [bt.initial_concentration for bt in self.breakthroughs]
        )
        self.initial_mass_fraction = np.array(
            [bt.initial_mass_fraction for bt in self.breakthroughs]
        )
        self.initial_pore_concentration = np.array(
            [bt.initial_pore_concentration for bt in self.breakthroughs]
        )
        self.inlet_concentration = np.array(
            [bt.mean_feed_concentration() for bt in self.breakthroughs]
        )

        # film transfer
        if isinstance(k_film, (float, FilmTransfer)):
            k_films = [k_film] * self.n_species
        else:
            k_films = list(k_film)

        if len(k_films) != self.n_species:
            raise ValueError(
                f"k_film has {len(k_films)} entries but there are "
                f"{self.n_species} breakthroughs."
            )

        self.k_film = [
            k_f.k_film if isinstance(k_f, FilmTransfer) else float(k_f)
            for k_f in k_films
        ]

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
        self.center_bc = [bc(node=0) for bc in self.center_bcs]

        # Numerics
        self.column_numerics = column_numerics
        self.particle_numerics = particle_numerics

        # Discretization
        self.axial_nodes = len(self.column_numerics.collocation.nodes)
        self.radial_nodes = len(self.particle_numerics.collocation.nodes)

        # Kinetics
        if not isinstance(kinetics, LocalEquilibriumPSDM):
            raise TypeError(
                "Only local equilibrium kinetics is currently supported for the PSDM."
            )
        kinetics.configure(
            self.column,
            isotherm,
            self.inlet_concentration,
            self.n_species,
            self.inlet_bc,
            column_numerics,
            particle_numerics,
            self.k_film,
            self.pore_diffusion,
            self.surface_diffusion,
        )
        self.kinetics = kinetics

        self.assert_parameters_set()

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, p_var = self.kinetics._split(y)
        dcdt, dp_vardt = self.kinetics._split(ydot)

        result = result.reshape(
            self.n_species,
            self.axial_nodes + self.axial_nodes * self.radial_nodes,
        )

        sink = np.zeros_like(c)
        result[:] = 0.0

        # bulk phase - inlet
        gradient = self.column_numerics.collocation.evaluate_gradient(c)
        for i, bc in enumerate(self.inlet_bc):
            result[i, 0] = bc.residual(c[i, 0], gradient[i, 0])

        # bulk phase - internal and outlet
        transport = (
            self.column.porosity * dcdt[:, 1:]
            + self.column.porosity
            * self.velocity
            * self.column_numerics.evaluate_gradient(c)[:, 1:]
            - self.column.porosity
            * self.axial_diffusion[:, None]
            * self.column_numerics.evaluate_second_derivative(c)[:, 1:]
        )

        # particle phase
        for k in range(self.axial_nodes):
            offset = self.axial_nodes + k * self.radial_nodes

            # apply chain rule
            cp, q, dcpdt, dqdt = self.kinetics._state_variables(p_var, dp_vardt, k)

            for i, bc in enumerate(self.center_bc):
                # calculate cp and q for each species
                cp_i, q_i = self.kinetics._species_variables(cp, q, k, i)

                # center: symmetry
                center_bc_var = self.kinetics._center_bc_var(cp_i, q_i)
                result[i, offset] = bc.residual(
                    gradient_concentration_0=self.particle_numerics.evaluate_gradient(
                        center_bc_var, 0
                    )
                )

                # particle phase - internal
                lap_cp = self.particle_numerics.evaluate_radial_operator(cp_i)
                lap_q = self.particle_numerics.evaluate_radial_operator(q_i)

                # calculate pore and surface diffusion terms
                Dp_term, Ds_term = self.kinetics._diffusion_residuals(
                    dcpdt, dqdt, lap_cp, lap_q, k, i
                )

                result[i, offset + 1 : offset + self.radial_nodes - 1] = (
                    Dp_term + Ds_term
                )

                # boundary condition
                grad_q = self.particle_numerics.evaluate_gradient(q_i, -1)
                grad_cp = self.particle_numerics.evaluate_gradient(cp_i, -1)

                diffusive_flux = (
                    self.column.media.particle_porosity
                    * self.pore_diffusion[i]
                    * grad_cp
                    + self.column.media.particle_density
                    * self.surface_diffusion[i]
                    * grad_q
                )

                film_flux = self.k_film[i] * (c[i, k] - cp_i[-1])

                result[i, offset + self.radial_nodes - 1] = diffusive_flux - film_flux

                sink[i, k] = (
                    6
                    * film_flux
                    * (1 - self.column.porosity)
                    / self.column.media.particle_diameter
                )

        result[:, 1 : self.axial_nodes] = transport + sink[:, 1:]

    def _jacobian(self, t, y, ydot, result, cj, jac):
        _, p_var = self.kinetics._split(y)
        _, dp_vardt = self.kinetics._split(ydot)

        S = self.n_species
        N = self.axial_nodes
        R = self.radial_nodes

        n = self.kinetics._n_vars()
        J = np.zeros((n, n))

        # Collocation operators
        D1 = self.column_numerics.collocation.first_derivative
        D2 = self.column_numerics.collocation.second_derivative

        L = self.particle_numerics.collocation.radial_operator_matrix
        G0 = self.particle_numerics.collocation.first_derivative[0, :]
        Gsurf = self.particle_numerics.collocation.first_derivative[-1, :]

        # Each species occupies this many entries in the state vector.
        species_size = N + N * R

        # Bulk phase
        for i, bc in enumerate(self.inlet_bc):
            species_offset = i * species_size

            # Inlet boundary condition
            J[species_offset, species_offset : species_offset + N] = bc.jacobian_row(
                D1[0, :]
            )

            # Bulk transport
            transport_jac = (
                self.column.porosity * self.velocity * D1
                - self.column.porosity * self.axial_diffusion[i] * D2
            )

            rows = slice(species_offset + 1, species_offset + N)
            cols = slice(species_offset, species_offset + N)

            J[rows, cols] = transport_jac[1:, :]

            # cj * dF/d(dC/dt)
            J[
                species_offset + 1 : species_offset + N,
                species_offset + 1 : species_offset + N,
            ] += (
                cj * self.column.porosity * np.eye(N - 1)
            )

        # Particle phase
        for k in range(N):
            # dqdCp:(S, S, R); d2qdCp2: (S, S, S, R)
            # d2qdCp2[i, j, m, r] = d2 q_i / (dCp_j dCp_m)
            for i in range(S):
                species_offset = i * species_size
                particle_offset = species_offset + N + k * R

                # Center boundary condition
                J[particle_offset, :] = 0.0

                J[particle_offset, particle_offset : particle_offset + R] = (
                    self.center_bc[i].jacobian_row(G0)
                )

                # Particle interior and surface
                self.kinetics._jacobian_kinetics(p_var, dp_vardt, J, cj, L, Gsurf, i, k)

        # bulk coupling
        for i in range(S):
            species_offset = i * species_size
            coef = (
                6
                * (1 - self.column.porosity)
                * self.k_film[i]
                / self.column.media.particle_diameter
            )
            for k in range(1, N):
                bulk_row = species_offset + k
                J[bulk_row, bulk_row] += coef

                dCpdq = self.kinetics._jacobian_sink_term(p_var, i, k, R)

                for j in range(S):
                    j_species_offset = j * species_size
                    j_q_offset = j_species_offset + N + k * R

                    J[bulk_row, j_q_offset + R - 1] -= coef * dCpdq[j]

        jac[:, :] = J

        return 0

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C_init = np.asarray(self.initial_concentration, dtype=float)
        p_var_init = self.kinetics._initialize_particle(
            self.initial_pore_concentration, self.initial_mass_fraction
        )

        C0 = np.broadcast_to(
            C_init[:, None],
            (self.n_species, self.axial_nodes),
        ).copy()

        gradient0 = self.column_numerics.collocation.evaluate_gradient(C0, 0)
        for i, bc in enumerate(self.inlet_bc):
            C0[i, 0] = bc.apply(gradient0[i])

        p_var0 = np.broadcast_to(
            p_var_init[:, None, None],
            (self.n_species, self.axial_nodes, self.radial_nodes),
        ).copy()

        y0 = np.concatenate(
            [
                C0,
                p_var0.reshape(self.n_species, self.axial_nodes * self.radial_nodes),
            ],
            axis=1,
        ).ravel()

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def _algebraic_vars_idx(self) -> list[int]:
        """Return indices of algebraic (non-differential) equations.

        Inlet boundary condition, plus the particle-center
        and particle-edge boundary conditions for every column node.
        """
        var_idxs = []
        for k in range(self.n_species):
            species_offset = k * (
                self.axial_nodes + self.axial_nodes * self.radial_nodes
            )
            # Liquid-phase inlet
            var_idxs.append(species_offset)
            # Particle center and edge
            for i in range(self.axial_nodes):
                particle_offset = (
                    species_offset + self.axial_nodes + i * self.radial_nodes
                )
                # particle center
                var_idxs.append(particle_offset)
                # particle edge
                var_idxs.append(particle_offset + self.radial_nodes - 1)
        return var_idxs

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
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )

        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # result.values.y has shape (n_out, n_vars); skip the t=t_span[0] row
        y_out = result.values.y[1:]  # (n_times, n_vars)

        y_out = y_out.reshape(
            len(y_out),
            self.n_species,
            self.axial_nodes + self.axial_nodes * self.radial_nodes,
        )

        C_out = y_out[:, :, : self.axial_nodes]  # (n_times, n_species, axial_nodes)

        Cp_out, q_out = self.kinetics._particle_out(
            y_out
        )  # (n_times, n_species, axial_nodes, radial_nodes)

        return (
            self.column_numerics.collocation.nodes,
            self.particle_numerics.collocation.nodes,
            C_out,
            Cp_out,
            q_out,
        )

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
