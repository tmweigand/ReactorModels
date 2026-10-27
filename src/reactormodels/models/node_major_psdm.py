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

__all__ = ["NodePSDM"]


class NodePSDM(NumericModel):
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
        if not hasattr(self, "_last_progress"):
            self._last_progress = -1.0

        progress = 100 * t / self.breakthroughs[0].time[-1]
        progress = np.clip(progress, 0.0, 100.0)
        progress = round(progress, 1)

        if progress > self._last_progress:
            print(
                f"Progress: {progress:.1f}% complete",
                flush=True,
            )
            self._last_progress = progress

        S = self.n_species
        N = self.axial_nodes
        R = self.radial_nodes

        # -------------------------------------------------------------
        # Unpack Node-Major state vector back into (S, N) and (S, N, R)
        # -------------------------------------------------------------
        # Node-Major layout per node k: [C_0..C_S, Cp_0..Cp_S (radial R)]
        node_block_size = S * (1 + R)
        y_reshaped = y.reshape(N, node_block_size)
        ydot_reshaped = ydot.reshape(N, node_block_size)

        c = np.zeros((S, N))
        dcdt = np.zeros((S, N))
        p_var = np.zeros((S, N, R))
        dp_vardt = np.zeros((S, N, R))

        for k in range(N):
            node_y = y_reshaped[k]
            node_ydot = ydot_reshaped[k]
            c[:, k] = node_y[:S]
            dcdt[:, k] = node_ydot[:S]
            p_var[:, k, :] = node_y[S:].reshape(S, R)
            dp_vardt[:, k, :] = node_ydot[S:].reshape(S, R)

        # Allocate internal results in species-major format for physics
        res_c = np.zeros((S, N))
        res_p = np.zeros((S, N, R))
        sink = np.zeros((S, N))

        # bulk phase - inlet
        self.kinetics._inlet_bc_loop(c, res_c)

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
        for k in range(N):
            # apply chain rule
            cp, q, dcpdt, dqdt = self.kinetics._state_variables(p_var, dp_vardt, k)

            for i, bc in enumerate(self.center_bc):
                # calculate cp and q for each species
                cp_i, q_i = self.kinetics._species_variables(cp, q, k, i)

                # center: symmetry
                center_bc_var = self.kinetics._center_bc_var(cp_i, q_i)
                res_p[i, k, 0] = bc.residual(
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

                res_p[i, k, 1 : R - 1] = Dp_term + Ds_term

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

                res_p[i, k, R - 1] = diffusive_flux - film_flux

                sink[i, k] = (
                    6
                    * film_flux
                    * (1 - self.column.porosity)
                    / self.column.media.particle_diameter
                )

        res_c[:, 1:N] = transport + sink[:, 1:]

        # -------------------------------------------------------------
        # Pack residual back into Node-Major order in result[:]
        # -------------------------------------------------------------
        res_reshaped = result.reshape(N, node_block_size)
        res_reshaped[:] = 0.0
        for k in range(N):
            res_reshaped[k, :S] = res_c[:, k]
            res_reshaped[k, S:] = res_p[:, k, :].ravel()

    def _jacobian(self, t, y, ydot, result, cj, jac):
        S = self.n_species
        N = self.axial_nodes
        R = self.radial_nodes

        n = self.kinetics._n_vars()
        J = np.zeros((n, n))

        node_block_size = S * (1 + R)

        # -------------------------------------------------------------
        # Unpack state vector for kinetics Jacobian evaluation
        # -------------------------------------------------------------
        y_reshaped = y.reshape(N, node_block_size)
        ydot_reshaped = ydot.reshape(N, node_block_size)

        p_var = np.zeros((S, N, R))
        dp_vardt = np.zeros((S, N, R))

        for k in range(N):
            p_var[:, k, :] = y_reshaped[k, S:].reshape(S, R)
            dp_vardt[:, k, :] = ydot_reshaped[k, S:].reshape(S, R)

        # Collocation operators
        D1 = self.column_numerics.collocation.first_derivative
        D2 = self.column_numerics.collocation.second_derivative

        L = self.particle_numerics.collocation.radial_operator_matrix
        G0 = self.particle_numerics.collocation.first_derivative[0, :]
        Gsurf = self.particle_numerics.collocation.first_derivative[-1, :]

        # Bulk phase
        for i, bc in enumerate(self.inlet_bc):
            # Inlet BC (Node 0)
            inlet_row = 0 * node_block_size + i
            for k in range(N):
                col = k * node_block_size + i
                J[inlet_row, col] = bc.jacobian_row(D1[0, :])[k]

            # Transport derivatives for node k > 0
            for k in range(1, N):
                row = k * node_block_size + i
                for m in range(N):
                    col = m * node_block_size + i
                    val = (
                        self.column.porosity * self.velocity * D1[k, m]
                        - self.column.porosity * self.axial_diffusion[i] * D2[k, m]
                    )
                    J[row, col] += val

                # cj term for differential bulk concentration
                J[row, row] += cj * self.column.porosity

        # -------------------------------------------------------------
        # 2. Particle Kinetics Derivatives
        # -------------------------------------------------------------
        for k in range(N):
            node_offset = k * node_block_size
            for i in range(S):
                p_offset = node_offset + S + i * R

                # Center BC
                J[p_offset, :] = 0.0
                J[p_offset, p_offset : p_offset + R] = self.center_bc[i].jacobian_row(
                    G0
                )

                # Kinetics interior + surface terms
                _, dqdt_k, _, dCpdq, _, d2Cpdq2 = self.kinetics._jacobian_vars(
                    p_var, dp_vardt, k
                )
                S = self.kinetics.n_species
                R = self.kinetics.radial_nodes
                node_block_size = S * (1 + R)
                node_offset = k * node_block_size

                particle_offset = node_offset + S + i * R

                for r in range(1, R - 1):
                    row = particle_offset + r

                    # Pore diffusion: -epsilon_p * Dp_i * L(Cp_i)
                    for j in range(S):
                        j_p_offset = node_offset + S + j * R

                        J[row, j_p_offset : j_p_offset + R] += (
                            -self.kinetics.column.media.particle_porosity
                            * self.kinetics.pore_diffusion[i]
                            * (L @ np.diag(dCpdq[i, j, :]))
                        )[r, :]

                    # Surface diffusion term: -rho_p * Ds_i * L(q_i)
                    J[row, particle_offset : particle_offset + R] += (
                        -self.kinetics.column.media.particle_density
                        * self.kinetics.surface_diffusion[i]
                        * L[r, :]
                    )

                    # Time derivative terms
                    for j in range(S):
                        j_p_offset = node_offset + S + j * R

                        # cj * epsilon_p * dCp_i/dq_j
                        J[row, j_p_offset + r] += (
                            cj
                            * self.kinetics.column.media.particle_porosity
                            * dCpdq[i, j, r]
                        )

                        # Derivative of dCp_i/dt with respect to q_j
                        J[
                            row, j_p_offset + r
                        ] += self.kinetics.column.media.particle_porosity * np.sum(
                            d2Cpdq2[i, j, :, r] * dqdt_k[:, r]
                        )

                    # cj * rho_p * dq_i/dt
                    J[row, particle_offset + r] += (
                        cj * self.kinetics.column.media.particle_density
                    )

                # Particle surface
                surface = particle_offset + R - 1

                # Diffusive pore flux: +epsilon_p * Dp_i * dCp_i/dr
                for j in range(S):
                    j_p_offset = node_offset + S + j * R

                    J[surface, j_p_offset : j_p_offset + R] += (
                        self.kinetics.column.media.particle_porosity
                        * self.kinetics.pore_diffusion[i]
                        * (Gsurf @ np.diag(dCpdq[i, j, :]))
                    )

                # Surface diffusion flux: +rho_p * Ds_i * dq_i/dr
                J[surface, particle_offset : particle_offset + R] += (
                    self.kinetics.column.media.particle_density
                    * self.kinetics.surface_diffusion[i]
                    * Gsurf
                )

                # Film flux coupling with bulk concentration
                C_col = node_offset + i

                # -kf_i * C_i,k
                J[surface, C_col] -= self.kinetics.k_film[i]

                # +kf_i * Cp_i,surface (w.r.t q_j)
                for j in range(S):
                    j_p_offset = node_offset + S + j * R

                    J[surface, j_p_offset + R - 1] += (
                        self.kinetics.k_film[i] * dCpdq[i, j, R - 1]
                    )

        # -------------------------------------------------------------
        # 3. Inter-phase Mass Transfer Sink Coupling
        # -------------------------------------------------------------
        for i in range(S):
            coef = (
                6
                * (1 - self.column.porosity)
                * self.k_film[i]
                / self.column.media.particle_diameter
            )
            for k in range(1, N):
                node_offset = k * node_block_size
                bulk_row = node_offset + i

                J[bulk_row, bulk_row] += coef
                dCpdq = self.kinetics._jacobian_sink_term(p_var, i, k, R)

                for j in range(S):
                    q_surf_col = node_offset + S + j * R + (R - 1)
                    J[bulk_row, q_surf_col] -= coef * dCpdq[j]

        jac[:, :] = J

        return 0

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        C_init = np.asarray(self.initial_concentration, dtype=float)
        p_var_init = self.kinetics._initialize_particle(
            self.initial_pore_concentration, self.initial_mass_fraction
        )

        S = self.n_species
        N = self.axial_nodes
        R = self.radial_nodes

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

        node_block_size = S * (1 + R)
        y0 = np.zeros(N * node_block_size)

        for k in range(N):
            offset = k * node_block_size
            y0[offset : offset + S] = C0[:, k]
            y0[offset + S : offset + node_block_size] = p_var0[:, k, :].ravel()

        ydot0 = np.zeros_like(y0)
        return y0, ydot0

    def _algebraic_vars_idx(self) -> list[int]:
        """Return indices of algebraic (non-differential) equations.

        Inlet boundary condition, plus the particle-center
        and particle-edge boundary conditions for every column node.
        """
        var_idxs = []
        S = self.n_species
        N = self.axial_nodes
        R = self.radial_nodes
        node_block_size = S * (1 + R)

        # 1. Inlet boundary nodes at k = 0
        for i in range(S):
            var_idxs.append(i)  # bulk inlet at k=0

        # 2. Particle center (r=0) and particle edge (r=R-1) for every node k and species i
        for k in range(N):
            node_offset = k * node_block_size
            for i in range(S):
                p_offset = node_offset + S + i * R
                var_idxs.append(p_offset)  # Particle center
                var_idxs.append(p_offset + R - 1)  # Particle outer surface

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
        n_times = len(y_out)

        S = self.n_species
        N = self.axial_nodes
        R = self.radial_nodes
        node_block_size = S * (1 + R)

        C_out = np.zeros((n_times, S, N))
        p_var_out = np.zeros((n_times, S, N, R))

        # Reconstruct standard species-major arrays for downstream plotting
        for t_idx in range(n_times):
            y_t = y_out[t_idx].reshape(N, node_block_size)
            for k in range(N):
                C_out[t_idx, :, k] = y_t[k, :S]
                p_var_out[t_idx, :, k, :] = y_t[k, S:].reshape(S, R)

        # Unpack particle phase outputs through kinetics post-processing
        # (Pass reconstructed species-major array to kinetics)
        y_species_major = np.concatenate(
            [C_out.reshape(n_times, S * N), p_var_out.reshape(n_times, S * N * R)],
            axis=1,
        )

        Cp_out, q_out = self.kinetics._particle_out(y_species_major)

        return (
            self.column_numerics.collocation.nodes,
            self.particle_numerics.collocation.nodes,
            C_out,
            Cp_out,
            q_out,
        )
