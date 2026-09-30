"""adsorption_kinetics.py"""

import numpy as np

from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm
from ..numerics.config import NumericsConfig
from collections.abc import Sequence


def _is_implemented(obj, method_name):
    method = getattr(type(obj), method_name)
    base_method = getattr(MultiSpeciesIsotherm, method_name)
    return method is not base_method


class AdsorptionKinetics:
    """Set the form of the adsorption kinetics."""

    def __init__(self):
        self.column = None
        self.inlet_concentration = None
        self.solution_variable = None
        self.n_species = None
        self.k_film = None
        self.pore_diffusion = None
        self.surface_diffusion = None

    def configure(
        self,
        column,
        isotherm,
        inlet_concentration,
        n_species,
        inlet_bc,
        column_numerics,
        particle_numerics=None,
        k_film: Sequence[float] | None = None,
        pore_diffusion=None,
        surface_diffusion=None,
    ):
        """Initialize convenient classes/quantities."""
        self.column = column
        self.isotherm: Isotherm | MultiSpeciesIsotherm = isotherm
        self.column_numerics: NumericsConfig = column_numerics
        self.particle_numerics = particle_numerics
        self.axial_nodes = len(self.column_numerics.collocation.nodes)
        if particle_numerics is not None:
            self.radial_nodes = len(self.particle_numerics.collocation.nodes)
        self.inlet_concentration = inlet_concentration
        self.n_species = n_species
        self.inlet_bc = inlet_bc
        self.k_film = k_film
        self.pore_diffusion = pore_diffusion
        self.surface_diffusion = surface_diffusion

        if isinstance(self.isotherm, MultiSpeciesIsotherm) and isinstance(
            self, (LinearDrivingForce, SecondOrder)
        ):
            assert all(
                _is_implemented(self.isotherm, name)
                for name in ("q", "dq_dC", "d2q_dC2")
            ), (
                "Linear driving force and second order kinetics is not supported"
                " for multi-species isotherms with q as the solution variable."
            )

        if isinstance(self.isotherm, Isotherm):
            self.solution_variable = LiquidPhaseSolve(self)

        elif isinstance(self.isotherm, MultiSpeciesIsotherm):
            if all(
                _is_implemented(self.isotherm, name)
                for name in ("q", "dq_dC", "d2q_dC2")
            ):
                self.solution_variable = LiquidPhaseSolve(self)
            else:
                self.solution_variable = SolidPhaseSolve(self)

    def _n_vars(self):
        """Total length of the IDA state vector."""
        raise NotImplementedError

    def _split(self, y):
        """Return (C, q)."""
        raise NotImplementedError

    def _reshape_result(self, result):
        """Set the shape of residual matrix."""
        raise NotImplementedError

    def _state_variables(self, y, ydot):
        """Compute time derivative variables."""
        raise NotImplementedError

    def _species_variables(self, cp, q, k, i):
        """State variables in species loop."""
        raise NotImplementedError

    def _diffusion_residuals(self, dcpdt, dqdt, lap_cp, lap_q, k, i):
        """Compute intraparticle diffusion residuals."""
        raise NotImplementedError

    def _inlet_bc_loop(self, c, result) -> None:
        """Set inlet boundary condition for each species."""
        raise NotImplementedError

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid phase residual."""
        raise NotImplementedError

    def _jacobian_kinetics(self, c, q, J, cj, D1, D2, diffusion, velocity) -> None:
        """Return liquid phase jacobian."""
        raise NotImplementedError

    def _jacobian_vars(self, p_var, dp_vardt, k):
        """Compute variables for jacobian."""
        raise NotImplementedError

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic."""
        raise NotImplementedError

    def _set_initial_conditions(self, C0, q_init):
        """Return y0 consistent with the algebraic constraint."""
        raise NotImplementedError

    def _initialize_particle(self, cp_init, q_init):
        """Initialize particle concentrations."""
        raise NotImplementedError

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        raise NotImplementedError

    def _particle_out(self, y_out):
        """Return particle concentrations."""
        raise NotImplementedError


class LocalEquilibrium(AdsorptionKinetics):
    """Methods for local equilibrium assumption:

    dq/dt = dq/dC * dC/dt

    """

    def _n_vars(self):
        """Total length of the IDA state vector."""
        return self.axial_nodes * self.n_species

    def _split(self, y):
        """Return (C, q)."""
        y = y.reshape(self.n_species, self.axial_nodes)
        return self.solution_variable._split_state(y)

    def _reshape_result(self, result):
        """Set the shape of residual matrix."""
        result = result.reshape(self.n_species, self.axial_nodes)
        return result

    def _state_variables(self, y, ydot):
        """Compute time derivative variables."""
        c, q = self._split(y)
        dcdt, dqdt = self._split(ydot)
        return self.solution_variable._time_derivatives(c, q, dcdt, dqdt)

    def _inlet_bc_loop(self, c, result) -> None:
        """Set inlet boundary condition for each species."""
        gradient = self.column_numerics.collocation.evaluate_gradient(c)
        for i, bc in enumerate(self.inlet_bc):
            result[i, 0] = bc.residual(c[i, 0], gradient[i, 0])

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid phase residual."""
        result[:, 1:] = transport + self.column.media.bed_density * dqdt

    def _jacobian_kinetics(self, c, q, J, cj, D1, D2, diffusion, velocity) -> None:
        """Return liquid phase jacobian."""
        # Inlet boundary conditions
        for i in range(self.n_species):
            row = i * self.axial_nodes
            J[row, :] = 0.0

            col_start = i * self.axial_nodes
            col_end = (i + 1) * self.axial_nodes

            J[row, col_start:col_end] = self.inlet_bc[i].jacobian_row(D1[0, :])

        interior_J = self.solution_variable._interior_jacobian(c, q, cj)

        for k in range(1, self.axial_nodes):
            indices = np.arange(self.n_species) * self.axial_nodes + k
            J[np.ix_(indices, indices)] += interior_J[:, :, k - 1]
            self.solution_variable._transport_jacobian(
                J, k, D1, D2, velocity, diffusion
            )

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic."""
        return [i * self.axial_nodes for i in range(self.n_species)]

    def _set_initial_conditions(self, C0, q_init):
        """Return y0 consistent with the algebraic constraint."""
        return self.solution_variable._set_initial_states(C0, q_init)

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        # (n_times, n_species, axial_nodes)
        y_out = y_out.reshape(len(y_out), self.n_species, self.axial_nodes)
        return self.solution_variable._model_outs(y_out)


class LocalEquilibriumPSDM(AdsorptionKinetics):
    """Local equilibrium methods for the PSDM."""

    def _n_vars(self):
        """Total length of the IDA state vector."""
        return (
            self.axial_nodes + self.radial_nodes * self.axial_nodes
        ) * self.n_species

    def _split(self, y):
        """Return (C, q)."""
        y = y.reshape(
            self.n_species,
            self.axial_nodes + self.axial_nodes * self.radial_nodes,
        )
        C = y[:, : self.axial_nodes]
        particle_state = y[:, self.axial_nodes :].reshape(
            self.n_species, self.axial_nodes, self.radial_nodes
        )
        return C, particle_state

    def _reshape_result(self, result):
        """Set the shape of residual matrix."""
        raise NotImplementedError

    def _state_variables(self, p_var, dp_vardt, k):
        """Compute time derivative variables."""
        return self.solution_variable._psdm_time_derivatives(p_var, dp_vardt, k)

    def _species_variables(self, cp, q, k, i):
        """State variables in species loop."""
        return self.solution_variable._particle_concentrations(cp, q, k, i)

    def _diffusion_residuals(self, dcpdt, dqdt, lap_cp, lap_q, k, i):
        """Compute intraparticle diffusion residuals."""
        return self.solution_variable._diffusion_terms(dcpdt, dqdt, lap_cp, lap_q, k, i)

    def _inlet_bc_loop(self, c, result) -> None:
        """Set inlet boundary condition for each species."""
        raise NotImplementedError

    def _center_bc_var(self, cp_i, q_i):
        """Return variable for center bc."""
        return self.solution_variable._center_state(cp_i, q_i)

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid phase residual."""
        raise NotImplementedError

    def _jacobian_vars(self, p_var, dp_vardt, k):
        """Compute variables for jacobian."""
        return self.solution_variable._jacobian_concentrations(p_var, dp_vardt, k)

    def _jacobian_kinetics(self, p_var, dp_vardt, J, cj, L, Gsurf, i, k) -> None:
        """Return liquid phase jacobian."""
        self.solution_variable._particle_jacobian(
            p_var, dp_vardt, J, cj, L, Gsurf, i, k
        )

    def _jacobian_sink_term(self, p_var, i, k, R):
        """Jacobian to couple with bulk domain."""
        return self.solution_variable._particle_sink_term(p_var, i, k, R)

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic."""
        raise NotImplementedError

    def _set_initial_conditions(self, C0, q_init):
        """Return y0 consistent with the algebraic constraint."""
        raise NotImplementedError

    def _initialize_particle(self, cp_init, q_init):
        """Initialize particle concentrations."""
        return self.solution_variable._initial_particle_concentration(cp_init, q_init)

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        raise NotImplementedError

    def _particle_out(self, y_out):
        """Return particle concentrations."""
        p_var_out = y_out[:, :, self.axial_nodes :].reshape(
            len(y_out),
            self.n_species,
            self.axial_nodes,
            self.radial_nodes,
        )
        return self.solution_variable._compute_particle(p_var_out)


class SolutionVariable:
    """State dependent variable"""

    def __init__(self, kinetics: AdsorptionKinetics):
        self.kinetics = kinetics

    def _split_state(self, y):
        """Split state for liquid or solid phase solve."""
        raise NotImplementedError

    def _time_derivatives(self, c, q, dcdt, dqdt):
        """Compute time derivative variables."""
        raise NotImplementedError

    def _psdm_time_derivatives(self, p_var, dp_vardt, k):
        """Compute intraparticle time derivatives."""
        raise NotImplementedError

    def _particle_concentrations(self, cp, q, k, i):
        """Compute cp and q per species."""
        raise NotImplementedError

    def _diffusion_terms(self, dcpdt, dqdt, lap_cp, lap_q, k, i):
        """Compute intraparticle diffusion terms."""
        raise NotImplementedError

    def _jacobian_concentrations(self, p_var, dp_vardt, k):
        """Compute q or Cp for jacobian."""
        raise NotImplementedError

    def _interior_jacobian(self, c, q, cj):
        """Return the interior jacobian"""
        raise NotImplementedError

    def _transport_jacobian(self, J, k, D1, D2, velocity, diffusion) -> None:
        """Return the transport jacobian"""
        raise NotImplementedError

    def _set_initial_states(self, C0, q_init):
        """Set initial state for C and q."""
        raise NotImplementedError

    def _initial_particle_concentration(self, cp_init, q_init):
        """Set initial particle concentrations."""
        raise NotImplementedError

    def _model_outs(self, y_out):
        """Return output for C and q."""
        raise NotImplementedError

    def _compute_particle(self, p_var_out):
        """Compute output particle concentrations."""
        raise NotImplementedError


class LiquidPhaseSolve(SolutionVariable):
    """Methods to solve for liquid phase concentration."""

    def _split_state(self, y):
        """Split state for liquid or solid phase solve."""
        C = y[:, :]
        q = np.empty_like(y)
        return C, q

    def _time_derivatives(self, c, q, dcdt, dqdt):
        """Compute time derivative variables."""
        dq_dC = self.kinetics.isotherm.dq_dC(c[:, 1:])
        dqdt = np.einsum("ijn,jn->in", dq_dC, dcdt[:, 1:])
        return c, q, dcdt, dqdt

    def _psdm_time_derivatives(self, p_var, dp_vardt, k):
        """Compute intraparticle time derivatives."""
        q = self.kinetics.isotherm.q(p_var[:, k, :])
        cp = p_var

        dq_dC = self.kinetics.isotherm.dq_dC(p_var[:, k, :])

        dcpdt = dp_vardt
        dqdt = np.einsum(
            "ijr,jr->ir",
            dq_dC,
            dp_vardt[:, k, :],
        )
        return cp, q, dcpdt, dqdt

    def _particle_concentrations(self, cp, q, k, i):
        """Compute cp and q per species."""
        cp_i = cp[i, k, :]
        q_i = q[i, :]
        return cp_i, q_i

    def _center_state(self, cp_i, q_i):
        """Choose variable for center bc."""
        return cp_i

    def _diffusion_terms(self, dcpdt, dqdt, lap_cp, lap_q, k, i):
        """Compute intraparticle diffusion terms."""
        Dp_term = (
            self.kinetics.column.media.particle_porosity * dcpdt[i, k, 1:-1]
            - self.kinetics.column.media.particle_porosity
            * self.kinetics.pore_diffusion[i]
            * lap_cp[1:-1]
        )

        Ds_term = (
            self.kinetics.column.media.particle_density * dqdt[i, 1:-1]
            - self.kinetics.column.media.particle_density
            * self.kinetics.surface_diffusion[i]
            * lap_q[1:-1]
        )
        return Dp_term, Ds_term

    def _jacobian_concentrations(self, p_var, dp_vardt, k):
        """Compute q or Cp for jacobian."""
        Cp_k = p_var[:, k, :]
        dCpdt_k = dp_vardt[:, k, :]
        dqdt_k = None
        dqdCp = self.kinetics.isotherm.dq_dC(Cp_k)
        dCpdq = None
        d2qdCp2 = self.kinetics.isotherm.d2q_dC2(Cp_k)
        d2Cpdq2 = None
        return dCpdt_k, dqdt_k, dqdCp, dCpdq, d2qdCp2, d2Cpdq2

    def _interior_jacobian(self, c, q, cj):
        """Return the interior jacobian"""
        # dq/dC
        dq_dC = self.kinetics.isotherm.dq_dC(c[:, 1:])

        # Time-derivative contribution:
        interior_J = (
            cj
            * self.kinetics.column.porosity
            * np.eye(self.kinetics.n_species)[:, :, None]
            + cj * self.kinetics.column.media.bed_density * dq_dC
        )
        return interior_J

    def _particle_jacobian(self, p_var, dp_vardt, J, cj, L, Gsurf, i, k) -> None:
        """Jacobian loop for particle interior."""
        dCpdt_k, _, dqdCp, _, d2qdCp2, _ = self._jacobian_concentrations(
            p_var, dp_vardt, k
        )
        S = self.kinetics.n_species
        N = self.kinetics.axial_nodes
        R = self.kinetics.radial_nodes
        species_size = N + N * R
        for r in range(1, R - 1):
            species_offset = i * species_size
            particle_offset = species_offset + N + k * R
            row = particle_offset + r
            # pore diffusion
            J[row, particle_offset : particle_offset + R] += (
                -self.kinetics.column.media.particle_porosity
                * self.kinetics.pore_diffusion[i]
                * L[r, :]
            )
            # surface diffusion
            for j in range(S):
                j_species_offset = j * species_size
                j_Cp_offset = j_species_offset + N + k * R

                J[
                    row,
                    j_Cp_offset : j_Cp_offset + R,
                ] += (
                    -self.kinetics.column.media.particle_density
                    * self.kinetics.surface_diffusion[i]
                    * (L @ np.diag(dqdCp[i, j, :]))
                )[r, :]
            # Time derivative terms
            # epsilon_p * dCp_i/dt + rho_p * dq_i/dt
            # dq_i/dt = sum_j dq_i/dCp_j * dCp_j/dt
            for j in range(S):
                j_species_offset = j * species_size
                j_Cp_offset = j_species_offset + N + k * R

                # cj * rho_p * dq_i/dCp_j
                J[row, j_Cp_offset + r] += (
                    cj * self.kinetics.column.media.particle_density * dqdCp[i, j, r]
                )

                # Derivative of dq_i/dt with respect to Cp_j:
                # sum_m [d2q_i/(dCp_j dCp_m) * dCp_m/dt]
                J[
                    row, j_Cp_offset + r
                ] += self.kinetics.column.media.particle_density * np.sum(
                    d2qdCp2[i, j, :, r] * dCpdt_k[:, r]
                )

            # cj * epsilon_p * dCp_i/dCp_j
            J[row, j_Cp_offset + r] += cj * self.kinetics.column.media.particle_porosity

        # particle surface
        surface = particle_offset + R - 1
        J[surface, particle_offset : particle_offset + R] += (
            self.kinetics.column.media.particle_porosity
            * self.kinetics.pore_diffusion[i]
            * Gsurf
        )
        # Surface diffusion flux:
        # rho_p * Ds_i * dq_i/dr
        for j in range(S):
            j_species_offset = j * species_size
            j_Cp_offset = j_species_offset + N + k * R

            J[
                surface,
                j_Cp_offset : j_Cp_offset + R,
            ] += (
                self.kinetics.column.media.particle_density
                * self.kinetics.surface_diffusion[i]
                * (Gsurf @ np.diag(dqdCp[i, j, :]))
            )

        # Film flux
        # film_flux_i = kf_i * (C_i,k - Cp_i,surface)
        C_col = species_offset + k

        # -kf_i * C_i,k
        J[surface, C_col] -= self.kinetics.k_film[i]

        # +kf_i * Cp_i,surface
        J[surface, particle_offset + R - 1] += self.kinetics.k_film[i]

    def _particle_sink_term(self, p_var, i, k, R):
        """Compute jacobian for sink term."""
        return np.ones(self.kinetics.n_species)

    def _transport_jacobian(self, J, k, D1, D2, velocity, diffusion) -> None:
        """Return the transport jacobian"""
        for m in range(self.kinetics.axial_nodes):
            for i in range(self.kinetics.n_species):
                row = i * self.kinetics.axial_nodes + k
                # Spatial derivative for species i
                J[row, i * self.kinetics.axial_nodes + m] += (
                    self.kinetics.column.porosity * velocity * D1[k, m]
                    - self.kinetics.column.porosity * diffusion[i] * D2[k, m]
                )

    def _set_initial_states(self, C0, q_init):
        """Set initial state for C and q."""
        return C0.ravel()

    def _initial_particle_concentration(self, cp_init, q_init):
        """Set initial particle concentrations."""
        p_var_init = np.asarray(cp_init, dtype=float)
        return p_var_init

    def _model_outs(self, y_out):
        """Return output for C and q."""
        C_out = y_out
        q_out = np.empty_like(y_out)
        for k in range(len(y_out)):
            q_out[k, :, :] = self.kinetics.isotherm.q(C_out[k, :, :])
        return C_out, q_out

    def _compute_particle(self, p_var_out):
        """Compute output particle concentrations."""
        Cp_out = p_var_out
        q_out = np.empty_like(Cp_out)
        for t in range(len(Cp_out)):
            for k in range(self.kinetics.axial_nodes):
                q_out[t, :, k, :] = self.kinetics.isotherm.q(Cp_out[t, :, k, :])
        return Cp_out, q_out


class SolidPhaseSolve(SolutionVariable):
    """Methods to solve for solid phase concentration."""

    def _split_state(self, y):
        """Split state for liquid or solid phase solve."""
        C = np.empty_like(y)
        C[:, 0] = y[:, 0]
        q = y[:, 1:]
        return C, q

    def _time_derivatives(self, c, q, dcdt, dqdt):
        """Compute time derivative variables."""
        c[:, 1:] = self.kinetics.isotherm.C(q)
        dC_dq = self.kinetics.isotherm.dC_dq(q)
        dcdt[:, 1:] = np.einsum("ijn,jn->in", dC_dq, dqdt)
        return c, q, dcdt, dqdt

    def _psdm_time_derivatives(self, p_var, dp_vardt, k):
        """Compute intraparticle time derivatives."""
        cp = self.kinetics.isotherm.C(p_var[:, k, :])
        q = p_var

        dCp_dq = self.kinetics.isotherm.dC_dq(p_var[:, k, :])

        dqdt = dp_vardt
        dcpdt = np.einsum(
            "ijr,jr->ir",
            dCp_dq,
            dqdt[:, k, :],
        )
        return cp, q, dcpdt, dqdt

    def _particle_concentrations(self, cp, q, k, i):
        """Compute cp and q per species."""
        q_i = q[i, k, :]
        cp_i = cp[i, :]
        return cp_i, q_i

    def _center_state(self, cp_i, q_i):
        """Choose variable for center bc."""
        return q_i

    def _diffusion_terms(self, dcpdt, dqdt, lap_cp, lap_q, k, i):
        """Compute intraparticle diffusion terms."""
        Dp_term = (
            self.kinetics.column.media.particle_porosity * dcpdt[i, 1:-1]
            - self.kinetics.column.media.particle_porosity
            * self.kinetics.pore_diffusion[i]
            * lap_cp[1:-1]
        )

        Ds_term = (
            self.kinetics.column.media.particle_density * dqdt[i, k, 1:-1]
            - self.kinetics.column.media.particle_density
            * self.kinetics.surface_diffusion[i]
            * lap_q[1:-1]
        )
        return Dp_term, Ds_term

    def _jacobian_concentrations(self, p_var, dp_vardt, k):
        """Compute q or Cp for jacobian."""
        q_k = p_var[:, k, :]
        dCpdt_k = None
        dqdt_k = dp_vardt[:, k, :]
        dqdCp = None
        dCpdq = self.kinetics.isotherm.dC_dq(q_k)
        d2qdCp2 = None
        d2Cpdq2 = self.kinetics.isotherm.d2C_dq2(q_k)
        return dCpdt_k, dqdt_k, dqdCp, dCpdq, d2qdCp2, d2Cpdq2

    def _interior_jacobian(self, c, q, cj):
        """Return the interior jacobian"""
        # dC/dq
        dC_dq = self.kinetics.isotherm.dC_dq(q)

        # Interior equations
        interior_J = (
            cj * self.kinetics.column.porosity * dC_dq
            + cj
            * self.kinetics.column.media.bed_density
            * np.eye(self.kinetics.n_species)[:, :, None]
        )
        return interior_J

    def _particle_jacobian(self, p_var, dp_vardt, J, cj, L, Gsurf, i, k) -> None:
        """Jacobian loop for particle interior."""
        _, dqdt_k, _, dCpdq, _, d2Cpdq2 = self._jacobian_concentrations(
            p_var, dp_vardt, k
        )
        S = self.kinetics.n_species
        N = self.kinetics.axial_nodes
        R = self.kinetics.radial_nodes
        species_size = N + N * R
        species_offset = i * species_size
        particle_offset = species_offset + N + k * R
        for r in range(1, R - 1):
            row = particle_offset + r
            surface = particle_offset + R - 1

            # -epsilon_p * Dp_i * L(Cp_i)
            for j in range(S):
                j_species_offset = j * species_size
                j_q_offset = j_species_offset + N + k * R

                J[
                    row,
                    j_q_offset : j_q_offset + R,
                ] += (
                    -self.kinetics.column.media.particle_porosity
                    * self.kinetics.pore_diffusion[i]
                    * (L @ np.diag(dCpdq[i, j, :]))
                )[r, :]
            # Surface diffusion term
            # -rho_p * Ds_i * L(q_i)
            J[
                row,
                particle_offset : particle_offset + R,
            ] += (
                -self.kinetics.column.media.particle_density
                * self.kinetics.surface_diffusion[i]
                * L[r, :]
            )
            # Time derivative terms
            # epsilon_p * dCp_i/dt + rho_p * dq_i/dt
            # dCp_i/dt = sum_j dCp_i/dq_j * dq_j/dt
            for j in range(S):
                j_species_offset = j * species_size
                j_q_offset = j_species_offset + N + k * R

                # cj * epsilon_p * dCp_i/dq_j
                J[row, j_q_offset + r] += (
                    cj * self.kinetics.column.media.particle_porosity * dCpdq[i, j, r]
                )

                # Derivative of dCp_i/dt with respect to q_j:
                # sum_m [d2Cp_i/(dq_j dq_m) * dq_m/dt]
                J[
                    row, j_q_offset + r
                ] += self.kinetics.column.media.particle_porosity * np.sum(
                    d2Cpdq2[i, j, :, r] * dqdt_k[:, r]
                )

            # cj * rho_p * dq_i/dt
            J[row, particle_offset + r] += (
                cj * self.kinetics.column.media.particle_density
            )
            # ===============================================================

        # ----------------------------------------------------------
        # Particle surface boundary condition
        # ----------------------------------------------------------

        # Diffusive pore flux:
        #
        # epsilon_p * Dp_i * dCp_i/dr
        #
        # Cp_i depends on all q_j.
        for j in range(S):
            j_species_offset = j * species_size
            j_q_offset = j_species_offset + N + k * R

            J[
                surface,
                j_q_offset : j_q_offset + R,
            ] += (
                self.kinetics.column.media.particle_porosity
                * self.kinetics.pore_diffusion[i]
                * (Gsurf @ np.diag(dCpdq[i, j, :]))
            )

        # Surface diffusion flux:
        #
        # rho_p * Ds_i * dq_i/dr
        J[
            surface,
            particle_offset : particle_offset + R,
        ] += (
            self.kinetics.column.media.particle_density
            * self.kinetics.surface_diffusion[i]
            * Gsurf
        )

        # ----------------------------------------------------------
        # Film flux
        #
        # film_flux_i =
        #     kf_i * (C_i,k - Cp_i,surface)
        #
        # residual:
        #     diffusive_flux - film_flux
        # ----------------------------------------------------------

        C_col = species_offset + k

        # -kf_i * C_i,k
        J[surface, C_col] -= self.kinetics.k_film[i]

        # +kf_i * Cp_i,surface
        #
        # Cp_i,surface depends on every q_j,surface.
        for j in range(S):
            j_species_offset = j * species_size
            j_q_offset = j_species_offset + N + k * R

            J[
                surface,
                j_q_offset + R - 1,
            ] += (
                self.kinetics.k_film[i] * dCpdq[i, j, R - 1]
            )

    def _particle_sink_term(self, p_var, i, k, R):
        """Compute jacobian for sink term."""
        dCpdq = self.kinetics.isotherm.dC_dq(p_var[:, k, :])
        return dCpdq[i, :, R - 1]

    def _transport_jacobian(self, J, k, D1, D2, velocity, diffusion) -> None:
        """Return the transport jacobian"""
        pass

    def _set_initial_states(self, C0, q_init):
        """Set initial state for C and q."""
        q0 = np.broadcast_to(
            q_init[:, None], (self.kinetics.n_species, self.kinetics.axial_nodes - 1)
        ).copy()
        y0 = np.concatenate([C0[:, 0:1], q0], axis=1).ravel()
        return y0

    def _initial_particle_concentration(self, cp_init, q_init):
        """Set initial particle concentrations."""
        p_var_init = np.asarray(q_init, dtype=float)
        return p_var_init

    def _model_outs(self, y_out):
        """Return output for C and q."""
        # State:
        # y[:, :, 0] = inlet C
        # y[:, :, 1:] = q
        C_out = np.empty_like(y_out)
        C_out[:, :, 0] = y_out[:, :, 0]
        for k in range(len(y_out)):
            C_out[k, :, 1:] = self.kinetics.isotherm.C(y_out[k, :, 1:])
        q_out = y_out[:, :, 1:]
        return C_out, q_out

    def _compute_particle(self, p_var_out):
        """Compute output particle concentrations."""
        q_out = p_var_out
        Cp_out = np.empty_like(q_out)
        for t in range(len(q_out)):
            for k in range(self.kinetics.axial_nodes):
                Cp_out[t, :, k, :] = self.kinetics.isotherm.C(q_out[t, :, k, :])
        return Cp_out, q_out


class DynamicAdsorptionKinetics(AdsorptionKinetics):
    """Methods for non-local equilibrium assumption."""

    def __init__(self, rate_constant: float):
        if rate_constant <= 0:
            raise ValueError("Rate constant must be greater than zero.")
        self.rate_constant = rate_constant

    def _n_vars(self):
        """Total length of the IDA state vector."""
        return 2 * self.axial_nodes * self.n_species

    def _split(self, y):
        """Return (C, q)."""
        y = y.reshape(self.n_species, 2, self.axial_nodes)
        C = y[:, 0, :]
        q = y[:, 1, :]
        return C, q

    def _reshape_result(self, result):
        """Set the shape of residual matrix."""
        result = result.reshape(self.n_species, 2, self.axial_nodes)
        return result

    def _state_variables(self, y, ydot):
        """Compute time derivative variables."""
        c, q = self._split(y)
        dcdt, dqdt = self._split(ydot)
        return c, q, dcdt, dqdt

    def _inlet_bc_loop(self, c, result) -> None:
        """Set inlet boundary condition for each species."""
        gradient = self.column_numerics.collocation.evaluate_gradient(c)
        for i, bc in enumerate(self.inlet_bc):
            result[i, 0, 0] = bc.residual(c[i, 0], gradient[i, 0])

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid and solid phase residuals."""
        # liquid phase
        result[:, 0, 1:] = transport + self.column.media.bed_density * dqdt[:, 1:]
        # solid phase
        result[:, 1, :] = dqdt - self._kinetic_expression(c, q)

    def _jacobian_kinetics(self, c, q, J, cj, D1, D2, diffusion, velocity) -> None:
        """Return liquid phase jacobian."""
        # Inlet boundary conditions
        for i in range(self.n_species):
            row = i * 2 * self.axial_nodes

            bc_row = self.inlet_bc[i].jacobian_row(D1[0, :])
            col_start = i * 2 * self.axial_nodes
            J[row, col_start : col_start + self.axial_nodes] = bc_row

        # Interior liquid-phase equations
        for i in range(self.n_species):
            c_start = i * 2 * self.axial_nodes
            q_start = c_start + self.axial_nodes

            for k in range(1, self.axial_nodes):
                row = c_start + k

                # Spatial transport: dF_C / dC
                J[row, c_start : c_start + self.axial_nodes] += (
                    self.column.porosity * velocity * D1[k, :]
                    - self.column.porosity * diffusion[i] * D2[k, :]
                )

                # Accumulation: cj * eps * dC/dt
                J[row, c_start + k] += cj * self.column.porosity

                # Adsorption: cj * rho_b * dq/dt
                J[row, q_start + k] += cj * self.column.media.bed_density

        # Solid-phase kinetic equations
        for i in range(self.n_species):
            for k in range(self.axial_nodes):
                row_c = i * 2 * self.axial_nodes + k
                row_q = row_c + self.axial_nodes

                self._solid_phase(c, q, J, cj, i, k, row_q)

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic."""
        return [2 * i * self.axial_nodes for i in range(self.n_species)]

    def _set_initial_conditions(self, C0, q_init):
        """Return y0 consistent with the algebraic constraint."""
        q0 = np.broadcast_to(
            q_init[:, None],
            (self.n_species, self.axial_nodes),
        ).copy()
        y0 = np.concatenate([C0, q0], axis=1).ravel()
        return y0

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        y_out = y_out.reshape(
            len(y_out),
            self.n_species,
            2,
            self.axial_nodes,
        )
        # Multi-species linear isotherm
        C_out = y_out[:, :, 0, :]
        q_out = y_out[:, :, 1, :]
        return C_out, q_out

    def _kinetic_expression(self, c, q):
        raise NotImplementedError

    def _solid_phase(self, c, q, J, cj, i, k, row_q) -> None:
        raise NotImplementedError


class LinearDrivingForce(DynamicAdsorptionKinetics):
    """Methods for linear driving force assumption:

    dq/dt = rate_constant * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q):
        return self.rate_constant * (self.isotherm.q(c) - q)

    def _solid_phase(self, c, q, J, cj, i, k, row_q) -> None:
        dq_dC = self.isotherm.dq_dC(c)

        q_start = i * 2 * self.axial_nodes + self.axial_nodes

        # dF_q_i / dC_j
        for j in range(self.n_species):
            c_start_j = j * 2 * self.axial_nodes

            J[row_q, c_start_j + k] = -self.rate_constant * dq_dC[i, j, k]

        # dF_q_i / dq_i
        J[row_q, q_start + k] = self.rate_constant + cj


class SecondOrder(DynamicAdsorptionKinetics):
    """Methods for second order assumption:

    dq/dt = rate_constant * C * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q):
        return self.rate_constant * c * (self.isotherm.q(c) - q)

    def _solid_phase(self, c, q, J, cj, i, k, row_q) -> None:
        q_eq = self.isotherm.q(c)
        dq_dC = self.isotherm.dq_dC(c)

        C = c[i, k]
        q_i = q[i, k]

        c_start = i * 2 * self.axial_nodes
        q_start = c_start + self.axial_nodes

        # dF_q_i / dC_j from q_eq
        for j in range(self.n_species):
            c_start_j = j * 2 * self.axial_nodes

            J[row_q, c_start_j + k] = -self.rate_constant * C * dq_dC[i, j, k]

        # dF_q_i / dC_i from explicit C_i
        J[row_q, c_start + k] += -self.rate_constant * (q_eq[i, k] - q_i)

        # dF_q_i / dq_i
        J[row_q, q_start + k] = self.rate_constant * C + cj
