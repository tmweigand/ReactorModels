"""solution_variable.py"""

from .adsorption_kinetics import LocalEquilibrium
import numpy as np


class Liquid(LocalEquilibrium):
    """Solve local equilibrium where liquid phase is the solution variable."""

    def _split(self, y):
        """Return (C, q)."""
        C = y.reshape(self.n_species, self.n_nodes)
        q = None
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt) -> None:
        """Return liquid phase residual."""
        for i in range(1, self.n_nodes):
            dqdC = self.isotherm.dq_dC(c[:, i])  # (n_species, n_species)
            dqdt = dqdC @ dcdt[:, i]
            result[:, i] += self.column.bed_density * dqdt

    def _jacobian_kinetics(self, C, q, dCdt, J, cj) -> None:
        """Return liquid phase jacobian."""
        # adsorption contribution to q residual w.r.t. q
        for i in range(1, self.n_nodes):
            interior_nodes = np.arange(self.n_species) * self.n_nodes + i

            dqdC = self.isotherm.dq_dC(C[:, i])
            d2qdC2 = self.isotherm.d2q_dC2(C[:, i])

            for j in range(self.n_species):
                J[interior_nodes[j], interior_nodes] += (
                    self.column.bed_density * (d2qdC2[j] @ dCdt[:, i])
                    + cj * self.column.bed_density * dqdC[j]
                )
                J[interior_nodes[j], interior_nodes[j]] += cj * self.column.porosity

        for i, bc in enumerate(self.inlet_bc):
            # Species inlet rows: algebraic constraint, no Cdot term
            inlet_node = i * self.axial_nodes
            outlet_node = inlet_node + self.axial_nodes
            species_nodes = slice(inlet_node, outlet_node)

            J[inlet_node, species_nodes] = bc.jacobian_row(
                self.numerics.collocation.first_derivative[0, :]
            )
            # Species interior and outlet rows: change in interior residual w.r.t to C
            interior_nodes = slice(inlet_node + 1, outlet_node)
            d_transport = (
                -self.column.porosity
                * self.velocity
                * self.numerics.collocation.first_derivative
                + self.column.porosity
                * self.axial_diffusion[i]
                * self.numerics.collocation.second_derivative
            )
            J[interior_nodes, species_nodes] = d_transport[1:, :]


class Solid(LocalEquilibrium):
    """Solve local equilibrium where solid phase is the solution variable."""

    def _split(self, y):
        """Return (C, q)."""
        y = y.reshape(self.n_species, self.n_nodes)
        q = y[:, 1:]
        C = np.empty_like(y)
        C[:, 0] = y[:, 0]
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt) -> None:
        """Return solid phase residual."""
        for i in range(1, self.n_nodes):
            q_idx = i - 1
            c[:, i] = self.isotherm.C(q[:, q_idx])
            dCdq = self.isotherm.dC_dq(q[:, q_idx])
            dcdt[:, i] = dCdq @ dqdt[:, q_idx]
            result[:, i] = self.column.bed_density * dqdt[:, q_idx]

    def _jacobian_kinetics(self, C, q, dCdt, dqdt, J, cj) -> None:
        """Return solid phase jacobian."""
        dCdq = np.zeros((self.n_species, self.n_species, self.axial_nodes))
        # adsorption contribution to q residual w.r.t. q
        for i in range(1, self.n_nodes):
            interior_nodes = np.arange(self.n_species) * self.n_nodes + i
            q_idx = i - 1

            dCdq[:, :, i] = self.isotherm.dC_dq(q[:, q_idx])
            d2Cdq2 = self.isotherm.d2C_dq2(q[:, q_idx])

            for j in range(self.n_species):
                J[interior_nodes[j], interior_nodes] += (
                    self.column.porosity * (d2Cdq2[j] @ dqdt[:, q_idx])
                    + cj * self.column.porosity * dCdq[j, :, i]
                )
                J[interior_nodes[j], interior_nodes[j]] += cj * self.column.bed_density

        for i, bc in enumerate(self.inlet_bc):
            # Species inlet rows: algebraic constraint, no Cdot term
            inlet_node = i * self.axial_nodes
            outlet_node = inlet_node + self.axial_nodes
            interior_nodes = slice(inlet_node + 1, outlet_node)

            # inlet state variable. bc_row is shape (axial_nodes,)
            bc_row = bc.jacobian_row(self.numerics.collocation.first_derivative[0, :])
            J[inlet_node, inlet_node] = bc_row[0]

            # change in transport w.r.t. q. shape (axial_nodes, axial_nodes)
            d_transport = (
                self.column.porosity
                * self.velocity
                * self.numerics.collocation.first_derivative
                - self.column.porosity
                * self.axial_diffusion[i]
                * self.numerics.collocation.second_derivative
            )
            J[interior_nodes, inlet_node] += d_transport[1:, 0]

            # Species interior and outlet rows: change in interior residual w.r.t to q
            for j in range(self.n_species):
                species_start = j * self.axial_nodes
                species_interior = slice(
                    species_start + 1, species_start + self.axial_nodes
                )
                # change in inlet C w.r.t. q
                J[inlet_node, species_interior] += bc_row[1:] * dCdq[i, j, 1:]

                # change in transport w.r.t. interior q
                J[interior_nodes, species_interior] += (
                    d_transport[1:, 1:] * dCdq[i, j, 1:]
                )
