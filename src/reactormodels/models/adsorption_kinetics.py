"""adsorption_kinetics.py"""

import numpy as np
from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm
from ..properties.breakthrough import Breakthrough
from ..properties.column import Column


def _is_implemented(isotherm, method_name: str) -> bool:
    base_class = (
        MultiSpeciesIsotherm if isinstance(isotherm, MultiSpeciesIsotherm) else Isotherm
    )
    return getattr(type(isotherm), method_name) is not getattr(base_class, method_name)


class AdsorptionKinetics:
    """Set the form of the adsorption kinetics."""

    def __init__(self) -> None:
        self.column: Column
        self.breakthrough: Breakthrough
        self.isotherm: Isotherm | MultiSpeciesIsotherm
        self.n_nodes: int
        self.inlet_concentration: float
        self.n_species = int

    def configure(
        self,
        breakthrough: Breakthrough,
        isotherm: Isotherm | MultiSpeciesIsotherm,
        n_nodes: int,
        inlet_concentration: float,
        n_species,
    ) -> None:
        """Initialize convenient classes/quantities."""
        self.column = breakthrough[0].column
        self.breakthrough: Breakthrough = breakthrough
        self.isotherm: Isotherm | MultiSpeciesIsotherm = isotherm
        self.n_nodes = n_nodes
        self.inlet_concentration = inlet_concentration
        self.n_species = n_species

    def _n_vars(self):
        """Total length of the IDA state vector."""
        raise NotImplementedError

    def _split(self, y):
        """Return (C, q)."""
        raise NotImplementedError

    def _residual_kinetics(self, result, c, q, dcdt, dqdt) -> None:
        """Return liquid phase residual."""
        raise NotImplementedError

    def _jacobian_kinetics(self, C, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        raise NotImplementedError

    def _set_initial_conditions(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        raise NotImplementedError

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        raise NotImplementedError


class LocalEquilibrium(AdsorptionKinetics):
    """Methods for local equilibrium assumption:

    dq/dt = dq/dC * dC/dt

    """

    def _n_vars(self):
        """Total length of the IDA state vector."""
        return self.n_nodes * self.n_species

    @classmethod
    def solution_variable(cls, isotherm):
        from .solution_variable import Liquid, Solid

        if all(
            _is_implemented(isotherm, method) for method in ("q", "dq_dC", "d2q_dC2")
        ):
            return Liquid()

        if all(
            _is_implemented(isotherm, method) for method in ("C", "dC_dq", "d2C_dq2")
        ):
            return Solid()

        raise ValueError("Unsupported isotherm formulation.")


class DynamicAdsorptionKinetics(AdsorptionKinetics):
    """Methods for non-local equilibrium assumption."""

    def __init__(self, rate_constant: float):
        if rate_constant <= 0:
            raise ValueError("Rate constant must be greater than zero.")
        self.rate_constant = rate_constant

    def _n_vars(self):
        """Total length of the IDA state vector."""
        return 2 * self.n_nodes

    def _split(self, y):
        """Return (C, q)."""
        C = y[: self.n_nodes]
        q = y[self.n_nodes :]
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid and solid phase residuals."""
        # liquid phase
        result[1 : self.n_nodes] = transport + self.column.media.bed_density * dqdt[1:]
        # solid phase
        result[self.n_nodes :] = dqdt - self._kinetic_expression(c, q)

    def _jacobian_kinetics(self, C, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        # liquid phase
        for i in range(1, self.n_nodes):
            J[i, i] += cj * self.column.porosity
            J[i, self.n_nodes + i] += cj * self.column.media.bed_density
        # solid phase
        self._solid_phase(C, q, J, cj)

    def _set_initial_conditions(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        q0 = np.full(self.n_nodes, self.breakthrough.initial_mass_fraction)
        q0[0] = self.isotherm.q(self.inlet_concentration)
        return np.concatenate([C0, q0])

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        C_out = y_out[:, : self.n_nodes]
        q_out = y_out[:, self.n_nodes :]
        return C_out, q_out

    def _kinetic_expression(self, c, q):
        raise NotImplementedError

    def _solid_phase(self, C, q, J, cj) -> None:
        raise NotImplementedError


class LinearDrivingForce(DynamicAdsorptionKinetics):
    """Methods for linear driving force assumption:

    dq/dt = rate_constant * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q):
        return self.rate_constant * (self.isotherm.q(c) - q)

    def _solid_phase(self, C, q, J, cj) -> None:
        for i in range(self.n_nodes):
            J[self.n_nodes + i, i] = -self.rate_constant * self.isotherm.dq_dC(C[i])
            J[self.n_nodes + i, self.n_nodes + i] = self.rate_constant + cj


class SecondOrder(DynamicAdsorptionKinetics):
    """Methods for second order assumption:

    dq/dt = rate_constant * C * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q):
        return self.rate_constant * c * (self.isotherm.q(c) - q)

    def _solid_phase(self, C, q, J, cj) -> None:
        for i in range(self.n_nodes):
            J[self.n_nodes + i, i] = (
                -self.rate_constant
                * (self.isotherm.q(C[i]) + C[i] * self.isotherm.dq_dC(C[i]))
                + self.rate_constant * q[i]
            )
            J[self.n_nodes + i, self.n_nodes + i] = self.rate_constant * C[i] + cj
