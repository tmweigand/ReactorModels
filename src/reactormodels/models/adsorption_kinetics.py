"""adsorption_kinetics.py"""

import numpy as np
from enum import Enum, auto

from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm


class SolutionVariable(Enum):
    """State dependent variable"""

    C = auto()
    Q = auto()


def _is_implemented(method):
    try:
        method(1.0)  # dummy scalar probe
        return True
    except NotImplementedError:
        return False


class AdsorptionKinetics:
    """Set the form of the adsorption kinetics."""

    def __init__(self):
        self.column = None
        self.breakthrough = None
        self.isotherm = None
        self.n_nodes = None
        self.inlet_concentration = None
        self.solution_variable = None

    def configure(self, breakthrough, isotherm, n_nodes, inlet_concentration):
        """Initialize convenient classes/quantities."""
        self.column = breakthrough.column
        self.breakthrough = breakthrough
        self.isotherm: Isotherm | MultiSpeciesIsotherm = isotherm
        self.n_nodes = n_nodes
        self.inlet_concentration = inlet_concentration

        if (
            _is_implemented(self.isotherm.q)
            and _is_implemented(self.isotherm.dq_dC)
            and _is_implemented(self.isotherm.d2q_dC2)
        ):
            self.solution_variable = SolutionVariable.C
        elif (
            _is_implemented(self.isotherm.C)
            and _is_implemented(self.isotherm.dC_dq)
            and _is_implemented(self.isotherm.d2C_dq2)
        ):
            self.solution_variable = SolutionVariable.Q

    def _n_vars(self):
        """Total length of the IDA state vector."""
        raise NotImplementedError

    def _split(self, y):
        """Return (C, q)."""
        raise NotImplementedError

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
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
        return self.n_nodes

    def _split(self, y):
        """Return (C, q)."""
        C = y[: self.n_nodes]
        q = None
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid phase residual."""
        result[1 : self.n_nodes] = (
            transport
            + self.column.media.bed_density * (self.isotherm.dq_dC(c) * dcdt)[1:]
        )

    def _jacobian_kinetics(self, C, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        for i in range(1, self.n_nodes):
            J[i, i] += cj * (
                self.column.porosity
                + self.column.media.bed_density * self.isotherm.dq_dC(C[i])
            )

    def _set_initial_conditions(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        return C0.copy()

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        C_out = y_out[:, : self.n_nodes]
        q_out = np.array(
            [self.isotherm.q(C_out[i]) for i in range(len(self.breakthrough.time))]
        )
        return C_out, q_out


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
