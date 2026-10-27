"""adsorption_kinetics.py"""

import numpy as np
from enum import Enum, auto

<<<<<<< HEAD
=======
from ..properties.breakthrough import Breakthrough
from .isotherm import Isotherm
from .multi_species_isotherm import MultiSpeciesIsotherm

>>>>>>> 84f9ea2 (start multi-species advection-diffusion-adsorption)

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
        self.n_species = None

    def configure(
        self, breakthrough, isotherm, n_nodes, inlet_concentration, n_species
    ):
        """Initialize convenient classes/quantities."""
        self.column = breakthrough.column
<<<<<<< HEAD
        self.breakthrough = breakthrough
        self.isotherm = isotherm
=======
        self.breakthrough: Breakthrough = breakthrough
        self.isotherm: Isotherm | MultiSpeciesIsotherm = isotherm
>>>>>>> 84f9ea2 (start multi-species advection-diffusion-adsorption)
        self.n_nodes = n_nodes
        self.inlet_concentration = inlet_concentration
        self.n_species = n_species

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

    def _jacobian_kinetics(self, C, dcdt, q, J, cj) -> None:
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

    def _split(self, y):
        """Return (C, q)."""
        C = y.reshape(self.n_species, self.n_nodes)
        q = None
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid phase residual."""
<<<<<<< HEAD
=======
        for i in range(1, self.n_nodes):
            dqdC = self.isotherm.dq_dC(c[:, i])  # (n_species, n_species)
            dqdt = dqdC @ dcdt[:, i]
            result[:, i] += self.column.media.bed_density * dqdt

    def _jacobian_kinetics(self, C, q, J, cj) -> None:
        """Return liquid phase jacobian."""
>>>>>>> 84f9ea2 (start multi-species advection-diffusion-adsorption)
        for i in range(1, self.n_nodes):
            result[i] = transport[i - 1] + self.column.bed_density * (
                self.isotherm.dq_dC(c[i]) * dcdt[i]
            )

    def _jacobian_kinetics(self, C, dcdt, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        # liquid phase residual
        for i in range(1, self.n_nodes):
            # liquid phase derivative
            J[i, i] += self.column.bed_density * self.isotherm.d2q_dC2(C[i]) * dcdt[
                i
            ] + cj * (
                self.column.porosity
                + self.column.bed_density * self.isotherm.dq_dC(C[i])
            )

    def _set_initial_conditions(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        return C0.copy()

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        C_out = y_out[:, : self.n_nodes]
        # calculate solid phase loading from isotherm equation
        q_out = np.empty_like(C_out)
        for t in range(len(self.breakthrough.time)):
            for i in range(self.n_nodes):
                q_out[t, i] = self.isotherm.q(C_out[t, i])
        return C_out, q_out


class DynamicAdsorptionKinetics(AdsorptionKinetics):
    """Methods for non-local equilibrium assumption."""

    def __init__(self, rate_constant: float):
        if rate_constant <= 0:
            raise ValueError("Rate constant must be greater than zero.")
        self.rate_constant = rate_constant

    def _n_vars(self):
        """Total length of the IDA state vector."""
        # liquid and solid phase concentration at each node
        return 2 * self.n_nodes

    def _split(self, y):
        """Return (C, q)."""
        C = y[: self.n_nodes]
        q = y[self.n_nodes :]
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid and solid phase residuals."""
        # liquid phase
        for i in range(1, self.n_nodes):
            result[i] = transport[i - 1] + self.column.bed_density * dqdt[i]
        # solid phase
        for i in range(self.n_nodes):
            result[self.n_nodes + i] = dqdt[i] - self._kinetic_expression(c, q, i)

    def _jacobian_kinetics(self, C, dcdt, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        # liquid phase residual
        for i in range(1, self.n_nodes):
            # liquid phase derivatives
            J[i, i] += cj * self.column.porosity
            # solid phase derivatives
            J[i, self.n_nodes + i] += cj * self.column.bed_density
        # solid phase residual
        self._solid_phase(C, q, J, cj)

    def _set_initial_conditions(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        # set initial solid phase loading at all nodes
        q0 = np.full(self.n_nodes, self.breakthrough.initial_solid_phase_loading)
        # replace inlet with solid phase loading calculated from isotherm
        q0[0] = self.isotherm.q(self.inlet_concentration)
        return np.concatenate([C0, q0])

    def _parse_variables(self, y_out):
        """Return C_out and q_out."""
        C_out = y_out[:, : self.n_nodes]
        q_out = y_out[:, self.n_nodes :]
        return C_out, q_out

    def _kinetic_expression(self, c, q, i):
        raise NotImplementedError

    def _solid_phase(self, C, q, J, cj) -> None:
        raise NotImplementedError


class LinearDrivingForce(DynamicAdsorptionKinetics):
    """Methods for linear driving force assumption:

    dq/dt = rate_constant * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q, i):
        return self.rate_constant * (self.isotherm.q(c[i]) - q[i])

    def _solid_phase(self, C, q, J, cj) -> None:
        # LDF solid phase residual
        for i in range(self.n_nodes):
            # liquid phase derivatives
            J[self.n_nodes + i, i] = -self.rate_constant * self.isotherm.dq_dC(C[i])
            # solid phase derivatives
            J[self.n_nodes + i, self.n_nodes + i] = self.rate_constant + cj


class SecondOrder(DynamicAdsorptionKinetics):
    """Methods for second order assumption:

    dq/dt = rate_constant * C * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q, i):
        return self.rate_constant * c[i] * (self.isotherm.q(c[i]) - q[i])

    def _solid_phase(self, C, q, J, cj) -> None:
        # second-order solid phase residual
        for i in range(self.n_nodes):
            # liquid phase derivatives
            J[self.n_nodes + i, i] = (
                -self.rate_constant
                * (self.isotherm.q(C[i]) + C[i] * self.isotherm.dq_dC(C[i]))
                + self.rate_constant * q[i]
            )
            # solid phase derivatives
            J[self.n_nodes + i, self.n_nodes + i] = self.rate_constant * C[i] + cj
