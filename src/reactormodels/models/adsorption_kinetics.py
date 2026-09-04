"""adsorption_kinetics.py"""

import numpy as np


class AdsorptionKinetics:
    """Set the form of the adsorption kinetics."""

    def bind(self, column, breakthrough, numerics, isotherm):
        """Initialize convenient classes/quantities."""
        self.column = column
        self.breakthrough = breakthrough
        self.numerics = numerics
        self.isotherm = isotherm
        self.N = len(numerics.collocation.nodes)
        self.inlet_concentration = breakthrough.mean_feed_concentration()

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

    def _y0(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        raise NotImplementedError

    def _solve_kinetics(self, y_out):
        """Return C_out and q_out."""
        raise NotImplementedError


class LocalEquilibrium(AdsorptionKinetics):
    """Methods for local equilibrium assumption:

    dq/dt = dq/dC * dC/dt

    """

    def _n_vars(self):
        """Total length of the IDA state vector."""
        return self.N

    def _split(self, y):
        """Return (C, q)."""
        C = y[: self.N]
        q = None
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid phase residual."""
        result[1 : self.N] = (
            transport
            + self.column.media.bed_density * (self.isotherm.dq_dC(c) * dcdt)[1:]
        )

    def _jacobian_kinetics(self, C, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        for i in range(1, self.N):
            J[i, i] += cj * (
                self.column.porosity
                + self.column.media.bed_density * self.isotherm.dq_dC(C[i])
            )

    def _y0(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        return C0.copy()

    def _solve_kinetics(self, y_out):
        """Return C_out and q_out."""
        C_out = y_out[:, : self.N]
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
        return 2 * self.N

    def _split(self, y):
        """Return (C, q)."""
        C = y[: self.N]
        q = y[self.N :]
        return C, q

    def _residual_kinetics(self, result, c, q, dcdt, dqdt, transport) -> None:
        """Return liquid and solid phase residuals."""
        # liquid phase
        result[1 : self.N] = transport + self.column.media.bed_density * dqdt[1:]
        # solid phase
        result[self.N :] = dqdt - self._kinetic_expression(c, q)

    def _jacobian_kinetics(self, C, q, J, cj) -> None:
        """Return liquid phase jacobian."""
        # liquid phase
        for i in range(1, self.N):
            J[i, i] += cj * self.column.porosity
            J[i, self.N + i] += cj * self.column.media.bed_density
        # solid phase
        self._solid_phase(C, q, J, cj)

    def _y0(self, C0):
        """Return y0 consistent with the algebraic constraint."""
        q0 = np.full(self.N, self.breakthrough.initial_mass_fraction)
        q0[0] = self.isotherm.q(self.inlet_concentration)
        return np.concatenate([C0, q0])

    def _solve_kinetics(self, y_out):
        """Return C_out and q_out."""
        C_out = y_out[:, : self.N]
        q_out = y_out[:, self.N :]
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
        for i in range(self.N):
            J[self.N + i, i] = -self.rate_constant * self.isotherm.dq_dC(C[i])
            J[self.N + i, self.N + i] = self.rate_constant + cj


class SecondOrder(DynamicAdsorptionKinetics):
    """Methods for second order assumption:

    dq/dt = rate_constant * C * (q_e - q)

    """

    def __init__(self, rate_constant: float):
        super().__init__(rate_constant)

    def _kinetic_expression(self, c, q):
        return self.rate_constant * c * (self.isotherm.q(c) - q)

    def _solid_phase(self, C, q, J, cj) -> None:
        for i in range(self.N):
            J[self.N + i, i] = (
                -self.rate_constant
                * (self.isotherm.q(C[i]) + C[i] * self.isotherm.dq_dC(C[i]))
                + self.rate_constant * q[i]
            )
            J[self.N + i, self.N + i] = self.rate_constant * C[i] + cj
