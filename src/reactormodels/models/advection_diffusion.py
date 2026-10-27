"""advection_diffusion.py"""

from typing import Type
import numpy as np
from collections.abc import Sequence
from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .boundary_conditions import BoundaryConditions, DirichletBC


class AdvectionDiffusion(NumericModel):
    """Advection-diffusion model.

    Governing equation:
        dC/dt + v * dC/dx - D * d2C/dx2 = 0

    """

    _param_names = (
        "velocity",
        "axial_diffusion",
    )

    def __init__(
        self,
        breakthrough: Breakthrough | Sequence[Breakthrough],
        numerics: NumericsConfig,
        inlet_bc: Type[InletBC] = DirichletBC,
        t_eval: np.ndarray | None = None,
    ):
        # Normalize breakthroughs to a list
        if isinstance(breakthrough, Breakthrough):
            self.breakthroughs = [breakthrough]
        else:
            self.breakthroughs = list(breakthrough)

        if not self.breakthroughs:
            raise ValueError("At least one breakthrough must be provided.")
        if not all(isinstance(b, Breakthrough) for b in self.breakthroughs):
            raise TypeError("breakthrough must be one or more Breakthroughs.")

        # number of species
        self.n_species = len(self.breakthroughs)

        # Physical parameters
        self.column = self.breakthroughs[0].column
        if not all(bt.column is self.column for bt in self.breakthroughs):
            raise ValueError("All breakthroughs must share the same Column object.")

        self.velocity = self.breakthroughs[0].interstitial_velocity
        if not all(
            np.isclose(bt.interstitial_velocity, self.velocity)
            for bt in self.breakthroughs
        ):
            raise ValueError(
                "All breakthroughs must have the same interstitial velocity."
            )

        # Species-specific parameters
        self.axial_diffusion = np.array(
            [bt.chemical.axial_diffusion for bt in self.breakthroughs]
        )
        if not np.any(self.axial_diffusion):
            raise ValueError("Axial diffusion is required for each species.")

        self.initial_concentration = np.array(
            [bt.initial_concentration for bt in self.breakthroughs]
        )
        self.inlet_concentration = np.array(
            [bt.mean_feed_concentration() for bt in self.breakthroughs]
        )

        # Normalize inlet BCs to a list
        if isinstance(inlet_bc, type):
            self.inlet_bcs = [inlet_bc] * len(self.breakthroughs)
        else:
            self.inlet_bcs = list(inlet_bc)

        if len(self.inlet_bcs) != len(self.breakthroughs):
            raise ValueError(
                "The number of inlet boundary conditions must match "
                "the number of breakthroughs."
            )
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

        # Numerics
        self.numerics = numerics

        # Discretization
        self.axial_nodes = len(self.numerics.collocation.nodes)

        # Set where t is output
        self.t_eval = (
            self._get_default_t_eval() if t_eval is None else np.asarray(t_eval)
        )

        self.assert_parameters_set()

    def _n_vars(self) -> int:
        """Total number of unknowns"""
        return self.axial_nodes * self.n_species

    def _reshape(self, y):
        """Reshape vectors for multi-species."""
        y = y.reshape(self.n_species, self.axial_nodes)
        return y

    def _residual(self, t, c, cdot, result):
        """IDA residual callback.  Writes into `result` in-place."""
        # reshape C and result
        c = self._reshape(c)
        dcdt = self._reshape(cdot)
        result = self._reshape(result)

        # Inlet boundary condition
        G = self.numerics.collocation.evaluate_gradient(c)
        for i, bc in enumerate(self.inlet_bc):
            result[i, 0] = bc.residual(c[i, 0], G[i, 0])

        # Interior nodes and outlet
        result[:, 1:] = (
            dcdt[:, 1:]
            + self.velocity * self.numerics.evaluate_gradient(c)[:, 1:]
            - self.axial_diffusion[:, None]
            * self.numerics.evaluate_second_derivative(c)[:, 1:]
        )

        return 0

    def _jacobian(self, t, C, Cdot, result, cj, jac):
        """Build jacobian of _residual."""
        n_vars = self._n_vars()
        J = np.zeros((n_vars, n_vars))

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
                self.velocity * self.numerics.collocation.first_derivative
                - self.axial_diffusion[i] * self.numerics.collocation.second_derivative
            )
            J[interior_nodes, species_nodes] = d_transport[1:, :]

            diagonal = np.arange(inlet_node + 1, outlet_node)
            J[diagonal, diagonal] += cj

        jac[:, :] = J

        return 0

    def _initial_conditions(self):
        """Set the initial concentration and dcdt."""
        c_init = np.asarray(self.initial_concentration, dtype=float)

        c_0 = np.broadcast_to(
            c_init[:, None],
            (self.n_species, self.axial_nodes),
        ).copy()

        first_derivative_matrix = self.numerics.collocation.first_derivative[0, :]
        for i, bc in enumerate(self.inlet_bc):
            c_0[i, 0] = bc.apply(c_0[i, :], first_derivative_matrix)

        c_0 = c_0.ravel()
        dcdt_0 = np.zeros_like(c_0)
        return c_0, dcdt_0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Only the inlet boundary condition for each species.
        """
        return [i * self.axial_nodes for i in range(self.n_species)]

    def solve(self):
        """Integrate from t_span[0] to t_span[1], returning results at t_eval."""
        c, dcdt = self._initial_conditions()

        result = self.numerics.integrate(
            residual=self._residual,
            jacobian=self._jacobian,
            y0=c,
            yp0=dcdt,
            t_span=[0, self.t_eval[-1]],
            t_eval=self.t_eval,
            algebraic_vars_idx=self._algebraic_vars_idx(),
        )
        if result.flag < 0:
            raise RuntimeError(
                f"IDA solver failed with flag {result.flag}: {result.message}"
            )

        # (n_times, n_species, axial_nodes), skip the t=t_span[0] row
        c_out = result.values.y[1:]
        c_out = c_out.reshape(len(c_out), self.n_species, self.axial_nodes)

        return self.numerics.collocation.nodes, c_out
