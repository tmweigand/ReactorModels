"""batch_psdm.py"""

from __future__ import annotations
from typing import Type
import numpy as np

from ..properties.breakthrough import Breakthrough
from ..numerics.config import NumericsConfig
from .numeric_model_base import NumericModel
from .isotherm import Isotherm
from .boundary_conditions import BoundaryConditions, SymmetryBC
from ..properties.film_transfer import FilmTransfer

__all__ = ["BatchPSDM"]


class BatchPSDM(NumericModel):
    """1D transport through pore and/or surface diffusion in a spherical particle.

    The governing equations are:

        Fluid phase: eps_p*dCp/dt - 1/(r^2)*d/dr[r^2*eps_p*D_p*dC_p/dr]
                    + rho_p*dq/dt  - 1/(r^2)*d/dr[r^2*rho_s*D_s*dq/dr] = 0

        Solid phase: dq_dt = dq*/dc*dc/dt

    """

    _param_names = (
        "pore_diffusion",
        "surface_diffusion",
        "iso",
        "numerics",
        "center_bc",
    )

    def __init__(
        self,
        breakthrough: Breakthrough,
        isotherm: Isotherm,
        numerics: NumericsConfig,
        k_film: FilmTransfer,
        sorbent_dosage: float,
        center_bc: Type[BoundaryConditions] = SymmetryBC,
    ):
        self.breakthrough = breakthrough
        self.media = breakthrough.column.media
        self.pore_diffusion = breakthrough.chemical.pore_diffusion
        self.surface_diffusion = breakthrough.chemical.surface_diffusion
        self.iso = isotherm
        self.numerics = numerics
        self.N = len(self.numerics.collocation.nodes)
        self.bulk_initial_concentration = breakthrough.mean_feed_concentration()
        self.k_film = k_film.k_film if isinstance(k_film, FilmTransfer) else k_film
        self.sorbent_dosage = sorbent_dosage
        self.center_bc = center_bc(node=0)
        self.assert_parameters_set()

    def _n_vars(self) -> int:
        """Total length of the IDA state vector."""
        return self.N + 1

    def _split(self, y: np.ndarray):
        """Return (C, Cp)."""
        C = y[0]
        Cp = y[1 : self.N + 1]
        return C, Cp

    def _residual(self, t, y, ydot, result):
        """IDA residual F(t, y, ydot) = 0.  Writes into `result` in-place."""
        c, cp = self._split(y)
        dcdt, dcpdt = self._split(ydot)

        # Center: symmetry
        result[1] = self.center_bc.residual(
            first_derivative_row=self.numerics.collocation.evaluate_gradient(cp, 0)
        )

        # Surface: bulk
        result[0] = dcdt + 6 * self.k_film * self.sorbent_dosage * (c - cp[-1]) / (
            self.media.particle_diameter * self.media.particle_density
        )

        # fluid phase - internal and outlet
        q = np.empty_like(cp)
        for r in range(self.N):
            q[r] = self.iso.q(cp[r])
        lap_q = self.numerics.evaluate_radial_operator(q)
        lap_cp = self.numerics.evaluate_radial_operator(cp)
        grad_cp = self.numerics.evaluate_gradient(cp, -1)
        grad_q = self.numerics.evaluate_gradient(q, -1)

        # Surface: particle
        result[-1] = (
            self.k_film * (c - cp[-1])
            - self.media.particle_porosity * self.pore_diffusion * grad_cp
            - self.media.particle_density * self.surface_diffusion * grad_q
        )

        for r in range(2, self.N):
            transport = (
                self.media.particle_porosity * dcpdt[r - 1]
                - self.media.particle_porosity * self.pore_diffusion * lap_cp[r - 1]
            )

            dqdC = self.iso.dq_dC(cp[r - 1])

            result[r] = (
                transport
                + self.media.particle_density * dqdC * dcpdt[r - 1]
                - self.media.particle_density * self.surface_diffusion * lap_q[r - 1]
            )

    def _jacobian(self, t, y, ydot, result, cj, jac):
        _, Cp = self._split(y)
        _, dCpdt = self._split(ydot)
        n = self._n_vars()
        J = np.zeros((n, n))

        # change in bulk c w.r.t. bulk c
        film_coef = (
            6
            * self.k_film
            * self.sorbent_dosage
            / (self.media.particle_diameter * self.media.particle_density)
        )
        J[0, 0] = film_coef + cj

        # change in bulk c w.r.t. surface cp
        J[0, -1] = -film_coef

        # center bc jacobian row
        J[1, 1:] = self.center_bc.jacobian_row(
            self.numerics.collocation.first_derivative[0, :]
        )

        # change in surface cp w.r.t. surface cp
        G = self.numerics.collocation.first_derivative[-1, :]
        for r in range(self.N):
            dqdCp = self.iso.dq_dC(Cp[r])
            J[-1, r + 1] = (
                -self.media.particle_porosity * self.pore_diffusion * G[r]
                - self.media.particle_density * self.surface_diffusion * G[r] * dqdCp
            )
        J[-1, -1] -= self.k_film

        # change in surface cp w.r.t. bulk c
        J[-1, 0] = self.k_film

        # change in interior cp w.r.t. interior cp
        for r in range(1, self.N - 1):
            dqdCp = self.iso.dq_dC(Cp[r])
            d2qdCp2 = self.iso.d2q_dC2(Cp[r])

            L = self.numerics.collocation.radial_operator_matrix

            for k in range(self.N):
                dqdCp_k = self.iso.dq_dC(Cp[k])
                J[1 + r, 1 + k] = (
                    -self.media.particle_porosity * self.pore_diffusion * L[r, k]
                    - self.media.particle_density
                    * self.surface_diffusion
                    * L[r, k]
                    * dqdCp_k
                )
            J[1 + r, 1 + r] += self.media.particle_density * d2qdCp2 * dCpdt[r]
            J[1 + r, 1 + r] += cj * (
                self.media.particle_porosity + self.media.particle_density * dqdCp
            )

        jac[:, :] = J

        return 0

    def _algebraic_vars_idx(self):
        """Create list identifying which equations are algebraic.

        Particle surface and center nodes.
        """
        return [1, self.N]

    def _initial_conditions(self):
        """Return (y0, ydot0) consistent with the algebraic constraint."""
        y0 = np.empty(self.N + 1)
        y0[0] = self.bulk_initial_concentration
        y0[1:] = self.breakthrough.initial_concentration
        ydot0 = np.zeros_like(y0)
        return y0, ydot0

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

        C_out = y_out[:, 0]
        Cp_out = y_out[:, 1:]  # (n_times, N)

        return self.numerics.collocation.nodes, C_out, Cp_out

    def get_solid_phase_loading(
        self, pore_concentration: np.ndarray | float
    ) -> np.ndarray | float:
        """Calculate the solid phase loading q"""
        return self.iso.q(pore_concentration)

    def get_radial_average(self, data_in: np.ndarray) -> np.ndarray | float:
        """Return the volume-weighted average over a spherical particle."""
        assert self.media.particle_radius is not None

        if data_in.ndim != 3:
            raise ValueError(
                f"get_radial_average expects a 3D array, but got "
                f"an array with shape {data_in.shape}."
            )

        numerator = self.numerics.collocation.integrate(
            data_in * self.numerics.collocation.nodes**2, axis=2
        )
        return numerator / (self.media.particle_radius**3 / 3.0)
