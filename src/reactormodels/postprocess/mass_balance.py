"""mass_balance.py"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from ..models.numeric_model_base import NumericModel

__all__ = ["MassBalance"]


@dataclass
class MassBalance:
    """Compute and verify mass balance for a 1D adsorption column, over time.

        mass_in - mass_out = mass_fluid + mass_adsorbed

    mass_in  = v * eps * C_in * t
    mass_out = v * eps * integral_0^t C(L, t') dt'
    """

    model: NumericModel
    liquid_concentration: np.ndarray  # shape (n_t, n_x) — column concentration history
    sorbent_mass_fraction: np.ndarray  # shape (n_t, n_x) — adsorbed-phase history
    pore_concentration: np.ndarray | None = None  # shape (n_t, n_x_particle)

    def __post_init__(self) -> None:
        """Determine if model simulates pore processes.

        If true perform radial averages.
        """
        self.sorbent_mass_fraction = self._as_radially_averaged(
            self.sorbent_mass_fraction, "sorbent_mass_fraction"
        )
        if self.pore_concentration is not None:
            self.pore_concentration = self._as_radially_averaged(
                self.pore_concentration, "pore_concentration"
            )

    def _as_radially_averaged(self, array: np.ndarray, name: str) -> np.ndarray:
        """Collapse a (n_t, n_x, n_r) radially-resolved field to (n_t, n_x).

        Some models track a within-particle radial profile, so
        `sorbent_mass_fraction`/`pore_concentration` may arrive as 3D
        arrays with a trailing radial axis. MassBalance only operates on
        column-averaged quantities, so a 3D array is radially averaged
        via `model.get_radial_average` automatically; a 2D array is
        assumed already averaged and used as-is.
        """
        if array.ndim == 3:
            if not hasattr(self.model, "get_radial_average"):
                raise ValueError(
                    f"{name} has shape {array.shape} (3D, radially-resolved) "
                    f"but {type(self.model).__name__} has no get_radial_average "
                    "method to collapse it."
                )
            array = self.model.get_radial_average(array)
        if array.ndim != 2:
            raise ValueError(
                f"{name} must be 2D (n_t, n_x) or 3D (n_t, n_x, n_r); "
                f"got shape {array.shape}"
            )
        if array.shape[0] != self.liquid_concentration.shape[0]:
            raise ValueError(
                f"{name} has {array.shape[0]} time points but "
                f"liquid_concentration has {self.liquid_concentration.shape[0]}"
            )
        return array

    @property
    def time(self):
        """The time of input data."""
        return self.model.breakthrough.time

    @property
    def breakthrough(self):
        """The Breakthrough backing `model`."""
        return self.model.breakthrough

    @property
    def velocity(self) -> float:
        """Interstitial velocity"""
        return self.breakthrough.interstitial_velocity

    @property
    def porosity(self) -> float:
        """Column porosity"""
        return self.breakthrough.column.porosity

    @property
    def particle_porosity(self) -> float | None:
        """Particle porosity"""
        return self.breakthrough.column.media.particle_porosity

    @property
    def bulk_density(self) -> float:
        """Bulk density"""
        return self.breakthrough.column.get_bulk_density()

    @property
    def cross_section_area(self) -> float:
        """Cross-section area"""
        return self.breakthrough.column.cross_section_area()

    @property
    def C_in(self) -> float:
        """Inlet concentration"""
        return self.breakthrough.mean_feed_concentration()

    @property
    def column_numerics(self):
        """Spatial-discretization numerics for the column domain.

        `DomainCoupling`-style models expose `column_numerics` (distinct
        from `particle_numerics`); single-domain models like
        `AdvectionDiffusionAdsorption` just expose `numerics`. Prefer the
        former when present.
        """
        return getattr(self.model, "column_numerics", None) or self.model.numerics

    @property
    def C_outlet(self) -> np.ndarray:
        """Outlet concentration C(x=L) at every time, shape (n_t,)."""
        return self.liquid_concentration[:, -1]

    @property
    def mass_in(self) -> np.ndarray:
        """Mass entered at every time: v * eps * C_in * t * A, shape (n_t,)."""
        return (
            self.velocity
            * self.porosity
            * self.C_in
            * self.time
            * self.cross_section_area
        )

    @property
    def mass_out(self) -> np.ndarray:
        """Mass exited via outlet at every time, shape (n_t,).

        Computes the cumulative mass exiting through the outlet:

            v * eps * A * integral_0^t C(L, t') dt'

        using a cumulative trapezoidal integral.
        """
        concentration = np.asarray(self.C_outlet, dtype=float)
        time = np.asarray(self.time, dtype=float)

        integral = integrate.cumulative_trapezoid(
            concentration,
            x=time,
            initial=0,
        )

        return self.velocity * self.porosity * self.cross_section_area * integral

    def _spatial_integrate_history(self, history: np.ndarray) -> np.ndarray:
        """Apply column_numerics.spatial_integrate to each time row.

        `history` has shape (n_t, n_x); returns shape (n_t,).
        """
        return np.apply_along_axis(self.column_numerics.spatial_integrate, 1, history)

    @property
    def mass_fluid(self) -> np.ndarray:
        """Mass in the fluid phase at every time, shape (n_t,).

        Column pore water, plus particle pore water if tracked.
        """
        mass = (
            self.porosity
            * self.cross_section_area
            * self._spatial_integrate_history(self.liquid_concentration)
        )
        if self.particle_porosity is not None and self.pore_concentration is not None:
            mass = mass + (
                (1.0 - self.porosity)
                * self.particle_porosity
                * self.cross_section_area
                * self._spatial_integrate_history(self.pore_concentration)
            )
        return mass

    @property
    def mass_adsorbed(self) -> np.ndarray:
        """Mass in the solid phase at every time, shape (n_t,)."""
        return (
            self.bulk_density
            * self.cross_section_area
            * self._spatial_integrate_history(self.sorbent_mass_fraction)
        )

    @property
    def mass_stored(self) -> np.ndarray:
        """Total mass in fluid and solid phases, shape (n_t,)."""
        return self.mass_fluid + self.mass_adsorbed

    @property
    def error(self) -> np.ndarray:
        """Absolute error: (mass_in - mass_out) - mass_stored, shape (n_t,)."""
        return (self.mass_in - self.mass_out) - self.mass_stored

    @property
    def relative_error(self) -> np.ndarray:
        """Relative mass balance error at every time, shape (n_t,).

        Entries where net mass in is ~0 (e.g. t=0) are set to inf.
        """
        net = self.mass_in - self.mass_out
        return np.where(np.abs(net) < 1e-30, np.inf, np.abs(self.error) / np.abs(net))

    def is_balanced(self, rel_tol: float = 0.05) -> np.ndarray:
        """Element-wise mass balance check within specified tolerance, shape (n_t,).

        Use `.all()` on the result to check that balance holds at every time.
        """
        return self.relative_error < rel_tol

    def summary(self, index: int = -1) -> str:
        """Provide a mass balance summary at a single time point.

        Defaults to the last time in the history; pass an index to
        inspect an earlier point.
        """
        t = self.time[index]
        mass_in = self.mass_in[index]
        mass_out = self.mass_out[index]
        mass_fluid = self.mass_fluid[index]
        mass_adsorbed = self.mass_adsorbed[index]
        mass_stored = self.mass_stored[index]
        error = self.error[index]
        relative_error = self.relative_error[index]
        balanced = self.is_balanced()[index]
        return (
            f"MassBalance at t={t:.2f}s\n"
            f"  mass_in       = {mass_in:.4f}\n"
            f"  mass_out      = {mass_out:.4f}\n"
            f"  net_in        = {mass_in - mass_out:.4f}\n"
            f"  mass_fluid    = {mass_fluid:.4f}\n"
            f"  mass_adsorbed = {mass_adsorbed:.4f}\n"
            f"  mass_stored   = {mass_stored:.4f}\n"
            f"  abs error     = {error:.4f}\n"
            f"  rel error     = {relative_error:.2%}\n"
            f"  balanced      = {balanced}"
        )
