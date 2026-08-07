"""mass_balance.py"""

from dataclasses import dataclass
from ..properties.breakthrough import Breakthrough
import numpy as np
import scipy


@dataclass
class MassBalance:
    """Compute and verify mass balance for a 1D adsorption column.

        mass_in - mass_out = mass_fluid + mass_adsorbed

    mass_in  = v * eps * C_in * t
    mass_out = v * eps * integral_0^t C(L, t') dt'   ← needs outlet history
    """

    x: np.ndarray
    C: np.ndarray  # concentration profile at current t (N,)
    q: np.ndarray  # solid loading profile at current t (N,)
    t: float
    velocity: float
    porosity: float
    bulk_density: float
    C_in: float
    t_history: np.ndarray  # all times up to current t (n_times,)
    C_outlet_history: np.ndarray  # C(x=L) at each time in t_history (n_times,)

    @property
    def mass_in(self) -> float:
        """Mass entered: v * eps * C_in * t."""
        if self.porosity is not None:
            return self.velocity * self.porosity * self.C_in * self.t

    @property
    def mass_out(self) -> float:
        """Mass exited via outlet: v * eps * integral C(L, t') dt'."""
        if self.porosity is not None:
            return (
                self.velocity
                * self.porosity
                * scipy.integrate.trapezoid(self.C_outlet_history, self.t_history)
            )

    @property
    def mass_fluid(self) -> float:
        """Mass in the fluid phase"""
        if self.porosity is not None:
            return self.porosity * scipy.integrate.trapezoid(self.C, self.x)

    @property
    def mass_adsorbed(self) -> float:
        """Mass in the solid phase."""
        return self.bulk_density * scipy.integrate.trapezoid(self.q, self.x)

    @property
    def mass_stored(self) -> float:
        """Total mass in fluid and solid phases."""
        return self.mass_fluid + self.mass_adsorbed

    @property
    def error(self) -> float:
        """Absolute error: (mass_in - mass_out) - mass_stored."""
        return (self.mass_in - self.mass_out) - self.mass_stored

    @property
    def relative_error(self) -> float:
        """Relative mass balance error."""
        net = self.mass_in - self.mass_out
        if abs(net) < 1e-30:
            return float("inf")
        return abs(self.error) / net

    def is_balanced(self, rel_tol=0.05) -> bool:
        """Check for mass balance within specified tolerance."""
        return self.relative_error < rel_tol

    def summary(self) -> str:
        """Provide mass balance summary."""
        return (
            f"MassBalance at t={self.t:.2f}s\n"
            f"  mass_in       = {self.mass_in:.4f}\n"
            f"  mass_out      = {self.mass_out:.4f}\n"
            f"  net_in        = {self.mass_in - self.mass_out:.4f}\n"
            f"  mass_fluid    = {self.mass_fluid:.4f}\n"
            f"  mass_adsorbed = {self.mass_adsorbed:.4f}\n"
            f"  mass_stored   = {self.mass_stored:.4f}\n"
            f"  abs error     = {self.error:.4f}\n"
            f"  rel error     = {self.relative_error:.2%}\n"
            f"  balanced      = {self.is_balanced()}"
        )

    @classmethod
    def from_solution(
        cls,
        breakthrough: Breakthrough,
        x: np.ndarray,
        t_eval: np.ndarray,
        C_history: np.ndarray,  # (n_times, N)
        q_history: np.ndarray,  # (n_times, N)
    ) -> list["MassBalance"]:
        """Build a MassBalance for every time in t_eval.

        Outlet concentration C(x=L) is the last spatial node at each time.
        """
        C_outlet_history = C_history[:, -1]  # C at x=L for all times

        return [
            cls(
                x=x,
                C=C_history[i],
                q=q_history[i],
                t=t_eval[i],
                velocity=breakthrough.interstitial_velocity(),
                porosity=breakthrough.column.porosity,
                bulk_density=breakthrough.column.get_bulk_density(),
                C_in=breakthrough.mean_feed_concentration(),
                t_history=t_eval[: i + 1],  # times up to current
                C_outlet_history=C_outlet_history[: i + 1],  # outlet up to current
            )
            for i in range(len(t_eval))
        ]
