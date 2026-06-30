"""analytic_models.py"""

import warnings
import numpy as np
from scipy.special import erfc, erfcx
from scipy.integrate import quad
from scipy.special import i0
import reactormodels


class AnalyticModels:
    """Base class for analytical solutions to fixed bed adsorption models."""

    def __init__(
        self,
        length: float | None = None,
        porosity: float | None = None,
        bulk_density: float | None = None,
        particle_density: float | None = None,
        diameter: float | None = None,
        feed_concentrations: float | np.ndarray | None = None,
        flow_rate: float | None = None,
    ):
        """Column and breakthrough inputs."""
        self.length = length
        self.porosity = porosity
        self.bulk_density = bulk_density
        self.particle_density = particle_density
        self.diameter = diameter
        self.feed_concentrations = feed_concentrations
        self.flow_rate = flow_rate

    def model(self, x: float | np.ndarray, time: float | np.ndarray):
        """Analytic model that return C/C_0"""
        raise NotImplementedError

    def spatial_profile(self, x: np.ndarray, time: float) -> np.ndarray:
        """Return concentration profile with respect to fixed time."""
        return self.model(x, time)

    def breakthrough_profile(self, time: np.ndarray, x: float) -> np.ndarray:
        """Return breakthrough profile with respect to time fixed bed length."""
        return self.model(x, time)


class OgataBanks(AnalyticModels):
    """Ogata Banks solution for 1D advection-diffusion with step input.

    D: diffusion
    v: interstital velocity
    """

    def __init__(
        self,
        diffusion: float,
    ):
        self.interstitial_velocity = reactormodels.Column.interstitial_velocity(
            flow_rate=self.flow_rate
        )
        self.diffusion = diffusion
        self.inlet_concentration = reactormodels.Breakthrough.mean_feed_concentration()

    def model(
        self,
        x: np.ndarray | float,
        time: np.ndarray | float,
    ):
        """Equation:

        C/Co =  1/2 * {erfc[(x - v*t)/(2*sqrt(D*t))]
                + exp(v*x/D)*erfc[(x + v*t)/(2*sqrt(D*t))]}
        """
        Pe_local = self.interstitial_velocity * x / self.diffusion
        arg1 = (x - self.interstitial_velocity * time) / (
            2 * np.sqrt(self.diffusion * time)
        )
        arg2 = (x + self.interstitial_velocity * time) / (
            2 * np.sqrt(self.diffusion * time)
        )
        exponent = Pe_local - arg2**2
        term2 = np.where(
            exponent > 500,
            0.0,
            erfcx(arg2) * np.exp(exponent),
        )
        return self.inlet_concentration * 0.5 * (erfc(arg1) + term2)


class YoonNelson(AnalyticModels):
    """Yoon-Nelson Model for fixed bed adsorption.

    t_50: time to 50% breakthrough
    k_YN: curve shaping coefficient
    """

    def __init__(self, k_YN: float, t_50: float):
        self.k_YN = k_YN
        self.t_50 = t_50

    def model(
        self,
        x: np.ndarray | float,
        time: np.ndarray | float,
    ):
        """Equation:

        C/Co = 1 / (1 + exp[k_YN*(t_50 - t)])
        """
        return 1 / (1 + np.exp(self.k_YN * (self.t_50 - time)))

    def breakthrough_profile(self, time: np.ndarray, x: float = 0) -> np.ndarray:
        """Yoon-Nelson has no spatial dependence; x is ignored."""
        if x != 0:
            warnings.warn("Yoon-Nelson has no spatial term; x argument is ignored.")
        return super().breakthrough_profile(time, 0)

    def spatial_profile(self, x: np.ndarray, time: float):
        """Not defined: Yoon-Nelson has no spatial dependence."""
        raise NotImplementedError(
            "Yoon-Nelson has no spatial term; use breakthrough_profile instead."
        )


class Clark(AnalyticModels):
    """Clark Model.

    A: capacity-like model coefficient
    r: kinetic-like model coefficient
    n: curve asymmetry coefficient
    """

    def __init__(
        self,
        r: float,
        A: float,
        n: float,
    ):
        self.r = r
        self.A = A
        self.n = n

    def model(
        self,
        x: np.ndarray | float,
        time: np.ndarray | float,
    ):
        """Equation:

        C/Co = 1 / [1 + A*exp(-r*t)]^(1 / (n - 1))
        """
        return 1 / (1 + self.A * np.exp(-self.r * time)) ** (1 / (self.n - 1))

    def breakthrough_profile(self, time: np.ndarray, x: float = 0) -> np.ndarray:
        """Clark has no spatial dependence; x is ignored."""
        if x != 0:
            warnings.warn("Yoon-Nelson has no spatial term; x argument is ignored.")
        return super().breakthrough_profile(time, 0)

    def spatial_profile(self, x: np.ndarray, time: float):
        """Not defined: Clark has no spatial dependence."""
        raise NotImplementedError(
            "Clark has no spatial term; use breakthrough_profile instead."
        )


class BohartAdams(AnalyticModels):
    """Bohart-Adams Model.

    sorbent_loading (m_o): dry mass of sorbent per volume bed
    k_BA: Bohart-Adams lumped rate constant
    sorbent_capacity: removal capacity per mass sorbent
    velocity: superficial velocity
    """

    def __init__(
        self,
        k_BA: float,
        sorbent_capacity: float,
    ):
        self.bed_density = reactormodels.Column.get_bulk_density()
        self.k_BA = k_BA
        self.sorbent_capacity = sorbent_capacity
        self.velocity = reactormodels.Column.superficial_velocity(
            flow_rate=self.flow_rate
        )
        self.inlet_concentration = reactormodels.Breakthrough.mean_feed_concentration()

    def model(
        self,
        x: np.ndarray | float,
        time: np.ndarray | float,
    ):
        """Equation:

        C/Co = 1 / [1 + exp(rho_b*k_BA*q_m*L/u - k_BA*Co*t)]
        """
        arg1 = self.bed_density * self.k_BA * self.sorbent_capacity * x / self.velocity
        arg2 = self.k_BA * self.inlet_concentration * time
        return 1 / (1 + np.exp(arg1 - arg2))


class ThomasRectangular(AnalyticModels):
    """Thomas Model with rectangular isotherm.

    k_Th: Thomas Model rate constant
    sorbent_capacity: removal capacity per mass sorbent
    BV: bed volumes treated
    """

    def __init__(
        self,
        k_Th: float,
        sorbent_capacity: float,
    ):
        self.sorbent_mass = (
            reactormodels.Column.column_volume()
            * reactormodels.Column.get_bulk_density()
        )
        self.k_Th = k_Th
        self.sorbent_capacity = sorbent_capacity
        self.bed_volume = reactormodels.Column.column_volume()
        self.bed_volumes_treated = reactormodels.Breakthrough.time_to_bed_volumes(
            column_volume=self.bed_volume
        )
        self.inlet_concentration = reactormodels.Breakthrough.mean_feed_concentration()

    def model(self):
        """Equation:

        C/Co = 1 / [1 + exp(k_Th*q_e*x/Q - k_Th*Co*BV)]
        """
        arg1 = self.k_Th * self.sorbent_capacity * self.sorbent_mass / self.bed_volume
        arg2 = self.k_Th * self.inlet_concentration * self.bed_volumes_treated
        return 1 / (1 + np.exp(arg1 - arg2))


class ThomasLangmuir(AnalyticModels):
    """Thomas Model with Langmuir isotherm."""

    def __init__(
        self,
        langmuir_constant: float,
        sorbent_capacity: float,
        k_Th: float,
    ):
        self.langmuir_constant = langmuir_constant
        self.particle_density = self.particle_density
        self.inlet_concentration = reactormodels.Breakthrough.mean_feed_concentration()
        self.sorbent_capacity = sorbent_capacity
        self.k_Th = k_Th
        self.bed_void_fraction = self.porosity
        self.interstitial_velocity = reactormodels.Column.interstitial_velocity(
            flow_rate=self.flow_rate
        )

    def _J_function(self, a: float, b: float) -> float:
        """Equation:

        J(x, y) = 1 - int(exp(-y - tau)*I_0*(2*sqrt(y*tau)) dtau) from 0 to x
        """

        def integrand(tau):
            return np.exp(-(b + tau)) * i0(2 * np.sqrt(b * tau))

        integral, _ = quad(integrand, 0, a)
        return 1.0 - integral

    def model(self, x: float | np.ndarray, time: float | np.ndarray):
        """Equation:

        C/Co = J((n/r), nT) /
            [J((n/r), nT) + [1 - J(n, (nT/r))]exp[(1 - (1/r))(n - nT)]]
        """
        n_arg1 = (
            self.particle_density
            * self.sorbent_capacity
            * self.k_Th
            * x
            * (1 - self.bed_void_fraction)
        )
        n_arg2 = self.bed_void_fraction * self.interstitial_velocity
        n = n_arg1 / n_arg2
        r = 1 + self.langmuir_constant * self.inlet_concentration
        T_arg1 = self.bed_void_fraction * (
            (1 / self.langmuir_constant) + self.inlet_concentration
        )
        T_arg2 = self.interstitial_velocity * time / x - 1
        T_arg3 = (
            self.particle_density * self.sorbent_capacity * (1 - self.bed_void_fraction)
        )
        T = T_arg1 * T_arg2 / T_arg3
        J = np.vectorize(self._J_function)  # wraps scalar quad calls to handle arrays
        J_arg1 = J(n / r, n * T)
        J_arg2 = J(n, n * T / r)

        return J_arg1 / (J_arg1 + (1 - J_arg2) * np.exp((1 - (1 / r)) * (n - n * T)))
