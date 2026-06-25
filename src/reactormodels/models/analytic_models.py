"""analytic_models.py"""

import warnings
import numpy as np
from scipy.special import erfc, erfcx
from scipy.integrate import quad
from scipy.special import i0


class AnalyticModels:
    """Base class for analytical solutions to fixed bed adsorption models."""

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
        interstitial_velocity: float,
        diffusion: float,
        inlet_concentration: float,
    ):
        self.interstitial_velocity = interstitial_velocity
        self.diffusion = diffusion
        self.inlet_concentration = inlet_concentration

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
        sorbent_loading: float,
        k_BA: float,
        sorbent_capacity: float,
        velocity: float,
        inlet_concentration: float,
    ):
        self.sorbent_loading = sorbent_loading
        self.k_BA = k_BA
        self.sorbent_capacity = sorbent_capacity
        self.velocity = velocity
        self.inlet_concentration = inlet_concentration

    def model(
        self,
        x: np.ndarray | float,
        time: np.ndarray | float,
    ):
        """Equation:

        C/Co = 1 / [1 + exp(m_o*k_BA*q_m*L/u - k_BA*Co*t)]
        """
        arg1 = (
            self.sorbent_loading * self.k_BA * self.sorbent_capacity * x / self.velocity
        )
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
        sorbent_mass: float,
        k_Th: float,
        sorbent_capacity: float,
        bed_volume: float,
        bed_volumes_treated: float,
        inlet_concentration: float,
    ):
        self.sorbent_mass = sorbent_mass
        self.k_Th = k_Th
        self.sorbent_capacity = sorbent_capacity
        self.bed_volume = bed_volume
        self.bed_volumes_treated = bed_volumes_treated
        self.inlet_concentration = inlet_concentration

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
        apparent_density: float,
        inlet_concentration: float,
        sorbent_capacity: float,
        k_Th: float,
        bed_void_fraction: float,
        interstitial_velocity: float,
    ):
        self.langmuir_constant = langmuir_constant
        self.apparent_density = apparent_density
        self.inlet_concentration = inlet_concentration
        self.sorbent_capacity = sorbent_capacity
        self.k_Th = k_Th
        self.bed_void_fraction = bed_void_fraction
        self.interstitial_velocity = interstitial_velocity

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
            self.apparent_density
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
            self.apparent_density * self.sorbent_capacity * (1 - self.bed_void_fraction)
        )
        T = T_arg1 * T_arg2 / T_arg3
        J = np.vectorize(self._J_function)  # wraps scalar quad calls to handle arrays
        J_arg1 = J(n / r, n * T)
        J_arg2 = J(n, n * T / r)

        return J_arg1 / (J_arg1 + (1 - J_arg2) * np.exp((1 - (1 / r)) * (n - n * T)))
