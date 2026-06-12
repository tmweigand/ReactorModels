"""analytic_models.py"""

import numpy as np
from scipy.special import erfc, erfcx
from scipy.integrate import quad
from scipy.special import i0


class AnalyticModels:
    """Analytical solutions to fixed bed adsorption models."""

    @staticmethod
    def ogata_banks(
        x: np.ndarray,
        time: float,
        velocity: float,
        diffusion: float,
        inlet_concentration: float = 1.0,
    ):
        """Ogata Banks solution for 1D advection-diffusion with step input."""
        Pe_local = velocity * x / diffusion
        arg1 = (x - velocity * time) / (2 * np.sqrt(diffusion * time))
        arg2 = (x + velocity * time) / (2 * np.sqrt(diffusion * time))
        exponent = Pe_local - arg2**2
        term2 = np.where(
            exponent > 500,
            0.0,
            erfcx(arg2) * np.exp(exponent),
        )
        return inlet_concentration * 0.5 * (erfc(arg1) + term2)

    @staticmethod
    def yoon_nelson(time: np.ndarray, tau: float, k_YN: float):
        """Yoon-Nelson Model for fixed bed adsorption.

        tau: time to 50% breakthrough
        k_YN: curve shaping coefficient

        Equation:
            C/Co = 1 / (1 + exp[k_YN*(tau - t)])

        """
        return 1 / (1 + np.exp(k_YN * (tau - time)))

    @staticmethod
    def clark(time: np.ndarray, r: float, A: float, n: float):
        """Clark Model.

        A: capacity-like model coefficient
        r: kintic-like model coefficient
        n: curve asymmetry coefficient

        Equation:
            C/Co = 1 / [1 + A*exp(-r*t)]^(1 / (n -1))

        """
        return 1 / (1 + A * np.exp(-r * time)) ** (1 / (n - 1))

    @staticmethod
    def bohart_adams(
        sorbent_loading: float,
        k_BA: float,
        sorbent_capacity: float,
        bed_length: float,
        velocity: float,
        time: float,
        inlet_concentration: float,
    ):
        """Bohart-Adams Model.

        sorbent_loading: dry mass of sorbent in bed
        k_BA: Bohart-Adams lumped rate constant
        sorbent_capacity: removal capacity per mass sorbent
        velocity: superficial velocity

        Equation:
            C/Co = 1 / [1 + exp(m_o*k_BA*q_m*L/u - k_BA*Co*t)]

        """
        arg1 = sorbent_loading * k_BA * sorbent_capacity * bed_length / velocity
        arg2 = k_BA * inlet_concentration * time
        return 1 / (1 + np.exp(arg1 - arg2))

    @staticmethod
    def thomas_rectangular(
        sorbent_mass: float,
        k_Th: float,
        sorbent_capacity: float,
        bed_volume: float,
        bed_volumes_treated: float,
        inlet_concentration: float,
    ):
        """Thomas Model with rectangular isotherm.

        k_Th: Thomas Model rate constant
        sorbent_capacity: removal capacity per mass sorbent

        Equation:
            C/Co = 1 / [1 + exp(k_Th*q_e*x/Q - k_Th*Co*t)]

        """
        arg1 = k_Th * sorbent_capacity * sorbent_mass / bed_volume
        arg2 = k_Th * inlet_concentration * bed_volumes_treated
        return 1 / (1 + np.exp(arg1 - arg2))

    @staticmethod
    def _J_function(x, y):
        """Used to solve Thomas with Langmuir."""

        def integrand(tau):
            return np.exp(-(y + tau)) * i0(2 * np.sqrt(y * tau))

        integral, _ = quad(integrand, 0, x)
        return 1.0 - integral

    @staticmethod
    def thomas_langmuir(
        langmuir_constant: float,
        apparent_density: float,
        inlet_concentration: float,
        sorbent_capacity: float,
        k_Th: float,
        bed_length: float,
        bed_void_fraction: float,
        interstitial_velocity: float,
        time: float,
    ):
        """Thomas Model with Langmuir isotherm.

        Equation:
            C/Co = J((n/r), nT) / [J((n/r), nT) + [1 - J(n, (nT/r))]exp[(1 - (1/r))(n - nT)]]

            J(x, y) = 1 - int(exp(-y - tau)*I_0*(2*sqrt(y*tau)) dtau) from 0 to x

        """
        n_arg1 = (
            apparent_density
            * sorbent_capacity
            * k_Th
            * bed_length
            * (1 - bed_void_fraction)
        )
        n_arg2 = bed_void_fraction * interstitial_velocity
        n = n_arg1 / n_arg2
        r = 1 + langmuir_constant * inlet_concentration
        T_arg1 = bed_void_fraction * ((1 / langmuir_constant) + inlet_concentration)
        T_arg2 = interstitial_velocity * time / bed_length - 1
        T_arg3 = apparent_density * sorbent_capacity * (1 - bed_void_fraction)
        T = T_arg1 * T_arg2 / T_arg3
        print(f"time={time:.2f}, " f"n={n:.3e}, " f"T={T:.3e}, " f"nT={n*T:.3e}")
        J_arg1 = AnalyticModels._J_function(n / r, n * T)
        J_arg2 = AnalyticModels._J_function(n, n * T / r)

        exp_arg = (1 - (1 / r)) * (n - n * T)
        print(exp_arg)
        return J_arg1 / (J_arg1 + (1 - J_arg2) * np.exp((1 - (1 / r)) * (n - n * T)))
