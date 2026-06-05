"""Equilibrium isotherm models and parameter fitting."""

import numpy as np
from scipy.optimize import curve_fit


class Isotherm:
    """Equilibrium isotherm models."""

    def linear(
        self,
        concentrations: np.ndarray,
        K: float,
    ) -> np.ndarray:
        """Return sorbent concentrations for the linear isotherm.

        q = K * C
        """

        return K * concentrations

    def freundlich(
        self,
        concentrations: np.ndarray,
        K: float,
        n: float,
    ) -> np.ndarray:
        """Return sorbent concentrations for the Freundlich isotherm.

        q = K * C**n
        """

        return K * concentrations**n

    def langmuir(
        self,
        concentrations: np.ndarray,
        q_max: float,
        K: float,
    ) -> np.ndarray:
        """Return sorbent concentrations for the Langmuir isotherm.

        q = q_max * K * C / (1 + K * C)
        """

        return (q_max * K * concentrations) / (1 + K * concentrations)

    def q(
        self,
        concentrations: np.ndarray,
        isotherm_type: str,
        **parameters: float,
    ) -> np.ndarray:
        """Return sorbent concentrations for a selected isotherm model."""

        if isotherm_type == "linear":
            return self.linear(
                concentrations=concentrations,
                K=parameters["K"],
            )

        if isotherm_type == "freundlich":
            return self.freundlich(
                concentrations=concentrations,
                K=parameters["K"],
                n=parameters["n"],
            )

        if isotherm_type == "langmuir":
            return self.langmuir(
                concentrations=concentrations,
                q_max=parameters["q_max"],
                K=parameters["K"],
            )

        raise ValueError("isotherm_type must be 'linear', 'freundlich', or 'langmuir'.")


def fit_isotherm_parameters(
    isotherm: Isotherm,
    isotherm_type: str,
    concentrations: np.ndarray,
    sorbent_concentrations: np.ndarray,
    parameter_names: list[str],
    initial_guess: list[float],
    lower_bounds: list[float] | None = None,
    upper_bounds: list[float] | None = None,
) -> dict[str, float]:
    """Fit parameters for a selected isotherm model."""

    if concentrations.size == 0:
        raise ValueError("concentrations cannot be empty.")

    if sorbent_concentrations.size == 0:
        raise ValueError("sorbent_concentrations cannot be empty.")

    if concentrations.shape != sorbent_concentrations.shape:
        raise ValueError(
            "concentrations and sorbent_concentrations must have the same shape."
        )

    if len(parameter_names) != len(initial_guess):
        raise ValueError("parameter_names and initial_guess must have the same length.")

    if lower_bounds is None:
        lower_bounds = [0.0] * len(initial_guess)

    if upper_bounds is None:
        upper_bounds = [np.inf] * len(initial_guess)

    def model(
        concentrations: np.ndarray,
        *parameter_values: float,
    ) -> np.ndarray:
        parameters = dict(zip(parameter_names, parameter_values))

        return isotherm.q(
            concentrations=concentrations,
            isotherm_type=isotherm_type,
            **parameters,
        )

    optimized_parameters, _ = curve_fit(
        model,
        concentrations,
        sorbent_concentrations,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )

    return dict(zip(parameter_names, optimized_parameters))
