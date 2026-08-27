"""isotherm.py"""

import numpy as np
from scipy.optimize import curve_fit


class Isotherm:
    """Base class for adsorption isotherms."""

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        raise NotImplementedError

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        raise NotImplementedError

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        raise NotImplementedError

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration"""
        raise NotImplementedError

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        raise NotImplementedError

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        raise NotImplementedError


class LangmuirIsotherm(Isotherm):
    """Langmuir isotherm:

    q* = q_m * K * C / (1 + K * C) or C* = q / (K (q_m - q))

    Parameters
    ----------
    K: float
        Langmuir dissociation constant
    q_m: float
        Maximum sorbent capacity.

    """

    def __init__(self, K: float, q_m: float):
        self.K = K
        self.q_m = q_m

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        return (
            self.q_m * self.K * np.maximum(C, 0.0) / (1 + self.K * np.maximum(C, 0.0))
        )

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C = np.asarray(C, dtype=float)
        return self.q_m * self.K / (1 + self.K * np.maximum(C, 0.0)) ** 2

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.asarray(C, dtype=float)
        return -2 * self.q_m * self.K**2 / (1 + self.K * np.maximum(C, 0.0)) ** 3

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration"""
        q = np.asarray(q, dtype=float)
        return q / (self.K * (self.q_m - q))

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        q = np.asarray(q, dtype=float)
        return self.q_m / (self.K * (self.q_m - q) ** 2)

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        q = np.asarray(q, dtype=float)
        return 2 * self.q_m / (self.K * (self.q_m - q) ** 3)


class FreundlichIsotherm(Isotherm):
    """Freundlich isotherm: q* = K * C^(1/n)

    Parameters
    ----------
    K : float
        Freundlich capacity factor
    n : float
        Freundlich intensity factor.

    """

    def __init__(self, K: float, n: float):
        self.K = K
        self.n = n

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        C = np.asarray(C, dtype=float)
        return self.K * np.maximum(C, 0.0) ** (1.0 / self.n)

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        C = np.asarray(C, dtype=float)
        C = np.maximum(C, 1e-30)
        return (self.K / self.n) * C ** (1.0 / self.n - 1.0)

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        C = np.maximum(np.asarray(C, dtype=float), 1e-30)
        return (self.K / self.n) * (1.0 / self.n - 1.0) * C ** (1.0 / self.n - 2.0)

    def C(self, q):
        """Return liquid phase concentration"""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        return (q / self.K) ** self.n

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)

        return self.n * (1 / self.K) ** self.n * q ** (self.n - 1)

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        q = np.asarray(q, dtype=float)
        q = np.maximum(q, 0.0)
        return self.n * (self.n - 1) * (1 / self.K) ** self.n * q ** (self.n - 2)


class LinearIsotherm(Isotherm):
    """Linear isotherm: q* = K * C

    Parameters
    ----------
    K : float
        Henry constant  [mg/g / (mg/L)]

    """

    def __init__(self, K: float):
        self.K = K

    def q(self, C: float | np.ndarray) -> np.ndarray:
        """Return sorbed mass concentration"""
        return self.K * np.asarray(C, dtype=float)

    def dq_dC(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the derivative of sorbed mass concentration by concentration."""
        return self.K * np.ones_like(np.asarray(C, dtype=float))

    def d2q_dC2(self, C: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(C, dtype=float))

    def C(self, q: float | np.ndarray) -> np.ndarray:
        """Return liquid phase concentration."""
        return np.asarray(q, dtype=float) / self.K

    def dC_dq(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of liquid concentration by sorbed mass concentration."""
        return np.ones_like(np.asarray(q, dtype=float)) / self.K

    def d2C_dq2(self, q: float | np.ndarray) -> np.ndarray:
        """Calculate the second derivative."""
        return np.zeros_like(np.asarray(q, dtype=float))


def flatten_parameters(parameters):
    """Flatten multi-species isotherm parameters for fitting."""
    flat = []

    for parameter in parameters:
        flat.extend(np.asarray(parameter).ravel())

    return np.asarray(flat, dtype=float)


def unflatten_parameters(flat, template):
    """Return fitting parameters to shape of initial guesses."""
    flat = np.asarray(flat, dtype=float)

    parameters = []
    i = 0

    for parameter in template:
        parameter = np.asarray(parameter)

        n = parameter.size

        values = flat[i : i + n]
        i += n

        if parameter.ndim == 0:
            parameters.append(values[0])
        else:
            parameters.append(values.reshape(parameter.shape))

    return tuple(parameters)


def fit_isotherm(
    isotherm_class: type[Isotherm],
    xdata: float | np.ndarray,
    ydata: float | np.ndarray,
    initial_guess: tuple[float, ...],
    fit_indices: tuple[int, ...],
    parameter_template: tuple[object, ...],
    output=None,
) -> Isotherm:
    """Fit an isotherm to equilibrium concentration data.

    Parameters
    ----------
    isotherm_class : type[Isotherm]
        Isotherm class to fit.
    xdata : float | np.ndarray
        Independent equilibrium concentration data.
    ydata : float | np.ndarray
        Dependent equilibrium concentration data.
    initial_guess : tuple[float, ...]
        Initial guess for fitting parameters.
    fit_indices : tuple[int, ...]
        Indices of the constructor parameters to be fitted.
    parameter_template : tuple[object, ...]
        Complete set of isotherm constructor parameters, in the same
        order as the isotherm constructor. Values corresponding to
        ``fit_indices`` are replaced during fitting.
    output : str, optional
        Variable being fitted. If ``"q"``, fit q as a function of C.
        If ``"C"``, fit C as a function of q. If None, use the
        default output specified by the isotherm class.

    Isotherm
        Fitted isotherm instance.

    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    if xdata.shape != ydata.shape:
        raise ValueError("C and q must have the same shape.")
    if np.any(xdata < 0) or np.any(ydata < 0):
        raise ValueError("C and q values must be nonnegative.")

    if output is None:
        output = isotherm_class.output

    if any(i < 0 or i >= len(parameter_template) for i in fit_indices):
        raise ValueError(
            "All fit_indices must correspond to parameters in " "parameter_template."
        )

    if len(initial_guess) != len(fit_indices):
        raise ValueError(
            "initial_guess must contain one value for each " "parameter in fit_indices."
        )

    template = tuple(
        np.asarray(parameter) if np.asarray(parameter).ndim > 0 else parameter
        for parameter in initial_guess
    )

    p0 = flatten_parameters(template)

    def model(x, *flat_parameters):
        fitted_parameters = unflatten_parameters(
            flat_parameters,
            template,
        )

        parameters = list(parameter_template)

        for index, value in zip(fit_indices, fitted_parameters):
            parameters[index] = value

        isotherm = isotherm_class(*parameters)

        function = isotherm.q if output == "q" else isotherm.C

        return np.array([function(x_i) for x_i in x]).ravel()

    bounds = (np.full(len(p0), 1e-30), np.full(len(p0), np.inf))

    popt, _ = curve_fit(
        model,
        xdata,
        ydata.ravel(),
        p0=p0,
        bounds=bounds,
    )

    fitted_parameters = unflatten_parameters(
        popt,
        template,
    )

    parameters = list(parameter_template)

    for index, value in zip(fit_indices, fitted_parameters):
        parameters[index] = value

    return isotherm_class(*parameters)
