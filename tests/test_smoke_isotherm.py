"""Tests for equilibrium isotherm models and fitting."""

import numpy as np

from reactormodels.Isotherm_class import Isotherm, fit_isotherm_parameters


def test_isotherm_fitting_smoke_test():
    """Smoke test that confirms the fitting function runs."""

    isotherm = Isotherm()

    concentrations = np.array([1.0, 2.0, 3.0])
    expected_K = 5.0

    sorbent_concentrations = isotherm.q(
        concentrations=concentrations,
        isotherm_type="linear",
        K=expected_K,
    )

    result = fit_isotherm_parameters(
        isotherm=isotherm,
        isotherm_type="linear",
        concentrations=concentrations,
        sorbent_concentrations=sorbent_concentrations,
        parameter_names=["K"],
        initial_guess=[1.0],
    )

    assert np.isclose(result["K"], expected_K)


def test_fit_linear_isotherm_parameters():
    """Fit K for synthetic linear isotherm data."""

    isotherm = Isotherm()

    concentrations = np.linspace(0.1, 10.0, 50)
    expected_K = 5.0

    sorbent_concentrations = isotherm.q(
        concentrations=concentrations,
        isotherm_type="linear",
        K=expected_K,
    )

    result = fit_isotherm_parameters(
        isotherm=isotherm,
        isotherm_type="linear",
        concentrations=concentrations,
        sorbent_concentrations=sorbent_concentrations,
        parameter_names=["K"],
        initial_guess=[1.0],
    )

    assert np.isclose(result["K"], expected_K)


def test_fit_freundlich_isotherm_parameters():
    """Fit K and n for synthetic Freundlich isotherm data."""

    isotherm = Isotherm()

    concentrations = np.linspace(0.1, 10.0, 50)
    expected_K = 3.0
    expected_n = 0.8

    sorbent_concentrations = isotherm.q(
        concentrations=concentrations,
        isotherm_type="freundlich",
        K=expected_K,
        n=expected_n,
    )

    result = fit_isotherm_parameters(
        isotherm=isotherm,
        isotherm_type="freundlich",
        concentrations=concentrations,
        sorbent_concentrations=sorbent_concentrations,
        parameter_names=["K", "n"],
        initial_guess=[1.0, 1.0],
    )

    assert np.isclose(result["K"], expected_K)
    assert np.isclose(result["n"], expected_n)


def test_fit_langmuir_isotherm_parameters():
    """Fit q_max and K for synthetic Langmuir isotherm data."""

    isotherm = Isotherm()

    concentrations = np.linspace(0.1, 10.0, 50)
    expected_q_max = 20.0
    expected_K = 2.5

    sorbent_concentrations = isotherm.q(
        concentrations=concentrations,
        isotherm_type="langmuir",
        q_max=expected_q_max,
        K=expected_K,
    )

    result = fit_isotherm_parameters(
        isotherm=isotherm,
        isotherm_type="langmuir",
        concentrations=concentrations,
        sorbent_concentrations=sorbent_concentrations,
        parameter_names=["q_max", "K"],
        initial_guess=[10.0, 1.0],
    )

    assert np.isclose(result["q_max"], expected_q_max)
    assert np.isclose(result["K"], expected_K)
