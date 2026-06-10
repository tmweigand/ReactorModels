import numpy as np
import pytest

from reactormodels.models import FreundlichIsotherm, LinearIsotherm
from reactormodels.models.isotherm import fit_isotherm


def test_linear_isotherm():
    iso = LinearIsotherm(K=5.0)
    C = np.array([0.0, 1.0, 2.0, 10.0])

    np.testing.assert_allclose(iso.q(C), 5.0 * C)
    np.testing.assert_allclose(iso.dq_dC(C), 5.0 * np.ones_like(C))


def test_freundlich_isotherm():
    K, n = 10.0, 2.0
    iso = FreundlichIsotherm(K=K, n=n)
    C = np.array([1.0, 4.0, 9.0])
    expected = K * C ** (1.0 / n)

    np.testing.assert_allclose(iso.q(C), expected)


def test_freundlich_derivative():
    K, n = 10.0, 2.0
    iso = FreundlichIsotherm(K=K, n=n)
    C = np.array([1.0, 2.0, 5.0])
    dC = 1e-6
    numerical = (iso.q(C + dC) - iso.q(C - dC)) / (2 * dC)

    np.testing.assert_allclose(iso.dq_dC(C), numerical, rtol=1e-5)


def test_freundlich_zero_concentration():
    iso = FreundlichIsotherm(K=5.0, n=2.0)
    result = iso.q(np.array([0.0]))

    assert result[0] == 0.0


def test_linear_zero_concentration():
    iso = LinearIsotherm(K=3.0)
    result = iso.q(np.array([0.0]))

    assert result[0] == 0.0


def test_fit_linear_isotherm_recovers_parameter():
    """Test fitting a linear isotherm to synthetic equilibrium data."""
    C = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    expected = LinearIsotherm(K=2.5)
    q = expected.q(C)

    isotherm = fit_isotherm(LinearIsotherm, C, q, initial_guess=(1.0,))

    np.testing.assert_allclose(isotherm.K, 2.5)


def test_fit_freundlich_isotherm_recovers_parameters():
    """Test fitting a Freundlich isotherm to synthetic equilibrium data."""
    C = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    expected = FreundlichIsotherm(K=3.0, n=2.0)
    q = expected.q(C)

    isotherm = fit_isotherm(FreundlichIsotherm, C, q, initial_guess=(1.0, 1.0))

    np.testing.assert_allclose(isotherm.K, 3.0)
    np.testing.assert_allclose(isotherm.n, 2.0)


def test_fit_isotherm_rejects_mismatched_shapes():
    """Test that fitting rejects C and q arrays with different shapes."""
    C = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="C and q must have the same shape."):
        fit_isotherm(LinearIsotherm, C, q, initial_guess=(1.0,))


def test_fit_isotherm_rejects_negative_values():
    """Test that fitting rejects negative equilibrium data."""
    C = np.array([1.0, -2.0, 3.0])
    q = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="C and q values must be nonnegative."):
        fit_isotherm(FreundlichIsotherm, C, q, initial_guess=(1.0, 1.0))
