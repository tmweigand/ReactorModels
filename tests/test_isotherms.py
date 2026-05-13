import reactormodels

import numpy as np
import pytest


def test_linear_isotherm():
    iso = reactormodels.models.LinearIsotherm(K=5.0)
    C = np.array([0.0, 1.0, 2.0, 10.0])
    np.testing.assert_allclose(iso.q(C), 5.0 * C)
    np.testing.assert_allclose(iso.dq_dC(C), 5.0 * np.ones_like(C))


def test_freundlich_isotherm():
    K, n = 10.0, 2.0
    iso = reactormodels.models.FreundlichIsotherm(K=K, n=n)
    C = np.array([1.0, 4.0, 9.0])
    expected = K * C ** (1.0 / n)
    np.testing.assert_allclose(iso.q(C), expected)


def test_freundlich_derivative():
    K, n = 10.0, 2.0
    iso = reactormodels.models.FreundlichIsotherm(K=K, n=n)
    C = np.array([1.0, 2.0, 5.0])
    # Numerical derivative
    dC = 1e-6
    numerical = (iso.q(C + dC) - iso.q(C - dC)) / (2 * dC)
    np.testing.assert_allclose(iso.dq_dC(C), numerical, rtol=1e-5)


def test_freundlich_zero_concentration():
    iso = reactormodels.models.FreundlichIsotherm(K=5.0, n=2.0)
    result = iso.q(np.array([0.0]))
    assert result[0] == 0.0


def test_linear_zero_concentration():
    iso = reactormodels.models.LinearIsotherm(K=3.0)
    result = iso.q(np.array([0.0]))
    assert result[0] == 0.0
