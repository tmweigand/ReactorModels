"""test_helpers.py"""

import numpy as np

import reactormodels


def test_convergence_order_first_order():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    error = h

    order = reactormodels.numerics.helpers.convergence_order(h, error)

    assert np.isnan(order[0])
    np.testing.assert_allclose(order[1:], 1.0)


def test_convergence_order_second_order():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    error = h**2

    order = reactormodels.numerics.helpers.convergence_order(h, error)

    assert np.isnan(order[0])
    np.testing.assert_allclose(order[1:], 2.0)
