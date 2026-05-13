import reactormodels
from scipy.special import roots_jacobi
import numpy as np
import pytest


def test_point_count():
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=4)
    assert len(oc.nodes) == 5  # 4 interior + 1 surface


def test_jabobi_roots_legendre():
    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=4, alpha=0, beta=0
    )
    oc_roots = oc.jacobi_roots()

    root_1 = np.sqrt((3.0 / 7.0) - (2.0 / 7.0) * np.sqrt(6 / 5))
    root_2 = np.sqrt((3.0 / 7.0) + (2.0 / 7.0) * np.sqrt(6 / 5))
    shift_and_scale = lambda x: 0.5 * (x + 1.0)
    true_roots = sorted([-root_1, root_1, -root_2, root_2])
    scaled_roots = [shift_and_scale(x) for x in true_roots]

    assert oc_roots == pytest.approx(scaled_roots, abs=1.0e-12)


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (1.0, 0.0),  # skewed right
        (0.0, 1.0),  # skewed left
        (1.0, 1.0),  # symmetric, interior Lobatto nodes
        (2.0, 0.0),  # higher alpha, stronger right skew
        (1.0, 2.0),  # mixed
    ],
)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10])
def test_jacobi_roots_asymmetric_vs_scipy(n, alpha, beta):
    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=n, alpha=alpha, beta=beta
    )
    oc_roots = oc.jacobi_roots()

    scipy_roots, _ = roots_jacobi(n, alpha, beta)
    scaled = 0.5 * (np.sort(scipy_roots) + 1.0)

    assert oc_roots == pytest.approx(scaled, abs=1e-12)


def test_jacobi_roots_chebyshev_u():
    """Roots of U_n(x) = sin((k)π/(n+1)), k=1..n"""
    n = 4
    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=n, alpha=0.5, beta=0.5
    )
    oc_roots = oc.jacobi_roots()

    shift_and_scale = lambda x: 0.5 * (x + 1.0)
    true_roots = sorted([np.cos(k * np.pi / (n + 1)) for k in range(1, n + 1)])
    scaled_roots = [shift_and_scale(x) for x in true_roots]

    assert oc_roots == pytest.approx(scaled_roots, abs=1e-12)


def test_surface_point_is_one():
    for n in [3, 5, 8]:
        oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=n)
        assert oc.nodes[-1] == pytest.approx(1.0, abs=1e-12)


def test_interior_points_in_range():
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=5)
    for xi in oc.nodes[:-1]:
        assert xi > 0.0
        assert xi < 1.0


def test_differentiation_matrix_linear():
    """first_derivative @ nodes should equal ones (derivative of identity function)."""
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=5)
    dfdx = oc.first_derivative @ oc.nodes
    np.testing.assert_allclose(dfdx, np.ones_like(oc.nodes), atol=1e-10)


def test_differentiation_matrix_quadratic():
    """first_derivative @ nodes^2 ≈ 2*nodes."""
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=5)
    dfdx = oc.first_derivative @ (oc.nodes**2)
    np.testing.assert_allclose(dfdx, 2 * oc.nodes, atol=1e-9)


def test_second_derivative_quadratic():
    """second_derivative @ nodes^2 ≈ 2 (constant)."""
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=5)
    d2fdx2 = oc.second_derivative @ (oc.nodes**2)
    np.testing.assert_allclose(d2fdx2, 2.0 * np.ones_like(oc.nodes), atol=1e-8)


def test_radial_operator_constant():
    """Laplacian of a constant should be zero."""
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=5)
    L = oc.radial_operator()
    u = np.ones(len(oc.nodes))
    result = L @ u
    np.testing.assert_allclose(result, np.zeros_like(result), atol=1e-10)


def test_multi_element_nodes():
    """Node count and bounds."""
    oc = reactormodels.numerics.OrthogonalCollocation(n_interior_points=3, n_elements=4)
    n_local = 3 + 2  # interior + 2 boundaries
    n_expected = 4 * (n_local - 1) + 1  # = 17
    assert len(oc.nodes) == n_expected
    assert oc.nodes[0] == pytest.approx(0.0)
    assert oc.nodes[-1] == pytest.approx(1.0)
    assert np.all(np.diff(oc.nodes) > 0)


def test_multi_element_vs_single_smooth():
    """For a smooth function, multi-element derivative matches single element."""
    f = lambda x: np.sin(np.pi * x)
    df = lambda x: np.pi * np.cos(np.pi * x)

    oc = reactormodels.numerics.OrthogonalCollocation(
        n_interior_points=4, n_elements=5, add_inlet=True
    )
    x = oc.nodes
    dfdx_numerical = oc.first_derivative @ f(x)
    dfdx_exact = df(x)

    # Multi-element should be accurate on smooth functions
    assert dfdx_numerical == pytest.approx(dfdx_exact, abs=1e-4)
