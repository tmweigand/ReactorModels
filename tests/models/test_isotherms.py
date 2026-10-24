import numpy as np
import pytest

from reactormodels.models import (
    FreundlichIsotherm,
    LinearIsotherm,
    LangmuirIsotherm,
    CompetitiveLangmuirIsotherm,
    CompetitiveIonIsotherm,
    CompetitiveFreundlichIsotherm,
    CompetitiveLangmuirFreundlichIsotherm,
    CompetitiveStoichiometricIsotherm,
    MultiCapacityIsotherm,
    AdsorbateComplexIsotherm,
)


def test_linear_isotherm():
    iso = LinearIsotherm(K=5.0)
    C = np.array([0.0, 1.0, 2.0, 10.0])

    np.testing.assert_allclose(iso.q(C), 5.0 * C)
    np.testing.assert_allclose(iso.dq_dC(C), 5.0 * np.ones_like(C))


def test_langmuir_isotherm():
    K, q_m = 10.0, 2.0
    iso = LangmuirIsotherm(K=K, q_m=q_m)
    C = np.array([1.0, 4.0, 9.0])
    expected = q_m * K * C / (1 + K * C)

    np.testing.assert_allclose(iso.q(C), expected)


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


def test_c_from_q_equals_q_from_c():
    """Test that C functions return values input to q and vice versa."""
    C = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    q = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    linear = LinearIsotherm(K=3)
    freundlich = FreundlichIsotherm(K=3.0, n=2.0)
    langmuir = LangmuirIsotherm(K=3, q_m=20)

    # q -> c -> q
    np.testing.assert_allclose(linear.q(linear.C(q)), q)
    np.testing.assert_allclose(freundlich.q(freundlich.C(q)), q)
    np.testing.assert_allclose(langmuir.q(langmuir.C(q)), q)

    # c -> q -> c
    np.testing.assert_allclose(linear.C(linear.q(C)), C)
    np.testing.assert_allclose(freundlich.C(freundlich.q(C)), C)
    np.testing.assert_allclose(langmuir.C(langmuir.q(C)), C)


def test_first_derivatives_cancel():
    """Test that dC_dq * dq_dC = 1."""
    C = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    q = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    linear = LinearIsotherm(K=3)
    freundlich = FreundlichIsotherm(K=3.0, n=2.0)
    langmuir = LangmuirIsotherm(K=3, q_m=20)

    # q input
    np.testing.assert_allclose(linear.dC_dq(linear.q(C)) * linear.dq_dC(C), 1)
    np.testing.assert_allclose(
        freundlich.dC_dq(freundlich.q(C)) * freundlich.dq_dC(C), 1
    )
    np.testing.assert_allclose(langmuir.dC_dq(langmuir.q(C)) * langmuir.dq_dC(C), 1)

    # C input
    np.testing.assert_allclose(linear.dC_dq(q) * linear.dq_dC(linear.C(q)), 1)
    np.testing.assert_allclose(
        freundlich.dC_dq(q) * freundlich.dq_dC(freundlich.C(q)), 1
    )
    np.testing.assert_allclose(langmuir.dC_dq(q) * langmuir.dq_dC(langmuir.C(q)), 1)


def assert_jacobian_matches_numerical(
    isotherm,
    var,
    eps=1e-7,
    rtol=1e-5,
    atol=1e-7,
):
    x = np.array([[0.3], [0.1]])

    if var == "q":
        function = isotherm.C
        jacobian = isotherm.dC_dq
    else:
        function = isotherm.q
        jacobian = isotherm.dq_dC

    analytical = jacobian(x)

    n_species = x.size

    numerical = np.zeros((n_species, n_species))

    for j in range(n_species):
        x_plus = x.copy()
        x_minus = x.copy()

        # Perturb species j at node k
        x_plus[j] += eps
        x_minus[j] -= eps

        numerical[:, j] = (function(x_plus) - function(x_minus)) / (2 * eps)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def assert_chain_rule_matches_numerical(
    isotherm,
    var,
    eps=1e-7,
    rtol=1e-5,
    atol=1e-7,
):
    x = np.array([[0.3], [0.1]])

    dxdt = np.array([[0.12], [0.08]])

    if var == "q":
        function = isotherm.C
        J = isotherm.dC_dq(x)
    else:
        function = isotherm.q
        J = isotherm.dq_dC(x)

    analytical = J @ dxdt

    x_plus = x + eps * dxdt
    x_minus = x - eps * dxdt

    numerical = (function(x_plus) - function(x_minus)) / (2 * eps)
    if numerical.ndim == 1:
        numerical = numerical.reshape(-1, 1)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def test_competitive_freundlich_jacobian():
    n = np.array([0.7, 0.9])
    K = np.array([0.5, 0.8])

    isotherm = CompetitiveFreundlichIsotherm(n=n, K=K)

    assert_jacobian_matches_numerical(isotherm, "q")
    assert_chain_rule_matches_numerical(isotherm, "q")


def test_iexcm_jacobian():
    K = np.array([1.2, 0.8])
    MW = np.array([35.45, 40.08])
    valence = np.array([1, 2])
    inlet_concentrations = np.array([100.0, 50.0])
    q_m = 2.0
    bed_density = 1.0

    isotherm = CompetitiveIonIsotherm(
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        q_m=q_m,
        bed_density=bed_density,
    )

    assert_jacobian_matches_numerical(isotherm, "q")
    assert_chain_rule_matches_numerical(isotherm, "q")


def test_competitive_langmuir_jacobian():
    q_m = 20
    K = np.array([0.5, 0.8])

    isotherm = CompetitiveLangmuirIsotherm(q_m=q_m, K=K)

    assert_jacobian_matches_numerical(isotherm, "C")
    assert_chain_rule_matches_numerical(isotherm, "C")


def test_competitive_langmuir_freundlich_jacobian():
    q_m = 20
    K = np.array([0.5, 0.8])
    n = np.array([0.7, 0.9])

    isotherm = CompetitiveLangmuirFreundlichIsotherm(q_m=q_m, K=K, n=n)

    assert_jacobian_matches_numerical(isotherm, "C")
    assert_chain_rule_matches_numerical(isotherm, "C")


def test_competitive_stoichiometric_jacobian():
    q_m = 20
    K = np.array([0.5, 0.8])
    n = np.array([0.7, 0.9])

    isotherm = CompetitiveStoichiometricIsotherm(q_m=q_m, K=K, n=n)

    assert_jacobian_matches_numerical(isotherm, "C")
    assert_chain_rule_matches_numerical(isotherm, "C")


def test_multi_capacity_jacobian():
    q_m = np.array([5, 10])
    K = np.array([0.5, 0.8])

    isotherm = MultiCapacityIsotherm(q_m=q_m, K=K)

    assert_jacobian_matches_numerical(isotherm, "C")
    assert_chain_rule_matches_numerical(isotherm, "C")


def test_adsorbate_complex_jacobian():
    q_m = 10
    K = np.array([0.5, 0.8])
    K_x = 5

    isotherm = AdsorbateComplexIsotherm(q_m=q_m, K=K, K_x=K_x)

    assert_jacobian_matches_numerical(isotherm, "C")
    assert_chain_rule_matches_numerical(isotherm, "C")


def assert_hessian_matches_numerical(
    isotherm,
    var,
    eps=1e-4,
    rtol=1e-4,
    atol=1e-7,
):
    # Shape: (n_species, 1)
    x = np.array([[0.3], [0.4], [0.1]])

    if var == "q":
        function = isotherm.C
        hessian = isotherm.d2C_dq2
    else:
        function = isotherm.q
        hessian = isotherm.d2q_dC2

    analytical = hessian(x)

    n_species = x.size
    numerical = np.zeros((n_species, n_species, n_species))

    for j in range(n_species):
        for k in range(n_species):
            ej = np.zeros_like(x)
            ek = np.zeros_like(x)

            # Perturb species j or k at every node
            ej[j, :] = eps
            ek[k, :] = eps

            if j == k:
                numerical[:, j, k] = (
                    function(x + ej) - 2.0 * function(x) + function(x - ej)
                ) / eps**2
            else:
                numerical[:, j, k] = (
                    function(x + ej + ek)
                    - function(x + ej - ek)
                    - function(x - ej + ek)
                    + function(x - ej - ek)
                ) / (4.0 * eps**2)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def assert_hessian_matches_numerical_binary(
    isotherm,
    var,
    eps=1e-4,
    rtol=1e-4,
    atol=1e-7,
):
    # Shape: (n_species, 1)
    x = np.array([[0.3], [0.4]])

    if var == "q":
        function = isotherm.C
        hessian = isotherm.d2C_dq2
    else:
        function = isotherm.q
        hessian = isotherm.d2q_dC2

    analytical = hessian(x)

    n_species = x.size
    numerical = np.zeros((n_species, n_species, n_species))

    for j in range(n_species):
        for k in range(n_species):
            ej = np.zeros_like(x)
            ek = np.zeros_like(x)

            # Perturb species j or k at every node
            ej[j, :] = eps
            ek[k, :] = eps

            if j == k:
                numerical[:, j, k] = (
                    function(x + ej) - 2.0 * function(x) + function(x - ej)
                ) / eps**2
            else:
                numerical[:, j, k] = (
                    function(x + ej + ek)
                    - function(x + ej - ek)
                    - function(x - ej + ek)
                    + function(x - ej - ek)
                ) / (4.0 * eps**2)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def assert_hessian_contraction_matches_gradient_difference(
    isotherm,
    var,
    eps=1e-5,
    rtol=1e-4,
    atol=1e-7,
):
    dxdt = np.array([[0.12], [0.08], [0.09]])
    x = np.array([[0.3], [0.4], [0.1]])

    if var == "q":
        gradient = isotherm.dC_dq
        hessian = isotherm.d2C_dq2
    else:
        gradient = isotherm.dq_dC
        hessian = isotherm.d2q_dC2

    H = hessian(x)

    # Analytical contraction: sum over k.
    analytical = np.einsum(
        "ijk,k->ij",
        H,
        dxdt.ravel(),
    )

    # Numerical directional derivative of the gradient.
    numerical = (gradient(x + eps * dxdt) - gradient(x - eps * dxdt)) / (2 * eps)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def assert_hessian_contraction_matches_binary(
    isotherm,
    var,
    eps=1e-5,
    rtol=1e-4,
    atol=1e-7,
):
    dxdt = np.array([[0.12], [0.08]])
    x = np.array([[0.3], [0.4]])

    if var == "q":
        gradient = isotherm.dC_dq
        hessian = isotherm.d2C_dq2
    else:
        gradient = isotherm.dq_dC
        hessian = isotherm.d2q_dC2

    H = hessian(x)

    # Analytical contraction: sum over k.
    analytical = np.einsum(
        "ijk,k->ij",
        H,
        dxdt.ravel(),
    )

    # Numerical directional derivative of the gradient.
    numerical = (gradient(x + eps * dxdt) - gradient(x - eps * dxdt)) / (2 * eps)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def test_competitive_freundlich_hessian():
    n = np.array([1.0, 1.1, 1.5])
    K = np.array([0.5, 0.8, 1.2])

    isotherm = CompetitiveFreundlichIsotherm(n=n, K=K)

    assert_hessian_matches_numerical(isotherm, "q")
    assert_hessian_contraction_matches_gradient_difference(isotherm, "q")


def test_iexcm_hessian():
    K = np.array([1.0, 1.1, 1.5])
    MW = np.array([35.45, 96.06, 62.01])
    valence = np.array([1, 2, 1])
    inlet_concentrations = np.array([100.0, 50.0, 25.0])
    q_m = 2.0
    bed_density = 1.0

    isotherm = CompetitiveIonIsotherm(
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        q_m=q_m,
        bed_density=bed_density,
    )

    assert_hessian_matches_numerical(isotherm, "q")
    assert_hessian_contraction_matches_gradient_difference(isotherm, "q")


def test_competitive_langmuir_hessian():
    q_m = 20
    K = np.array([0.5, 0.8, 1.0])

    isotherm = CompetitiveLangmuirIsotherm(q_m=q_m, K=K)

    assert_hessian_matches_numerical(isotherm, "C")
    assert_hessian_contraction_matches_gradient_difference(isotherm, "C")


def test_competitive_langmuir_freundlich_hessian():
    q_m = 20
    K = np.array([0.5, 0.8, 0.6])
    n = np.array([0.7, 0.9, 0.5])

    isotherm = CompetitiveLangmuirFreundlichIsotherm(q_m=q_m, K=K, n=n)

    assert_hessian_matches_numerical(isotherm, "C")
    assert_hessian_contraction_matches_gradient_difference(isotherm, "C")


def test_competitive_stoichiometric_hessian():
    q_m = 20
    K = np.array([0.5, 0.8, 0.6])
    n = np.array([0.7, 0.9, 1.0])

    isotherm = CompetitiveStoichiometricIsotherm(q_m=q_m, K=K, n=n)

    assert_hessian_matches_numerical(isotherm, "C")
    assert_hessian_contraction_matches_gradient_difference(isotherm, "C")


def test_multi_capacity_hessian():
    q_m = np.array([5, 10])
    K = np.array([0.5, 0.8])

    isotherm = MultiCapacityIsotherm(q_m=q_m, K=K)

    assert_hessian_matches_numerical_binary(isotherm, "C")
    assert_hessian_contraction_matches_binary(isotherm, "C")


def test_adsorbate_complex_hessian():
    q_m = 10
    K = np.array([0.5, 0.8])
    K_x = 5

    isotherm = AdsorbateComplexIsotherm(q_m=q_m, K=K, K_x=K_x)

    assert_hessian_matches_numerical_binary(isotherm, "C")
    assert_hessian_contraction_matches_binary(isotherm, "C")


def test_n_species_1_reduces_to_single_species_freundlich():
    q = np.array([[0.3, 0.5, 0.7]])
    n = np.array([1.1])
    K = np.array([0.5])

    iso_single = FreundlichIsotherm(K, n)
    iso_multi = CompetitiveFreundlichIsotherm(K, n)

    np.testing.assert_allclose(iso_single.C(q), iso_multi.C(q))
    np.testing.assert_allclose(iso_single.multi_dC_dq(q), iso_multi.dC_dq(q))
    np.testing.assert_allclose(iso_single.multi_d2C_dq2(q), iso_multi.d2C_dq2(q))


def test_n_species_1_reduces_to_single_species_linear():
    q = np.array([[0.3, 0.5, 0.7]])
    n = np.array([1.0])
    K = np.array([0.5])

    iso_linear = LinearIsotherm(K)
    iso_single = FreundlichIsotherm(K, n)
    iso_multi = CompetitiveFreundlichIsotherm(K, n)

    np.testing.assert_allclose(iso_linear.C(q), iso_multi.C(q))
    np.testing.assert_allclose(iso_linear.multi_dC_dq(q), iso_multi.dC_dq(q))
    np.testing.assert_allclose(
        iso_linear.multi_d2C_dq2(q), iso_multi.d2C_dq2(q), atol=1e-12
    )

    np.testing.assert_allclose(iso_single.C(q), iso_multi.C(q))
    np.testing.assert_allclose(iso_single.multi_dC_dq(q), iso_multi.dC_dq(q))
    np.testing.assert_allclose(
        iso_single.multi_d2C_dq2(q), iso_multi.d2C_dq2(q), atol=1e-12
    )


def test_n_species_1_reduces_to_single_species_langmuir():
    C = np.array([[0.3, 0.5, 0.7]])
    K = np.array([0.5])
    q_m = 20

    iso_single = LangmuirIsotherm(K, q_m)
    iso_multi = CompetitiveLangmuirIsotherm(K, q_m)

    np.testing.assert_allclose(iso_single.q(C), iso_multi.q(C))
    np.testing.assert_allclose(iso_single.multi_dq_dC(C), iso_multi.dq_dC(C))
    np.testing.assert_allclose(iso_single.multi_d2q_dC2(C), iso_multi.d2q_dC2(C))


def test_langmuir_freundlich_reduces_to_langmuir():
    C = np.array([[0.3, 0.5, 0.7], [0.2, 0.4, 0.5], [0.1, 0.2, 0.3]])
    K = np.array([0.5, 0.6, 0.7])
    n = np.array([1.0, 1.0, 1.0])
    q_m = 20

    iso_langmuir = CompetitiveLangmuirIsotherm(K, q_m)
    iso_lf = CompetitiveLangmuirFreundlichIsotherm(K, n, q_m)

    np.testing.assert_allclose(iso_langmuir.q(C), iso_lf.q(C))
    np.testing.assert_allclose(iso_langmuir.dq_dC(C), iso_lf.dq_dC(C))
    np.testing.assert_allclose(iso_langmuir.d2q_dC2(C), iso_lf.d2q_dC2(C))


def test_stoichiometric_reduces_to_langmuir():
    C = np.array([[0.3, 0.5, 0.7], [0.2, 0.4, 0.5], [0.1, 0.2, 0.3]])
    K = np.array([0.5, 0.6, 0.7])
    n = np.array([1.0, 1.0, 1.0])
    q_m = 20

    iso_langmuir = CompetitiveLangmuirIsotherm(K, q_m)
    iso_stoich = CompetitiveStoichiometricIsotherm(K, n, q_m)

    np.testing.assert_allclose(iso_langmuir.q(C), iso_stoich.q(C))
    np.testing.assert_allclose(iso_langmuir.dq_dC(C), iso_stoich.dq_dC(C))
    np.testing.assert_allclose(iso_langmuir.d2q_dC2(C), iso_stoich.d2q_dC2(C))


def test_multi_capacity_reduces_to_langmuir():
    C = np.array([[0.3, 0.5, 0.7], [0.2, 0.4, 0.5]])
    K = np.array([0.5, 0.6])
    q_tot = 20
    q_m = np.array([20, 20])

    iso_langmuir = CompetitiveLangmuirIsotherm(K, q_tot)
    iso_multi = MultiCapacityIsotherm(K, q_m)

    np.testing.assert_allclose(iso_langmuir.q(C), iso_multi.q(C))
    np.testing.assert_allclose(iso_langmuir.dq_dC(C), iso_multi.dq_dC(C))
    np.testing.assert_allclose(iso_langmuir.d2q_dC2(C), iso_multi.d2q_dC2(C))


def test_adsorbate_complex_reduces_to_langmuir():
    C = np.array([[0.3, 0.5, 0.7], [0.2, 0.4, 0.5]])
    K = np.array([0.5, 0.6])
    q_m = 20
    K_x = 0

    iso_langmuir = CompetitiveLangmuirIsotherm(K, q_m)
    iso_ac = AdsorbateComplexIsotherm(K, q_m, K_x)

    np.testing.assert_allclose(iso_langmuir.q(C), iso_ac.q(C))
    np.testing.assert_allclose(iso_langmuir.dq_dC(C), iso_ac.dq_dC(C))
    np.testing.assert_allclose(iso_langmuir.d2q_dC2(C), iso_ac.d2q_dC2(C))


def epa_ceq(
    q,
    K,
    MW,
    valence,
    inlet_concentrations,
    q_m,
    bed_density,
    reference_concentration,
    reference_mw,
    reference_z=1,
):
    """Calculate equilibrium concentrations using EPA HSDMIX equations.

    q: resin loading, shape (n_species, n_nodes)
    Returns: EPA-equivalent concentrations, shape (n_species, n_nodes)
    """
    # Convert resin loadings to equivalent loadings.
    q_eq = q * bed_density * np.asarray(valence)[:, None] / np.asarray(MW)[:, None]
    q_ref = q_m - q_eq.sum(axis=0)

    # Total liquid-phase equivalent concentration.
    CT = (
        np.sum(np.asarray(inlet_concentrations) * np.asarray(valence) / np.asarray(MW))
        + reference_concentration * reference_z / reference_mw
    )

    K = np.asarray(K)
    valence = np.asarray(valence)

    if np.all(valence == 1) and reference_z == 1:
        # EPA calc_Ceq_mv: include the reference ion with K_ref = 1.
        q_all = np.concatenate([q_ref[None, :], q_eq], axis=0)
        K_all = np.concatenate([[1.0], K])

        alpha = K_all[:, None] / K_all[None, :]
        C_all = q_all * CT / (alpha @ q_all)

        return C_all[1:]

    # EPA calc_Ceq_dv.
    divalent = valence == 2
    monovalent = valence == 1

    a = np.sum(q_eq[divalent] / K[divalent, None], axis=0) / q_ref**2
    b = 1.0 + np.sum(q_eq[monovalent] / K[monovalent, None], axis=0) / q_ref
    c = -CT

    C_ref = 2 * c / (-b - np.sqrt(b**2 - 4 * a * c))

    return q_eq / K[:, None] * (C_ref[None, :] / q_ref[None, :]) ** valence[:, None]


@pytest.mark.parametrize(
    "valence",
    [
        np.array([1, 1]),  # EPA monovalent branch
        np.array([1, 2]),  # EPA divalent branch
        np.array([2, 1, 2]),  # Multiple divalent species
    ],
)
def test_competitive_ion_isotherm_C_matches_epa(valence):
    K = np.array([0.7, 1.8] if len(valence) == 2 else [0.7, 1.8, 1.2])
    MW = np.array([35.45, 100.0] if len(valence) == 2 else [100.0, 35.45, 150.0])
    inlet_concentrations = np.array(
        [10.0, 20.0] if len(valence) == 2 else [20.0, 10.0, 15.0]
    )

    q_m = 2.0
    bed_density = 500.0
    reference_concentration = 5.0
    reference_mw = 35.45
    reference_z = 1

    isotherm = CompetitiveIonIsotherm(
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        q_m=q_m,
        bed_density=bed_density,
        reference_concentration=reference_concentration,
        reference_mw=reference_mw,
        reference_z=reference_z,
    )

    # Several loading states, all below the total exchange capacity.
    q = np.array(
        [
            [0.01, 0.02, 0.03],
            [0.02, 0.01, 0.02],
        ]
    )
    if len(valence) == 3:
        q = np.array(
            [
                [0.01, 0.02, 0.03],
                [0.02, 0.01, 0.02],
                [0.01, 0.015, 0.02],
            ]
        )

    expected = epa_ceq(
        q=q,
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        q_m=q_m,
        bed_density=bed_density,
        reference_concentration=reference_concentration,
        reference_mw=reference_mw,
        reference_z=reference_z,
    )

    actual = isotherm.C(q)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
