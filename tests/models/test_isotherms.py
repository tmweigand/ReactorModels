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
from reactormodels.models.isotherm import fit_isotherm


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


def test_fit_linear_isotherm_recovers_parameter():
    """Test fitting a linear isotherm to synthetic equilibrium data."""
    C = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    K = 2.5
    expected = LinearIsotherm(K=K)
    q = expected.q(C)

    fit = fit_isotherm(
        LinearIsotherm,
        C,
        q,
        initial_guess=(1.0,),
        fit_indices=(0,),
        parameter_template=(K,),
    )

    np.testing.assert_allclose(fit.K, K)


def test_fit_freundlich_isotherm_recovers_parameters():
    """Test fitting a Freundlich isotherm to synthetic equilibrium data."""
    C = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    K = 3.0
    n = 2.0
    expected = FreundlichIsotherm(K, n)
    q = expected.q(C)
    params = K, n

    isotherm = fit_isotherm(
        FreundlichIsotherm,
        C,
        q,
        initial_guess=(1.0, 1.0),
        fit_indices=(0, 1),
        parameter_template=params,
    )

    np.testing.assert_allclose(isotherm.K, K)
    np.testing.assert_allclose(isotherm.n, n)


def test_fit_isotherm_rejects_mismatched_shapes():
    """Test that fitting rejects C and q arrays with different shapes."""
    C = np.array([1.0, 2.0, 3.0])
    q = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="C and q must have the same shape."):
        fit_isotherm(
            LinearIsotherm,
            C,
            q,
            initial_guess=(1.0,),
            fit_indices=(0, 1),
            parameter_template=(1,),
        )


def test_fit_isotherm_rejects_negative_values():
    """Test that fitting rejects negative equilibrium data."""
    C = np.array([1.0, -2.0, 3.0])
    q = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="C and q values must be nonnegative."):
        fit_isotherm(
            FreundlichIsotherm,
            C,
            q,
            initial_guess=(1.0, 1.0),
            fit_indices=(0, 1),
            parameter_template=(1, 1),
        )


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


def assert_jacobian_matches_numerical(isotherm, var, eps=1e-7, rtol=1e-5, atol=1e-7):
    if var == "q":
        x = np.array([0.3, 0.5])
        function = isotherm.C
        jacobian = isotherm.dC_dq

    else:
        x = np.array([0.3, 0.5])
        function = isotherm.q
        jacobian = isotherm.dq_dC

    analytical = jacobian(x)
    numerical = np.zeros_like(analytical)

    for j in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[j] += eps
        x_minus[j] -= eps

        numerical[:, j] = (function(x_plus) - function(x_minus)) / (2 * eps)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=rtol,
        atol=atol,
    )


def assert_chain_rule_matches_numerical(isotherm, var, eps=1e-7, rtol=1e-5, atol=1e-7):
    if var == "q":
        q = np.array([0.3, 0.5])
        dqdt = np.array([0.12, -0.07])

        J = isotherm.dC_dq(q)

        analytical = J @ dqdt

        C_plus = isotherm.C(q + eps * dqdt)
        C_minus = isotherm.C(q - eps * dqdt)

        numerical = (C_plus - C_minus) / (2 * eps)

    else:
        C = np.array([0.3, 0.5])
        dCdt = np.array([0.12, -0.07])
        J = isotherm.dq_dC(C)

        analytical = J @ dCdt

        q_plus = isotherm.q(C + eps * dCdt)
        q_minus = isotherm.q(C - eps * dCdt)

        numerical = (q_plus - q_minus) / (2 * eps)

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
    bulk_density = 1.0

    isotherm = CompetitiveIonIsotherm(
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        q_m=q_m,
        bulk_density=bulk_density,
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


def assert_multi_species_fit_recovers_params(
    model, params, guess, fit_indices, fit_names
):
    isotherm = model(*params)

    if isotherm.output == "q":
        iso = isotherm.q
    else:
        iso = isotherm.C

    # experimental data
    data_in = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.3],
            [0.3, 0.5],
            [0.5, 0.2],
            [0.8, 0.7],
            [1.0, 1.0],
        ]
    )
    data_out = np.array([iso(x) for x in data_in])

    fitted = fit_isotherm(
        model,
        data_in,
        data_out,
        initial_guess=guess,
        fit_indices=fit_indices,
        parameter_template=params,
    )

    for index, name in zip(fit_indices, fit_names):
        np.testing.assert_allclose(
            getattr(fitted, name),
            params[index],
            rtol=1e-5,
            atol=1e-7,
        )


def test_adsorbate_complex_fit():
    K = np.array([0.5, 0.8])
    q_m = 10.0
    K_x = 5.0

    params = K, q_m, K_x
    fit_names = "K", "q_m", "K_x"
    fit_indices = (0, 1, 2)

    model = AdsorbateComplexIsotherm
    guess = np.array([0.4, 0.7]), 8.0, 4.0

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )


def test_competitive_freundlich_fit():
    K = np.array([0.5, 0.8])
    n = np.array([0.7, 0.9])

    params = K, n
    fit_names = "K", "n"
    fit_indices = (0, 1)

    model = CompetitiveFreundlichIsotherm
    guess = (np.array([0.4, 0.7]), np.array([1, 1]))

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )


def test_iexcm_fit():
    K = np.array([1.2, 0.8])
    MW = np.array([35.45, 40.08])
    z = np.array([1, 2])
    C_o = np.array([100.0, 50.0])
    q_m = 2.0
    rho_b = 1.0

    params = K, MW, z, C_o, q_m, rho_b
    fit_names = "K", "q_m"
    fit_indices = (0, 4)

    model = CompetitiveIonIsotherm
    guess = (np.array([0.9, 0.7]), 3)

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )


def test_competitive_langmuir_fit():
    q_m = 20
    K = np.array([0.5, 0.8])

    params = K, q_m
    fit_names = "K", "q_m"
    fit_indices = (0, 1)

    model = CompetitiveLangmuirIsotherm
    guess = (np.array([0.9, 0.7]), 10)

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )


def test_competitive_langmuir_freundlich_fit():
    q_m = 20
    K = np.array([0.5, 0.8])
    n = np.array([0.7, 0.9])

    params = K, n, q_m
    fit_names = "K", "n", "q_m"
    fit_indices = (0, 1, 2)

    model = CompetitiveLangmuirFreundlichIsotherm
    guess = (np.array([0.4, 0.9]), np.array([1, 0.6]), 10)

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )


def test_competitive_stoichiometric_fit():
    q_m = 20
    K = np.array([0.5, 0.8])
    n = np.array([0.7, 0.9])

    params = K, n, q_m
    fit_names = "K", "n", "q_m"
    fit_indices = (0, 1, 2)

    model = CompetitiveStoichiometricIsotherm
    guess = (np.array([0.4, 0.9]), np.array([1, 0.6]), 10)

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )


def test_multi_capacity_fit():
    q_m = np.array([5, 10])
    K = np.array([0.5, 0.8])

    params = K, q_m
    fit_names = "K", "q_m"
    fit_indices = (0, 1)

    model = MultiCapacityIsotherm
    guess = (np.array([0.4, 0.9]), np.array([7, 8]))

    assert_multi_species_fit_recovers_params(
        model, params, guess, fit_indices, fit_names
    )
