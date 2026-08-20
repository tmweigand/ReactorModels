import numpy as np
import pytest

from reactormodels.models import FreundlichIsotherm, CompetitiveIonIsotherm


def test_uncoupled_freundlich_jacobian():
    n = np.array([0.7, 0.9])
    K = np.array([0.5, 0.8])

    isotherm = FreundlichIsotherm(
        n=n,
        K=K,
    )

    q = np.array([0.3, 0.5])

    analytical = isotherm.dC_dq(q)

    numerical = np.zeros((2, 2))
    eps = 1e-7

    for j in range(2):
        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[j] += eps
        q_minus[j] -= eps

        numerical[:, j] = (isotherm.C(q_plus) - isotherm.C(q_minus)) / (2 * eps)

    print("C:", isotherm.C(q))
    print("\nAnalytical diagonal dC/dq:")
    print(analytical)

    print("\nNumerical full Jacobian:")
    print(numerical)

    # Diagonal terms
    np.testing.assert_allclose(
        analytical,
        np.diag(numerical),
        rtol=1e-5,
        atol=1e-7,
    )

    # Cross-species terms must be zero
    off_diagonal = numerical - np.diag(np.diag(numerical))

    np.testing.assert_allclose(
        off_diagonal,
        0.0,
        rtol=0,
        atol=1e-7,
    )


def test_coupled_freundlich_jacobian():
    n = np.array([0.7, 0.9])
    K = np.array([0.5, 0.8])

    isotherm = FreundlichIsotherm(
        n=n,
        K=K,
    )

    q = np.array([0.3, 0.5])

    # Analytical Jacobian
    analytical = isotherm.dC_dq_coupled(q)

    # Numerical Jacobian
    numerical = np.zeros((2, 2))
    eps = 1e-7

    for j in range(2):
        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[j] += eps
        q_minus[j] -= eps

        numerical[:, j] = (isotherm.C_coupled(q_plus) - isotherm.C_coupled(q_minus)) / (
            2 * eps
        )

    print("q:")
    print(q)

    print("\nC:")
    print(isotherm.C_coupled(q))

    print("\nAnalytical Jacobian:")
    print(analytical)

    print("\nNumerical Jacobian:")
    print(numerical)

    print("\nDifference:")
    print(analytical - numerical)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


def test_coupled_freundlich_chain_rule():
    n = np.array([0.7, 0.9])
    K = np.array([0.5, 0.8])

    isotherm = FreundlichIsotherm(
        n=n,
        K=K,
    )

    q = np.array([0.3, 0.5])
    dqdt = np.array([0.12, -0.07])

    # Analytical dC/dt from Jacobian
    J = isotherm.dC_dq_coupled(q)
    analytical = J @ dqdt

    # Numerical dC/dt using the chain rule directly
    eps = 1e-7

    C_plus = isotherm.C_coupled(q + eps * dqdt)
    C_minus = isotherm.C_coupled(q - eps * dqdt)

    numerical = (C_plus - C_minus) / (2 * eps)

    print("q:")
    print(q)

    print("\ndqdt:")
    print(dqdt)

    print("\nJacobian:")
    print(J)

    print("\nAnalytical dC/dt:")
    print(analytical)

    print("\nNumerical dC/dt:")
    print(numerical)

    print("\nDifference:")
    print(analytical - numerical)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


def test_iexcm_jacobian():
    K = np.array([1.2, 0.8])
    MW = np.array([35.45, 40.08])
    valence = np.array([1, 2])
    inlet_concentrations = np.array([100.0, 50.0])
    capacity = 2.0
    bulk_density = 1.0

    isotherm = CompetitiveIonIsotherm(
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        capacity=capacity,
        bulk_density=bulk_density,
    )

    # Sorbed concentrations, e.g. mg/g
    q = np.array([0.3, 0.2])

    # Analytical Jacobian
    analytical = isotherm.dC_dq_coupled(q)

    # Numerical Jacobian
    numerical = np.zeros((2, 2))
    eps = 1e-7

    for j in range(2):
        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[j] += eps
        q_minus[j] -= eps

        numerical[:, j] = (isotherm.C_coupled(q_plus) - isotherm.C_coupled(q_minus)) / (
            2 * eps
        )

    print("q:")
    print(q)

    print("\nC:")
    print(isotherm.C_coupled(q))

    print("\nAnalytical Jacobian:")
    print(analytical)

    print("\nNumerical Jacobian:")
    print(numerical)

    print("\nDifference:")
    print(analytical - numerical)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


def test_iexcm_chain_rule():
    K = np.array([1.2, 0.8])
    MW = np.array([35.45, 40.08])
    valence = np.array([1, 2])
    inlet_concentrations = np.array([100.0, 50.0])
    capacity = 2.0
    bulk_density = 1.0

    isotherm = CompetitiveIonIsotherm(
        K=K,
        MW=MW,
        valence=valence,
        inlet_concentrations=inlet_concentrations,
        capacity=capacity,
        bulk_density=bulk_density,
    )

    q = np.array([0.3, 0.2])
    dqdt = np.array([0.12, -0.07])

    # Analytical dC/dt
    J = isotherm.dC_dq_coupled(q)
    analytical = J @ dqdt

    # Numerical dC/dt
    eps = 1e-7

    C_plus = isotherm.C_coupled(q + eps * dqdt)
    C_minus = isotherm.C_coupled(q - eps * dqdt)

    numerical = (C_plus - C_minus) / (2 * eps)

    print("q:")
    print(q)

    print("\ndqdt:")
    print(dqdt)

    print("\nJacobian:")
    print(J)

    print("\nAnalytical dC/dt:")
    print(analytical)

    print("\nNumerical dC/dt:")
    print(numerical)

    print("\nDifference:")
    print(analytical - numerical)

    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )
