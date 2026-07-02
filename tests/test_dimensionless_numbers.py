import numpy as np
import pytest

import reactormodels


def test_reynolds_number():
    """Test Reynolds number calculation."""
    density = 1000.0
    velocity = 0.5
    diameter = 0.02
    viscosity = 0.001

    expected = density * velocity * diameter / viscosity

    np.testing.assert_allclose(
        reactormodels.reynolds_number(density, velocity, diameter, viscosity),
        expected,
    )


def test_schmidt_number():
    """Test Schmidt number calculation."""
    viscosity = 0.001
    density = 1000.0
    diffusion_coefficient = 1.0e-9

    expected = viscosity / (density * diffusion_coefficient)

    np.testing.assert_allclose(
        reactormodels.schmidt_number(viscosity, density, diffusion_coefficient),
        expected,
    )


def test_peclet_sherwood_and_stanton_numbers():
    """Test Peclet, Sherwood, and Stanton number calculations."""
    velocity = 0.5
    length = 1.2
    diffusion_coefficient = 1.0e-9
    mass_transfer_coefficient = 2.0e-5
    diameter = 0.02

    expected_peclet = velocity * length / diffusion_coefficient
    expected_sherwood = mass_transfer_coefficient * diameter / diffusion_coefficient
    expected_stanton = mass_transfer_coefficient / velocity

    np.testing.assert_allclose(
        reactormodels.peclet_number(velocity, length, diffusion_coefficient),
        expected_peclet,
    )
    np.testing.assert_allclose(
        reactormodels.sherwood_number(
            mass_transfer_coefficient,
            diameter,
            diffusion_coefficient,
        ),
        expected_sherwood,
    )
    np.testing.assert_allclose(
        reactormodels.stanton_number(mass_transfer_coefficient, velocity),
        expected_stanton,
    )


def test_dimensionless_numbers_reject_invalid_inputs():
    """Test that dimensionless number functions reject non-positive inputs."""
    with pytest.raises(AssertionError, match="density must be positive"):
        reactormodels.reynolds_number(0.0, 0.5, 0.02, 0.001)

    with pytest.raises(AssertionError, match="diffusion_coefficient must be positive"):
        reactormodels.schmidt_number(0.001, 1000.0, 0.0)

    with pytest.raises(AssertionError, match="length must be positive"):
        reactormodels.peclet_number(0.5, 0.0, 1.0e-9)

    with pytest.raises(
        AssertionError,
        match="mass_transfer_coefficient must be positive",
    ):
        reactormodels.sherwood_number(0.0, 0.02, 1.0e-9)

    with pytest.raises(AssertionError, match="velocity must be positive"):
        reactormodels.stanton_number(2.0e-5, 0.0)
