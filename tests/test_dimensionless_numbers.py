"""Tests for dimensionless_numbers.py."""

import numpy as np
import pytest

from reactormodels import (
    Breakthrough,
    Chemical,
    Column,
    Media,
    Water,
    dimensionless_numbers,
)


@pytest.fixture
def parameter_objects():
    """Create objects used to test parameter recall."""
    media = Media(
        particle_porosity=0.3,
        particle_density=1.2,
        mean_diameter=0.001,
        sphericity=0.9,
    )

    water = Water(
        water_matrix="Test water",
        density=1.0,
        viscosity=1.0,
        temperature=25.0,
    )

    chemical = Chemical(
        compound="Test compound",
        molar_volume=100.0,
    )

    column = Column(
        length=1.2,
        porosity=0.4,
        diameter=0.2,
        media=media,
        water=water,
        chemical=chemical,
    )

    breakthrough = Breakthrough(
        column=column,
        compound="Test compound",
        feed_concentrations=100.0,
        effluent_concentrations=np.array([0.0, 25.0, 50.0, 100.0]),
        flow_rate=0.01,
        time=np.array([0.0, 10.0, 20.0, 30.0]),
        water_matrix="Test water",
    )

    return (
        column,
        media,
        water,
        chemical,
        breakthrough,
    )


def test_reynolds_number_direct():
    """Test Reynolds number using direct parameters."""
    density = 1000.0
    interstitial_velocity = 0.5
    diameter = 0.02
    viscosity = 0.001

    expected = density * interstitial_velocity * diameter / viscosity

    result = dimensionless_numbers.reynolds_number(
        density,
        interstitial_velocity,
        diameter,
        viscosity,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_reynolds_number_using_parameter_objects(
    parameter_objects,
):
    """Test Reynolds number using existing parameter classes."""
    column, media, water, _, breakthrough = parameter_objects

    superficial_velocity = breakthrough.calculate_superficial_velocity(
        breakthrough.flow_rate,
        column.cross_section_area(),
    )

    interstitial_velocity = breakthrough.calculate_interstitial_velocity(
        superficial_velocity,
        column.porosity,
    )

    expected = (
        water.density
        * media.sphericity
        * media.mean_diameter
        * interstitial_velocity
        / water.viscosity
    )

    result = dimensionless_numbers.reynolds_number(
        water=water,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_schmidt_number_using_chemical(
    parameter_objects,
):
    """Test Schmidt number using Chemical liquid diffusion."""
    _, _, water, chemical, _ = parameter_objects

    liquid_diffusion_coefficient = chemical.liquid_diffusion_coefficient(
        water.viscosity
    )

    expected = water.viscosity / (water.density * liquid_diffusion_coefficient)

    result = dimensionless_numbers.schmidt_number(
        water=water,
        chemical=chemical,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_schmidt_number_with_direct_water_properties(
    parameter_objects,
):
    """Test Schmidt using direct fluid properties and Chemical."""
    _, _, _, chemical, _ = parameter_objects

    viscosity = 0.001
    density = 1000.0

    liquid_diffusion_coefficient = chemical.liquid_diffusion_coefficient(viscosity)

    expected = viscosity / (density * liquid_diffusion_coefficient)

    result = dimensionless_numbers.schmidt_number(
        viscosity=viscosity,
        density=density,
        chemical=chemical,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_peclet_number_direct():
    """Test Peclet number from Reynolds and Schmidt numbers."""
    reynolds = 25.0
    schmidt = 1000.0

    expected = reynolds * schmidt

    result = dimensionless_numbers.peclet_number(
        reynolds=reynolds,
        schmidt=schmidt,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_peclet_number_using_parameter_objects(
    parameter_objects,
):
    """Test Peclet number using existing parameter classes."""
    column, media, water, chemical, breakthrough = parameter_objects

    superficial_velocity = breakthrough.calculate_superficial_velocity(
        breakthrough.flow_rate,
        column.cross_section_area(),
    )

    interstitial_velocity = breakthrough.calculate_interstitial_velocity(
        superficial_velocity,
        column.porosity,
    )

    reynolds = (
        water.density
        * media.sphericity
        * media.mean_diameter
        * interstitial_velocity
        / water.viscosity
    )

    liquid_diffusion_coefficient = chemical.liquid_diffusion_coefficient(
        water.viscosity
    )

    schmidt = water.viscosity / (water.density * liquid_diffusion_coefficient)

    expected = reynolds * schmidt

    result = dimensionless_numbers.peclet_number(
        water=water,
        chemical=chemical,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_chern_chien_sherwood_number():
    """Test a Sherwood correlation using supplied Re and Sc."""
    reynolds = 25.0
    schmidt = 1000.0
    bed_porosity = 0.4

    expected = (2 + 0.644 * reynolds**0.5 * schmidt ** (1 / 3)) * (
        1 + 1.5 * (1 - bed_porosity)
    )

    result = dimensionless_numbers.sherwood_number(
        method="chern_chien",
        reynolds=reynolds,
        schmidt=schmidt,
        bed_porosity=bed_porosity,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_sherwood_number_using_parameter_objects(
    parameter_objects,
):
    """Test Sherwood using recalled parameters."""
    column, media, water, chemical, breakthrough = parameter_objects

    superficial_velocity = breakthrough.calculate_superficial_velocity(
        breakthrough.flow_rate,
        column.cross_section_area(),
    )

    interstitial_velocity = breakthrough.calculate_interstitial_velocity(
        superficial_velocity,
        column.porosity,
    )

    reynolds = (
        water.density
        * media.sphericity
        * media.mean_diameter
        * interstitial_velocity
        / water.viscosity
    )

    liquid_diffusion_coefficient = chemical.liquid_diffusion_coefficient(
        water.viscosity
    )

    schmidt = water.viscosity / (water.density * liquid_diffusion_coefficient)

    expected = (2 + 0.644 * reynolds**0.5 * schmidt ** (1 / 3)) * (
        1 + 1.5 * (1 - column.porosity)
    )

    result = dimensionless_numbers.sherwood_number(
        method="chern_chien",
        water=water,
        chemical=chemical,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_sherwood_constraints():
    """Test Reynolds and Schmidt correlation constraints."""
    with pytest.raises(
        ValueError,
        match="Wilson-Geankoplis requires",
    ):
        dimensionless_numbers.sherwood_number(
            method="wilson_geankoplis",
            reynolds=0.001,
            schmidt=1000.0,
            bed_porosity=0.4,
        )


def test_invalid_inputs():
    """Test rejection of essential invalid inputs."""
    with pytest.raises(
        AssertionError,
        match="density must be positive",
    ):
        dimensionless_numbers.reynolds_number(
            density=0.0,
            interstitial_velocity=0.5,
            diameter=0.02,
            viscosity=0.001,
        )

    with pytest.raises(
        ValueError,
        match="chemical is required",
    ):
        dimensionless_numbers.schmidt_number(
            viscosity=0.001,
            density=1000.0,
        )

    with pytest.raises(
        ValueError,
        match="Unknown Sherwood method",
    ):
        dimensionless_numbers.sherwood_number(
            method="unknown",
            reynolds=25.0,
            schmidt=1000.0,
            bed_porosity=0.4,
        )
