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

    return column, media, water, chemical, breakthrough


def test_reynolds_number_direct():
    """Test Reynolds number using direct parameters."""
    density = 1000.0
    interstitial_velocity = 0.5
    diameter = 0.02
    viscosity = 0.001

    expected = density * interstitial_velocity * diameter / viscosity

    result = dimensionless_numbers.reynolds_number(
        density=density,
        interstitial_velocity=interstitial_velocity,
        diameter=diameter,
        viscosity=viscosity,
    )

    np.testing.assert_allclose(result, expected)


def test_reynolds_number_using_parameter_objects(parameter_objects):
    """Test Reynolds number using existing parameter classes."""
    column, media, water, _, breakthrough = parameter_objects

    interstitial_velocity = breakthrough.interstitial_velocity()

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

    np.testing.assert_allclose(result, expected)


def test_reynolds_number_with_superficial_velocity():
    """Test conversion from superficial to interstitial velocity."""
    density = 1000.0
    superficial_velocity = 0.01
    diameter = 0.001
    viscosity = 0.001
    bed_porosity = 0.4
    sphericity = 0.9

    expected = (
        density
        * sphericity
        * diameter
        * superficial_velocity
        / (bed_porosity * viscosity)
    )

    result = dimensionless_numbers.reynolds_number(
        density=density,
        superficial_velocity=superficial_velocity,
        diameter=diameter,
        viscosity=viscosity,
        bed_porosity=bed_porosity,
        sphericity=sphericity,
    )

    np.testing.assert_allclose(result, expected)


def test_reynolds_uses_column_porosity_for_superficial_velocity(
    parameter_objects,
):
    """Test retrieval of bed porosity from Column."""
    column, _, _, _, _ = parameter_objects

    density = 1000.0
    superficial_velocity = 0.01
    diameter = 0.001
    viscosity = 0.001

    expected = density * diameter * superficial_velocity / (column.porosity * viscosity)

    result = dimensionless_numbers.reynolds_number(
        density=density,
        superficial_velocity=superficial_velocity,
        diameter=diameter,
        viscosity=viscosity,
        column=column,
    )

    np.testing.assert_allclose(result, expected)


def test_reynolds_rejects_two_velocity_types():
    """Test rejection of simultaneous velocity definitions."""
    with pytest.raises(ValueError, match="not both"):
        dimensionless_numbers.reynolds_number(
            density=1000.0,
            interstitial_velocity=0.02,
            superficial_velocity=0.01,
            diameter=0.001,
            viscosity=0.001,
            bed_porosity=0.4,
        )


def test_reynolds_requires_porosity_with_superficial_velocity():
    """Test that superficial velocity requires bed porosity."""
    with pytest.raises(ValueError, match="bed_porosity is required"):
        dimensionless_numbers.reynolds_number(
            density=1000.0,
            superficial_velocity=0.01,
            diameter=0.001,
            viscosity=0.001,
        )


@pytest.mark.parametrize("bed_porosity", [0.0, 1.0, 1.2])
def test_reynolds_rejects_invalid_porosity(bed_porosity):
    """Test rejection of invalid bed porosity."""
    with pytest.raises(
        AssertionError,
        match="bed_porosity must be between 0 and 1",
    ):
        dimensionless_numbers.reynolds_number(
            density=1000.0,
            superficial_velocity=0.01,
            diameter=0.001,
            viscosity=0.001,
            bed_porosity=bed_porosity,
        )


@pytest.mark.parametrize("sphericity", [0.0, 1.1])
def test_reynolds_rejects_invalid_sphericity(sphericity):
    """Test rejection of invalid particle sphericity."""
    with pytest.raises(AssertionError, match="sphericity must be in"):
        dimensionless_numbers.reynolds_number(
            density=1000.0,
            interstitial_velocity=0.5,
            diameter=0.001,
            viscosity=0.001,
            sphericity=sphericity,
        )


def test_schmidt_number_using_chemical(parameter_objects):
    """Test Schmidt number using Chemical liquid diffusion."""
    _, _, water, chemical, _ = parameter_objects

    diffusion_coefficient = chemical.liquid_diffusion_coefficient(water.viscosity)
    expected = water.viscosity / (water.density * diffusion_coefficient)

    result = dimensionless_numbers.schmidt_number(
        water=water,
        chemical=chemical,
    )

    np.testing.assert_allclose(result, expected)


def test_schmidt_number_with_direct_water_properties(parameter_objects):
    """Test Schmidt number using direct fluid properties and Chemical."""
    _, _, _, chemical, _ = parameter_objects

    viscosity = 0.001
    density = 1000.0
    diffusion_coefficient = chemical.liquid_diffusion_coefficient(viscosity)

    expected = viscosity / (density * diffusion_coefficient)

    result = dimensionless_numbers.schmidt_number(
        viscosity=viscosity,
        density=density,
        chemical=chemical,
    )

    np.testing.assert_allclose(result, expected)


def test_schmidt_number_with_direct_diffusion_coefficient():
    """Test Schmidt number using a supplied diffusion coefficient."""
    viscosity = 0.001
    density = 1000.0
    diffusion_coefficient = 1.0e-9

    expected = viscosity / (density * diffusion_coefficient)

    result = dimensionless_numbers.schmidt_number(
        viscosity=viscosity,
        density=density,
        diffusion_coefficient=diffusion_coefficient,
    )

    np.testing.assert_allclose(result, expected)


def test_schmidt_rejects_invalid_diffusion_coefficient():
    """Test rejection of a nonpositive diffusion coefficient."""
    with pytest.raises(
        AssertionError,
        match="diffusion_coefficient must be positive",
    ):
        dimensionless_numbers.schmidt_number(
            viscosity=0.001,
            density=1000.0,
            diffusion_coefficient=0.0,
        )


def test_peclet_number_direct():
    """Test mass-transfer Peclet number from Reynolds and Schmidt."""
    reynolds = 25.0
    schmidt = 1000.0

    expected = reynolds * schmidt

    result = dimensionless_numbers.peclet_number(
        reynolds=reynolds,
        schmidt=schmidt,
    )

    np.testing.assert_allclose(result, expected)


def test_peclet_number_using_parameter_objects(parameter_objects):
    """Test mass-transfer Peclet number using parameter classes."""
    column, media, water, chemical, breakthrough = parameter_objects

    reynolds = dimensionless_numbers.reynolds_number(
        water=water,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )
    schmidt = dimensionless_numbers.schmidt_number(
        water=water,
        chemical=chemical,
    )
    expected = reynolds * schmidt

    result = dimensionless_numbers.peclet_number(
        water=water,
        chemical=chemical,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )

    np.testing.assert_allclose(result, expected)


def test_peclet_number_axial_form():
    """Test the axial-dispersion Peclet-number form."""
    interstitial_velocity = 0.5
    length = 2.0
    axial_dispersion_coefficient = 0.1

    expected = length * interstitial_velocity / axial_dispersion_coefficient

    result = dimensionless_numbers.peclet_number(
        interstitial_velocity=interstitial_velocity,
        length=length,
        axial_dispersion_coefficient=axial_dispersion_coefficient,
    )

    np.testing.assert_allclose(result, expected)


def test_peclet_number_axial_form_using_objects(parameter_objects):
    """Test retrieval of velocity and length for axial Peclet."""
    column, _, _, _, breakthrough = parameter_objects

    axial_dispersion_coefficient = 0.02
    expected = (
        breakthrough.interstitial_velocity()
        * column.length
        / axial_dispersion_coefficient
    )

    result = dimensionless_numbers.peclet_number(
        axial_dispersion_coefficient=axial_dispersion_coefficient,
        column=column,
        breakthrough=breakthrough,
    )

    np.testing.assert_allclose(result, expected)


def test_peclet_rejects_mixed_parameter_sets():
    """Test rejection of mixed axial and mass-transfer inputs."""
    with pytest.raises(ValueError, match="not both"):
        dimensionless_numbers.peclet_number(
            interstitial_velocity=0.5,
            length=2.0,
            axial_dispersion_coefficient=0.1,
            reynolds=10.0,
            schmidt=1000.0,
        )


def test_peclet_requires_axial_dispersion_coefficient():
    """Test rejection of incomplete axial parameters."""
    with pytest.raises(
        ValueError,
        match="axial_dispersion_coefficient is required",
    ):
        dimensionless_numbers.peclet_number(
            interstitial_velocity=0.5,
            length=2.0,
        )


def test_chern_chien_sherwood_number():
    """Test the Chern-Chien Sherwood correlation."""
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

    np.testing.assert_allclose(result, expected)


def test_sherwood_number_using_parameter_objects(parameter_objects):
    """Test Sherwood number using recalled parameters."""
    column, media, water, chemical, breakthrough = parameter_objects

    reynolds = dimensionless_numbers.reynolds_number(
        water=water,
        media=media,
        column=column,
        breakthrough=breakthrough,
    )
    schmidt = dimensionless_numbers.schmidt_number(
        water=water,
        chemical=chemical,
    )
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

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("tan", 1.1 * (10.0 * 1000.0) ** (1 / 3) / 0.4),
        (
            "wilson_geankoplis",
            1.09 * (10.0 * 1000.0) ** (1 / 3) / 0.4,
        ),
        (
            "williamson",
            2.4 * 0.4 * 10.0**0.3 * 1000.0**0.42,
        ),
        (
            "ko",
            0.325 / (0.4 * 10.0**0.36 * 1000.0 ** (1 / 3)),
        ),
        (
            "wakao_funazkri",
            2 + 1.1 * 10.0**0.6 * 1000.0 ** (1 / 3),
        ),
        (
            "kataoka",
            1.85 * ((1 - 0.4) / 0.4) ** (1 / 3) * 10.0 ** (1 / 3) * 1000.0 ** (1 / 3),
        ),
    ],
)
def test_additional_sherwood_correlations(method, expected):
    """Test the additional supported Sherwood correlations."""
    result = dimensionless_numbers.sherwood_number(
        method=method,
        reynolds=10.0,
        schmidt=1000.0,
        bed_porosity=0.4,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("reynolds", "coefficient", "exponent"),
    [
        (1.0, 1.58, 0.4),
        (10.0, 1.21, 0.5),
        (500.0, 0.59, 0.6),
    ],
)
def test_ohashi_piecewise_correlations(
    reynolds,
    coefficient,
    exponent,
):
    """Test all three Ohashi Reynolds-number ranges."""
    schmidt = 1000.0
    expected = 2 + coefficient * reynolds**exponent * schmidt ** (1 / 3)

    result = dimensionless_numbers.sherwood_number(
        method="ohashi",
        reynolds=reynolds,
        schmidt=schmidt,
    )

    np.testing.assert_allclose(result, expected)


def test_gnielinski_sherwood_number():
    """Test the Gnielinski packed-bed correlation."""
    reynolds = 10.0
    schmidt = 1000.0
    bed_porosity = 0.4

    sherwood_laminar = 0.644 * reynolds**0.5 * schmidt ** (1 / 3)
    sherwood_turbulent = (
        0.037
        * reynolds**0.8
        * schmidt
        / (1 + 2.443 * reynolds ** (-0.1) * (schmidt ** (2 / 3) - 1))
    )
    expected = (2 + (sherwood_laminar**2 + sherwood_turbulent**2) ** 0.5) * (
        1 + 1.5 * (1 - bed_porosity)
    )

    result = dimensionless_numbers.sherwood_number(
        method="gnielinski",
        reynolds=reynolds,
        schmidt=schmidt,
        bed_porosity=bed_porosity,
    )

    np.testing.assert_allclose(result, expected)


def test_sherwood_method_is_case_insensitive():
    """Test normalization of the selected Sherwood method."""
    expected = 1.1 * (10.0 * 1000.0) ** (1 / 3) / 0.4

    result = dimensionless_numbers.sherwood_number(
        method="TAN",
        reynolds=10.0,
        schmidt=1000.0,
        bed_porosity=0.4,
    )

    np.testing.assert_allclose(result, expected)


def test_wilson_geankoplis_reynolds_constraint():
    """Test the Wilson-Geankoplis Reynolds constraint."""
    with pytest.raises(ValueError, match="Wilson-Geankoplis requires"):
        dimensionless_numbers.sherwood_number(
            method="wilson_geankoplis",
            reynolds=0.001,
            schmidt=1000.0,
            bed_porosity=0.4,
        )


def test_wilson_geankoplis_schmidt_constraint():
    """Test the Wilson-Geankoplis Schmidt constraint."""
    with pytest.raises(ValueError, match="950 < Schmidt"):
        dimensionless_numbers.sherwood_number(
            method="wilson_geankoplis",
            reynolds=10.0,
            schmidt=100.0,
            bed_porosity=0.4,
        )


@pytest.mark.parametrize(
    (
        "method",
        "reynolds",
        "schmidt",
        "bed_porosity",
        "message",
    ),
    [
        ("ohashi", 0.001, 1000.0, None, "Ohashi requires"),
        (
            "williamson",
            0.08,
            1000.0,
            0.4,
            "Williamson requires",
        ),
        (
            "williamson",
            10.0,
            150.0,
            0.4,
            "Williamson requires",
        ),
        (
            "wakao_funazkri",
            3.0,
            1000.0,
            None,
            "Wakao-Funazkri requires",
        ),
        (
            "kataoka",
            200.0,
            1000.0,
            0.4,
            "Kataoka requires",
        ),
        (
            "gnielinski",
            0.1,
            1000.0,
            0.4,
            "Gnielinski requires",
        ),
        (
            "gnielinski",
            1.0,
            12000.0,
            0.4,
            "Gnielinski requires",
        ),
    ],
)
def test_sherwood_correlation_constraints(
    method,
    reynolds,
    schmidt,
    bed_porosity,
    message,
):
    """Test validity constraints for Sherwood correlations."""
    with pytest.raises(ValueError, match=message):
        dimensionless_numbers.sherwood_number(
            method=method,
            reynolds=reynolds,
            schmidt=schmidt,
            bed_porosity=bed_porosity,
        )


def test_sherwood_requires_bed_porosity():
    """Test correlations that require bed porosity."""
    with pytest.raises(ValueError, match="bed_porosity is required"):
        dimensionless_numbers.sherwood_number(
            method="tan",
            reynolds=10.0,
            schmidt=1000.0,
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

    with pytest.raises(ValueError, match="chemical is required"):
        dimensionless_numbers.schmidt_number(
            viscosity=0.001,
            density=1000.0,
        )

    with pytest.raises(ValueError, match="Unknown Sherwood method"):
        dimensionless_numbers.sherwood_number(
            method="unknown",
            reynolds=25.0,
            schmidt=1000.0,
            bed_porosity=0.4,
        )
