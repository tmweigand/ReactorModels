import numpy as np
import pytest

from reactormodels import dimensionless_numbers


def test_reynolds_number():
    """Test the Reynolds-number calculation."""
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


def test_reynolds_number_with_sphericity():
    """Test the packed-bed Reynolds number with particle sphericity."""
    density = 1000.0
    interstitial_velocity = 0.01
    diameter = 0.001
    viscosity = 0.001
    sphericity = 0.9

    expected = density * sphericity * interstitial_velocity * diameter / viscosity

    result = dimensionless_numbers.reynolds_number(
        density=density,
        interstitial_velocity=interstitial_velocity,
        diameter=diameter,
        viscosity=viscosity,
        sphericity=sphericity,
    )

    np.testing.assert_allclose(result, expected)


def test_schmidt_number():
    """Test the Schmidt-number calculation."""
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


def test_peclet_number_from_reynolds_and_schmidt():
    """Test the mass-transfer Peclet-number form."""
    reynolds = 25.0
    schmidt = 1000.0

    result = dimensionless_numbers.peclet_number(
        reynolds=reynolds,
        schmidt=schmidt,
    )

    np.testing.assert_allclose(result, reynolds * schmidt)


def test_axial_peclet_number():
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


def test_peclet_rejects_mixed_parameter_sets():
    """Test rejection of mixed Peclet-number definitions."""
    with pytest.raises(ValueError, match="not both"):
        dimensionless_numbers.peclet_number(
            interstitial_velocity=0.5,
            length=2.0,
            axial_dispersion_coefficient=0.1,
            reynolds=10.0,
            schmidt=1000.0,
        )


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
        (
            "chern_chien",
            (2 + 0.644 * 10.0**0.5 * 1000.0 ** (1 / 3)) * (1 + 1.5 * (1 - 0.4)),
        ),
    ],
)
def test_sherwood_correlations(method, expected):
    """Test Sherwood-number correlations with direct inputs."""
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


@pytest.mark.parametrize(
    ("method", "reynolds", "schmidt", "message"),
    [
        (
            "wilson_geankoplis",
            0.001,
            1000.0,
            "Wilson-Geankoplis requires",
        ),
        (
            "wilson_geankoplis",
            10.0,
            100.0,
            "950 < Schmidt",
        ),
        ("ohashi", 0.001, 1000.0, "Ohashi requires"),
        ("williamson", 0.08, 1000.0, "Williamson requires"),
        ("williamson", 10.0, 150.0, "Williamson requires"),
        (
            "wakao_funazkri",
            3.0,
            1000.0,
            "Wakao-Funazkri requires",
        ),
        ("kataoka", 200.0, 1000.0, "Kataoka requires"),
        ("gnielinski", 0.1, 1000.0, "Gnielinski requires"),
        ("gnielinski", 1.0, 12000.0, "Gnielinski requires"),
    ],
)
def test_sherwood_correlation_constraints(
    method,
    reynolds,
    schmidt,
    message,
):
    """Test the applicability limits of Sherwood correlations."""
    with pytest.raises(ValueError, match=message):
        dimensionless_numbers.sherwood_number(
            method=method,
            reynolds=reynolds,
            schmidt=schmidt,
            bed_porosity=0.4,
        )


def test_sherwood_rejects_unknown_method():
    """Test rejection of an unsupported Sherwood correlation."""
    with pytest.raises(ValueError, match="Unknown Sherwood method"):
        dimensionless_numbers.sherwood_number(
            method="unknown",
            reynolds=25.0,
            schmidt=1000.0,
            bed_porosity=0.4,
        )
