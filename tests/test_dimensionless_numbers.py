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


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        ("density", "density must be positive"),
        ("interstitial_velocity", "interstitial_velocity must be positive"),
        ("diameter", "diameter must be positive"),
        ("viscosity", "viscosity must be positive"),
    ],
)
def test_reynolds_rejects_nonpositive_parameters(parameter, message):
    """Test rejection of nonpositive Reynolds-number parameters."""
    parameters = {
        "density": 1000.0,
        "interstitial_velocity": 0.5,
        "diameter": 0.02,
        "viscosity": 0.001,
    }
    parameters[parameter] = 0.0

    with pytest.raises(AssertionError, match=message):
        dimensionless_numbers.reynolds_number(**parameters)


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


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        ("viscosity", "viscosity must be positive"),
        ("density", "density must be positive"),
        (
            "diffusion_coefficient",
            "diffusion_coefficient must be positive",
        ),
    ],
)
def test_schmidt_rejects_nonpositive_parameters(parameter, message):
    """Test rejection of nonpositive Schmidt-number parameters."""
    parameters = {
        "viscosity": 0.001,
        "density": 1000.0,
        "diffusion_coefficient": 1.0e-9,
    }
    parameters[parameter] = 0.0

    with pytest.raises(AssertionError, match=message):
        dimensionless_numbers.schmidt_number(**parameters)


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
    ("parameters", "message"),
    [
        (
            {
                "length": 2.0,
                "axial_dispersion_coefficient": 0.1,
            },
            "interstitial_velocity is required",
        ),
        (
            {
                "interstitial_velocity": 0.5,
                "axial_dispersion_coefficient": 0.1,
            },
            "length is required",
        ),
        (
            {
                "interstitial_velocity": 0.5,
                "length": 2.0,
            },
            "axial_dispersion_coefficient is required",
        ),
    ],
)
def test_peclet_requires_complete_axial_parameters(parameters, message):
    """Test rejection of incomplete axial Peclet parameters."""
    with pytest.raises(ValueError, match=message):
        dimensionless_numbers.peclet_number(**parameters)


def test_peclet_requires_reynolds():
    """Test that the mass-transfer form requires Reynolds number."""
    with pytest.raises(ValueError, match="reynolds is required"):
        dimensionless_numbers.peclet_number()


def test_peclet_requires_schmidt():
    """Test that the mass-transfer form requires Schmidt number."""
    with pytest.raises(ValueError, match="schmidt is required"):
        dimensionless_numbers.peclet_number(reynolds=10.0)


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        (
            "interstitial_velocity",
            "interstitial_velocity must be positive",
        ),
        ("length", "length must be positive"),
        (
            "axial_dispersion_coefficient",
            "axial_dispersion_coefficient must be positive",
        ),
    ],
)
def test_axial_peclet_rejects_nonpositive_parameters(parameter, message):
    """Test rejection of nonpositive axial Peclet parameters."""
    parameters = {
        "interstitial_velocity": 0.5,
        "length": 2.0,
        "axial_dispersion_coefficient": 0.1,
    }
    parameters[parameter] = 0.0

    with pytest.raises(AssertionError, match=message):
        dimensionless_numbers.peclet_number(**parameters)


@pytest.mark.parametrize(
    ("reynolds", "schmidt", "message"),
    [
        (0.0, 1000.0, "reynolds must be positive"),
        (10.0, 0.0, "schmidt must be positive"),
    ],
)
def test_mass_transfer_peclet_rejects_nonpositive_parameters(
    reynolds,
    schmidt,
    message,
):
    """Test rejection of nonpositive mass-transfer parameters."""
    with pytest.raises(AssertionError, match=message):
        dimensionless_numbers.peclet_number(
            reynolds=reynolds,
            schmidt=schmidt,
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


@pytest.mark.parametrize(
    "method",
    [
        "tan",
        "wilson_geankoplis",
        "williamson",
        "ko",
        "kataoka",
        "chern_chien",
        "gnielinski",
    ],
)
def test_sherwood_requires_bed_porosity(method):
    """Test correlations that require bed porosity."""
    with pytest.raises(ValueError, match="bed_porosity is required"):
        dimensionless_numbers.sherwood_number(
            method=method,
            reynolds=10.0,
            schmidt=1000.0,
        )


@pytest.mark.parametrize(
    "method",
    [
        "tan",
        "wilson_geankoplis",
        "williamson",
        "ko",
        "kataoka",
        "chern_chien",
        "gnielinski",
    ],
)
@pytest.mark.parametrize("bed_porosity", [0.0, 1.0])
def test_sherwood_rejects_invalid_bed_porosity(method, bed_porosity):
    """Test rejection of invalid bed porosity."""
    with pytest.raises(
        AssertionError,
        match="bed_porosity must be between 0 and 1",
    ):
        dimensionless_numbers.sherwood_number(
            method=method,
            reynolds=10.0,
            schmidt=1000.0,
            bed_porosity=bed_porosity,
        )


@pytest.mark.parametrize(
    ("reynolds", "schmidt", "message"),
    [
        (0.0, 1000.0, "reynolds must be positive"),
        (10.0, 0.0, "schmidt must be positive"),
    ],
)
def test_sherwood_rejects_nonpositive_inputs(
    reynolds,
    schmidt,
    message,
):
    """Test rejection of nonpositive Reynolds or Schmidt numbers."""
    with pytest.raises(AssertionError, match=message):
        dimensionless_numbers.sherwood_number(
            method="ohashi",
            reynolds=reynolds,
            schmidt=schmidt,
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
