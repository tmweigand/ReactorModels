import numpy as np
import pytest

from reactormodels import dimensionless_numbers as dn


def test_reynolds_number():
    """Test Reynolds number calculation using interstitial velocity."""
    density = 1000.0
    interstitial_velocity = 0.5
    diameter = 0.02
    viscosity = 0.001

    expected = density * interstitial_velocity * diameter / viscosity

    np.testing.assert_allclose(
        dn.reynolds_number(
            density,
            interstitial_velocity,
            diameter,
            viscosity,
        ),
        expected,
    )


def test_packed_bed_reynolds_number():
    """Test packed-bed Reynolds number calculation."""
    density = 1000.0
    superficial_velocity = 0.01
    particle_diameter = 0.001
    viscosity = 0.001
    bed_porosity = 0.4
    sphericity = 1.0

    expected = (
        density
        * sphericity
        * particle_diameter
        * superficial_velocity
        / (bed_porosity * viscosity)
    )

    np.testing.assert_allclose(
        dn.packed_bed_reynolds_number(
            density,
            superficial_velocity,
            particle_diameter,
            viscosity,
            bed_porosity,
            sphericity,
        ),
        expected,
    )


def test_schmidt_number():
    """Test Schmidt number calculation."""
    viscosity = 0.001
    density = 1000.0
    diffusion_coefficient = 1.0e-9

    expected = viscosity / (density * diffusion_coefficient)

    np.testing.assert_allclose(
        dn.schmidt_number(
            viscosity,
            density,
            diffusion_coefficient,
        ),
        expected,
    )


def test_sherwood_number():
    """Test Sherwood number calculation."""
    mass_transfer_coefficient = 2.0e-5
    characteristic_length = 0.02
    diffusion_coefficient = 1.0e-9

    expected = mass_transfer_coefficient * characteristic_length / diffusion_coefficient

    np.testing.assert_allclose(
        dn.sherwood_number(
            mass_transfer_coefficient,
            characteristic_length,
            diffusion_coefficient,
        ),
        expected,
    )


def test_gnielinski_sherwood_number():
    """Test packed-bed Gnielinski Sherwood number correlation."""
    reynolds = 25.0
    schmidt = 1000.0
    bed_porosity = 0.4

    expected = (1 + 1.5 * (1 - bed_porosity)) * (
        2 + 0.644 * reynolds**0.5 * schmidt ** (1 / 3)
    )

    np.testing.assert_allclose(
        dn.gnielinski_sherwood_number(
            reynolds,
            schmidt,
            bed_porosity,
        ),
        expected,
    )


def test_film_transfer_coefficient_from_sherwood():
    """Test external film mass-transfer coefficient from Sherwood number."""
    sherwood = 50.0
    diffusion_coefficient = 1.0e-9
    particle_diameter = 0.001

    expected = sherwood * diffusion_coefficient / particle_diameter

    np.testing.assert_allclose(
        dn.film_transfer_coefficient_from_sherwood(
            sherwood,
            diffusion_coefficient,
            particle_diameter,
        ),
        expected,
    )


def test_peclet_number():
    """Test axial Peclet number calculation using interstitial velocity."""
    interstitial_velocity = 0.5
    length = 1.2
    axial_dispersion_coefficient = 2.0e-5

    expected = interstitial_velocity * length / axial_dispersion_coefficient

    np.testing.assert_allclose(
        dn.peclet_number(
            interstitial_velocity,
            length,
            axial_dispersion_coefficient,
        ),
        expected,
    )


def test_packed_bed_mass_transfer_peclet_number():
    """Test packed-bed mass-transfer Peclet number."""
    reynolds = 25.0
    schmidt = 1000.0

    expected = reynolds * schmidt

    np.testing.assert_allclose(
        dn.packed_bed_mass_transfer_peclet_number(
            reynolds,
            schmidt,
        ),
        expected,
    )


def test_stanton_number():
    """Test fixed-bed Stanton number calculation."""
    mass_transfer_coefficient = 2.0e-5
    residence_time = 300.0
    bed_porosity = 0.4
    particle_radius = 0.0005

    expected = (
        mass_transfer_coefficient
        * residence_time
        * (1 - bed_porosity)
        / (bed_porosity * particle_radius)
    )

    np.testing.assert_allclose(
        dn.stanton_number(
            mass_transfer_coefficient,
            residence_time,
            bed_porosity,
            particle_radius,
        ),
        expected,
    )


def test_stanton_number_from_ebct():
    """Test fixed-bed Stanton number using EBCT."""
    mass_transfer_coefficient = 2.0e-5
    empty_bed_contact_time = 750.0
    bed_porosity = 0.4
    particle_radius = 0.0005

    expected = (
        mass_transfer_coefficient
        * empty_bed_contact_time
        * (1 - bed_porosity)
        / particle_radius
    )

    np.testing.assert_allclose(
        dn.stanton_number_from_ebct(
            mass_transfer_coefficient,
            empty_bed_contact_time,
            bed_porosity,
            particle_radius,
        ),
        expected,
    )


def test_helfferich_number():
    """Test Helfferich number calculation for ion exchange."""
    total_resin_capacity = 1000.0
    intraparticle_diffusion_coefficient = 1.0e-11
    film_thickness = 1.0e-5
    liquid_concentration = 1.0
    liquid_diffusion_coefficient = 1.0e-9
    particle_radius = 0.0005
    separation_factor = 2.0

    expected = (
        total_resin_capacity
        * intraparticle_diffusion_coefficient
        * film_thickness
        / (liquid_concentration * liquid_diffusion_coefficient * particle_radius)
        * (5 + 2 * separation_factor)
    )

    np.testing.assert_allclose(
        dn.helfferich_number(
            total_resin_capacity,
            intraparticle_diffusion_coefficient,
            film_thickness,
            liquid_concentration,
            liquid_diffusion_coefficient,
            particle_radius,
            separation_factor,
        ),
        expected,
    )


def test_dimensionless_numbers_reject_invalid_inputs():
    """Test that dimensionless number functions reject invalid inputs."""
    with pytest.raises(AssertionError, match="density must be positive"):
        dn.reynolds_number(0.0, 0.5, 0.02, 0.001)

    with pytest.raises(AssertionError, match="interstitial_velocity must be positive"):
        dn.reynolds_number(1000.0, 0.0, 0.02, 0.001)

    with pytest.raises(AssertionError, match="bed_porosity must be between 0 and 1"):
        dn.packed_bed_reynolds_number(
            1000.0,
            0.01,
            0.001,
            0.001,
            1.0,
        )

    with pytest.raises(AssertionError, match="diffusion_coefficient must be positive"):
        dn.schmidt_number(0.001, 1000.0, 0.0)

    with pytest.raises(
        AssertionError,
        match="mass_transfer_coefficient must be positive",
    ):
        dn.sherwood_number(0.0, 0.02, 1.0e-9)

    with pytest.raises(AssertionError, match="bed_porosity must be between 0 and 1"):
        dn.gnielinski_sherwood_number(25.0, 1000.0, 0.0)

    with pytest.raises(AssertionError, match="sherwood must be positive"):
        dn.film_transfer_coefficient_from_sherwood(
            0.0,
            1.0e-9,
            0.001,
        )

    with pytest.raises(
        AssertionError,
        match="interstitial_velocity must be positive",
    ):
        dn.peclet_number(0.0, 1.2, 2.0e-5)

    with pytest.raises(
        AssertionError,
        match="axial_dispersion_coefficient must be positive",
    ):
        dn.peclet_number(0.5, 1.2, 0.0)

    with pytest.raises(AssertionError, match="reynolds must be positive"):
        dn.packed_bed_mass_transfer_peclet_number(0.0, 1000.0)

    with pytest.raises(AssertionError, match="residence_time must be positive"):
        dn.stanton_number(2.0e-5, 0.0, 0.4, 0.0005)

    with pytest.raises(AssertionError, match="empty_bed_contact_time must be positive"):
        dn.stanton_number_from_ebct(2.0e-5, 0.0, 0.4, 0.0005)

    with pytest.raises(AssertionError, match="separation_factor must be positive"):
        dn.helfferich_number(
            1000.0,
            1.0e-11,
            1.0e-5,
            1.0,
            1.0e-9,
            0.0005,
            0.0,
        )
