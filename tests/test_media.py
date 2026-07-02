import numpy as np
import pytest

import reactormodels


def test_media_stores_properties():
    """Test that Media stores particle and bed properties."""
    media = reactormodels.Media(
        particle_porosity=0.35,
        particle_density=1.5,
        mean_diameter=0.001,
        bed_density=0.9,
    )

    assert media.particle_porosity == 0.35
    assert media.particle_density == 1.5
    assert media.mean_diameter == 0.001
    assert media.bed_density == 0.9


def test_media_get_bed_density_and_total_porosity():
    """Test bed density and total porosity calculations."""
    media = reactormodels.Media(
        particle_porosity=0.35,
        particle_density=1.5,
        mean_diameter=0.001,
    )
    bed_porosity = 0.4

    expected_bed_density = (1.0 - bed_porosity) * media.particle_density
    expected_total_porosity = (
        bed_porosity + (1.0 - bed_porosity) * media.particle_porosity
    )

    np.testing.assert_allclose(
        media.get_bed_density(bed_porosity),
        expected_bed_density,
    )
    np.testing.assert_allclose(
        media.total_porosity(bed_porosity),
        expected_total_porosity,
    )


def test_media_as_array_and_as_dict():
    """Test that media properties can be returned as an array and dictionary."""
    media = reactormodels.Media(
        particle_porosity=0.35,
        particle_density=1.5,
        mean_diameter=0.001,
        bed_density=0.9,
    )

    expected_array = np.array([0.35, 1.5, 0.001, 0.9])

    np.testing.assert_allclose(media.as_array(), expected_array)

    assert media.as_dict() == {
        "particle_porosity": 0.35,
        "particle_density": 1.5,
        "mean_diameter": 0.001,
        "bed_density": 0.9,
    }


def test_media_rejects_invalid_inputs():
    """Test that Media rejects invalid particle and bed properties."""
    with pytest.raises(AssertionError, match="particle_porosity must be in"):
        reactormodels.Media(
            particle_porosity=1.0,
            particle_density=1.5,
            mean_diameter=0.001,
        )

    with pytest.raises(AssertionError, match="particle_density must be positive"):
        reactormodels.Media(
            particle_porosity=0.35,
            particle_density=0.0,
            mean_diameter=0.001,
        )

    with pytest.raises(AssertionError, match="mean_diameter must be positive"):
        reactormodels.Media(
            particle_porosity=0.35,
            particle_density=1.5,
            mean_diameter=0.0,
        )

    media = reactormodels.Media(
        particle_porosity=0.35,
        particle_density=1.5,
        mean_diameter=0.001,
    )

    with pytest.raises(AssertionError, match="bed_porosity must be in"):
        media.get_bed_density(1.0)
