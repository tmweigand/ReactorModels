import numpy as np

import reactormodels


def test_media_stores_properties():
    """Test that Media stores particle and bed properties."""
    media = reactormodels.Media(
        particle_porosity=0.35,
        particle_density=1.5,
        particle_diameter=0.001,
    )

    assert media.particle_porosity == 0.35
    assert media.particle_density == 1.5
    assert media.particle_diameter == 0.001
