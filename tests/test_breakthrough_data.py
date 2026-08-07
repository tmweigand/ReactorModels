import reactormodels
import numpy as np
import pytest


def test_breakthrough_class():

    feed_concentrations = [101, 99]
    compound = "PFOA"
    water_matrix = "Surface_Water"
    bed_volumes = np.array([0, 100, 200, 250, 350, 400, 500, 600])
    time = np.array(bed_volumes) * np.pi / 8
    effluent_concentrations = np.array([0, 0, 10, 15, 25, 80, 100, 100])
    flow_rate = 2.5
    length = 5
    diameter = 0.5
    porosity = 0.5
    threshold = 0.2
    bulk_density = 0.5

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        bulk_density=bulk_density,
        media=reactormodels.Media(),
        water=reactormodels.Water(),
        chemical=reactormodels.Chemical(),
    )

    breakthrough = reactormodels.Breakthrough(
        feed_concentrations=feed_concentrations,
        compound=compound,
        time=time,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
        column=column,
    )

    assert breakthrough.column.column_volume() == pytest.approx(
        5 * np.pi * 0.25**2, abs=1e-5
    )
    assert breakthrough.column.get_bulk_density() == pytest.approx(0.5, abs=1e-5)
    assert breakthrough.column.get_particle_density() == pytest.approx(1, abs=1e-5)
    assert breakthrough.column.get_sorbent_mass() == pytest.approx(
        0.5 * 5 * np.pi * 0.25**2, abs=1e-5
    )

    assert breakthrough.mean_feed_concentration() == pytest.approx(100, abs=1e-5)

    expected = [0, 0, 0.1, 0.15, 0.25, 0.8, 1, 1]
    assert breakthrough.normalize_concentration() == pytest.approx(expected, abs=1e-5)

    expected = np.pi / 8
    assert breakthrough.empty_bed_contact_time() == expected

    breakthrough.time_to_bed_volumes()
    assert breakthrough.bed_volumes == pytest.approx(bed_volumes, abs=0.05)

    breakthrough.bed_volumes_to_time()
    assert breakthrough.time == pytest.approx(time, abs=0.05)

    threshold, index = breakthrough.breakthrough_threshold(threshold, return_index=True)
    assert threshold == pytest.approx(300, abs=1e-5)
    assert index == 4

    assert breakthrough.has_breakthrough(0.01)
