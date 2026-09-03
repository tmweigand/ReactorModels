import reactormodels
import numpy as np
import pytest


def test_breakthrough_class():

    feed_concentrations = [101, 99]
    bed_volumes = np.array([0, 100, 200, 250, 350, 400, 500, 600])
    time = np.array(bed_volumes) * np.pi / 8
    effluent_concentrations = np.array([0, 0, 10, 15, 25, 80, 100, 100])
    flow_rate = 2.5
    length = 5
    diameter = 0.5
    porosity = 0.4
    threshold = 0.2

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        media=reactormodels.Media,
        water=reactormodels.Water,
    )

    column_volume = column.column_volume()

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical,
        feed_concentrations=feed_concentrations,
        time=time,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
    )

    assert breakthrough.mean_feed_concentration() == pytest.approx(100, abs=1e-5)

    expected = [0, 0, 0.1, 0.15, 0.25, 0.8, 1, 1]
    assert breakthrough.normalize_concentration() == pytest.approx(expected, abs=1e-5)

    expected = np.pi / 8
    assert breakthrough.empty_bed_contact_time(column_volume) == expected

    breakthrough.time_to_bed_volumes(column_volume)
    assert breakthrough.bed_volumes == pytest.approx(bed_volumes, abs=0.05)

    breakthrough.bed_volumes_to_time(column_volume)
    assert breakthrough.time == pytest.approx(time, abs=0.05)

    threshold, index = breakthrough.breakthrough_threshold(
        column_volume, threshold, return_index=True
    )
    assert threshold == pytest.approx(300, abs=1e-5)
    assert index == 4

    assert breakthrough.has_breakthrough(5, 0.01)


def test_breakthough_time():

    feed_concentrations = [101, 99]
    bed_volumes = np.array([0, 100, 200, 250, 350, 400, 500, 600], dtype=float)
    time = np.array(bed_volumes) * np.pi / 8
    effluent_concentrations = np.array([0, 0, 10, 15, 25, 80, 100, 100])
    flow_rate = 2.5
    length = 5
    diameter = 0.5
    porosity = 0.4

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        media=reactormodels.Media,
        water=reactormodels.Water,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical,
        feed_concentrations=feed_concentrations,
        time=time,
        bed_volumes=None,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
    )

    np.testing.assert_array_almost_equal(breakthrough.bed_volumes, bed_volumes)


def test_breakthough_bed_volumes():

    feed_concentrations = [101, 99]
    bed_volumes = np.array([0, 100, 200, 250, 350, 400, 500, 600], dtype=float)
    time = np.array(bed_volumes) * np.pi / 8
    effluent_concentrations = np.array([0, 0, 10, 15, 25, 80, 100, 100])
    flow_rate = 2.5
    length = 5
    diameter = 0.5
    porosity = 0.4

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity,
        media=reactormodels.Media,
        water=reactormodels.Water,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=reactormodels.Chemical,
        feed_concentrations=feed_concentrations,
        time=None,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
    )

    np.testing.assert_array_almost_equal(breakthrough.time, time)
