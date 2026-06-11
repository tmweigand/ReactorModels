import reactormodels
import numpy as np
import pytest


def test_time_conversion_class():

    feed_concentrations = [101, 103]
    compound = "PFOA"
    water_matrix = "Surface_Water"
    time = [0, 1.257, 2.513, 3.77, 5.027, 6.283, 7.54]
    bed_volumes = [0, 100, 200, 300, 400, 500, 600]
    effluent_concentrations = [0, 0, 10, 20, 80, 100, 100]
    flow_rate = 5
    length = 2
    diameter = 0.2
    porosity = 0.4

    column = reactormodels.Column(length=length, diameter=diameter, porosity=porosity)

    column_volume = column.column_volume()

    breakthrough = reactormodels.Breakthrough(
        feed_concentrations=feed_concentrations,
        compound=compound,
        water_matrix=water_matrix,
        time=time,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
    )

    print(
        f"Converted bed volumes: {np.array2string(breakthrough.time_to_bed_volumes(column_volume), precision=3)}\n"
    )
    print(
        f"Converted time: {np.array2string(breakthrough.bed_volumes_to_time(column_volume), precision=3)}\n"
    )

    assert breakthrough.bed_volumes == pytest.approx(
        breakthrough.time_to_bed_volumes(column_volume), abs=0.05
    ), f"Failed: max error = {np.abs(breakthrough.bed_volumes - breakthrough.time_to_bed_volumes(column_volume)).max():.2e}"

    assert breakthrough.time == pytest.approx(
        breakthrough.bed_volumes_to_time(column_volume), abs=0.05
    ), f"Failed: max error = {np.abs(breakthrough.time - breakthrough.bed_volumes_to_time(column_volume)).max():.2e}"
