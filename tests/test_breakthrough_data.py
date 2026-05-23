import reactormodels
import numpy as np
import pytest

def test_breakthrough_class():

    initial_feed_concentration = 100
    midpoint_feed_concentration = 105
    compound = "PFOA"
    water_matrix = "NaCl"
    bed_volumes = [0, 100, 200, 300, 400, 500, 600]
    effluent_concentrations = [0, 0, 10, 20, 80, 100, 100]
    flow_rate = 5
    length = 2
    diameter = 0.2
    porosity = 0.4

    column = reactormodels.Column(
        length=length,
        diameter=diameter,
        porosity=porosity
    )

    column_volume = column.column_volume()

    breakthrough = reactormodels.Breakthrough(
        initial_feed_concentration=initial_feed_concentration,
        midpoint_feed_concentration=midpoint_feed_concentration,
        compound=compound,
        water_matrix=water_matrix,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate
    )

    print(breakthrough.summary(column_volume))