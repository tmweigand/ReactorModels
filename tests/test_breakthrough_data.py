import reactormodels

import numpy as np
import pytest


def test_breakthrough_class():

    initial_feed_concentration: 100
    midpoint_feed_concentration: 105
    compound: "PFOA"
    water_matrix: "NaCl"
    bed_volumes: [0, 100, 200, 300, 400, 500]
    effluent_concentrations: [0, 0, 0.1, 0.3, 0.7, 1]
    flow_rate: 5
    length: 2
    diameter: 0.2

    column = reactormodels.Column(
        length=length, diameter=diameter
    )
    
    breakthrough = reactormodels.Breakthrough(
        initial_feed_concentration=initial_feed_concentration,
        midpoint_feed_concentration=midpoint_feed_concentration,
        compound=compound,
        water_matrix=water_matrix,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        flow_rate=flow_rate,
        column_volume = column.column_volume
    )

    breakthrough.summary()