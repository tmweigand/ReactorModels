"""test_film_transfer.py"""

import pytest
import reactormodels


def test_film_tranfer():

    water = reactormodels.Water(density=1000, viscosity=1e-4)

    br = reactormodels.fixtures.make_breakthrough(water=water, superficial_velocity=100)

    film_transfer = reactormodels.FilmTransfer(br)

    assert film_transfer.k_film == pytest.approx(7.698598390771386)
