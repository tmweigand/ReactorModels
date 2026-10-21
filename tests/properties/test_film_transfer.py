"""test_film_transfer.py"""

import pytest
import reactormodels


def test_film_tranfer():

    water = reactormodels.Water(density=1000, viscosity=1e-2)

    chemical = reactormodels.Chemical(molar_volume=1)

    br = reactormodels.fixtures.make_breakthrough(
        water=water, chemical=chemical, superficial_velocity=10
    )

    film_transfer = reactormodels.FilmTransfer(br, method="gnielinski")
    assert film_transfer.k_film == pytest.approx(2.13906, rel=1e-5, abs=1e-8)

    film_transfer = reactormodels.FilmTransfer(br, k_film=8.04)
    assert film_transfer.k_film == pytest.approx(8.04, rel=1e-5, abs=1e-8)
