import pytest

import reactormodels


def test_water_stores_properties():
    """Water should store name, density, viscosity, and temperature."""
    water = reactormodels.Water(
        name="Example water",
        density=1000.0,
        viscosity=0.001,
        temperature=25.0,
    )

    assert water.name == "Example water"
    assert water.density == 1000.0
    assert water.viscosity == 0.001
    assert water.temperature == 25.0


def test_water_accepts_positive_density_and_viscosity():
    """Water should accept valid positive density and viscosity values."""
    water = reactormodels.Water(
        name="Valid water",
        density=998.2,
        viscosity=0.00089,
        temperature=20.0,
    )

    assert water.density == pytest.approx(998.2)
    assert water.viscosity == pytest.approx(0.00089)


def test_water_rejects_nonpositive_density():
    """Water should reject zero or negative density."""
    with pytest.raises(AssertionError, match="density must be positive"):
        reactormodels.Water(
            name="Invalid water",
            density=0.0,
            viscosity=0.001,
            temperature=25.0,
        )

    with pytest.raises(AssertionError, match="density must be positive"):
        reactormodels.Water(
            name="Invalid water",
            density=-1000.0,
            viscosity=0.001,
            temperature=25.0,
        )


def test_water_rejects_nonpositive_viscosity():
    """Water should reject zero or negative viscosity."""
    with pytest.raises(AssertionError, match="viscosity must be positive"):
        reactormodels.Water(
            name="Invalid water",
            density=1000.0,
            viscosity=0.0,
            temperature=25.0,
        )

    with pytest.raises(AssertionError, match="viscosity must be positive"):
        reactormodels.Water(
            name="Invalid water",
            density=1000.0,
            viscosity=-0.001,
            temperature=25.0,
        )
