import numpy as np
import pytest

import reactormodels


def test_chemical_stores_properties():
    """Test that Chemical stores required and optional properties."""
    chem = reactormodels.Chemical(
        name="Example chemical",
        molecular_weight=100.0,
        molar_volume=75.0,
        density=1.2,
        solubility=10.0,
        vapor_pressure=0.5,
        boiling_point=100.0,
        diffusion_parameter=2.0,
    )

    assert chem.name == "Example chemical"
    assert chem.molecular_weight == 100.0
    assert chem.molar_volume == 75.0
    assert chem.density == 1.2
    assert chem.solubility == 10.0
    assert chem.vapor_pressure == 0.5
    assert chem.boiling_point == 100.0
    assert chem.diffusion_parameter == 2.0


def test_liquid_diffusion_coefficient():
    """Test liquid diffusion coefficient using the AdDesign equation."""
    chem = reactormodels.Chemical(
        name="Example chemical",
        molecular_weight=100.0,
        molar_volume=75.0,
    )
    viscosity = 1.0

    expected = 13.26e-5 / (
        np.power(viscosity, 1.14) * np.power(chem.molar_volume, 0.589)
    )

    np.testing.assert_allclose(
        chem.liquid_diffusion_coefficient(viscosity),
        expected,
    )


def test_chemical_rejects_invalid_required_values():
    """Test that Chemical rejects invalid molecular weight and molar volume."""
    with pytest.raises(AssertionError, match="molecular_weight must be positive"):
        reactormodels.Chemical(
            name="Example chemical",
            molecular_weight=0.0,
            molar_volume=75.0,
        )

    with pytest.raises(AssertionError, match="molar_volume must be positive"):
        reactormodels.Chemical(
            name="Example chemical",
            molecular_weight=100.0,
            molar_volume=0.0,
        )


def test_liquid_diffusion_coefficient_rejects_invalid_viscosity():
    """Test that liquid diffusion coefficient rejects non-positive viscosity."""
    chem = reactormodels.Chemical(
        name="Example chemical",
        molecular_weight=100.0,
        molar_volume=75.0,
    )

    with pytest.raises(AssertionError, match="viscosity must be positive"):
        chem.liquid_diffusion_coefficient(0.0)
