"""Test Input.py."""

from pathlib import Path
import numpy as np
import pytest
import reactormodels
from reactormodels.Input.Input import load_input_file

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "io" / "Input_data_test"
PARAMETER_FILE = DATA_DIRECTORY / "input_parameter_test.xlsx"
BREAKTHROUGH_FILE = DATA_DIRECTORY / "input_breakthrough_test.xlsx"
ISOTHERM_FILE = DATA_DIRECTORY / "input_isotherm_test.xlsx"


def test_load_input_file():
    """Test loading experimental input data against Excel values."""
    data = load_input_file(
        parameter_file=PARAMETER_FILE,
        breakthrough_file=BREAKTHROUGH_FILE,
        breakthrough_sheet="effluent_concentration",
        isotherm_file=ISOTHERM_FILE,
    )

    water = data["water"]

    assert isinstance(water, reactormodels.Water)

    assert water.name == "water"
    assert water.density == pytest.approx(997.0)
    assert water.viscosity == pytest.approx(0.00089)
    assert water.temperature == pytest.approx(25.0)

    media = data["media"]
    assert isinstance(media, reactormodels.Media)

    assert media.bed_density == pytest.approx(500.0)
    assert media.sphericity == pytest.approx(1.0)
    assert media.particle_porosity == pytest.approx(0.5)
    assert media.particle_density == pytest.approx(500.0)
    assert media.particle_radius == pytest.approx(2.5e-5)
    assert media.particle_diameter == pytest.approx(5.0e-5)

    column = data["column"]

    assert isinstance(column, reactormodels.Column)

    assert column.length == pytest.approx(0.1)
    assert column.diameter == pytest.approx(0.05)
    assert column.porosity == pytest.approx(0.4)

    assert column.bulk_density is None
    assert column.sorbent_mass is None
    assert column.water is water
    assert column.media is media

    assert set(data["chemicals"]) == {"chemical1"}

    chemical = data["chemicals"]["chemical1"]
    assert isinstance(
        chemical,
        reactormodels.Chemical,
    )
    assert chemical.name == "chemical 1"
    assert chemical.axial_diffusion is None
    assert chemical.molecular_weight == pytest.approx(200.0)
    assert chemical.molar_volume == pytest.approx(133.33333333333334)
    assert chemical.density == pytest.approx(1.5)
    assert chemical.solubility == pytest.approx(0.05)
    assert chemical.vapor_pressure == pytest.approx(0.0012)
    assert chemical.boiling_point == pytest.approx(173.0)
    assert set(data["breakthroughs"]) == {"chemical1"}
    breakthrough = data["breakthroughs"]["chemical1"]
    assert isinstance(
        breakthrough,
        reactormodels.Breakthrough,
    )
    assert breakthrough.column is column
    assert breakthrough.chemical is chemical

    assert breakthrough.flow_rate == pytest.approx(0.1)
    expected_concentrations = np.array(
        [
            0.10,
            0.09,
            0.08,
            0.07,
            0.06,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
            0.00,
        ]
    )

    np.testing.assert_allclose(
        breakthrough.feed_concentrations,
        expected_concentrations,
    )

    np.testing.assert_allclose(
        breakthrough.effluent_concentrations,
        expected_concentrations,
    )

    np.testing.assert_allclose(
        breakthrough.bed_volumes,
        np.array(
            [
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
            ]
        ),
    )
    np.testing.assert_allclose(
        breakthrough.time,
        np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
            ]
        ),
    )

    assert breakthrough.initial_concentration == pytest.approx(1.0)

    assert breakthrough._superficial_velocity == pytest.approx(0.001)
    assert breakthrough.initial_mass_fraction == pytest.approx(0.5)

    assert set(data["isotherms"]) == {"chemical1"}

    chemical_isotherms = data["isotherms"]["chemical1"]

    assert set(chemical_isotherms) == {
        "linear",
        "freundlich",
        "langmuir",
    }

    linear = chemical_isotherms["linear"]

    assert isinstance(
        linear,
        reactormodels.models.LinearIsotherm,
    )
    assert linear.K == pytest.approx(1.0)

    freundlich = chemical_isotherms["freundlich"]

    assert isinstance(
        freundlich,
        reactormodels.models.FreundlichIsotherm,
    )
    assert freundlich.K == pytest.approx(1.0)
    assert freundlich.n == pytest.approx(1.0)
    langmuir = chemical_isotherms["langmuir"]
    assert isinstance(
        langmuir,
        reactormodels.models.LangmuirIsotherm,
    )
    assert langmuir.K == pytest.approx(1.0)
    assert langmuir.q_m == pytest.approx(10.0)
