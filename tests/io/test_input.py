"""Test Input.py."""

from pathlib import Path
import numpy as np
import pytest
import reactormodels
from reactormodels.Input.Input import load_input_file, identify_curve_outliers

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "io" / "input_data_test"
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


def _make_particle(
    Ds: float = 5e-9,
    C_in: float = 1,
    time: np.ndarray | None = None,
) -> reactormodels.models.PSDM:
    """Create a PSDM model for testing."""
    # Particle properties
    particle_porosity = 0.5
    particle_density = 600  # g/mL
    particle_diameter = 0.07  # cm
    pore_diffusion = 5e-6  # cm²/s
    k_film = 0.1  # cm/s

    # Column properties
    axial_diffusion = 0
    K = 100  # (mg/g) * (L/mg)
    initial_concentration = 0
    length = 100  # cm
    diameter = 10  # cm
    porosity = 0.334
    bulk_density = 399.8  # g/mL
    flow_rate = 40  # cm³/s

    isotherm = reactormodels.models.LinearIsotherm(K=K)

    media = reactormodels.Media(
        particle_porosity=particle_porosity,
        particle_diameter=particle_diameter,
        particle_density=particle_density,
    )

    column = reactormodels.Column(
        length=length,
        porosity=porosity,
        diameter=diameter,
        bulk_density=bulk_density,
        media=media,
        water=reactormodels.Water(),
    )

    chemical = reactormodels.Chemical(
        axial_diffusion=axial_diffusion,
        pore_diffusion=pore_diffusion,
        surface_diffusion=Ds,
    )

    breakthrough = reactormodels.Breakthrough(
        column=column,
        chemical=chemical,
        feed_concentrations=C_in,
        initial_concentration=initial_concentration,
        flow_rate=flow_rate,
        time=time,
    )

    column_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=column.length,
        n_interior_points=3,
        n_elements=8,
        add_inlet=True,
    )

    particle_numerics = reactormodels.numerics.NumericsConfig(
        domain_length=media.particle_radius,
        n_interior_points=3,
        n_elements=1,
        add_inlet=True,
    )

    return reactormodels.models.PSDM(
        isotherm=isotherm,
        breakthrough=breakthrough,
        column_numerics=column_numerics,
        particle_numerics=particle_numerics,
        k_film=k_film,
    )


def test_identify_curve_outliers():
    time = np.linspace(1e-10, 25 * 1440 * 60, 50)

    model = _make_particle(time=time)

    _, _, C, _ = model.solve()

    # add noise every third point
    rng = np.random.default_rng(0)
    c = np.maximum(C[:, -1], 0)
    c_outliers = c.copy()

    indices = np.arange(0, c.size, 3)

    # make proportional to concentration
    noise = rng.normal(0, 0.7 * c[indices], size=indices.size)
    c_outliers[indices] += noise

    # apply outlier identification helper
    outliers, _, removed = identify_curve_outliers(
        time,
        c_outliers,
        absolute_tolerance=0.02,
        relative_tolerance=0.4,
        window_size=5,
        max_outliers=10,
    )

    detected_indices = np.where(outliers)

    # Every detected outlier was intentionally perturbed
    assert np.all(np.isin(detected_indices, indices))

    # At least one intentionally perturbed point was detected
    assert len(detected_indices) > 0

    # Outliers are where noise was introduced
    assert np.all(np.isin(np.where(outliers), indices))
