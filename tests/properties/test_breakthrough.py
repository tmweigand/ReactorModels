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

    assert breakthrough.has_breakthrough(0.01)


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

    # Add known NaNs
    nan_indices = np.array([5, 10])
    c_outliers[nan_indices] = np.nan

    # Confirm NaNs were added
    assert np.all(np.isnan(c_outliers[nan_indices]))

    breakthrough = reactormodels.Breakthrough(
        column=model.breakthrough.column,
        chemical=model.breakthrough.chemical,
        feed_concentrations=model.breakthrough.feed_concentrations,
        flow_rate=model.breakthrough.flow_rate,
        time=time,
        effluent_concentrations=c_outliers,
    )

    valid_time, valid_concentration = breakthrough.valid_data()

    assert len(valid_time) == len(time) - len(nan_indices)
    assert len(valid_concentration) == len(c_outliers) - len(nan_indices)
    assert np.all(np.isfinite(valid_time))
    assert np.all(np.isfinite(valid_concentration))

    # apply outlier identification helper
    outliers, _, _ = breakthrough.identify_curve_outliers(
        valid_time,
        valid_concentration,
        n_mad=1.5,
        window_size=3,
        max_outliers=10,
        baseline_threshold=0.01,
    )

    detected_indices = np.where(outliers)

    # Every detected outlier was intentionally perturbed
    assert np.all(np.isin(detected_indices, indices))

    # At least one intentionally perturbed point was detected
    assert len(detected_indices) > 0
