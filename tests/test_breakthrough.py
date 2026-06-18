import reactormodels
import numpy as np
import pytest


def _base_curve():
    """Shared setup for breakthrough curve tests."""
    chemical_name = "Example chemical"
    feed_concentration = 10.0
    bed_volumes = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    effluent_concentrations = np.array([0.0, 0.2, 0.8, 2.0])
    return reactormodels.breakthrough_data(
        chemical_name=chemical_name,
        bed_volumes=bed_volumes,
        effluent_concentrations=effluent_concentrations,
        feed_concentration=feed_concentration,
    )


def _base_time_curve():
    """Shared setup for breakthrough curve tests using time."""
    chemical_name = "Example chemical"
    feed_concentration = 10.0
    times = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    effluent_concentrations = np.array([0.0, 0.2, 0.8, 2.0])
    return reactormodels.breakthrough_data(
        chemical_name=chemical_name,
        times=times,
        effluent_concentrations=effluent_concentrations,
        feed_concentration=feed_concentration,
    )


def test_normalized_concentrations():
    """Effluent concentrations should normalize as C/C0."""
    curve = _base_curve()
    expected = np.array([0.0, 0.02, 0.08, 0.2])
    assert curve.normalized_concentrations == pytest.approx(expected)


def test_breakthrough_detection():
    """Breakthrough should be detected when C/C0 reaches the threshold."""
    curve = _base_curve()
    assert curve.has_breakthrough(0.05) is True
    assert curve.breakthrough_index(0.05) == 2


def test_breakthrough_point():
    """The first breakthrough point should be returned correctly."""
    curve = _base_curve()
    point = curve.breakthrough_point(0.05)
    assert point["chemical_name"] == "Example chemical"
    assert point["bed_volume"] == pytest.approx(3000.0)
    assert point["effluent_concentration"] == pytest.approx(0.8)
    assert point["normalized_concentration"] == pytest.approx(0.08)


def test_summary_line():
    """Summary line should include the key breakthrough information."""
    curve = _base_curve()
    summary = curve.summary_line(0.05)
    assert "Example chemical" in summary
    assert "points=4" in summary
    assert "feed_concentration=10" in summary
    assert "breakthrough=True" in summary


def test_plot_breakthrough_curve(tmp_path):
    """Breakthrough curve should plot and save without error."""
    curve = _base_curve()
    output_path = tmp_path / "breakthrough_curve.png"
    curve.plot(
        x_axis="bed_volume",
        save_path=output_path,
        show=False,
    )
    assert output_path.exists()


def test_removal_efficiency_over_bed_volumes():
    """Removal efficiency should be returned over bed volumes."""
    curve = _base_curve()

    result = curve.breakthrough_removal_efficiency(
        x_axis="bed_volume",
        efficiency_step=10,
    )

    expected_efficiencies = np.array([100.0, 98.0, 92.0, 80.0])

    assert result["chemical_name"] == "Example chemical"
    assert result["x_axis"] == "bed_volume"
    assert result["removal_efficiencies"] == pytest.approx(expected_efficiencies)
    assert result["max_removal_efficiency"] == pytest.approx(100.0)
    assert result["bed_volume_at_max_removal_efficiency"] == pytest.approx(1000.0)

    expected_steps = {
        0.0: 1000.0,
        10.0: 1000.0,
        20.0: 1000.0,
        30.0: 1000.0,
        40.0: 1000.0,
        50.0: 1000.0,
        60.0: 1000.0,
        70.0: 1000.0,
        80.0: 1000.0,
        90.0: 1000.0,
        100.0: 1000.0,
    }

    assert result["bed_volume_at_each_efficiency_step"] == pytest.approx(expected_steps)


def test_removal_efficiency_over_time():
    """Removal efficiency should be returned over time when x_axis is time."""
    curve = _base_time_curve()

    result = curve.breakthrough_removal_efficiency(
        x_axis="time",
        efficiency_step=10,
    )

    expected_efficiencies = np.array([100.0, 98.0, 92.0, 80.0])

    assert result["chemical_name"] == "Example chemical"
    assert result["x_axis"] == "time"
    assert result["removal_efficiencies"] == pytest.approx(expected_efficiencies)
    assert result["max_removal_efficiency"] == pytest.approx(100.0)
    assert result["time_at_max_removal_efficiency"] == pytest.approx(1000.0)

    expected_steps = {
        0.0: 1000.0,
        10.0: 1000.0,
        20.0: 1000.0,
        30.0: 1000.0,
        40.0: 1000.0,
        50.0: 1000.0,
        60.0: 1000.0,
        70.0: 1000.0,
        80.0: 1000.0,
        90.0: 1000.0,
        100.0: 1000.0,
    }

    assert result["time_at_each_efficiency_step"] == pytest.approx(expected_steps)


def test_removal_efficiency_interpolates_each_10_percent():
    """The selected x-axis value should be interpolated at each 10% efficiency."""
    curve = reactormodels.breakthrough_data(
        chemical_name="Example chemical",
        feed_concentration=10.0,
        bed_volumes=np.array([0.0, 10.0, 20.0]),
        effluent_concentrations=np.array([10.0, 5.0, 0.0]),
    )

    result = curve.breakthrough_removal_efficiency(
        x_axis="bed_volume",
        efficiency_step=10,
    )

    expected_steps = {
        0.0: 0.0,
        10.0: 2.0,
        20.0: 4.0,
        30.0: 6.0,
        40.0: 8.0,
        50.0: 10.0,
        60.0: 12.0,
        70.0: 14.0,
        80.0: 16.0,
        90.0: 18.0,
        100.0: 20.0,
    }

    assert result["bed_volume_at_each_efficiency_step"] == pytest.approx(expected_steps)


def test_removal_efficiency_raises_error_for_missing_time():
    """Requesting time output should fail if times were not provided."""
    curve = _base_curve()

    with pytest.raises(ValueError, match="times were not provided"):
        curve.breakthrough_removal_efficiency(x_axis="time")


def test_removal_efficiency_raises_error_for_missing_bed_volumes():
    """Requesting bed volume output should fail if bed volumes were not provided."""
    curve = _base_time_curve()

    with pytest.raises(ValueError, match="bed_volumes were not provided"):
        curve.breakthrough_removal_efficiency(x_axis="bed_volume")


def test_removal_efficiency_raises_error_for_invalid_x_axis():
    """Invalid x_axis values should raise an error."""
    curve = _base_curve()

    with pytest.raises(ValueError, match="x_axis must be 'bed_volume' or 'time'"):
        curve.breakthrough_removal_efficiency(x_axis="invalid")


def test_removal_efficiency_raises_error_for_invalid_efficiency_step():
    """Efficiency step must be greater than zero."""
    curve = _base_curve()

    with pytest.raises(ValueError, match="efficiency_step must be greater than zero"):
        curve.breakthrough_removal_efficiency(
            x_axis="bed_volume",
            efficiency_step=0,
        )
