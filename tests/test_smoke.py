import reactormodels
import numpy as np
import pytest

def _base_curve():
    """Shared setup for breakthrough curve tests."""
    chemical_name = "Example chemical"
    feed_concentration = 10.0
    bed_volumes = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    effluent_concentrations = np.array([0.0, 0.2, 0.8, 2.0])
    return reactormodels.ExperimentalBreakthroughCurve(
        chemical_name=chemical_name,
        bed_volumes=bed_volumes,
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