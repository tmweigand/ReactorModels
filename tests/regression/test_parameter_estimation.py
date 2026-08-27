import numpy as np
import pytest

import reactormodels
from reactormodels.regression import make_parameters, fit_parameters


def _synthetic_yoon_nelson(k_yn=0.045, t50=480.0, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(50, 900, 60)
    true_model = reactormodels.models.YoonNelson(k_YN=k_yn, t_50=t50)
    c_true = true_model.breakthrough_profile(time, x=0)
    c_obs = np.clip(c_true + rng.normal(0, noise, size=time.size), 0, 1)
    return time, c_obs


def test_recovers_known_parameters():
    """Fitting synthetic (noisy) data recovers the generating parameters."""
    k_true, t50_true = 0.045, 480.0
    time, c_obs = _synthetic_yoon_nelson(k_true, t50_true)

    def model_func(p, time=time):
        m = reactormodels.models.YoonNelson(k_YN=p["k_YN"].value, t_50=p["t_50"].value)
        return m.breakthrough_profile(time, x=0)

    params = make_parameters(
        k_YN=dict(value=0.02, min=0),
        t_50=dict(value=400, min=0),
    )
    result = fit_parameters(model_func, params, y_data=c_obs)

    assert result.params["k_YN"].value == pytest.approx(k_true, rel=0.1)
    assert result.params["t_50"].value == pytest.approx(t50_true, rel=0.02)
    assert result.rmse() < 0.05


def test_confidence_interval_contains_truth():
    """The 95% CI should (usually) bracket the true generating parameters."""
    k_true, t50_true = 0.045, 480.0
    time, c_obs = _synthetic_yoon_nelson(k_true, t50_true, seed=1)

    def model_func(p, time=time):
        m = reactormodels.models.YoonNelson(k_YN=p["k_YN"].value, t_50=p["t_50"].value)
        return m.breakthrough_profile(time, x=0)

    params = make_parameters(
        k_YN=dict(value=0.02, min=0),
        t_50=dict(value=400, min=0),
    )
    result = fit_parameters(model_func, params, y_data=c_obs)
    ci = result.ci(level=0.95)

    lo, _, hi = ci["k_YN"]
    assert lo < k_true < hi
    lo, _, hi = ci["t_50"]
    assert lo < t50_true < hi


def test_fixed_parameter_is_not_fitted():
    """vary=False parameters should stay at their initial value."""
    k_true, t50_true = 0.045, 480.0
    time, c_obs = _synthetic_yoon_nelson(k_true, t50_true, seed=2)

    def model_func(p, time=time):
        m = reactormodels.models.YoonNelson(k_YN=p["k_YN"].value, t_50=p["t_50"].value)
        return m.breakthrough_profile(time, x=0)

    params = make_parameters(
        k_YN=dict(value=0.02, min=0),
        t_50=dict(value=t50_true, vary=False),
    )
    result = fit_parameters(model_func, params, y_data=c_obs)

    assert result.params["t_50"].value == t50_true
    assert "t_50" not in result.ci()  # only varying params get a CI
    assert result.params["k_YN"].value == pytest.approx(k_true, rel=0.1)


def test_shape_mismatch_raises():
    def model_func(p):
        return np.zeros(5)

    params = make_parameters(a=dict(value=1.0))
    with pytest.raises(ValueError):
        fit_parameters(model_func, params, y_data=np.zeros(10))


def test_solver_failure_is_penalized_not_fatal():
    """A model that raises RuntimeError for bad trial params shouldn't crash the fit.

    Mimics PSDM's IDA solver raising RuntimeError when a trial parameter
    set is numerically unreachable (e.g. during a finite-difference Jacobian
    probe or an aggressive line-search step).
    """
    k_true, t50_true = 0.045, 480.0
    time, c_obs = _synthetic_yoon_nelson(k_true, t50_true, seed=4)

    def flaky_model_func(p, time=time):
        if p["k_YN"].value <= 0 or p["k_YN"].value > 1:
            raise RuntimeError("solver failed to converge for this trial point")
        m = reactormodels.models.YoonNelson(k_YN=p["k_YN"].value, t_50=p["t_50"].value)
        return m.breakthrough_profile(time, x=0)

    # deliberately start right at the edge of the "bad" region so the
    # optimizer is likely to probe k_YN <= 0 at least once
    params = make_parameters(
        k_YN=dict(value=1e-6, min=-1, max=1),
        t_50=dict(value=400, min=0),
    )

    result = fit_parameters(flaky_model_func, params, y_data=c_obs)

    assert result.params["k_YN"].value == pytest.approx(k_true, rel=0.15)


def test_solver_failure_can_be_left_to_propagate():
    """catch_exceptions=() should let the original exception through."""

    def always_fails(p):
        raise RuntimeError("boom")

    params = make_parameters(a=dict(value=1.0))
    with pytest.raises(RuntimeError, match="boom"):
        fit_parameters(always_fails, params, y_data=np.zeros(5), catch_exceptions=())


def test_least_squares_default_handles_wide_scale_and_penalty():
    """Regression test for a real failure mode: decade-spanning parameters
    plus solver-failure penalties can make lmfit's legacy 'leastsq' method
    converge to nonsensical values, while the 'least_squares' default
    (SciPy's bounded trust-region solver) recovers the true values.
    """
    rng = np.random.default_rng(7)
    true_a, true_b, true_c = 5e-10, 5e-6, 0.1
    x = np.linspace(1, 100, 40)

    def true_curve(a, b, c):
        return (
            1
            / (1 + np.exp(-(np.log(a) + 15) * (x - 50) / 10))
            * 1
            / (1 + np.exp(-(np.log(b) + 13) * (x - 50) / 10))
            * (0.5 + 0.5 * np.tanh((c - 0.05) * 20))
        )

    y_obs = true_curve(true_a, true_b, true_c) + rng.normal(0, 0.01, x.size)

    fail_a, fail_b = 1e-6, 1e-2  # mimics a DAE solver breaking down out here

    def flaky_model_func(p, x=x):
        a, b, c = p["a"].value, p["b"].value, p["c"].value
        if not (0 < a < fail_a) or not (0 < b < fail_b) or c <= 0:
            raise RuntimeError("solver unstable for this trial point")
        return true_curve(a, b, c)

    # loose (min-only) bounds -- the demo-script bug this test guards against
    params = make_parameters(
        a=dict(value=1e-10, min=0),
        b=dict(value=1e-6, min=0),
        c=dict(value=0.05, min=0),
    )

    result = fit_parameters(flaky_model_func, params, y_data=y_obs)  # default method

    assert result.params["a"].value == pytest.approx(true_a, rel=0.5)
    assert result.params["b"].value == pytest.approx(true_b, rel=0.1)
    assert result.params["c"].value == pytest.approx(true_c, rel=0.2)


def test_predict_band_brackets_best_fit():
    time, c_obs = _synthetic_yoon_nelson(seed=3)

    def model_func(p, time=time):
        m = reactormodels.models.YoonNelson(k_YN=p["k_YN"].value, t_50=p["t_50"].value)
        return m.breakthrough_profile(time, x=0)

    params = make_parameters(
        k_YN=dict(value=0.02, min=0),
        t_50=dict(value=400, min=0),
    )
    result = fit_parameters(model_func, params, y_data=c_obs)
    lo, hi = result.predict_band()

    assert np.all(lo <= result.best_fit + 1e-9)
    assert np.all(hi >= result.best_fit - 1e-9)
