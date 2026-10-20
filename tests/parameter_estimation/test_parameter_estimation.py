"""Tests for the parameter_estimation package.

Sections:
  1. Fit recovery on Yoon-Nelson breakthrough data
  2. Confidence intervals and prediction band
  3. Automatic solver-failure handling
  4. Input validation
  5. Multistart fitting
"""

import numpy as np
import pytest

import reactormodels
from reactormodels.parameter_estimation import (
    fit_parameters,
    fit_parameters_multistart,
    make_parameters,
)
from reactormodels.parameter_estimation.fitting import _sample_start

K_TRUE, T50_TRUE = 0.045, 480.0


def _synthetic_yoon_nelson(k_yn=K_TRUE, t50=T50_TRUE, noise=0.02, seed=0):
    """Noisy Yoon-Nelson breakthrough data: returns (time, c_obs)."""
    rng = np.random.default_rng(seed)
    time = np.linspace(50, 900, 60)
    c_true = reactormodels.models.YoonNelson(k_YN=k_yn, t_50=t50).breakthrough_profile(
        time, x=0
    )
    c_obs = np.clip(c_true + rng.normal(0, noise, size=time.size), 0, 1)
    return time, c_obs


def _yoon_nelson_model(time):
    """Build a model_func(params) for Yoon-Nelson evaluated at `time`."""

    def model_func(p):
        m = reactormodels.models.YoonNelson(k_YN=p["k_YN"].value, t_50=p["t_50"].value)
        return m.breakthrough_profile(time, x=0)

    return model_func


def _yn_params(**overrides):
    """Default Yoon-Nelson starting parameters, with optional per-name overrides."""
    specs = dict(k_YN=dict(value=0.02, min=0), t_50=dict(value=400, min=0))
    specs.update(overrides)
    return make_parameters(**specs)


def test_make_parameters_exposes_lmfit_options():
    """Pass every supported lmfit option through to its Parameter."""
    params = make_parameters(
        fitted=dict(
            value=1.0,
            vary=True,
            min=0.0,
            max=2.0,
            brute_step=0.1,
            user_data="metadata",
        ),
        fixed=dict(value=3.0, vary=False),
        defaulted=dict(value=2.0),
        derived=dict(expr="2 * fitted"),
    )

    assert params["fitted"].vary is True
    assert params["fitted"].min == 0.0
    assert params["fitted"].max == 2.0
    assert params["fitted"].brute_step == 0.1
    assert params["fitted"].user_data == "metadata"
    assert params["fixed"].vary is False
    assert params["defaulted"].vary is True
    assert params["defaulted"].min == -np.inf
    assert params["defaulted"].max == np.inf
    assert params["derived"].value == 2.0
    assert params["derived"].vary is False


def test_make_parameters_rejects_unsupported_options():
    """Reject misspelled options instead of silently ignoring them."""
    with pytest.raises(TypeError, match="Unsupported option"):
        make_parameters(a=dict(value=1.0, typo=True))


@pytest.fixture
def yn_fit():
    """A completed Yoon-Nelson fit (seed 1) shared by the interval/band tests."""
    time, c_obs = _synthetic_yoon_nelson(seed=1)
    return fit_parameters(_yoon_nelson_model(time), _yn_params(), y_data=c_obs)


def test_recovers_known_parameters():
    """Fitting noisy synthetic data recovers the generating parameters."""
    time, c_obs = _synthetic_yoon_nelson(seed=0)
    params = _yn_params()
    result = fit_parameters(_yoon_nelson_model(time), params, y_data=c_obs)

    assert result.params["k_YN"].value == pytest.approx(K_TRUE, rel=0.05)
    assert result.params["t_50"].value == pytest.approx(T50_TRUE, rel=0.002)
    assert result.rmse() < 0.02


def test_fixed_parameter_is_not_fitted():
    """vary=False parameters stay put and get no confidence interval."""
    time, c_obs = _synthetic_yoon_nelson(seed=2)
    params = _yn_params(t_50=dict(value=T50_TRUE, vary=False))
    result = fit_parameters(_yoon_nelson_model(time), params, y_data=c_obs)

    assert result.params["t_50"].value == T50_TRUE
    assert "t_50" not in result.confidence_interval()
    assert result.params["k_YN"].value == pytest.approx(K_TRUE, rel=0.1)


def test_residual_is_data_minus_best_fit(yn_fit):
    np.testing.assert_allclose(yn_fit.residual, yn_fit.y_data - yn_fit.best_fit)


def test_weights_can_ignore_corrupted_points():
    """Zero-weighting gross outliers lets the fit recover the true parameters."""
    time, c_obs = _synthetic_yoon_nelson(seed=5)
    c_bad = c_obs.copy()
    bad = slice(10, 15)
    c_bad[bad] = 1000.0  # corrupt a stretch of the data
    weights = np.ones_like(c_bad)
    weights[bad] = 0.0

    result = fit_parameters(
        _yoon_nelson_model(time), _yn_params(), y_data=c_bad, weights=weights
    )

    assert result.params["k_YN"].value == pytest.approx(K_TRUE, rel=0.15)
    assert result.params["t_50"].value == pytest.approx(T50_TRUE, rel=0.03)


def test_confidence_interval_contains_truth(yn_fit):
    """The 95% CI should (usually) bracket the true generating parameters."""
    confidence_interval = yn_fit.confidence_interval(level=0.95)

    lo, _, hi = confidence_interval["k_YN"]
    assert lo < K_TRUE < hi
    lo, _, hi = confidence_interval["t_50"]
    assert lo < T50_TRUE < hi


def test_wider_level_gives_wider_interval(yn_fit):
    lo90, _, hi90 = yn_fit.confidence_interval(level=0.90)["k_YN"]
    lo99, _, hi99 = yn_fit.confidence_interval(level=0.99)["k_YN"]
    assert (hi99 - lo99) > (hi90 - lo90)


def test_profile_interval_brackets_best_value(yn_fit):
    """Profile Likelihood CI is finite, brackets the estimate, and is comparable to the Wald CI."""
    wald = yn_fit.confidence_interval(method="stderr")
    profile_likelihood = yn_fit.confidence_interval(method="profile_likelihood")

    for name in ("k_YN", "t_50"):
        lo, best, hi = profile_likelihood[name]
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo < best < hi
        wald_width = wald[name][2] - wald[name][0]
        assert 0.33 * wald_width < (hi - lo) < 3.0 * wald_width


def test_unknown_ci_method_raises(yn_fit):
    with pytest.raises(ValueError, match="Unknown ci method"):
        yn_fit.confidence_interval(method="bogus")


def test_predict_band_brackets_best_fit():
    time, c_obs = _synthetic_yoon_nelson(seed=3)
    result = fit_parameters(_yoon_nelson_model(time), _yn_params(), y_data=c_obs)
    lo, hi = result.predict_band()

    assert np.all(lo <= result.best_fit + 1e-9)
    assert np.all(hi >= result.best_fit - 1e-9)


def test_solver_failure_is_penalized_not_fatal():
    """A model that raises RuntimeError at bad trial points doesn't crash the fit.

    Mimics PSDM's IDA solver raising RuntimeError when a trial parameter set
    is numerically unreachable (e.g. during a Jacobian probe or line search).
    """
    time, c_obs = _synthetic_yoon_nelson(seed=4)
    good_model = _yoon_nelson_model(time)

    def flaky_model_func(p):
        if p["k_YN"].value <= 0 or p["k_YN"].value > 1:
            raise RuntimeError("solver failed to converge for this trial point")
        return good_model(p)

    # start at the edge of the "bad" region so the optimizer likely probes it
    params = _yn_params(k_YN=dict(value=1e-6, min=-1, max=1))
    result = fit_parameters(flaky_model_func, params, y_data=c_obs)

    assert result.params["k_YN"].value == pytest.approx(K_TRUE, rel=0.15)


def test_failure_at_initial_guess_raises_original_error():
    """Errors at the starting point surface immediately instead of being hidden."""

    def always_fails(p):
        raise RuntimeError("boom")

    params = make_parameters(a=dict(value=1.0))
    with pytest.raises(RuntimeError, match="boom"):
        fit_parameters(always_fails, params, y_data=np.zeros(5))


def test_non_finite_output_at_initial_guess_raises():
    def returns_nan(p):
        return np.full(5, np.nan)

    params = make_parameters(a=dict(value=1.0))
    with pytest.raises(ValueError, match="non-finite"):
        fit_parameters(returns_nan, params, y_data=np.zeros(5))


def test_non_solver_exceptions_are_not_swallowed():
    """A genuine bug (e.g. TypeError) must propagate, not be penalized."""

    def buggy(p):
        raise TypeError("bug in model")

    params = make_parameters(a=dict(value=1.0))
    with pytest.raises(TypeError, match="bug in model"):
        fit_parameters(buggy, params, y_data=np.zeros(5))


def test_least_squares_default_handles_wide_scale_and_penalty():
    """Regression test.

    Decade-spanning parameters plus solver-failure penalties can make
    lmfit's legacy 'leastsq' converge to nonsense, while the
    'least_squares' default (SciPy's bounded trust-region solver) recovers
    the true values.
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

    def flaky_model_func(p):
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
    result = fit_parameters(flaky_model_func, params, y_data=y_obs)

    assert result.params["a"].value == pytest.approx(true_a, rel=0.5)
    assert result.params["b"].value == pytest.approx(true_b, rel=0.1)
    assert result.params["c"].value == pytest.approx(true_c, rel=0.2)


def test_shape_mismatch_raises():
    def model_func(p):
        return np.zeros(5)

    params = make_parameters(a=dict(value=1.0))
    with pytest.raises(ValueError, match="shape"):
        fit_parameters(model_func, params, y_data=np.zeros(10))


def test_multistart_recovers_parameters_from_poor_guess():
    time, c_obs = _synthetic_yoon_nelson(seed=6)
    params = _yn_params(
        k_YN=dict(value=0.5, min=1e-3, max=1.0),
        t_50=dict(value=100, min=50, max=900),
    )
    best, all_results = fit_parameters_multistart(
        _yoon_nelson_model(time),
        params,
        y_data=c_obs,
        n_starts=6,
        log_uniform_names=("k_YN",),
    )

    assert best.params["k_YN"].value == pytest.approx(K_TRUE, rel=0.15)
    assert best.params["t_50"].value == pytest.approx(T50_TRUE, rel=0.03)
    assert len(all_results) >= 2
    assert best.result.chisqr == min(r.result.chisqr for r in all_results)


def test_multistart_requires_finite_bounds_or_start_ranges():
    time, c_obs = _synthetic_yoon_nelson(seed=6)
    with pytest.raises(ValueError, match="start_ranges"):
        fit_parameters_multistart(_yoon_nelson_model(time), _yn_params(), y_data=c_obs)


def test_multistart_log_uniform_requires_positive_lo():
    time, c_obs = _synthetic_yoon_nelson(seed=6)
    params = _yn_params(
        k_YN=dict(value=0.02, min=0, max=1.0), t_50=dict(value=400, min=0, max=900)
    )
    with pytest.raises(ValueError, match="lo > 0"):
        fit_parameters_multistart(
            _yoon_nelson_model(time), params, y_data=c_obs, log_uniform_names=("k_YN",)
        )


def test_multistart_unknown_names_raise():
    time, c_obs = _synthetic_yoon_nelson(seed=6)
    params = _yn_params(
        k_YN=dict(value=0.5, min=1e-3, max=1.0),
        t_50=dict(value=100, min=50, max=900),
    )
    with pytest.raises(ValueError, match="non-varying or unknown"):
        fit_parameters_multistart(
            _yoon_nelson_model(time),
            params,
            y_data=c_obs,
            start_ranges={"typo": (0.0, 1.0)},
        )
    with pytest.raises(ValueError, match="non-varying or unknown"):
        fit_parameters_multistart(
            _yoon_nelson_model(time),
            params,
            y_data=c_obs,
            log_uniform_names=("typo",),
        )


def test_multistart_empty_start_range_raises():
    time, c_obs = _synthetic_yoon_nelson(seed=6)
    params = _yn_params(
        k_YN=dict(value=0.5, min=1e-3, max=1.0),
        t_50=dict(value=100, min=50, max=900),
    )
    # Window lies entirely outside the parameter's bounds -> empty after intersection.
    with pytest.raises(ValueError, match="empty sampling range"):
        fit_parameters_multistart(
            _yoon_nelson_model(time),
            params,
            y_data=c_obs,
            start_ranges={"t_50": (1000.0, 2000.0)},
        )


def test_multistart_unbounded_with_start_ranges_recovers_parameters():
    time, c_obs = _synthetic_yoon_nelson(seed=6)
    params = _yn_params(
        k_YN=dict(value=0.5, min=1e-3),  # max = +inf
        t_50=dict(value=100, min=50),  # max = +inf
    )
    best, _ = fit_parameters_multistart(
        _yoon_nelson_model(time),
        params,
        y_data=c_obs,
        n_starts=6,
        start_ranges={"k_YN": (1e-3, 1.0), "t_50": (50.0, 900.0)},
        log_uniform_names=("k_YN",),
    )
    assert best.params["k_YN"].value == pytest.approx(K_TRUE, rel=0.15)
    assert best.params["t_50"].value == pytest.approx(T50_TRUE, rel=0.03)


def test_multistart_reproducible_with_seed():
    time, c_obs = _synthetic_yoon_nelson(seed=6)

    def run(seed):
        params = _yn_params(
            k_YN=dict(value=0.5, min=1e-3, max=1.0),
            t_50=dict(value=100, min=50, max=900),
        )
        _, results = fit_parameters_multistart(
            _yoon_nelson_model(time),
            params,
            y_data=c_obs,
            n_starts=4,
            seed=seed,
            log_uniform_names=("k_YN",),
        )
        return [r.result.chisqr for r in results]

    assert run(5) == run(5)


def test_multistart_keep_original_guess_controls_start_count():
    time, c_obs = _synthetic_yoon_nelson(seed=6)

    def n_results(keep):
        params = _yn_params(
            k_YN=dict(value=0.5, min=1e-3, max=1.0),
            t_50=dict(value=100, min=50, max=900),
        )
        _, results = fit_parameters_multistart(
            _yoon_nelson_model(time),
            params,
            y_data=c_obs,
            n_starts=3,
            keep_original_guess=keep,
        )
        return len(results)

    # Some random starts may fail and be skipped, so allow <=.
    assert n_results(True) <= 4
    assert n_results(False) <= 3


def test_sample_start_within_ranges():
    ranges = {"a": (-2.0, 3.0), "k": (1e-3, 1e3)}
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = _sample_start(ranges, ("k",), rng)
        assert set(s) == {"a", "k"}
        for name, (lo, hi) in ranges.items():
            assert lo <= s[name] <= hi


def test_sample_start_log_uniform_spreads_across_decades():
    ranges = {"k": (1e-3, 1e3)}
    rng = np.random.default_rng(0)
    log_vals = np.array([_sample_start(ranges, ("k",), rng)["k"] for _ in range(500)])
    lin_vals = np.array([_sample_start(ranges, (), rng)["k"] for _ in range(500)])
    # Geometric midpoint is 1.0: log sampling puts ~half below it, linear almost none.
    assert 0.4 < np.mean(log_vals < 1.0) < 0.6
    assert np.mean(lin_vals < 1.0) < 0.05
