"""confidence.py

Uncertainty estimates for a fit: parameter intervals and prediction bands.

Everything here takes a `FitResult` and returns plain numbers/arrays, so
you can call these directly or through the `FitResult.ci` / `.predict_band`
shortcuts.

    stderr_interval: fast linear (Wald) interval from the covariance matrix
    profile_likelihood_interval: likelihood-profile interval; slower, better for
                                    nonlinear models
    prediction_band: delta-method confidence band on the model curve

Intervals are returned as `{name: (lo, best, hi)}`.

Caveat: the F-test and covariance used here assume residuals are roughly
i.i.d. Gaussian in the space of the chosen error function(s).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lmfit
import numpy as np
from scipy import stats
from scipy.optimize import brentq

if TYPE_CHECKING:
    from .result import FitResult

Interval = tuple[float, float, float]


def _varying(fit: FitResult) -> list[tuple[str, lmfit.Parameter]]:
    return [(name, p) for name, p in fit.params.items() if p.vary]


def _dof(fit: FitResult) -> int:
    dof = fit.result.nfree
    if dof <= 0:
        raise ValueError(
            "No degrees of freedom left; cannot form a confidence interval."
        )
    return dof


def prediction_band(
    fit: FitResult, level: float = 0.95, step: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Delta-method confidence band on the model curve: `(lower, upper)`.

    band = best_fit +/- t_crit * sqrt(diag(J @ cov @ J.T)), where J is the
    central finite-difference Jacobian at the optimum. `step` overrides the
    default relative step (1e-4 * |value|).
    """
    if fit.result.covar is None:
        raise ValueError("No covariance matrix; the fit method must report one.")

    names = [name for name, _ in _varying(fit)]
    best = fit.params
    y0 = fit.model_func(best)
    objective = fit.objective

    J = np.zeros((y0.size, len(names)))
    for j, name in enumerate(names):
        value = best[name].value
        h = step or max(abs(value) * 1e-4, 1e-8)
        trial = best.copy()
        ys = []
        for sign in (+1, -1):
            trial[name].value = value + sign * h
            y = fit.model_func(trial)
            if objective.is_failed(y):
                raise RuntimeError(
                    f"Perturbing {name!r} by {sign * h:+g} made the model "
                    "fail to solve; try a smaller `step` or tighter bounds "
                    f"on {name!r}."
                )
            ys.append(y)
        J[:, j] = (ys[0] - ys[1]) / (2 * h)

    var = np.clip(np.einsum("ij,jk,ik->i", J, fit.result.covar, J), 0, None)
    half = stats.t.ppf(0.5 + level / 2, fit.result.nfree) * np.sqrt(var)
    return y0 - half, y0 + half


def stderr_interval(fit: FitResult, level: float = 0.95) -> dict[str, Interval]:
    """Best value +/- t_crit * stderr for every varying parameter."""
    t_crit = stats.t.ppf(0.5 + level / 2, _dof(fit))
    out = {}
    for name, p in _varying(fit):
        if p.stderr is None:
            raise ValueError(f"No stderr for {name!r}; use method='profile' instead.")
        half = t_crit * p.stderr
        out[name] = (p.value - half, p.value, p.value + half)
    return out


def profile_likelihood_interval(
    fit: FitResult,
    level: float = 0.95,
    max_expansions: int = 12,
    initial_step_in_stderr: float = 1.0,
    step_growth: float = 1.6,
    method: str = "least_squares",
    **minimize_kwargs: Any,
) -> dict[str, Interval]:
    """Profile-likelihood interval for every varying parameter.

    Each parameter is fixed at trial values (stepping outward from the best
    fit, first step `initial_step_in_stderr * stderr`, growing by
    `step_growth`) and the rest refit, until chi-square crosses the F-test
    threshold. A side that never crosses within `max_expansions` steps or
    the parameter bound is returned as -inf / +inf. `method` and
    `minimize_kwargs` configure the re-optimizer.
    """
    dof = _dof(fit)
    f_crit = stats.f.ppf(level, 1, dof)
    target = float(fit.result.chisqr) * (1.0 + f_crit / dof)
    scanner = ParameterScanner(fit, target, method, minimize_kwargs)

    out = {}
    for name, p in _varying(fit):
        if not p.stderr or p.stderr <= 0:
            raise ValueError(f"No usable stderr for {name!r}; cannot pick a scan step.")
        lo, hi = (
            scanner.scan_side(
                name,
                direction,
                p.value,
                p.stderr,
                initial_step_in_stderr,
                step_growth,
                max_expansions,
            )
            for direction in (-1, +1)
        )
        out[name] = (lo, p.value, hi)
    return out


class ParameterScanner:
    """Finds one profile-likelihood bound by refitting with a parameter fixed.

    The likelihood-ratio criterion for a confidence interval is

        2 * [l(theta_hat) - l_max(q)] <= chi2(1, 1 - alpha),

    where l(theta_hat) is the log-likelihood at the global best fit and
    l_max(q) is the profile log-likelihood: the parameter is fixed at q and
    all others are re-optimized.

    For Gaussian residuals with known sigma, l = -chi2/2 + const, so the
    constants cancel and the criterion becomes

        chi2_prof(q) - chi2_min <= chi2(1, 1 - alpha).

    Uses a bounded re-optimizer (default `least_squares`) rather than
    `lmfit.conf_interval`, which always uses legacy MINPACK `leastsq` and
    can stall at the penalty discontinuity in `Objective`.
    """

    def __init__(
        self, fit: FitResult, target: float, method: str, minimize_kwargs: dict
    ):
        self.fit = fit
        self.target = target
        self.method = method
        self.minimize_kwargs = minimize_kwargs

    def chisqr_at(self, name: str, value: float) -> float:
        """Refit with `name` fixed at `value`; return the resulting chisqr."""
        trial = self.fit.params.copy()
        trial[name].set(value=value, vary=False)
        out = lmfit.Minimizer(self.fit.objective.residual, trial).minimize(
            method=self.method, **self.minimize_kwargs
        )
        return float(out.chisqr)

    def scan_side(
        self,
        name: str,
        direction: int,
        best: float,
        stderr: float,
        initial_step_in_stderr: float,
        step_growth: float,
        max_expansions: int,
    ) -> float:
        """Return one bound (`direction` is -1 or +1); +/-inf if never crossed."""
        p = self.fit.params[name]
        bound = p.min if direction < 0 else p.max
        step = direction * stderr * initial_step_in_stderr
        prev = best

        for _ in range(max_expansions):
            trial = prev + step
            hit_bound = bound is not None and direction * (trial - bound) >= 0
            if hit_bound:
                trial = bound

            if self.chisqr_at(name, trial) >= self.target:
                lo, hi = sorted((prev, trial))
                return brentq(
                    lambda x: self.chisqr_at(name, x) - self.target,
                    lo,
                    hi,
                    xtol=abs(step) * 1e-3,
                )
            if hit_bound:
                break
            prev = trial
            step *= step_growth

        return direction * np.inf
