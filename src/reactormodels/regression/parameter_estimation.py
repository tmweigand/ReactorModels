"""parameter_estimation.py

Model-agnostic parameter estimation built on lmfit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import lmfit
from scipy import stats
from scipy.optimize import brentq

__all__ = [
    "make_parameters",
    "fit_parameters",
    "fit_parameters_multistart",
    "FitResult",
]

ModelFunc = Callable[[lmfit.Parameters], np.ndarray]


def make_parameters(**specs: dict[str, Any]) -> lmfit.Parameters:
    """Build lmfit.Parameters from plain kwarg dicts, e.g. value/min/max/vary."""
    params = lmfit.Parameters()
    for name, spec in specs.items():
        params.add(name, **spec)
    return params


class Objective:
    """Turns model_func into an lmfit-compatible residual, with a safety net.

    `predict` catches solver failures (e.g. a DAE that fails to converge on
    a trial point) and non-finite output, returning a constant `penalty`
    array instead of raising -- this lets the optimizer back away from a
    bad region rather than crashing the whole fit. `residual` is what gets
    passed to lmfit.Minimizer.
    """

    def __init__(
        self,
        model_func: ModelFunc,
        y_data: np.ndarray,
        weights: np.ndarray | None,
        catch_exceptions: tuple[type[BaseException], ...],
        penalty: float,
    ):
        self.model_func = model_func
        self.y_data = y_data
        self.weights = weights
        self.catch_exceptions = catch_exceptions
        self.penalty = penalty

    def predict(self, p: lmfit.Parameters) -> np.ndarray:
        try:
            y_pred = np.asarray(self.model_func(p), dtype=float)
        except self.catch_exceptions as exc:
            warnings.warn(
                f"model_func failed ({exc}); penalizing this point.", stacklevel=3
            )
            return np.full(self.y_data.shape, self.penalty)

        if y_pred.shape != self.y_data.shape:
            raise ValueError(
                f"model_func returned shape {y_pred.shape}, expected {self.y_data.shape}."
            )
        if not np.all(np.isfinite(y_pred)):
            warnings.warn(
                "model_func returned non-finite values; penalizing this point.",
                stacklevel=3,
            )
            return np.full(self.y_data.shape, self.penalty)
        return y_pred

    def residual(self, p: lmfit.Parameters) -> np.ndarray:
        resid = self.predict(p) - self.y_data
        if self.weights is not None:
            resid = resid * self.weights
        return resid


def fit_parameters(
    model_func: ModelFunc,
    params: lmfit.Parameters,
    y_data: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "least_squares",
    catch_exceptions: tuple[type[BaseException], ...] = (RuntimeError,),
    penalty: float | None = None,
    **minimize_kwargs: Any,
) -> FitResult:
    """Fit model_func to y_data and return a FitResult.

    Uses `method="least_squares"` (bounded, SciPy trust-region) by default
    rather than lmfit's legacy `"leastsq"` (MINPACK): it copes far better
    with per-parameter bounds and the penalty discontinuity above,
    especially when parameters span many orders of magnitude. Give every
    fitted parameter a realistic `min` and `max` -- that's what keeps the
    search out of regions where the model can't be solved. Consider
    `x_scale="jac"` if parameters span many orders of magnitude.

    `catch_exceptions`: exception types from model_func to treat as "bad
    fit" rather than a hard failure. `penalty`: prediction value substituted
    on failure; defaults to `1e3 * (1 + max(abs(y_data)))`.
    """
    y_data = np.asarray(y_data, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
    if penalty is None:
        penalty = 1e3 * (1 + np.max(np.abs(y_data)))

    objective = Objective(model_func, y_data, weights, catch_exceptions, penalty)
    result = lmfit.Minimizer(objective.residual, params).minimize(
        method=method, **minimize_kwargs
    )

    return FitResult(
        result=result,
        model_func=objective.predict,
        y_data=y_data,
        weights=weights,
        penalty=penalty,
    )


def fit_parameters_multistart(
    model_func: ModelFunc,
    params: lmfit.Parameters,
    y_data: np.ndarray,
    n_starts: int = 12,
    seed: int | None = 0,
    log_uniform_names: tuple[str, ...] = (),
    keep_original_guess: bool = True,
    **fit_parameters_kwargs: Any,
) -> tuple[FitResult, list[FitResult]]:
    """Run fit_parameters from many randomized starting points; keep the best.

    `fit_parameters` is a local optimizer, so which minimum it finds
    depends on the initial guess. This tries `n_starts` random starts
    (sampled within `params`' min/max) plus the original guess, and
    returns the best result plus every successful run (compare
    `[r.result.chisqr for r in all_results]` to check for multimodality).

    `log_uniform_names`: parameters whose bounds span >~2 orders of
    magnitude should be sampled log-uniformly (needs `min > 0`), or
    linear sampling wastes most starts in a narrow slice of the range.
    """
    for name in log_uniform_names:
        p = params[name]
        if p.min is None or p.min <= 0:
            raise ValueError(f"{name!r} needs min > 0 for log-uniform sampling.")

    rng = np.random.default_rng(seed)
    varying_names = [name for name, p in params.items() if p.vary]
    for name in varying_names:
        p = params[name]
        if p.min is None or p.max is None:
            raise ValueError(
                f"{name!r} needs both min and max to sample starting points."
            )

    starts: list[dict[str, float] | None] = [None] if keep_original_guess else []
    starts += [
        _sample_start(params, varying_names, log_uniform_names, rng)
        for _ in range(n_starts)
    ]

    all_results = []
    for start in starts:
        trial_params = params.copy()
        if start is not None:
            for name, value in start.items():
                trial_params[name].set(value=value)
        try:
            all_results.append(
                fit_parameters(
                    model_func, trial_params, y_data=y_data, **fit_parameters_kwargs
                )
            )
        except Exception:
            continue

    if not all_results:
        raise RuntimeError(
            f"All {len(starts)} multi-start fits failed; check bounds and model_func."
        )

    best_result = min(all_results, key=lambda r: r.result.chisqr)
    return best_result, all_results


def _sample_start(
    params: lmfit.Parameters,
    varying_names: list[str],
    log_uniform_names: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample one random starting point within each varying parameter's bounds."""
    start = {}
    for name in varying_names:
        p = params[name]
        if name in log_uniform_names:
            start[name] = float(np.exp(rng.uniform(np.log(p.min), np.log(p.max))))
        else:
            start[name] = float(rng.uniform(p.min, p.max))
    return start


class ProfileScanner:
    """Computes a likelihood-profile confidence interval for one parameter.

    Refits with the target parameter fixed at a series of trial values
    (expanding outward from the best fit) until the chi-square crosses the
    F-test threshold, then brackets the crossing with brentq. Uses this
    module's bounded `least_squares` re-optimizer rather than
    `lmfit.conf_interval` (which always re-optimizes with legacy MINPACK
    `leastsq`, and can stall at the penalty discontinuity in `Objective`).
    """

    def __init__(
        self,
        result: FitResult,
        target_chisqr: float,
        method: str,
        minimize_kwargs: dict[str, Any],
    ):
        self.result = result
        self.target_chisqr = target_chisqr
        self.method = method
        self.minimize_kwargs = minimize_kwargs

    def chisqr_at(self, name: str, trial_value: float) -> float:
        """Refit with `name` fixed at trial_value; return the resulting chisqr."""
        trial_params = self.result.params.copy()
        trial_params[name].set(value=trial_value, vary=False)
        objective = Objective(
            self.result.model_func,
            self.result.y_data,
            self.result.weights,
            catch_exceptions=(),
            penalty=0.0,
        )
        out = lmfit.Minimizer(objective.residual, trial_params).minimize(
            method=self.method, **self.minimize_kwargs
        )
        return float(out.chisqr)

    def scan_side(
        self,
        name: str,
        direction: int,
        best_value: float,
        stderr: float,
        initial_step_in_stderr: float,
        step_growth: float,
        max_expansions: int,
    ) -> float:
        """Expand outward from best_value until chisqr crosses the threshold."""
        p = self.result.params[name]
        bound = p.min if direction < 0 else p.max
        step = direction * stderr * initial_step_in_stderr
        prev_value = best_value

        for _ in range(max_expansions):
            trial_value = prev_value + step
            hit_bound = bound is not None and (
                (direction < 0 and trial_value <= bound)
                or (direction > 0 and trial_value >= bound)
            )
            if hit_bound:
                trial_value = bound

            if self.chisqr_at(name, trial_value) >= self.target_chisqr:
                lo, hi = sorted((prev_value, trial_value))
                return brentq(
                    lambda x: self.chisqr_at(name, x) - self.target_chisqr,
                    lo,
                    hi,
                    xtol=abs(step) * 1e-3,
                )
            if hit_bound:
                return (
                    direction * np.inf
                )  # chisqr never crossed threshold within bounds

            prev_value = trial_value
            step *= step_growth

        return direction * np.inf  # ran out of expansions


@dataclass
class FitResult:
    """Wraps an lmfit.MinimizerResult with convenience accessors."""

    result: lmfit.minimizer.MinimizerResult
    model_func: ModelFunc  # solver-failure-safe prediction function (Objective.predict)
    y_data: np.ndarray
    weights: np.ndarray | None = field(default=None, repr=False)
    penalty: float | None = field(default=None, repr=False)

    @property
    def params(self) -> lmfit.Parameters:
        return self.result.params

    @property
    def best_fit(self) -> np.ndarray:
        return self.model_func(self.result.params)

    @property
    def residual(self) -> np.ndarray:
        return self.y_data - self.best_fit

    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residual**2)))

    def summary(self) -> str:
        return lmfit.fit_report(self.result)

    def ci(
        self, level: float = 0.95, method: str = "stderr", **kwargs: Any
    ) -> dict[str, tuple[float, float, float]]:
        """Confidence interval for every varying fitted parameter: {name: (lo, best, hi)}.

        method="stderr" (default, fast): linear/Wald interval from the
        covariance matrix -- fine for well-conditioned fits.
        method="profile" (slower, more rigorous for nonlinear models): see
        `ProfileScanner`. Costs many extra solves per parameter.
        """
        if method == "stderr":
            return self._ci_stderr(level)
        if method == "profile":
            return self._ci_profile(level, **kwargs)
        raise ValueError(f"Unknown ci method {method!r}; use 'stderr' or 'profile'.")

    def _ci_stderr(self, level: float) -> dict[str, tuple[float, float, float]]:
        dof = self.result.nfree
        if dof <= 0:
            raise ValueError(
                "No degrees of freedom left; cannot form a confidence interval."
            )
        t_crit = stats.t.ppf(0.5 + level / 2, dof)

        out = {}
        for name, p in self.result.params.items():
            if not p.vary:
                continue
            if p.stderr is None:
                raise ValueError(
                    f"No stderr reported for {name!r}; use method='profile' instead."
                )
            half_width = t_crit * p.stderr
            out[name] = (p.value - half_width, p.value, p.value + half_width)
        return out

    def _ci_profile(
        self,
        level: float,
        max_expansions: int = 12,
        initial_step_in_stderr: float = 1.0,
        step_growth: float = 1.6,
        method: str = "least_squares",
        **minimize_kwargs: Any,
    ) -> dict[str, tuple[float, float, float]]:
        dof = self.result.nfree
        if dof <= 0:
            raise ValueError("No degrees of freedom left; cannot form a CI.")
        f_crit = stats.f.ppf(level, 1, dof)
        target_chisqr = float(self.result.chisqr) * (1.0 + f_crit / dof)
        scanner = ProfileScanner(self, target_chisqr, method, minimize_kwargs)

        out = {}
        for name, p in self.result.params.items():
            if not p.vary:
                continue
            if not p.stderr or p.stderr <= 0:
                raise ValueError(
                    f"No usable stderr for {name!r}; cannot pick a scan step size."
                )
            lo = scanner.scan_side(
                name,
                -1,
                p.value,
                p.stderr,
                initial_step_in_stderr,
                step_growth,
                max_expansions,
            )
            hi = scanner.scan_side(
                name,
                +1,
                p.value,
                p.stderr,
                initial_step_in_stderr,
                step_growth,
                max_expansions,
            )
            out[name] = (lo, p.value, hi)
        return out

    def predict_band(
        self, level: float = 0.95, step: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delta-method confidence band on the model curve: (lower, upper).

        band = best_fit +/- t_crit * sqrt(diag(J @ cov @ J.T)), with J the
        finite-difference Jacobian of model_func at the optimum.
        """
        if self.result.covar is None:
            raise ValueError(
                "No covariance matrix available (fit method must report one, e.g. 'least_squares')."
            )

        varying = [name for name, p in self.result.params.items() if p.vary]
        best = self.result.params
        y0 = self.model_func(best)

        J = np.zeros((y0.size, len(varying)))
        for j, name in enumerate(varying):
            p = best[name]
            h = step or max(abs(p.value) * 1e-4, 1e-8)
            trial = best.copy()
            trial[name].value = p.value + h
            y_plus = self.model_func(trial)
            trial[name].value = p.value - h
            y_minus = self.model_func(trial)

            for label, y_trial in (("+", y_plus), ("-", y_minus)):
                if self.penalty is not None and np.all(y_trial == self.penalty):
                    raise RuntimeError(
                        f"Perturbing {name!r} ({label}{h:g}) made the model fail to solve; "
                        "predict_band's Jacobian is unreliable there. Try a smaller `step` "
                        f"or tighten the bounds on {name!r}."
                    )
            J[:, j] = (y_plus - y_minus) / (2 * h)

        pred_var = np.clip(np.einsum("ij,jk,ik->i", J, self.result.covar, J), 0, None)
        t_crit = stats.t.ppf(0.5 + level / 2, self.result.nfree)
        half_width = t_crit * np.sqrt(pred_var)
        return y0 - half_width, y0 + half_width
