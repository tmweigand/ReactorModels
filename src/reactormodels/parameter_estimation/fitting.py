"""Entry points: build parameters and run (multi-start) fits."""

from __future__ import annotations

import warnings
from typing import Any

import lmfit
import numpy as np

from .objective import ModelFunc, Objective
from .result import FitResult


def make_parameters(**specs: dict[str, Any]) -> lmfit.Parameters:
    """Build `lmfit.Parameters` from per-parameter options.

    Supported options are:
        `value` (default `None`),
        `vary` (default `True`),
        `min`/`max` (default negative/positive infinity),
        `expr` (default `None`),
        `brute_step` (default `None`)
        `user_data` (default `None`).

    `vary` controls whether a parameter is optimized; expression-constrained
    parameters are fixed by lmfit.
    """
    params = lmfit.Parameters()
    for name, spec in specs.items():
        supported_options = {
            "value",
            "vary",
            "min",
            "max",
            "expr",
            "brute_step",
            "user_data",
        }
        unsupported_options = spec.keys() - supported_options
        if unsupported_options:
            options = ", ".join(sorted(unsupported_options))
            raise TypeError(f"Unsupported option(s) for parameter {name!r}: {options}")

        params.add(
            name=name,
            value=spec.get("value"),
            vary=spec.get("vary", True),
            min=spec.get("min", -np.inf),
            max=spec.get("max", np.inf),
            expr=spec.get("expr"),
            brute_step=spec.get("brute_step"),
            user_data=spec.get("user_data"),
        )
    return params


def fit_parameters(
    model_func: ModelFunc,
    params: lmfit.Parameters,
    y_data: np.ndarray,
    weights: np.ndarray | None = None,
    method: str = "least_squares",
    **minimize_kwargs: Any,
) -> FitResult:
    """Fit `model_func` to `y_data`.

    Model failures at trial points are handled automatically. The fit raises
    if the model fails at the initial guess or at the final answer.
    """
    return _run_fit(
        model_func,
        params,
        y_data,
        strict_start=True,
        weights=weights,
        method=method,
        **minimize_kwargs,
    )


def _run_fit(
    model_func: ModelFunc,
    params: lmfit.Parameters,
    y_data: np.ndarray,
    *,
    strict_start: bool,
    weights: np.ndarray | None = None,
    method: str = "least_squares",
    **minimize_kwargs: Any,
) -> FitResult:
    """Shared fit routine. `strict_start` raises model errors at the initial guess."""
    y_data = np.asarray(y_data, dtype=float)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)

    objective = Objective(model_func, y_data, weights)
    if strict_start:
        objective.check_start(params)

    result = lmfit.Minimizer(objective.residual, params).minimize(
        method=method, **minimize_kwargs
    )

    n_failures = objective.n_failures
    if objective.is_failed(objective.predict(result.params)):
        raise RuntimeError(
            "The fit ended where the model can't be solved; "
            "check parameter bounds and model_func."
        )
    if n_failures:
        warnings.warn(
            f"model_func could not be solved at {n_failures}"
            " trial point(s) during the fit; "
            "those points were avoided. Tighter parameter bounds may speed things up."
        )
    return FitResult(result=result, objective=objective)


def fit_parameters_multistart(
    model_func: ModelFunc,
    params: lmfit.Parameters,
    y_data: np.ndarray,
    n_starts: int = 12,
    seed: int | None = 0,
    log_uniform_names: tuple[str, ...] = (),
    keep_original_guess: bool = True,
    start_ranges: dict[str, tuple[float, float]] | None = None,
    **fit_kwargs: Any,
) -> tuple[FitResult, list[FitResult]]:
    """Run `fit_parameters` from several random starts; return `(best, all)`.

    Starts are sampled within a range for each varying parameter, plus the
    original guess if `keep_original_guess`. Compare
    `[r.result.chisqr for r in all_results]` to check for multimodality.

    `start_ranges`: `{name: (lo, hi)}` windows to sample starts from. Defaults
    to the parameter's `min`/`max`. Required for any varying parameter with an
    infinite bound; also useful for huge but finite bounds (e.g. [0, 1e12]).
    Ranges are intersected with the parameter's own bounds. The fit itself
    still uses the original bounds.

    `log_uniform_names`: parameters sampled log-uniformly (sampling range
    needs lo > 0); use for ranges spanning more than ~2 orders of magnitude.
    Other kwargs (`weights`, `method`, ...) go to `fit_parameters`.
    """
    varying = [name for name, p in params.items() if p.vary]
    start_ranges = start_ranges or {}

    unknown = (set(start_ranges) | set(log_uniform_names)) - set(varying)
    if unknown:
        raise ValueError(
            f"start_ranges/log_uniform_names reference non-varying or unknown "
            f"parameters: {sorted(unknown)}"
        )

    # Resolve and validate the sampling window for every varying parameter.
    ranges: dict[str, tuple[float, float]] = {}
    for name in varying:
        p = params[name]
        lo, hi = start_ranges.get(name, (p.min, p.max))
        lo, hi = max(lo, p.min), min(hi, p.max)  # stay inside the fit bounds
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(
                f"{name!r} has an infinite bound; pass "
                f"start_ranges={{{name!r}: (lo, hi)}} to sample starts."
            )
        if lo >= hi:
            raise ValueError(
                f"{name!r} has an empty sampling range (lo={lo}, hi={hi})."
            )
        if name in log_uniform_names and lo <= 0:
            raise ValueError(
                f"{name!r} needs a sampling range with lo > 0 for log-uniform "
                f"sampling (got lo={lo})."
            )
        ranges[name] = (lo, hi)

    rng = np.random.default_rng(seed)
    starts: list[dict[str, float] | None] = [None] if keep_original_guess else []
    starts += [_sample_start(ranges, log_uniform_names, rng) for _ in range(n_starts)]

    all_results = []
    for start in starts:
        trial = params.copy()
        for name, value in (start or {}).items():
            trial[name].set(value=value)
        try:
            all_results.append(
                _run_fit(
                    model_func,
                    trial,
                    y_data,
                    strict_start=start
                    is None,  # only the user's own guess must be solvable
                    **fit_kwargs,
                )
            )
        except Exception as exc:  # one bad start shouldn't kill the rest
            warnings.warn(f"A multistart fit failed ({exc}); skipping it.")

    if not all_results:
        raise RuntimeError(
            f"All {len(starts)} multi-start fits failed; check bounds and model_func."
        )
    best = min(all_results, key=lambda r: r.result.chisqr)
    return best, all_results


def _sample_start(
    ranges: dict[str, tuple[float, float]],
    log_uniform_names: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample one starting point uniformly (or log-uniformly) within `ranges`."""
    start = {}
    for name, (lo, hi) in ranges.items():
        if name in log_uniform_names:
            start[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            start[name] = float(rng.uniform(lo, hi))
    return start
