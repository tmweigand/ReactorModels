"""result.py

Fitted parameters and fit-quality accessors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lmfit
import numpy as np

from . import confidence
from .objective import ModelFunc, Objective


@dataclass
class FitResult:
    """An `lmfit.MinimizerResult` plus the `Objective` that produced it."""

    result: lmfit.minimizer.MinimizerResult
    objective: Objective

    @property
    def params(self) -> lmfit.Parameters:
        """Optimized parameters"""
        return self.result.params

    @property
    def y_data(self) -> np.ndarray:
        """Data to fit to"""
        return self.objective.y_data

    @property
    def model_func(self) -> ModelFunc:
        """Failure-safe prediction function."""
        return self.objective.predict

    @property
    def best_fit(self) -> np.ndarray:
        """Return model solution at optimized parameters"""
        return self.model_func(self.params)

    @property
    def residual(self) -> np.ndarray:
        """Return residual of fit"""
        """Data-space residual `y_data - best_fit` (not the optimized residual)."""
        return self.y_data - self.best_fit

    def rmse(self) -> float:
        """Return rmse of fit"""
        return float(np.sqrt(np.mean(self.residual**2)))

    def summary(self) -> str:
        """Return lmfit report"""
        return lmfit.fit_report(self.result)

    # -- uncertainty (see confidence.py) ------------------------------------

    def confidence_interval(
        self, level: float = 0.95, method: str = "stderr", **kwargs: Any
    ) -> dict[str, tuple[float, float, float]]:
        """Confidence interval per varying parameter: `{name: (lo, best, hi)}`.

        method="stderr" (fast, linear) or "profile_likelihood" (slower, better for
        nonlinear models; extra kwargs go to `confidence.profile_interval`).
        """
        if method == "stderr":
            return confidence.stderr_interval(self, level)
        if method == "profile_likelihood":
            return confidence.profile_likelihood_interval(self, level, **kwargs)
        raise ValueError(
            f"Unknown ci method {method!r}; use 'stderr' or 'profile_likelihood'."
        )

    def predict_band(
        self, level: float = 0.95, step: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Confidence band on the model curve: `(lower, upper)`."""
        return confidence.prediction_band(self, level, step)
