"""Objective: model + data -> the residual lmfit minimizes.

Solver failures are handled automatically: if the model can't be solved at
a trial point, that point is given a huge residual so the optimizer backs
away from it. Users never configure this.
"""

from __future__ import annotations

from typing import Callable

import lmfit
import numpy as np

ModelFunc = Callable[[lmfit.Parameters], np.ndarray]

_SOLVER_FAILURES = (RuntimeError, ArithmeticError, ValueError, np.linalg.LinAlgError)


class Objective:
    """Wraps a model and data with a solver-failure safety net."""

    def __init__(
        self,
        model_func: ModelFunc,
        y_data: np.ndarray,
        weights: np.ndarray | None,
    ):
        self.model_func = model_func
        self.y_data = y_data
        self.weights = weights
        self.penalty = 1e3 * (
            1 + np.max(np.abs(y_data))
        )  # value substituted on failure
        self.n_failures = 0  # trial points penalized so far

    # -- prediction ---------------------------------------------------------

    def predict(self, p: lmfit.Parameters) -> np.ndarray:
        """Model prediction at `p`"""
        try:
            y_pred = np.asarray(self.model_func(p), dtype=float)
        except _SOLVER_FAILURES:
            return self._penalized()
        self._check_shape(y_pred)
        if not np.all(np.isfinite(y_pred)):
            return self._penalized()
        return y_pred

    def is_failed(self, y_pred: np.ndarray) -> bool:
        """Determine if prediction failed."""
        return bool(np.all(y_pred == self.penalty))

    def check_start(self, p: lmfit.Parameters) -> None:
        """Evaluate the model once, unprotected, so real errors surface immediately."""
        y_pred = np.asarray(self.model_func(p), dtype=float)
        self._check_shape(y_pred)
        if not np.all(np.isfinite(y_pred)):
            raise ValueError(
                "model_func returned non-finite values at the initial guess."
            )

    def _penalized(self) -> np.ndarray:
        self.n_failures += 1
        return np.full(self.y_data.shape, self.penalty)

    def _check_shape(self, y_pred: np.ndarray) -> None:
        if y_pred.shape != self.y_data.shape:
            raise ValueError(
                f"model_func returned shape {y_pred.shape}, "
                f"expected {self.y_data.shape}."
            )

    def residual(self, p: lmfit.Parameters) -> np.ndarray:
        """Weighted difference `(y_pred - y_data) * weights`, flattened."""
        resid = self.predict(p) - self.y_data
        if self.weights is not None:
            resid = resid * self.weights
        return resid.ravel()
