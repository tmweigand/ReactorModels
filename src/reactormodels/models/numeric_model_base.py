"""base.py"""

from __future__ import annotations

from typing import Any

__all__ = ["NumericModel"]


class NumericModel:
    """Base class for IDA-based numeric transport models.

    Subclasses are expected to implement the DAE callbacks used by the IDA
    solver (`_residual`, `_jacobian`, `_initial_conditions`,
    `_algebraic_vars_idx`) and a `solve` method that drives the time
    integration. Signatures for `_initial_conditions` and `solve` differ per
    model, so they are intentionally not enforced here.

    Subclasses should set the class attribute `_param_names` to the tuple of
    attribute names that represent the model's required physical inputs, and
    call `self.assert_parameters_set()` at the end of `__init__` to fail fast
    if any required parameter was not supplied.
    """

    _param_names: tuple[str, ...] = ()

    def parameters(self) -> dict[str, Any]:
        """Return the model's configured parameters as {name: value}."""
        return {name: getattr(self, name) for name in self._param_names}

    def assert_parameters_set(self) -> None:
        """Raise ValueError if any required parameter is None."""
        missing = [name for name, value in self.parameters().items() if value is None]
        if missing:
            raise ValueError(
                f"{type(self).__name__} is missing required parameter(s): "
                f"{', '.join(missing)}"
            )
