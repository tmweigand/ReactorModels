"""base.py"""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ["NumericModel"]


class NumericModel:
    """Base class for IDA-based numeric transport models.

    Subclasses are expected to implement the DAE callbacks used by the IDA
    solver:
        _residual, _jacobian, _initial_conditions, and _algebraic_vars_idx

    and a `solve` method that drives the time integration. Signatures for
    `_initial_conditions` and `solve` differ per model, so they are
    intentionally not enforced here.

    Subclasses should set the class attribute `_param_names` to the tuple of
    attribute names that represent the model's required physical inputs, and
    call `self.assert_parameters_set()` at the start of `solve()` to fail
    fast if any required parameter was not supplied. It's checked in
    `solve()` rather than `__init__` so that models can be constructed with
    parameters still unset (e.g. while calibrating them) and only need to
    be complete once you actually try to run them.
    """

    _param_names: tuple[str, ...] = ()

    def parameters(self) -> dict[str, Any]:
        """Return the model's configured parameters as {name: value}."""
        return {name: getattr(self, name) for name in self._param_names}

    def assert_parameters_set(self) -> None:
        """Raise ValueError if any required parameter is None."""
        missing: list[str] = []

        for name, value in self.parameters().items():
            if value is None:
                missing.append(name)
                continue

            class_name = type(value).__name__

            missing.extend(
                f"{class_name}.{sub_name}"
                for sub_name in _unset_constructor_params(value)
            )

        if missing:
            raise ValueError(
                f"{type(self).__name__} is missing required parameter(s): "
                f"{', '.join(missing)}"
            )


def _unset_constructor_params(obj: Any) -> list[str]:
    """Names of obj's __init__ parameters whose current attribute value is None.

    One level only -- deliberately not recursive, so this stays a plain,
    fast check on a model's directly-declared parameters and their
    immediate composite objects (isotherm, boundary conditions, etc.),
    rather than a general object-graph walk.
    """
    sig = inspect.signature(type(obj).__init__)
    names = [
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    return [n for n in names if hasattr(obj, n) and getattr(obj, n) is None]
