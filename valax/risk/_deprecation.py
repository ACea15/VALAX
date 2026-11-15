"""Deprecation helpers for the ``*Ladder`` → ``*Sensitivities`` rename.

VALAX used to call its base-point bucketed Greeks a *ladder*
(``SensitivityLadder``, ``KeyedLadder``, ...).  Industry usage (FRTB SA,
ISDA SIMM) calls that object *sensitivities*, while desks use *ladder*
for revaluation at shifted levels.  The objects were renamed; the old
names stay importable with a :class:`DeprecationWarning` until the
removal the roadmap schedules (``docs/roadmap.md``, "Remove the
deprecated ``*Ladder`` aliases").

Everything here runs at the user boundary (import / call time, before
tracing), never inside traced code.
"""

import functools
import warnings
from typing import Any, Callable

#: Old public name → new public name, for every renamed symbol.
RENAMED_SYMBOLS: dict[str, str] = {
    "SensitivityLadder": "FlatSensitivities",
    "compute_ladder": "compute_flat_sensitivities",
    "KeyedLadder": "KeyedSensitivities",
    "keyed_ladder": "keyed_sensitivities",
    "BucketedLadder": "BucketedSensitivities",
    "bucket_sensitivity_ladder": "bucket_sensitivities",
}


def warn_renamed(old: str, new: str, stacklevel: int = 3) -> None:
    """Emit the standard rename :class:`DeprecationWarning`.

    Args:
        old: Deprecated name.
        new: Replacement name.
        stacklevel: Passed to :func:`warnings.warn` so the warning points
            at the caller's line.
    """
    warnings.warn(
        f"{old} is deprecated; use {new} instead. VALAX now calls base-point "
        "bucketed Greeks 'sensitivities' and reserves 'ladder' for "
        "revaluation at shifted levels.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def renamed_kwarg(old: str, new: str) -> Callable:
    """Decorator forwarding a renamed keyword argument with a warning.

    Positional calls are unaffected; ``f(..., old=x)`` becomes
    ``f(..., new=x)`` and warns.

    Args:
        old: Deprecated keyword name.
        new: Current keyword name.

    Returns:
        A decorator.

    Raises:
        TypeError: (from the wrapped call) if both names are passed.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if old in kwargs:
                if new in kwargs:
                    raise TypeError(
                        f"{fn.__name__}() got both {old!r} (deprecated) and "
                        f"{new!r}; pass only {new!r}."
                    )
                warn_renamed(
                    f"The {old!r} argument of {fn.__name__}()",
                    repr(new),
                )
                kwargs[new] = kwargs.pop(old)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_module_getattr(
    module_name: str,
    namespace: dict[str, Any],
    renamed: dict[str, str],
) -> Callable[[str], Any]:
    """Build a module-level ``__getattr__`` that serves deprecated aliases.

    Args:
        module_name: ``__name__`` of the module installing the hook (for
            the ``AttributeError`` message).
        namespace: The module's ``globals()``, which holds the new names.
        renamed: Old name → new name for the aliases this module serves.

    Returns:
        A ``__getattr__(name)`` function.
    """
    def __getattr__(name: str) -> Any:
        if name in renamed:
            new = renamed[name]
            warn_renamed(f"{module_name}.{name}", f"{module_name}.{new}")
            return namespace[new]
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
