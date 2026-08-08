"""Unified PDE dispatcher.

A registry layer over the finite-difference drivers, mirroring
:mod:`valax.pricing.mc.dispatch`. Users call one function,
:func:`pde_price_dispatch`, which looks up a **recipe** keyed on the pair
``(type(instrument), type(model))`` and runs the appropriate grid + operator +
boundary + terminal + backward-sweep sequence.

The registry is populated at import time by :mod:`valax.pricing.pde.recipes`.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
from jaxtyping import Float
from jax import Array


class PDEResult(eqx.Module):
    """Result of a PDE pricing call.

    Attributes:
        price: The PDE estimate of the price (scalar array).

    Convenience:
        ``float(result)`` returns ``float(result.price)`` so the result
        interoperates with scalar-expecting APIs without unpacking.
    """

    price: Float[Array, ""]

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.price)


# Registry: (instrument_cls, model_cls) -> recipe. Populated by recipes.py.
_REGISTRY: dict[tuple[type, type], Callable[..., PDEResult]] = {}


def register(
    instrument_cls: type,
    model_cls: type,
    *,
    overwrite: bool = False,
) -> Callable[[Callable[..., PDEResult]], Callable[..., PDEResult]]:
    """Decorator: register a PDE recipe for ``(instrument_cls, model_cls)``.

    The decorated function must accept keyword arguments ``instrument``,
    ``model``, ``config`` and whatever ``market_args`` the recipe needs
    (``spot`` for the equity recipes).

    Args:
        instrument_cls: The instrument pytree class (e.g. ``AmericanOption``).
        model_cls: The model pytree class (e.g. ``BlackScholesModel``).
        overwrite: If True, replace an existing registration; otherwise raise.

    Returns:
        The decorator to apply to the recipe function.
    """
    key = (instrument_cls, model_cls)

    def decorator(recipe: Callable[..., PDEResult]) -> Callable[..., PDEResult]:
        if key in _REGISTRY and not overwrite:
            raise ValueError(
                f"A PDE recipe is already registered for "
                f"({instrument_cls.__name__}, {model_cls.__name__}). "
                f"Pass overwrite=True to replace it.",
            )
        _REGISTRY[key] = recipe
        return recipe

    return decorator


def registered_recipes() -> list[tuple[str, str]]:
    """Return a sorted list of ``(instrument_name, model_name)`` for every
    currently-registered recipe."""
    return sorted((i.__name__, m.__name__) for i, m in _REGISTRY.keys())


def _format_available() -> str:
    recipes = registered_recipes()
    if not recipes:
        return "  (no recipes registered)"
    return "\n".join(f"  ({i}, {m})" for i, m in recipes)


def pde_price_dispatch(
    instrument: Any,
    model: Any,
    config: Any,
    **market_args: Any,
) -> PDEResult:
    """Price an instrument/model combination via finite differences.

    Looks up the registered recipe for ``(type(instrument), type(model))`` and
    runs it.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree.
        config: A :class:`~valax.pricing.pde.config.PDEConfig` (or
            ``PDEConfig2D`` for 2-D recipes).
        **market_args: Recipe-specific arguments (e.g. ``spot``).

    Returns:
        A :class:`PDEResult`.

    Raises:
        ValueError: If no recipe is registered for the instrument/model pair.
            The message lists all currently-registered combinations.
    """
    key_types = (type(instrument), type(model))
    recipe = _REGISTRY.get(key_types)
    if recipe is None:
        raise ValueError(
            f"No PDE recipe registered for "
            f"({type(instrument).__name__}, {type(model).__name__}).\n"
            f"Available recipes:\n{_format_available()}\n"
            f"To add a new recipe, use "
            f"`from valax.pricing.pde import register` and decorate a function "
            f"taking (instrument, model, config, **market_args) and returning a "
            f"PDEResult.",
        )
    return recipe(instrument=instrument, model=model, config=config, **market_args)
