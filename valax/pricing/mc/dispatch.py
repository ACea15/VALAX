"""Unified Monte Carlo dispatcher.

A thin registry layer on top of the path generators and payoff functions
already in ``valax/pricing/mc/``. Users call one function,
:func:`mc_price_dispatch`, which looks up a **recipe** keyed on the pair
``(type(instrument), type(model))`` and runs the appropriate path
generation + payoff + discounting sequence.

Design goals
------------

1. **One entry point.** Instead of remembering which path generator to
   pair with which payoff for each instrument, users pass
   ``mc_price_dispatch(instrument, model, config, key, **market_args)``.

2. **Extensible.** Contributors register new ``(instrument, model)``
   combinations with the :func:`register` decorator. The registry lives
   in the process and is populated at import time by
   :mod:`valax.pricing.mc.recipes`.

3. **Backward-compatible.** The existing :func:`valax.pricing.mc.mc_price`
   and :func:`mc_price_with_stderr` functions remain untouched. Users
   already using those keep working; the dispatcher is the new
   preferred API going forward.

4. **Composable with JAX transforms.** Because recipes call only
   JAX-pure functions, :func:`mc_price_dispatch` supports
   ``jax.grad`` / ``jax.jit`` / ``jax.vmap`` just like the underlying
   pieces did.

Usage
-----

.. code-block:: python

    from valax.instruments import EuropeanOption
    from valax.models import BlackScholesModel
    from valax.pricing.mc import mc_price_dispatch, MCConfig
    import jax, jax.numpy as jnp

    option = EuropeanOption(strike=jnp.array(100.0),
                            expiry=jnp.array(1.0), is_call=True)
    model = BlackScholesModel(vol=jnp.array(0.2),
                              rate=jnp.array(0.05),
                              dividend=jnp.array(0.02))

    result = mc_price_dispatch(
        option, model,
        config=MCConfig(n_paths=100_000, n_steps=100),
        key=jax.random.PRNGKey(42),
        spot=jnp.array(100.0),
    )
    print(f"{result.price:.4f} +/- {result.stderr:.4f}")

Registering a custom recipe
---------------------------

.. code-block:: python

    from valax.pricing.mc import register, MCResult

    @register(MyInstrument, MyModel)
    def _my_recipe(instrument, model, config, key, *, spot, **_):
        paths = generate_my_paths(model, spot, ...)
        cashflows = my_payoff(paths, instrument)
        df = jnp.exp(-model.rate * instrument.expiry)
        price = df * jnp.mean(cashflows)
        stderr = df * jnp.std(cashflows) / jnp.sqrt(config.n_paths)
        return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

# Re-export the existing MCConfig so users have a single import point
# for the new dispatcher API.
from valax.pricing.mc.engine import MCConfig


# ─────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────


class MCResult(eqx.Module):
    """Result of a Monte Carlo pricing call.

    Attributes:
        price: MC estimate of the price (scalar array).
        stderr: Standard error of the estimate (scalar array).
            Where standard-error estimation is not meaningful (e.g.,
            Longstaff-Schwartz inner regressions), this is ``0.0``.
        n_paths: Number of paths used (static int).

    Convenience:
        ``float(result)`` returns ``float(result.price)`` so the result
        interoperates with scalar-expecting APIs without unpacking.
    """

    price: Float[Array, ""]
    stderr: Float[Array, ""]
    n_paths: int = eqx.field(static=True)

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.price)


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────


# The registry maps (instrument_cls, model_cls) -> recipe function.
# Populated at import time by valax.pricing.mc.recipes.
_REGISTRY: dict[tuple[type, type], Callable[..., MCResult]] = {}


def register(
    instrument_cls: type,
    model_cls: type,
    *,
    overwrite: bool = False,
) -> Callable[[Callable[..., MCResult]], Callable[..., MCResult]]:
    """Decorator: register an MC recipe for ``(instrument_cls, model_cls)``.

    The decorated function must accept the keyword arguments
    ``instrument``, ``model``, ``config``, ``key`` and whatever
    ``market_args`` the recipe needs (e.g. ``spot`` for equity models,
    ``taus`` + ``forward_indices`` for LMM rate payoffs).

    Args:
        instrument_cls: The instrument pytree class (e.g. ``EuropeanOption``).
        model_cls: The model pytree class (e.g. ``BlackScholesModel``).
        overwrite: If ``True``, replace an existing registration for this
            pair. If ``False`` (default), raise ``ValueError`` on conflict.

    Returns:
        The decorator to apply to the recipe function.

    Example:
        >>> @register(MyInstrument, MyModel)
        ... def my_recipe(*, instrument, model, config, key, spot, **kwargs):
        ...     ...
        ...     return MCResult(price=..., stderr=..., n_paths=config.n_paths)
    """
    key = (instrument_cls, model_cls)

    def decorator(
        recipe: Callable[..., MCResult],
    ) -> Callable[..., MCResult]:
        if key in _REGISTRY and not overwrite:
            raise ValueError(
                f"An MC recipe is already registered for "
                f"({instrument_cls.__name__}, {model_cls.__name__}). "
                f"Pass overwrite=True to replace it.",
            )
        _REGISTRY[key] = recipe
        return recipe

    return decorator


def registered_recipes() -> list[tuple[str, str]]:
    """Return a sorted list of ``(instrument_name, model_name)`` tuples
    for every currently-registered recipe.

    Useful for introspection and error messages.
    """
    return sorted(
        (i.__name__, m.__name__) for i, m in _REGISTRY.keys()
    )


def _format_available() -> str:
    """Format the available recipes for error messages."""
    recipes = registered_recipes()
    if not recipes:
        return "  (no recipes registered)"
    return "\n".join(f"  ({i}, {m})" for i, m in recipes)


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def mc_price_dispatch(
    instrument: Any,
    model: Any,
    config: MCConfig,
    key: jax.Array,
    **market_args: Any,
) -> MCResult:
    """Price any instrument/model combination via Monte Carlo.

    Looks up the registered recipe for ``(type(instrument), type(model))``
    and runs it. The set of ``market_args`` required depends on the
    recipe — see :mod:`valax.pricing.mc.recipes` for the built-in set.

    Args:
        instrument: A VALAX instrument pytree (e.g. an ``EuropeanOption``,
            a ``Caplet``, a ``Swaption``).
        model: A VALAX stochastic-model pytree (e.g. ``BlackScholesModel``,
            ``HestonModel``, ``LMMModel``).
        config: :class:`MCConfig` with ``n_paths`` and ``n_steps``.
        key: JAX PRNG key.
        **market_args: Recipe-specific required arguments. Typical
            examples:

            - Equity recipes: ``spot`` (scalar spot price).
            - LMM rate recipes: ``forward_index`` / ``forward_indices``
              and ``taus`` (tenor mapping), plus the model carries its
              own initial curve.

    Returns:
        :class:`MCResult` with ``price``, ``stderr``, and ``n_paths``.

    Raises:
        ValueError: If no recipe is registered for
            ``(type(instrument), type(model))``. The error message lists
            all currently-registered combinations.

    Notes:
        The dispatcher is pure: you can wrap it in ``jax.grad``,
        ``jax.jit``, or ``jax.vmap`` as long as every recipe it could
        reach is itself pure (all built-in recipes are).
    """
    key_types = (type(instrument), type(model))
    recipe = _REGISTRY.get(key_types)
    if recipe is None:
        raise ValueError(
            f"No MC recipe registered for "
            f"({type(instrument).__name__}, {type(model).__name__}).\n"
            f"Available recipes:\n{_format_available()}\n"
            f"To add a new recipe, use "
            f"`from valax.pricing.mc import register` and decorate a "
            f"function taking (instrument, model, config, key, "
            f"**market_args) and returning an MCResult.",
        )
    return recipe(
        instrument=instrument,
        model=model,
        config=config,
        key=key,
        **market_args,
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers used by recipes
# ─────────────────────────────────────────────────────────────────────


def discounted_mean_and_stderr(
    cashflows: Float[Array, " n_paths"],
    discount_factor: Float[Array, ""] | Float[Array, " n_paths"],
    n_paths: int,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Discount path-wise cashflows and return ``(mean, stderr)``.

    For deterministic ``discount_factor`` (scalar): the standard MC
    formula ``DF * mean(cashflows)``, ``DF * std(cashflows) / sqrt(N)``.

    For path-wise ``discount_factor`` (array of length ``n_paths``):
    the product is taken per-path before averaging, which is the
    natural formulation for stochastic-rates pricing.

    Args:
        cashflows: Per-path payoffs at maturity.
        discount_factor: Either a scalar DF (deterministic rates) or
            a per-path DF array (stochastic rates).
        n_paths: Number of paths (static).

    Returns:
        ``(price, stderr)`` as JAX scalars.
    """
    sqrt_n = jnp.sqrt(jnp.array(n_paths, dtype=cashflows.dtype))
    if jnp.ndim(discount_factor) == 0:
        price = discount_factor * jnp.mean(cashflows)
        stderr = discount_factor * jnp.std(cashflows) / sqrt_n
    else:
        pv = discount_factor * cashflows
        price = jnp.mean(pv)
        stderr = jnp.std(pv) / sqrt_n
    return price, stderr
