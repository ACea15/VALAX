"""The unified pricing signature: one ``price(...)`` over every method.

Three call conventions used to coexist — loose positional scalars for the
analytic pricers, ``(instrument, model, config, **market_args)`` for the
MC/PDE dispatchers, and ``(instrument, market)`` for the risk layer.  This
module collapses them into a single contract.  The user-facing walkthrough is
``docs/guide/tutorial-unified-pricing.md`` (reference:
``docs/api/core.md``); the design record is ``PRICING_ONTOLOGY.md`` §2.3:

.. code-block:: python

    price(
        instrument,          # contract data + static market selectors
        model,               # dynamics + calibrated params (None ⇒ pure discounting)
        market,              # MarketState (curves + surfaces + spots + …)
        *,
        method="auto",       # "analytic" | "lattice" | "pde" | "mc" | "auto"
        config=None,         # method-specific numerical settings
        key=None,            # PRNG key for MC
    ) -> PricingResult

**No pricer was rewritten to make this work.**  Existing pricers are
reached through thin adapters (:mod:`valax.core.adapters`), and the MC /
PDE recipe registries are *bridged*, not duplicated: ``price(...,
method="mc")`` delegates to
:func:`~valax.pricing.mc.dispatch.mc_price_dispatch`, so every recipe
registered there is automatically reachable here and the two can never
drift apart.

Method selection (resolved question Q2)
---------------------------------------

Availability and preference are kept separate:

* the **registry** ``(instrument_cls, model_cls, method) -> adapter``
  answers *what can be priced*;
* the **policy table** :data:`_AUTO_POLICY` answers *what should be*,
  pair by pair, and is populated by :func:`set_auto_method`.

``method="auto"`` takes the pair's policy entry, and only that: every pair
the library registers has one (:mod:`valax.core.policy`), and a pair with
pricers but no entry is an error rather than a guess, so adding a pricer
never silently changes what ``auto`` means for an existing pair.

Growth happens in the dispatch, never in the signature: new methods and
new ``(instrument, model)`` pairs register; ``price``'s parameters do not
change.  Open ``**market_args`` survive only at the internal recipe
layer, where the legacy dispatchers keep them.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.core.binding import bind_model
from valax.core.market_state import MarketState
from valax.core.resolve import ResolvedMarket, resolve


# ── Result container (resolved question Q1) ──────────────────────────


class PricingResult(eqx.Module):
    """Result of a :func:`price` call — one container for every method.

    Generalises :class:`~valax.pricing.pde.dispatch.PDEResult` and
    :class:`~valax.pricing.mc.dispatch.MCResult`.  Those remain the
    back-ends' own return types; ``price`` lifts them here so a caller
    that said ``method="auto"`` never has to branch on the method the
    dispatcher picked for it.

    Attributes:
        price: The price (scalar array).
        stderr: Sampling standard error, or ``None`` for methods that
            have none (analytic, lattice, PDE).  ``None`` rather than
            ``0.0`` so "no sampling error" is distinguishable from
            "sampling error measured as zero".
        method: The method actually used — ``"analytic"``, ``"lattice"``,
            ``"pde"``, or ``"mc"``.  Static field.
        diagnostics: Optional method-specific extras (pytree of arrays).
        config: The config actually used — the caller's, else the pair's
            default (:func:`effective_config`); ``None`` for a configless
            analytic route.  Static field.

    Convenience:
        ``float(result)`` returns ``float(result.price)`` so the result
        interoperates with scalar-expecting APIs without unpacking.
    """

    price: Float[Array, ""]
    stderr: Float[Array, ""] | None = None
    method: str = eqx.field(static=True, default="")
    diagnostics: dict = eqx.field(default_factory=dict)
    config: Any = eqx.field(static=True, default=None)

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.price)


# ── Registry ─────────────────────────────────────────────────────────


#: The methods the unified layer knows about.
METHODS = ("analytic", "lattice", "pde", "mc")

#: Order in which :func:`available_methods` lists methods (exact-and-cheap
#: first, sampling last). Not a selection rule: ``method="auto"`` comes from
#: the explicit policy of each pair (:mod:`valax.core.policy`).
_METHOD_PRIORITY = METHODS

#: ``(instrument_cls, model_cls, method) -> adapter``.  Populated by
#: :func:`register_pricer` (see :mod:`valax.core.adapters`).
_REGISTRY: dict[tuple[type, type, str], Callable[..., PricingResult]] = {}

#: ``(instrument_cls, model_cls) -> method``.  The auto-selection policy,
#: deliberately separate from the registry (Q2).
_AUTO_POLICY: dict[tuple[type, type], str] = {}

#: ``(instrument_cls, model_cls, method) -> config``: the default config of
#: each pair's method (the config half of the policy), used when ``price``
#: is called with ``config=None``.  Populated by :func:`set_default_config`.
_DEFAULT_CONFIGS: dict[tuple[type, type, str], Any] = {}

#: Methods that need no numerical config: ``config=None`` is valid for them
#: when the pair registers no default.
CONFIGLESS_METHODS = ("analytic",)

#: Set once the analytic/lattice adapters and the legacy MC/PDE recipe
#: registries have been imported.  Kept lazy so ``import valax`` does not
#: pull in every recipe module.
_BACKENDS_LOADED = False


def _validate_method(method: str) -> None:
    """Raise :class:`ValueError` unless ``method`` is a known method."""
    if method not in METHODS:
        raise ValueError(
            f"Unknown pricing method {method!r}. Choose one of "
            f"{METHODS} or 'auto'."
        )


def register_pricer(
    instrument_cls: type,
    model_cls: type,
    method: str,
    *,
    overwrite: bool = False,
) -> Callable[[Callable[..., PricingResult]], Callable[..., PricingResult]]:
    """Decorator: register an adapter for one (instrument, model, method).

    The decorated function is called as
    ``adapter(instrument=..., model=..., market=..., resolved=...,
    config=..., key=...)`` and must return a :class:`PricingResult`.
    ``resolved`` is the :class:`~valax.core.resolve.ResolvedMarket` for
    the pair, so adapters never touch raw ids.

    Args:
        instrument_cls: Instrument pytree class.
        model_cls: Model pytree class; use ``type(None)`` for
            curve-discounted (model-free) pricing.
        method: One of :data:`METHODS`.
        overwrite: Replace an existing registration instead of raising.

    Returns:
        The decorator.

    Raises:
        ValueError: On an unknown ``method``, or a duplicate
            registration without ``overwrite=True``.
    """
    _validate_method(method)
    key = (instrument_cls, model_cls, method)

    def decorator(
        adapter: Callable[..., PricingResult],
    ) -> Callable[..., PricingResult]:
        if key in _REGISTRY and not overwrite:
            raise ValueError(
                f"A {method} pricer is already registered for "
                f"({instrument_cls.__name__}, {model_cls.__name__}). "
                f"Pass overwrite=True to replace it."
            )
        _REGISTRY[key] = adapter
        return adapter

    return decorator


def set_auto_method(
    instrument_cls: type,
    model_cls: type,
    method: str,
) -> None:
    """Declare the ``method="auto"`` default for one (instrument, model).

    Args:
        instrument_cls: Instrument pytree class.
        model_cls: Model pytree class (``type(None)`` for model-free).
        method: The preferred method for this pair.

    Raises:
        ValueError: If ``method`` is not a known method.
    """
    _validate_method(method)
    _AUTO_POLICY[(instrument_cls, model_cls)] = method


def set_default_config(
    instrument_cls: type,
    model_cls: type,
    method: str,
    config: Any,
) -> None:
    """Declare the default config of one (instrument, model, method).

    The config half of the pricing policy: ``price(..., config=None)`` uses
    it, so every numerical method can be called without settings, and an
    instrument that needs finer settings than its siblings gets its own.
    Configs are all-static, so the default is part of the compiled program.

    Args:
        instrument_cls: Instrument pytree class.
        model_cls: Model pytree class (``type(None)`` for model-free).
        method: One of :data:`METHODS`.
        config: The method's config (``PDEConfig``, ``MCConfig``, …).

    Raises:
        ValueError: If ``method`` is not a known method or ``config`` is
            ``None``.
    """
    _validate_method(method)
    if config is None:
        raise ValueError("A default config cannot be None.")
    _DEFAULT_CONFIGS[(instrument_cls, model_cls, method)] = config


def default_config(instrument: Any, model: Any, method: str) -> Any:
    """The registered default config of one (instrument, model, method).

    Looked up through the instrument's base classes, as the pricers are.

    Args:
        instrument: Instrument pytree (or class).
        model: Model pytree (or class), possibly ``None``.
        method: A concrete method (not ``"auto"``).

    Returns:
        The default config, or ``None`` if none is registered.
    """
    _ensure_backends()
    instrument_cls = instrument if isinstance(instrument, type) else type(instrument)
    model_cls = model if isinstance(model, type) else type(model)
    for cls in _instrument_bases(instrument_cls):
        found = _DEFAULT_CONFIGS.get((cls, model_cls, method))
        if found is not None:
            return found
    return None


def effective_config(
    instrument: Any,
    model: Any,
    method: str = "auto",
    config: Any = None,
) -> Any:
    """The config :func:`price` actually uses for this call.

    Precedence: an explicit ``config`` > the pair's registered default for
    the method (:func:`set_default_config`) > ``None`` for a configless method
    (:data:`CONFIGLESS_METHODS`) > error. Calibrations record this as their
    engine provenance, so a fit priced with ``config=None`` is bound to the
    grid it actually used.

    Args:
        instrument: Instrument pytree (or class).
        model: Model pytree (or class), possibly ``None``.
        method: The pricing method, or ``"auto"`` (resolved by the policy).
        config: The caller's config, possibly ``None``.

    Returns:
        The config used, or ``None`` for a configless method without one.

    Raises:
        ValueError: If a method that needs a config has none registered for
            the pair and none was passed.
    """
    _ensure_backends()
    if method == "auto":
        method = auto_method(instrument, model)
    else:
        _validate_method(method)
    if config is not None:
        return config
    found = default_config(instrument, model, method)
    if found is not None or method in CONFIGLESS_METHODS:
        return found
    instrument_cls = instrument if isinstance(instrument, type) else type(instrument)
    model_cls = model if isinstance(model, type) else type(model)
    raise ValueError(
        f"No default {method} config is registered for "
        f"({instrument_cls.__name__}, {model_cls.__name__}). Pass config=, or "
        f"declare one with set_default_config({instrument_cls.__name__}, "
        f"{model_cls.__name__}, {method!r}, ...)."
    )


def _ensure_backends() -> None:
    """Import the adapters and the legacy recipe registries once."""
    global _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return
    # Deferred: keeps ``import valax`` light and avoids an import cycle
    # (adapters import this module).
    from valax.core import adapters  # noqa: F401
    from valax.pricing import mc  # noqa: F401
    from valax.pricing import pde  # noqa: F401
    from valax.core import policy  # noqa: F401  (after every registry)

    _BACKENDS_LOADED = True


def _instrument_bases(instrument_cls: type) -> tuple[type, ...]:
    """Lookup order for ``instrument_cls``: itself, then its base classes.

    A subclass that only *adds* market selectors — the P3 pattern: same
    contract, now naming the curves it discounts and projects on —
    inherits its base class's pricers instead of needing a duplicate
    registration.  Resolution follows the MRO, so a registration on the
    subclass still wins over one on the base.

    Args:
        instrument_cls: The instrument class.

    Returns:
        The candidate classes, most specific first.
    """
    return tuple(
        cls for cls in instrument_cls.__mro__ if cls not in (object, eqx.Module)
    )


def _legacy_registry(method: str) -> dict:
    """The MC or PDE recipe registry, bridged rather than duplicated."""
    if method == "mc":
        from valax.pricing.mc.dispatch import _REGISTRY as mc_registry

        return mc_registry
    from valax.pricing.pde.dispatch import _REGISTRY as pde_registry

    return pde_registry


def available_methods(instrument: Any, model: Any) -> tuple[str, ...]:
    """Methods able to price this (instrument, model) pair.

    Unions the unified registry with the legacy MC/PDE recipe
    registries, so a recipe registered the old way is visible here with
    no extra bookkeeping.

    Args:
        instrument: Instrument pytree (or class).
        model: Model pytree (or class), possibly ``None``.

    Returns:
        The available method names, in :data:`_METHOD_PRIORITY` order.
    """
    _ensure_backends()
    instrument_cls = instrument if isinstance(instrument, type) else type(instrument)
    model_cls = model if isinstance(model, type) else type(model)

    candidates = _instrument_bases(instrument_cls)
    found = {
        method
        for (i_cls, m_cls, method) in _REGISTRY
        if i_cls in candidates and m_cls is model_cls
    }
    for method in ("pde", "mc"):
        registry = _legacy_registry(method)
        if any((cls, model_cls) in registry for cls in candidates):
            found.add(method)
    return tuple(m for m in _METHOD_PRIORITY if m in found)


def auto_method(instrument: Any, model: Any) -> str:
    """Resolve ``method="auto"`` for one (instrument, model) pair.

    The answer comes from the pair's policy only (:func:`set_auto_method`,
    looked up through the instrument's base classes; the library's entries
    are in :mod:`valax.core.policy`). There is no fallback order: a pair
    with pricers but no policy is an error, so adding a pricer never
    silently changes what ``auto`` means for an existing pair.

    Args:
        instrument: Instrument pytree (or class).
        model: Model pytree (or class), possibly ``None``.

    Returns:
        The chosen method name.

    Raises:
        ValueError: If no method can price the pair (the message lists
            everything that is registered), if the pair has no policy, or if
            its policy names a method no pricer is registered for.
    """
    instrument_cls = instrument if isinstance(instrument, type) else type(instrument)
    model_cls = model if isinstance(model, type) else type(model)
    methods = available_methods(instrument_cls, model_cls)
    if not methods:
        raise ValueError(
            f"No pricer registered for "
            f"({instrument_cls.__name__}, {model_cls.__name__}).\n"
            f"Available (instrument, model, method) combinations:\n"
            f"{_format_available()}"
        )
    for cls in _instrument_bases(instrument_cls):
        preferred = _AUTO_POLICY.get((cls, model_cls))
        if preferred is None:
            continue
        if preferred not in methods:
            raise ValueError(
                f"The auto policy of ({cls.__name__}, {model_cls.__name__}) "
                f"names {preferred!r}, but only {methods} are registered for "
                f"({instrument_cls.__name__}, {model_cls.__name__})."
            )
        return preferred
    raise ValueError(
        f"({instrument_cls.__name__}, {model_cls.__name__}) has pricers "
        f"{methods} but no auto policy. Pass method= explicitly, or declare "
        f"the default with set_auto_method({instrument_cls.__name__}, "
        f"{model_cls.__name__}, ...)."
    )


def registered_pricers() -> list[tuple[str, str, str]]:
    """Every ``(instrument, model, method)`` reachable through ``price``.

    Returns:
        A sorted list of name triples, including the recipes bridged
        from the legacy MC and PDE registries.
    """
    _ensure_backends()
    entries = {
        (i.__name__, m.__name__, method)
        for (i, m, method) in _REGISTRY
    }
    for method in ("pde", "mc"):
        for (i, m) in _legacy_registry(method):
            entries.add((i.__name__, m.__name__, method))
    return sorted(entries)


def _format_available() -> str:
    """Format the registered combinations for error messages."""
    pricers = registered_pricers()
    if not pricers:
        return "  (no pricers registered)"
    return "\n".join(f"  ({i}, {m}, {method})" for i, m, method in pricers)


# ── Legacy back-end bridging ─────────────────────────────────────────


#: Market arguments forwarded only to recipes that name them **explicitly**.
#:
#: ``spot`` may go to a recipe that merely declares ``**market_args`` — it is
#: the one argument the legacy convention always meant.  The curves may not:
#: nearly every rates recipe carries a ``**kwargs``, so a var-kwargs match
#: would hand a projection curve to a recipe that silently drops it, which is
#: a wrong price with no signal.  Requiring the parameter by name makes
#: multi-curve support an opt-in property of each recipe's signature, and
#: :func:`valax.pricing.multicurve.require_single_curve` covers the payoffs
#: that opt in only to reject.  The keyed views (D16) follow the same rule:
#: ``curves`` (every curve the trade reads, a ``CurveGraph``) and ``spots``
#: (every underlying's spot, stacked in the instrument's order) reach only
#: recipes that name them.
_EXPLICIT_ONLY_MARKET_ARGS = ("discount_curve", "forward_curve", "curves", "spots")


def _recipe_market_args(recipe: Callable, resolved: ResolvedMarket) -> dict:
    """Market arguments a legacy recipe accepts, taken from the market.

    The legacy recipes keep their open ``**market_args`` convention, but
    they do not all accept the same ones (the PDE equity recipes take a
    bare ``spot`` and nothing else).  Rather than guess, the recipe's own
    signature is inspected and only what it declares is passed.

    This is also how the resolved **discount** and **projection** curves
    reach the MC/PDE back-ends (ontology report §12, P5.1a): a rates recipe
    opts into multi-curve by naming ``discount_curve`` / ``forward_curve``
    in its signature, and recipes that do not are left exactly as they were.

    Args:
        recipe: The registered recipe function.
        resolved: Market objects bound to this trade.

    Returns:
        The keyword arguments to forward.
    """
    candidates = {}
    if resolved.spot is not None:
        candidates["spot"] = resolved.spot
    candidates["discount_curve"] = resolved.discount_curve
    candidates["forward_curve"] = resolved.forward_curve
    candidates["curves"] = resolved.curves

    parameters = inspect.signature(recipe).parameters
    if "spots" in parameters and resolved.spots:
        # Stacked in the instrument's ``underlying_ids`` order (dict order).
        candidates["spots"] = jnp.stack(list(resolved.spots.values()))
    accepts_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )

    def wanted(name: str) -> bool:
        if name in _EXPLICIT_ONLY_MARKET_ARGS:
            return name in parameters
        return accepts_var_kwargs or name in parameters

    return {
        name: value for name, value in candidates.items() if wanted(name)
    }


def _price_via_legacy(
    instrument: Any,
    model: Any,
    resolved: ResolvedMarket,
    method: str,
    config: Any,
    key: Any,
) -> PricingResult:
    """Delegate to the MC or PDE dispatcher and lift its result."""
    registry = _legacy_registry(method)
    recipe = None
    for cls in _instrument_bases(type(instrument)):
        recipe = registry.get((cls, type(model)))
        if recipe is not None:
            break
    if recipe is None:
        raise ValueError(
            f"No {method} recipe registered for "
            f"({type(instrument).__name__}, {type(model).__name__}).\n"
            f"Available (instrument, model, method) combinations:\n"
            f"{_format_available()}"
        )
    if (
        "spots" in inspect.signature(recipe).parameters
        and not getattr(instrument, "underlying_ids", None)
    ):
        # A multi-asset recipe needs every underlying's spot in model order,
        # which only the instrument's own static ids can say (D16).
        raise ValueError(
            f"The {method} recipe for ({type(instrument).__name__}, "
            f"{type(model).__name__}) needs the spot of every asset: set "
            f"{type(instrument).__name__}.underlying_ids to the asset ids "
            f"(keys of MarketState.spots) in model order."
        )
    market_args = _recipe_market_args(recipe, resolved)
    if (
        resolved.forward_curve is not None
        and resolved.forward_curve is not resolved.discount_curve
        and "forward_curve" not in market_args
    ):
        # The trade names a projection curve the recipe cannot consume:
        # pricing it single-curve would be a wrong price with no signal.
        raise ValueError(
            f"The {method} recipe for ({type(instrument).__name__}, "
            f"{type(model).__name__}) does not accept a projection curve, but "
            f"the instrument resolves one distinct from its discount curve. "
            f"Drop the instrument's forward_id to price it single-curve, or "
            f"use a model / method that supports the projection curve."
        )
    extra = [
        cid for cid in resolved.curves.keys()
        if cid not in (resolved.discount_id, resolved.forward_id)
    ]
    if extra and "curves" not in market_args:
        # A multi-index trade on a recipe that reads only the primary pair
        # would be priced on a subset of its curves (D16).
        raise ValueError(
            f"The {method} recipe for ({type(instrument).__name__}, "
            f"{type(model).__name__}) reads only the discount / projection "
            f"pair, but the instrument also touches {extra}. A recipe that "
            f"prices it must take every curve by naming `curves` in its "
            f"signature."
        )
    # The dispatchers look the recipe up by *exact* type, so a selector
    # subclass (§_instrument_bases) is routed to the base class's recipe
    # directly.  Same call the dispatcher would make — it only adds the
    # lookup this function has already done.
    exact = (type(instrument), type(model)) in registry

    if method == "mc":
        if key is None:
            raise ValueError(
                "method='mc' needs a PRNG key: pass key=jax.random.PRNGKey(...)."
            )
        if exact:
            from valax.pricing.mc.dispatch import mc_price_dispatch

            result = mc_price_dispatch(
                instrument, model, config, key, **market_args
            )
        else:
            result = recipe(
                instrument=instrument, model=model, config=config,
                key=key, **market_args,
            )
        return PricingResult(
            price=result.price, stderr=result.stderr, method="mc",
        )

    if exact:
        from valax.pricing.pde.dispatch import pde_price_dispatch

        result = pde_price_dispatch(instrument, model, config, **market_args)
    else:
        result = recipe(
            instrument=instrument, model=model, config=config, **market_args,
        )
    return PricingResult(price=result.price, method="pde")


# ── Public entry point ───────────────────────────────────────────────


def price(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> PricingResult:
    """Price any instrument/model pair against a calibrated market.

    The instrument's static selectors (discount curve, projection curve,
    underlying) and the model's static market references (surface) are
    bound against ``market`` by
    :func:`valax.core.resolve.resolve`, then the registered adapter — or
    the bridged MC/PDE recipe — is run.

    Single-curve trades need **zero** configuration: an instrument that
    declares no selectors resolves to the ``"_default_"`` curve with no
    projection curve, which is the pre-existing single-curve semantics
    bit-for-bit.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None`` for instruments priced
            by curve discounting alone (fixed-rate bonds, vanilla swaps).
        market: The calibrated :class:`~valax.core.market_state.MarketState`.
        method: ``"analytic"``, ``"lattice"``, ``"pde"``, ``"mc"``, or
            ``"auto"`` (default) to let the policy table choose.
        config: Method-specific numerical settings (``PDEConfig``,
            ``MCConfig``, ``BinomialConfig``, …); ``None`` uses the pair's
            registered default for the method (:func:`effective_config`,
            :mod:`valax.core.policy`). The config used is recorded on the
            result.
        key: PRNG key — required for ``method="mc"``.

    Returns:
        A :class:`PricingResult` carrying the price, the method used,
        and (for MC) the sampling standard error.

    Raises:
        ValueError: If the pair cannot be priced by the requested
            method, if the method name is unknown, or if ``method="mc"``
            is requested without a ``key``.  Messages list what *is*
            registered.
        KeyError: If a selector names a curve / spot / surface absent
            from ``market``.

    Examples:
        >>> market = MarketState.single_curve(curve, spot=jnp.array(100.0))
        >>> result = price(option, bs_model, market)       # analytic
        >>> result = price(option, bs_model, market,       # same trade, MC
        ...                method="mc", config=MCConfig(n_paths=10_000),
        ...                key=jax.random.PRNGKey(0))
    """
    _ensure_backends()
    if method == "auto":
        method = auto_method(instrument, model)
    else:
        _validate_method(method)

    resolved = resolve(instrument, model, market)

    # Layer 3 of 4: fill any model parameter left as ``None`` from the
    # resolved market, so every pricer below sees a fully-concrete model
    # and none of them needs to know about market sourcing.  A
    # fully-specified model is returned unchanged, which is what keeps
    # constant-mode prices bit-identical.  ``bind_model`` preserves the
    # model's class, so dispatch below is unaffected.
    model = bind_model(model, instrument, resolved, market)

    adapter = None
    for cls in _instrument_bases(type(instrument)):
        adapter = _REGISTRY.get((cls, type(model), method))
        if adapter is not None:
            break
    if adapter is None and method not in available_methods(instrument, model):
        kind = "recipe" if method in ("mc", "pde") else "pricer"
        raise ValueError(
            f"No {method} {kind} registered for "
            f"({type(instrument).__name__}, {type(model).__name__}).\n"
            f"Available (instrument, model, method) combinations:\n"
            f"{_format_available()}"
        )

    # The config actually used: the caller's, else the pair's default.
    config = effective_config(instrument, model, method, config)

    if adapter is not None:
        result = adapter(
            instrument=instrument,
            model=model,
            market=market,
            resolved=resolved,
            config=config,
            key=key,
        )
    else:
        result = _price_via_legacy(
            instrument, model, resolved, method, config, key,
        )
    return dataclasses.replace(result, config=config)
