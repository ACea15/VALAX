"""The unified pricing signature: one ``price(...)`` over every method.

Three call conventions used to coexist — loose positional scalars for the
analytic pricers, ``(instrument, model, config, **market_args)`` for the
MC/PDE dispatchers, and ``(instrument, market)`` for the risk layer.  This
module collapses them into the single contract specified in
``docs/architecture/pricing-ontology.md`` §2.3:

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

``method="auto"`` takes the policy entry when there is one and it is
available, else the first available method in :data:`_METHOD_PRIORITY`
(``analytic`` → ``lattice`` → ``pde`` → ``mc``: exact and cheap first).

Growth happens in the dispatch, never in the signature: new methods and
new ``(instrument, model)`` pairs register; ``price``'s parameters do not
change.  Open ``**market_args`` survive only at the internal recipe
layer, where the legacy dispatchers keep them.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import equinox as eqx
from jaxtyping import Float
from jax import Array

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

    Convenience:
        ``float(result)`` returns ``float(result.price)`` so the result
        interoperates with scalar-expecting APIs without unpacking.
    """

    price: Float[Array, ""]
    stderr: Float[Array, ""] | None = None
    method: str = eqx.field(static=True, default="")
    diagnostics: dict = eqx.field(default_factory=dict)

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.price)


# ── Registry ─────────────────────────────────────────────────────────


#: The methods the unified layer knows about.
METHODS = ("analytic", "lattice", "pde", "mc")

#: Preference order used by ``method="auto"`` when no policy entry
#: applies: exact-and-cheap first, sampling last.
_METHOD_PRIORITY = METHODS

#: ``(instrument_cls, model_cls, method) -> adapter``.  Populated by
#: :func:`register_pricer` (see :mod:`valax.core.adapters`).
_REGISTRY: dict[tuple[type, type, str], Callable[..., PricingResult]] = {}

#: ``(instrument_cls, model_cls) -> method``.  The auto-selection policy,
#: deliberately separate from the registry (Q2).
_AUTO_POLICY: dict[tuple[type, type], str] = {}

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

    Args:
        instrument: Instrument pytree (or class).
        model: Model pytree (or class), possibly ``None``.

    Returns:
        The chosen method name.

    Raises:
        ValueError: If no method can price the pair.  The message lists
            everything that is registered.
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
        if preferred is not None and preferred in methods:
            return preferred
    return methods[0]


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


def _recipe_market_args(recipe: Callable, resolved: ResolvedMarket) -> dict:
    """Market arguments a legacy recipe accepts, taken from the market.

    The legacy recipes keep their open ``**market_args`` convention, but
    they do not all accept the same ones (the PDE equity recipes take a
    bare ``spot`` and nothing else).  Rather than guess, the recipe's own
    signature is inspected and only what it declares is passed.

    Args:
        recipe: The registered recipe function.
        resolved: Market objects bound to this trade.

    Returns:
        The keyword arguments to forward.
    """
    candidates = {}
    if resolved.spot is not None:
        candidates["spot"] = resolved.spot
    if not candidates:
        return {}

    parameters = inspect.signature(recipe).parameters
    accepts_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    return {
        name: value
        for name, value in candidates.items()
        if accepts_var_kwargs or name in parameters
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
    market_args = _recipe_market_args(recipe, resolved)
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
            ``MCConfig``, ``BinomialConfig``); ``None`` uses that
            method's default.
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

    adapter = None
    for cls in _instrument_bases(type(instrument)):
        adapter = _REGISTRY.get((cls, type(model), method))
        if adapter is not None:
            break
    if adapter is not None:
        return adapter(
            instrument=instrument,
            model=model,
            market=market,
            resolved=resolved,
            config=config,
            key=key,
        )

    if method in ("mc", "pde"):
        return _price_via_legacy(
            instrument, model, resolved, method, config, key,
        )

    raise ValueError(
        f"No {method} pricer registered for "
        f"({type(instrument).__name__}, {type(model).__name__}).\n"
        f"Available (instrument, model, method) combinations:\n"
        f"{_format_available()}"
    )
