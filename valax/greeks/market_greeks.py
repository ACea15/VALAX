"""Unified sensitivities over the ontology's ``price(instrument, model, market)``.

Walkthrough: ``docs/guide/tutorial-unified-pricing.md`` §8; design record:
``PRICING_ONTOLOGY.md``
§6.  Where :mod:`valax.greeks.autodiff` differentiates a *scalar-argument*
analytic pricer, this module differentiates the **unified**
:func:`valax.core.pricing.price` and so gives sensitivities for every method
(analytic / lattice / PDE / MC) and every asset class behind one API.

Three sensitivity families (§6.1)
---------------------------------

* **Market greeks** — ``∇`` w.r.t. the :class:`~valax.core.market_state.MarketState`
  (spots, surface nodes, curve discount factors).  :func:`market_gradient`.
* **Model greeks** — ``∇`` w.r.t. the model's calibrated parameters
  (``v0``, ``kappa``, ``vol`` …).  :func:`model_gradient`.
* **Contract greeks** — ``∇`` w.r.t. the instrument's own leaves: dual-delta
  (``∂/∂strike``) and **theta** (``∂/∂expiry``), the latter following the
  autodiff-vs-bump policy of §6.2.

Everything is a thin wrapper over ``eqx.filter_grad`` of ``price(...).price``:
because ``price`` is a pure function of ``(instrument, model, market)``, one
backward pass yields a *pytree-shaped* gradient whose leaves are read out by
the convenience accessors below.  Nothing here re-prices by finite difference
except theta on the PDE/MC path, where the time grid is a static discretisation
that cannot be differentiated through.

The Jacobian stack (§6.3)
-------------------------

:func:`curve_pillar_deltas` returns the raw ``∂price/∂DF(pillar)`` vector; feed
it to :func:`bucketed_curve_deltas` with a :class:`~valax.risk.bucketing.BucketMap`
(e.g. from ``tenor_bucket_map``) to land the same sensitivity on the FRTB tenor
vertices — the ``J_bucket`` leg of the stack.

Quote-space (hedge) risk is the sibling ``J_build`` leg, and it lives in
:mod:`valax.greeks.adjoint`: :func:`~valax.greeks.adjoint.curve_quote_deltas`
pushes the pillar cotangent back through the bootstrap residual by the implicit
function theorem, giving ``∂price/∂quote`` for every instrument the curves were
built from.  Prefer it over multiplying these pillar deltas by a dense
``quote_jacobian``: the adjoint never forms the quote Jacobian, so its cost
does not grow with the number of quotes, and it reuses one factorisation across
a whole portfolio.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.core.pricing import auto_method, price
from valax.core.resolve import discount_id_of, underlying_id_of


# ── Core: gradient pytrees ───────────────────────────────────────────


def _price_scalar(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str,
    config: Any,
    key: Any,
) -> Float[Array, ""]:
    """The scalar price, the quantity every sensitivity differentiates."""
    return price(
        instrument, model, market, method=method, config=config, key=key
    ).price


def market_gradient(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> MarketState:
    """``∇`` of the price w.r.t. the market — a ``MarketState``-shaped pytree.

    Every inexact-array leaf of ``market`` (spots, dividends, each curve's
    ``discount_factors``, every surface node) is replaced by its sensitivity;
    integer/static leaves are ``None``.  Read specific greeks out with
    :func:`spot_delta` / :func:`curve_pillar_deltas`, or consume the whole
    pytree for an all-at-once risk pass.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None`` for curve-discounted pricing.
        market: The calibrated market.
        method: ``"analytic" | "lattice" | "pde" | "mc" | "auto"``.
        config: Method-specific numerical settings.
        key: PRNG key (required for ``method="mc"``).

    Returns:
        A ``MarketState`` gradient pytree.
    """
    def fn(m: MarketState) -> Float[Array, ""]:
        return _price_scalar(
            instrument, model, m, method=method, config=config, key=key
        )

    return eqx.filter_grad(fn)(market)


def model_gradient(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Any:
    """``∇`` of the price w.r.t. the model parameters (model greeks).

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree (must not be ``None``).
        market: The calibrated market.
        method: Pricing method.
        config: Method-specific numerical settings.
        key: PRNG key (required for ``method="mc"``).

    Returns:
        A model-shaped gradient pytree (e.g. ``grad.vol``, ``grad.v0``).

    Raises:
        ValueError: If ``model is None`` (model-free pricing has no model
            greeks).
    """
    if model is None:
        raise ValueError(
            "model_gradient needs a model; a model-free (curve-discounted) "
            "trade has no model parameters. Use market_gradient for its "
            "curve sensitivities."
        )

    def fn(mod: Any) -> Float[Array, ""]:
        return _price_scalar(
            instrument, mod, market, method=method, config=config, key=key
        )

    return eqx.filter_grad(fn)(model)


# ── Sentinel-aware leaf accessors (mirror MarketState resolution) ─────


def _lookup_asset_grad(grad_table: dict, market_table: dict, key: str):
    """Read a per-asset grad leaf, honouring the single-asset sentinel."""
    if key in grad_table:
        return grad_table[key]
    if key == DEFAULT_ID and len(market_table) == 1:
        return next(iter(grad_table.values()))
    raise KeyError(
        f"No sensitivity for asset {key!r} "
        f"(available: {sorted(market_table)})."
    )


def _lookup_curve_grad(grad_graph, market_graph, curve_id: str | None):
    """Read a curve grad, honouring the ``_default_`` / single-curve rule."""
    gc = grad_graph.curves
    mc = market_graph.curves
    if curve_id is None or curve_id == DEFAULT_ID:
        if DEFAULT_ID in gc:
            return gc[DEFAULT_ID]
        if len(gc) == 1:
            return next(iter(gc.values()))
        raise KeyError(
            f"MarketState holds {len(mc)} curves and none under the "
            f"{DEFAULT_ID!r} sentinel; pass curve_id explicitly "
            f"(available: {sorted(mc)})."
        )
    if curve_id not in gc:
        raise KeyError(
            f"Curve {curve_id!r} not in MarketState (available: {sorted(mc)})."
        )
    return gc[curve_id]


# ── Named market greeks ──────────────────────────────────────────────


def spot_delta(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    asset_id: str | None = None,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, ""]:
    """``∂price/∂spot`` for the instrument's underlying.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None``.
        market: The calibrated market (must carry a spot for the underlying).
        asset_id: Override the underlying id; defaults to the instrument's
            ``underlying_id`` selector.
        method: Pricing method.
        config: Method-specific numerical settings.
        key: PRNG key (MC only).

    Returns:
        The spot delta.

    Raises:
        KeyError: If the market carries no spot for the underlying.
    """
    grad = market_gradient(
        instrument, model, market, method=method, config=config, key=key
    )
    aid = asset_id if asset_id is not None else underlying_id_of(instrument)
    return _lookup_asset_grad(grad.spots, market.spots, aid)


def curve_pillar_deltas(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    curve_id: str | None = None,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, " n_pillars"]:
    """``∂price/∂DF(pillar)`` for one curve — the raw key-rate sensitivity.

    This is the ``∇_MarketState price`` leg of the Jacobian stack (§6.3):
    aggregate it to tenor buckets with :func:`bucketed_curve_deltas`, or
    compose it with a bootstrap quote-Jacobian for hedge-space risk.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None`` for curve-discounted pricing.
        market: The calibrated market.
        curve_id: Override the curve id; defaults to the instrument's
            ``discount_id`` selector.
        method: Pricing method.
        config: Method-specific numerical settings.
        key: PRNG key (MC only).

    Returns:
        The per-pillar discount-factor sensitivities.
    """
    grad = market_gradient(
        instrument, model, market, method=method, config=config, key=key
    )
    cid = curve_id if curve_id is not None else discount_id_of(instrument)
    return _lookup_curve_grad(grad.curves, market.curves, cid).discount_factors


def bucketed_curve_deltas(
    pillar_deltas: Float[Array, " n_pillars"],
    bucket_map: Any,
) -> Float[Array, " n_buckets"]:
    """Aggregate raw pillar deltas onto predefined tenor buckets.

    The ``J_bucket`` leg of §6.3: ``δ_b = A · δ_x`` where ``A`` is the
    :class:`~valax.risk.bucketing.BucketMap` matrix (e.g. FRTB vertices from
    ``tenor_bucket_map``).  With an indicator map the total is preserved
    (``sum(δ_b) == sum(δ_x)``).

    Args:
        pillar_deltas: Raw per-pillar sensitivities from
            :func:`curve_pillar_deltas`.
        bucket_map: A :class:`~valax.risk.bucketing.BucketMap`.

    Returns:
        The bucketed sensitivities.
    """
    # Lazy import keeps valax.greeks free of a valax.risk import at module load.
    from valax.risk.bucketing import aggregate

    return aggregate(bucket_map, pillar_deltas)


# ── Contract greeks ──────────────────────────────────────────────────


def dual_delta(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> Float[Array, ""]:
    """``∂price/∂strike`` — a contract greek (§6.1).

    Args:
        instrument: An instrument carrying a ``strike`` leaf.
        model: A VALAX model pytree, or ``None``.
        market: The calibrated market.
        method: Pricing method.
        config: Method-specific numerical settings.
        key: PRNG key (MC only).

    Returns:
        The dual delta.

    Raises:
        AttributeError: If the instrument has no ``strike`` field.
    """
    if not hasattr(instrument, "strike"):
        raise AttributeError(
            f"{type(instrument).__name__} has no strike; dual_delta is a "
            f"strike sensitivity."
        )

    def fn(strike: Float[Array, ""]) -> Float[Array, ""]:
        inst = eqx.tree_at(lambda i: i.strike, instrument, strike)
        return _price_scalar(
            inst, model, market, method=method, config=config, key=key
        )

    return jax.grad(fn)(jnp.asarray(instrument.strike))


def theta(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
    dt: float = 1.0 / 365.0,
) -> Float[Array, ""]:
    r"""Calendar theta ``dV/dt`` — value change per unit time passing (§6.2).

    Time passing advances calendar time by ``dt`` and shrinks time-to-expiry
    by ``dt``, so ``theta = -∂V/∂T``.  The mechanism depends on the method:

    * **analytic / lattice** — ``expiry`` is a smooth year-fraction float, so
      ``theta = -jax.grad(price, expiry)`` (exact autodiff).
    * **pde / mc** — ``expiry`` also fixes the time-grid / step count (a static
      discretisation you cannot differentiate through), so a finite-difference
      calendar bump ``(V(T - dt) - V(T)) / dt`` is used instead.

    Args:
        instrument: An instrument carrying a smooth ``expiry`` float.
        model: A VALAX model pytree, or ``None``.
        market: The calibrated market.
        method: Pricing method (``"auto"`` resolves via the policy table).
        config: Method-specific numerical settings.
        key: PRNG key (MC only).
        dt: Finite-difference step for the PDE/MC path (year fraction).

    Returns:
        The theta.

    Raises:
        NotImplementedError: For date-based (rates) instruments with no smooth
            ``expiry`` leaf — their theta is a valuation-date roll (shift the
            market ``reference_date``), which is not yet implemented here.
    """
    resolved = auto_method(instrument, model) if method == "auto" else method
    expiry = getattr(instrument, "expiry", None)
    if expiry is None:
        raise NotImplementedError(
            "theta for date-based (rates) instruments is a valuation-date "
            "roll (advance the market reference_date by dt and reprice), not "
            "a derivative of an integer date. That roll is not yet implemented "
            "in P4; compute it explicitly by rolling the MarketState, or use "
            "an instrument parameterised by a float time-to-expiry."
        )

    if resolved in ("analytic", "lattice"):
        def fn(t: Float[Array, ""]) -> Float[Array, ""]:
            inst = eqx.tree_at(lambda i: i.expiry, instrument, t)
            return _price_scalar(
                inst, model, market, method=resolved, config=config, key=key
            )

        return -jax.grad(fn)(jnp.asarray(expiry))

    # PDE / MC: finite-difference calendar bump (static time grid).
    base = _price_scalar(
        instrument, model, market, method=resolved, config=config, key=key
    )
    inst = eqx.tree_at(lambda i: i.expiry, instrument, expiry - dt)
    bumped = _price_scalar(
        inst, model, market, method=resolved, config=config, key=key
    )
    return (bumped - base) / dt


# ── Declared-entry risk report ───────────────────────────────────────


class MarketRisk(eqx.Module):
    """Price and its sensitivity to every market entry the trade declares.

    Keyed by the ids of :func:`~valax.core.resolve.declared_entries` — the
    instrument's curves (every one, not only the discount curve), spots,
    dividends, FX spots, inflation curves, and the model's curve fields and
    surface — so a newly routed instrument's risk report follows from its
    static selectors, with no per-instrument code.

    Attributes:
        price: The price.
        curves: ``curve_id -> ∂price/∂DF(pillar)``.
        spots: ``asset_id -> ∂price/∂spot``.
        dividends: ``asset_id -> ∂price/∂q``.
        fx_spots: ``pair_id -> ∂price/∂FX spot``.
        inflation_curves: ``id -> ∂price/∂forward CPI(pillar)``.
        surfaces: ``surface_id -> gradient pytree of the surface``.
        declared: The declared ids the report is keyed by.
    """

    price: Float[Array, ""]
    curves: dict
    spots: dict
    dividends: dict
    fx_spots: dict
    inflation_curves: dict
    surfaces: dict
    declared: Any = eqx.field(static=True)


def market_risk(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
) -> MarketRisk:
    """One backward pass: the price's sensitivity to each declared entry.

    The generic risk report of a ``price()`` route: whatever the instrument,
    the entries it may depend on are its declared ones
    (:func:`~valax.core.resolve.declared_entries`), and the market gradient
    is structurally zero everywhere else. Bucket a curve's pillar deltas with
    :func:`bucketed_curve_deltas`, or take quote-space risk through the
    calibration chain (:func:`~valax.greeks.quote_risk.quote_risk_adjoint`).

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None``.
        market: The calibrated market.
        method: Pricing method.
        config: Method-specific numerical settings (``None``: the default).
        key: PRNG key (MC only).

    Returns:
        A :class:`MarketRisk`.
    """
    from valax.core.resolve import declared_entries

    declared = declared_entries(instrument, model, market)

    def fn(m: MarketState) -> Float[Array, ""]:
        return _price_scalar(
            instrument, model, m, method=method, config=config, key=key
        )

    value, grad = eqx.filter_value_and_grad(fn)(market)
    return MarketRisk(
        price=value,
        curves={c: grad.curves.curves[c].discount_factors for c in declared.curves},
        spots={a: grad.spots[a] for a in declared.spots},
        dividends={a: grad.dividends[a] for a in declared.dividends},
        fx_spots={f: grad.fx_spots[f] for f in declared.fx_spots},
        inflation_curves={
            i: grad.inflation_curves[i].forward_cpis
            for i in declared.inflation_curves
        },
        surfaces={s: grad.surfaces[s] for s in declared.surfaces},
        declared=declared,
    )


# ── Aggregate convenience ────────────────────────────────────────────


def market_greeks(
    instrument: Any,
    model: Any,
    market: MarketState,
    *,
    method: str = "auto",
    config: Any = None,
    key: Any = None,
    include_theta: bool = True,
) -> dict[str, Float[Array, ""]]:
    """Price plus the common named greeks in as few passes as possible.

    Extracts what is meaningful for the given (instrument, model): a spot
    ``delta`` when the market carries the underlying's spot; ``vega`` / ``rho``
    / ``dividend_rho`` from the model parameters when present; and ``theta``
    when the instrument has a smooth ``expiry`` (skipped for date-based rates
    products).  Curve key-rate deltas are intentionally *not* flattened into
    this dict — use :func:`curve_pillar_deltas` for those.

    Args:
        instrument: A VALAX instrument pytree.
        model: A VALAX model pytree, or ``None``.
        market: The calibrated market.
        method: Pricing method.
        config: Method-specific numerical settings.
        key: PRNG key (MC only).
        include_theta: Whether to compute theta (extra pricing passes).

    Returns:
        A dict with ``price`` and whichever of ``delta`` / ``vega`` / ``rho``
        / ``dividend_rho`` / ``theta`` apply.
    """
    result: dict[str, Float[Array, ""]] = {}
    result["price"] = _price_scalar(
        instrument, model, market, method=method, config=config, key=key
    )

    if market.spots:
        try:
            result["delta"] = spot_delta(
                instrument, model, market,
                method=method, config=config, key=key,
            )
        except KeyError:
            pass

    if model is not None:
        mgrad = model_gradient(
            instrument, model, market, method=method, config=config, key=key
        )
        for name, field in (
            ("vega", "vol"),
            ("rho", "rate"),
            ("dividend_rho", "dividend"),
        ):
            leaf = getattr(mgrad, field, None)
            if leaf is not None:
                result[name] = leaf

    if include_theta and getattr(instrument, "expiry", None) is not None:
        result["theta"] = theta(
            instrument, model, market, method=method, config=config, key=key
        )

    return result
