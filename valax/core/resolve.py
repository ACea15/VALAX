"""Late binding of instrument/model market selectors to a ``MarketState``.

The ontology (``PRICING_ONTOLOGY.md`` §3.1, §3.4) says
a model should *reference* market observables by static string id rather
than embed copies of them, and an instrument should carry its own
**contractual** selectors — which curve *this trade* discounts on, which
index it projects, which underlying it is written on.  This module is the
one place those ids are turned into concrete objects:

.. code-block:: python

    resolved = resolve(instrument, model, market)
    resolved.discount_curve, resolved.forward_curve, resolved.spot, ...

Three properties make it safe to put between every instrument and every
pricer:

1. **Single-curve stays zero-config (§5).** Selectors are read with
   ``getattr`` defaults, so an instrument that declares none at all —
   every instrument in the library today — resolves to the ``"_default_"``
   discount curve and ``forward_id=None``, i.e. "project off the discount
   curve".  That is the existing single-curve semantics, bit-for-bit.
2. **Ids are static strings.** Selectors are ``eqx.field(static=True)``
   (or plain class attributes), never traced, so resolution happens at
   trace time and adds nothing to the gradient path.
3. **`curves_touched` is honoured.** The bootstrap instruments already
   carry ``curves_touched`` (``(discount_id,)`` or
   ``(discount_id, forward_id)``); the ontology calls that the seed of
   the selector pattern, so it is read directly rather than duplicated.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.curves.discount import DiscountCurve
from valax.curves.fixings import FixingHistory


class ResolvedMarket(eqx.Module):
    """Concrete market objects bound to one (instrument, model) pair.

    Attributes:
        discount_curve: Curve this trade discounts on.
        forward_curve: Projection curve, or ``None`` when the trade
            projects off its discount curve (single-curve semantics).
        spot: Underlying spot level, or ``None`` for trades with no
            spot input (linear rates products).
        surface: Volatility surface the model reads, or ``None``.
        fixings: Realised-fixing registry from the market.
        reference_date: Valuation-date ordinal of the discount curve.
    """

    discount_curve: DiscountCurve
    forward_curve: DiscountCurve | None
    spot: Float[Array, ""] | None
    surface: Any
    fixings: FixingHistory
    reference_date: Int[Array, ""]


# ── Selector readers ─────────────────────────────────────────────────


def discount_id_of(instrument: Any) -> str:
    """Curve id this instrument discounts on.

    Reads ``instrument.discount_id`` if present, else the first entry of
    ``curves_touched`` if present, else the ``"_default_"`` sentinel.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The curve identifier.
    """
    explicit = getattr(instrument, "discount_id", None)
    if explicit is not None:
        return explicit
    touched = getattr(instrument, "curves_touched", None)
    if touched:
        return touched[0]
    return DEFAULT_ID


def forward_id_of(instrument: Any) -> str | None:
    """Projection-curve id, or ``None`` for single-curve projection.

    Reads ``instrument.forward_id`` if present, else the second entry of
    ``curves_touched`` when it differs from the first, else ``None``.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The forward curve identifier, or ``None``.
    """
    if hasattr(instrument, "forward_id"):
        return instrument.forward_id
    touched = getattr(instrument, "curves_touched", None)
    if touched and len(touched) > 1 and touched[1] != touched[0]:
        return touched[1]
    return None


def underlying_id_of(instrument: Any) -> str:
    """Asset id this instrument is written on (sentinel by default).

    Args:
        instrument: Any instrument pytree.

    Returns:
        The asset identifier.
    """
    return getattr(instrument, "underlying_id", None) or DEFAULT_ID


def surface_id_of(model: Any) -> str | None:
    """Surface id this model reads, or ``None`` if it reads none.

    Args:
        model: Any model pytree, or ``None`` for curve-discounted
            (model-free) pricing.

    Returns:
        The surface identifier, or ``None``.
    """
    if model is None:
        return None
    return getattr(model, "surface_id", None)


# ── Resolution ───────────────────────────────────────────────────────


def resolve_curves(
    instrument: Any,
    market: MarketState,
) -> tuple[DiscountCurve, DiscountCurve | None]:
    """Bind an instrument's curve selectors against ``market``.

    Args:
        instrument: Any instrument pytree.
        market: The calibrated market.

    Returns:
        ``(discount_curve, forward_curve)`` where ``forward_curve`` is
        ``None`` when the instrument names no projection curve — the
        signal every VALAX rates pricer already interprets as "project
        off the discount curve".

    Raises:
        KeyError: If a named curve is absent from the market.
    """
    discount_curve = market.curve(discount_id_of(instrument))
    forward_id = forward_id_of(instrument)
    forward_curve = None if forward_id is None else market.curve(forward_id)
    return discount_curve, forward_curve


def resolve_spot(
    instrument: Any,
    market: MarketState,
) -> Float[Array, ""] | None:
    """Bind an instrument's underlying selector to a spot level.

    Args:
        instrument: Any instrument pytree.
        market: The calibrated market.

    Returns:
        The spot level, or ``None`` when the market carries no spots at
        all (a pure rates market) — pricers that need one raise their own
        error, which is more informative than a generic lookup failure.

    Raises:
        KeyError: If the market holds spots but not this instrument's.
    """
    if not market.spots:
        return None
    return market.spot(underlying_id_of(instrument))


def resolve_surface(model: Any, market: MarketState) -> Any:
    """Bind a model's surface reference against ``market``.

    Args:
        model: Any model pytree, or ``None``.
        market: The calibrated market.

    Returns:
        The surface object, or ``None`` when the model references none
        and the market has no default surface.

    Raises:
        KeyError: If the model names a surface the market lacks.
    """
    surface_id = surface_id_of(model)
    if surface_id is not None:
        return market.surface(surface_id)
    if DEFAULT_ID in market.surfaces:
        return market.surfaces[DEFAULT_ID]
    return None


def resolve(
    instrument: Any,
    model: Any,
    market: MarketState,
) -> ResolvedMarket:
    """Bind every selector of one (instrument, model) pair.

    The single entry point used by the pricing adapters in
    :mod:`valax.core.pricing`.

    Args:
        instrument: Any instrument pytree.
        model: Any model pytree, or ``None`` for curve-discounted
            pricing.
        market: The calibrated market.

    Returns:
        A :class:`ResolvedMarket`.

    Raises:
        KeyError: If any named curve / spot / surface is absent from the
            market.  The message lists what is available.
    """
    discount_curve, forward_curve = resolve_curves(instrument, market)
    return ResolvedMarket(
        discount_curve=discount_curve,
        forward_curve=forward_curve,
        spot=resolve_spot(instrument, market),
        surface=resolve_surface(model, market),
        fixings=market.fixings,
        reference_date=discount_curve.reference_date,
    )
