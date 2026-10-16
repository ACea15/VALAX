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

from typing import Any, NamedTuple

import equinox as eqx
from jaxtyping import Float, Int
from jax import Array

from valax.core.market_state import DEFAULT_ID, MarketState
from valax.curves.discount import DiscountCurve
from valax.curves.fixings import FixingHistory
from valax.curves.graph import CurveGraph


class ResolvedMarket(eqx.Module):
    """Concrete market objects bound to one (instrument, model) pair.

    Every market slot the trade reads by id is resolved into a **keyed
    view** holding exactly the entries the trade names (D16): its curves,
    its spots, its FX spots, its inflation curves. Which entries is decided
    by the instrument's static selectors only, so resolution happens at
    trace time and the views' values stay traced leaves of the market —
    ``jax.grad`` with respect to the market lands on each named entry.

    Attributes:
        curves: The curves the trade reads (:func:`curve_ids_of`), keyed by
            curve id — a sub-graph of the market's, holding the very same
            curve objects.
        spot: Spot of the trade's (first) underlying, or ``None`` when the
            market carries no spots.
        dividend: Continuous dividend yield of that underlying, or ``None``
            for a market with no assets; ``0.0`` when the market carries
            assets but no dividend entries.
        surface: Volatility surface the model reads, or ``None``.
        fixings: Realised-fixing registry from the market (the whole
            registry: fixings are looked up by index id, never used to find
            a curve).
        reference_date: Valuation-date ordinal of the discount curve.
        spots: Spots of every underlying the trade names
            (:func:`underlying_ids_of`), keyed by asset id; empty when the
            market carries no spots.
        fx_spots: FX spots the trade names (:func:`fx_ids_of`), keyed by
            pair id.
        inflation_curves: Inflation curves the trade names
            (:func:`inflation_ids_of`), keyed by id.
        discount_id: Static id of the trade's discount curve.
        forward_id: Static id of its projection curve, or ``None``.
    """

    curves: CurveGraph
    spot: Float[Array, ""] | None
    dividend: Float[Array, ""] | None
    surface: Any
    fixings: FixingHistory
    reference_date: Int[Array, ""]
    spots: dict = eqx.field(default_factory=dict)
    fx_spots: dict = eqx.field(default_factory=dict)
    inflation_curves: dict = eqx.field(default_factory=dict)
    discount_id: str = eqx.field(static=True, default=DEFAULT_ID)
    forward_id: str | None = eqx.field(static=True, default=None)

    @property
    def discount_curve(self) -> DiscountCurve:
        """The curve this trade discounts on (a view of ``curves``)."""
        return self.curves[self.discount_id]

    @property
    def forward_curve(self) -> DiscountCurve | None:
        """The projection curve, or ``None`` for single-curve projection."""
        return None if self.forward_id is None else self.curves[self.forward_id]


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

    Reads ``instrument.forward_id`` if present; else, for an instrument
    with no selectors whose ``curves_touched`` has exactly **two** entries
    (the documented ``(discount, forward)`` convention), the second when it
    differs from the first; else ``None``. A trade touching three or more
    curves (e.g. a tenor basis swap as legs) has no single projection
    curve: all of them are in :attr:`ResolvedMarket.curves`, and a pricer
    that reads only the primary pair is refused for it by ``price()``.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The forward curve identifier, or ``None``.
    """
    if hasattr(instrument, "forward_id"):
        return instrument.forward_id
    if hasattr(instrument, "discount_id"):
        # Explicit selectors and no projection selector: a curves_touched
        # pair is (domestic, foreign) or similar, not (discount, forward).
        return None
    touched = getattr(instrument, "curves_touched", None)
    if touched and len(touched) == 2 and touched[1] != touched[0]:
        return touched[1]
    return None


def curve_ids_of(instrument: Any) -> tuple[str, ...]:
    """Every curve id this instrument reads, primary pair first.

    The discount id and the projection id (if any), followed by the rest of
    ``curves_touched`` (when the instrument declares one), de-duplicated in
    order. Reads static fields only, so it is evaluated at trace time.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The curve identifiers.
    """
    ids = [discount_id_of(instrument)]
    forward_id = forward_id_of(instrument)
    if forward_id is not None:
        ids.append(forward_id)
    for curve_id in getattr(instrument, "curves_touched", None) or ():
        ids.append(curve_id)
    return tuple(dict.fromkeys(ids))


def underlying_id_of(instrument: Any) -> str:
    """Asset id this instrument is written on (sentinel by default).

    The first of ``underlying_ids`` for a multi-asset instrument.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The asset identifier.
    """
    ids = getattr(instrument, "underlying_ids", None)
    if ids:
        return ids[0]
    return getattr(instrument, "underlying_id", None) or DEFAULT_ID


def underlying_ids_of(instrument: Any) -> tuple[str, ...]:
    """Every asset id the instrument is written on, in model order.

    ``underlying_ids`` for a multi-asset instrument, else the single
    :func:`underlying_id_of`.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The asset identifiers.
    """
    ids = getattr(instrument, "underlying_ids", None)
    if ids:
        return tuple(ids)
    return (underlying_id_of(instrument),)


def fx_ids_of(instrument: Any) -> tuple[str, ...]:
    """FX spot ids (pair keys of ``MarketState.fx_spots``) the trade reads.

    Read from the instrument's ``fx_ids`` (a static property, e.g.
    ``CrossCurrencySwap``'s pair without the ``/``); empty otherwise.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The FX pair identifiers.
    """
    return tuple(getattr(instrument, "fx_ids", None) or ())


def inflation_ids_of(instrument: Any) -> tuple[str, ...]:
    """Inflation-curve ids (keys of ``MarketState.inflation_curves``) read.

    The instrument's ``inflation_id`` if it has one; empty otherwise.

    Args:
        instrument: Any instrument pytree.

    Returns:
        The inflation curve identifiers.
    """
    inflation_id = getattr(instrument, "inflation_id", None)
    return () if inflation_id is None else (inflation_id,)


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
        all (a pure rates market), or when the instrument names no
        underlying and the market has none under the sentinel (a rates
        trade on a market that also holds equities) — pricers that need
        one raise their own error, which is more informative than a
        generic lookup failure.

    Raises:
        KeyError: If the market holds spots but not this instrument's.
    """
    if not _reads_spot(instrument, market):
        return None
    return market.spot(underlying_id_of(instrument))


def _names_underlying(instrument: Any) -> bool:
    """Whether the instrument carries an underlying selector at all."""
    return hasattr(instrument, "underlying_id") or bool(
        getattr(instrument, "underlying_ids", None)
    )


def _reads_spot(instrument: Any, market: MarketState) -> bool:
    """Whether ``resolve`` should look up a spot for this trade."""
    if not market.spots:
        return False
    return _names_underlying(instrument) or DEFAULT_ID in market.spots


def resolve_dividend(
    instrument: Any,
    market: MarketState,
) -> Float[Array, ""] | None:
    """Bind an instrument's underlying selector to a dividend yield.

    Args:
        instrument: Any instrument pytree.
        market: The calibrated market.

    Returns:
        The continuous dividend yield, or ``None`` when the market carries
        no assets at all (a pure rates market).  A market that holds
        assets but no dividend entries yields ``0.0``, which is the
        economically correct default for a non-payer.

    Raises:
        KeyError: If the market holds dividends but not this instrument's.
    """
    if not _reads_spot(instrument, market):
        return None
    return market.dividend(underlying_id_of(instrument))


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
        KeyError: If any named curve / spot / FX spot / inflation curve /
            surface is absent from the market.  The message lists what is
            available.
    """
    discount_id = discount_id_of(instrument)
    curves = CurveGraph(
        curves={cid: market.curve(cid) for cid in curve_ids_of(instrument)}
    )
    spots = (
        {aid: market.spot(aid) for aid in underlying_ids_of(instrument)}
        if _reads_spot(instrument, market) else {}
    )
    return ResolvedMarket(
        curves=curves,
        spot=resolve_spot(instrument, market),
        dividend=resolve_dividend(instrument, market),
        surface=resolve_surface(model, market),
        fixings=market.fixings,
        reference_date=curves[discount_id].reference_date,
        spots=spots,
        fx_spots=_keyed(market.fx_spots, fx_ids_of(instrument), "FX spot"),
        inflation_curves=_keyed(
            market.inflation_curves, inflation_ids_of(instrument),
            "Inflation curve",
        ),
        discount_id=discount_id,
        forward_id=forward_id_of(instrument),
    )


class DeclaredEntries(NamedTuple):
    """Market entries one (instrument, model) pair reads, by slot (D16).

    Every id is decided by static fields of the instrument and the model, so
    this is known before (and independently of) any numerics. It is the
    exact set the pair's price may depend on: a gradient of the price with
    respect to the market is structurally zero outside it.

    Attributes:
        curves: Curve ids — the instrument's (:func:`curve_ids_of`) and the
            model's declared curve fields (``curve_requirements``).
        spots: Asset ids whose spot is read.
        dividends: Asset ids whose dividend yield is read.
        fx_spots: FX pair ids.
        inflation_curves: Inflation-curve ids.
        surfaces: Surface ids (the model's ``surface_id``, or the market's
            default surface when it has one and the model names none).
    """

    curves: tuple[str, ...]
    spots: tuple[str, ...]
    dividends: tuple[str, ...]
    fx_spots: tuple[str, ...]
    inflation_curves: tuple[str, ...]
    surfaces: tuple[str, ...]


def declared_entries(
    instrument: Any, model: Any, market: MarketState
) -> DeclaredEntries:
    """Every market entry ``price(instrument, model, market)`` may read.

    The same rules as :func:`resolve` plus the model's declared curve fields
    (:func:`~valax.core.binding.curve_requirements`, including keyed
    families), so risk reports (:func:`valax.greeks.market_risk`) and the
    route contract test can be driven by the declaration rather than by
    knowledge of each pricer.

    Args:
        instrument: Any instrument pytree.
        model: Any model pytree, or ``None``.
        market: The calibrated market (decides the default-surface rule).

    Returns:
        The :class:`DeclaredEntries`.
    """
    from valax.core.binding import curve_requirements

    curve_ids = list(curve_ids_of(instrument))
    if model is not None:
        for requirement in curve_requirements(model, instrument).values():
            ids = (requirement,) if isinstance(requirement, str) else tuple(
                requirement.values()
            )
            curve_ids.extend(ids)
    asset_ids = tuple(
        _market_key(market.spots, a) for a in underlying_ids_of(instrument)
    ) if _reads_spot(instrument, market) else ()
    surface_id = surface_id_of(model)
    if surface_id is not None:
        surfaces: tuple[str, ...] = (surface_id,)
    elif DEFAULT_ID in market.surfaces:
        surfaces = (DEFAULT_ID,)
    else:
        surfaces = ()
    curve_table = market.curves.curves
    return DeclaredEntries(
        curves=tuple(dict.fromkeys(_market_key(curve_table, c) for c in curve_ids)),
        spots=asset_ids,
        dividends=tuple(a for a in asset_ids if a in market.dividends),
        fx_spots=fx_ids_of(instrument),
        inflation_curves=inflation_ids_of(instrument),
        surfaces=surfaces,
    )


def _market_key(table: dict, key: str) -> str:
    """The key ``table`` actually files ``key`` under (sentinel rule).

    ``MarketState`` resolves the ``"_default_"`` sentinel to the only entry
    of a one-entry table, so a declared sentinel names that entry.
    """
    if key not in table and key == DEFAULT_ID and len(table) == 1:
        return next(iter(table))
    return key


def _keyed(slot: dict, ids: tuple[str, ...], what: str) -> dict:
    """``{id: slot[id]}`` for the trade's ids, naming what is available."""
    missing = [i for i in ids if i not in slot]
    if missing:
        raise KeyError(
            f"{what} {missing[0]!r} not in MarketState (available: "
            f"{sorted(slot)})."
        )
    return {i: slot[i] for i in ids}
