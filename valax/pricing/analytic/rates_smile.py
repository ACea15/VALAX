"""Curve-aware, smile-aware rates pricing via a callable vol source.

Unifies flat volatility and full SABR smiles behind one entry point per rates
instrument family. A *vol source* is any callable that returns an implied vol at
the instrument's coordinate and exposes an ``is_normal`` quoting flag --
satisfied by :class:`~valax.surfaces.swaption_cube.SwaptionCube`,
:class:`~valax.surfaces.optionlet_surface.OptionletVolSurface`, and
:class:`~valax.surfaces.constant.ConstantVol` (a bare scalar is auto-wrapped).

The vol-source protocol
-----------------------
A vol source is called as::

    src(strike, expiry[, tenor], forward=F)

where ``F`` is the **live forward computed off the curves** for this very
instrument -- the forward par swap rate for a swaption, the projected period
forward for an optionlet. ``forward`` is keyword-only and every source must
accept it; flat sources ignore it.

Passing the forward in is what makes a smile *curve-aware*: a vol surface's
smile is a function of moneyness, so which forward it is anchored to decides
whether a curve move reprices the smile or merely slides along it. The surface
objects own that policy via their ``dynamics`` field -- see the
[smile-dynamics section](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves)
of the vol-surfaces guide, which is the single source of truth for the three
conventions and the chosen default. This module's job is only to *supply* the
forward, not to choose what the surface does with it.

Each pricer therefore computes the instrument's coordinate (time-to-expiry
``T``, the swap ``tenor`` where applicable) **and its forward** exactly once,
reads the vol from the source there, and routes to the Black-76 or Bachelier
core per the source's convention -- handing the same forward back to the payoff
so vol and price can never see different ones. Annuity / discounting mechanics
are reused from the ``DiscountCurve``-based pricers. So passing a flat scalar
reproduces the existing pricers bit-for-bit, while passing a smile object makes
the price strike/expiry(/tenor)-aware.

This module sits in the *pricing* layer and deliberately does **not** import
the surfaces package: a bare scalar is wrapped in the self-contained
:class:`_FlatVol` source below rather than in
:class:`~valax.surfaces.constant.ConstantVol`.  This keeps the dependency
one-directional (surfaces -> pricing) and avoids a circular import, since the
surfaces package imports back into ``valax.calibration`` / ``valax.pricing``.
This mirrors :func:`valax.pricing.analytic.cms_convexity._resolve_vol_source`.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.instruments.rates import Cap, Caplet, Swaption
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import year_fraction
from valax.pricing.analytic.swaptions import (
    swaption_forward_and_annuity,
    swaption_black76_from_forward,
    swaption_bachelier_from_forward,
)
from valax.pricing.analytic.caplets import (
    caplet_forward,
    cap_forwards,
    caplet_price_black76,
    caplet_price_bachelier,
    cap_price_black76,
    cap_price_bachelier,
)


class _FlatVol(eqx.Module):
    """Self-contained flat-vol source used to wrap a bare scalar.

    Structurally identical to :class:`~valax.surfaces.constant.ConstantVol`
    (callable, carries an ``is_normal`` flag) but defined here in the pricing
    package so the curve-aware pricers never import the surfaces package,
    keeping the surfaces -> pricing dependency acyclic.

    Attributes:
        vol: The constant volatility.
        is_normal: Quoting convention -- True for normal (Bachelier), False for
            lognormal (Black-76).
    """

    vol: Float[Array, ""]
    is_normal: bool = eqx.field(static=True, default=False)

    def __call__(
        self,
        *coords: Float[Array, ""],
        forward: Float[Array, ""] | None = None,
    ) -> Float[Array, ""]:
        """Return the constant vol.

        Args:
            *coords: Ignored ``(strike, expiry[, tenor])`` coordinates.
            forward: Ignored live forward; accepted so a flat vol stays a
                drop-in for a curve-aware smile.

        Returns:
            The constant volatility.
        """
        return jnp.asarray(self.vol)


def _as_vol_source(vol):
    """Normalize a scalar-or-callable into a vol source.

    A vol source is a callable exposing an ``is_normal`` attribute (e.g.
    ``SwaptionCube``, ``OptionletVolSurface``,
    :class:`~valax.surfaces.constant.ConstantVol`). A bare scalar is wrapped in
    a lognormal :class:`_FlatVol`; pass a ``ConstantVol(v, is_normal=True)`` (or
    any callable with ``is_normal=True``) explicitly for a flat normal vol.
    """
    if callable(vol) and hasattr(vol, "is_normal"):
        return vol
    return _FlatVol(jnp.asarray(vol))


# ── Swaptions (swaption cube) ─────────────────────────────────────────

def swaption_price(
    swaption: Swaption,
    curve: DiscountCurve,
    vol,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Curve-aware European swaption price from a vol source.

    Computes the forward par swap rate and annuity off the curves once, reads
    the implied vol at ``(strike, expiry, tenor, forward=S)`` -- where ``tenor``
    is the underlying swap length -- and prices via Black-76 or Bachelier per
    the source's ``is_normal`` flag, reusing that same ``S`` and annuity.

    Args:
        swaption: Swaption contract.
        curve: Discount (OIS) curve; also supplies the forward when
            ``forward_curve`` is None.
        vol: A vol source (``SwaptionCube``/``ConstantVol``/callable with
            ``is_normal``) or a bare scalar (treated as flat lognormal).
        forward_curve: Optional forwarding curve for the swap's floating leg.

    Returns:
        Payer or receiver swaption price.
    """
    src = _as_vol_source(vol)
    T = year_fraction(curve.reference_date, swaption.expiry_date, swaption.day_count)
    tenor = year_fraction(
        swaption.expiry_date, swaption.fixed_dates[-1], swaption.day_count
    )
    S, ann = swaption_forward_and_annuity(swaption, curve, forward_curve)
    v = src(swaption.strike, T, tenor, forward=S)
    if src.is_normal:
        return swaption_bachelier_from_forward(swaption, S, ann, v, T)
    return swaption_black76_from_forward(swaption, S, ann, v, T)


# ── Caplets / floorlets (optionlet surface) ───────────────────────────

def caplet_price(
    caplet: Caplet,
    curve: DiscountCurve,
    vol,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Curve-aware caplet/floorlet price from a vol source.

    Projects the period forward off the curves once, reads the implied vol at
    ``(strike, expiry, forward=F)``, and prices via Black-76 or Bachelier per
    the source's ``is_normal`` flag with that same ``F``.

    Args:
        caplet: Caplet/floorlet contract.
        curve: Discount curve; also projects the forward when ``forward_curve``
            is None.
        vol: A vol source (``OptionletVolSurface``/``ConstantVol``/callable) or
            a bare scalar (flat lognormal).
        forward_curve: Optional forwarding curve for the forward rate.

    Returns:
        Caplet or floorlet price.
    """
    src = _as_vol_source(vol)
    T = year_fraction(curve.reference_date, caplet.fixing_date, caplet.day_count)
    F = caplet_forward(caplet, curve, forward_curve)
    v = src(caplet.strike, T, forward=F)
    if src.is_normal:
        return caplet_price_bachelier(
            caplet, curve, v, forward_curve, forward=F
        )
    return caplet_price_black76(caplet, curve, v, forward_curve, forward=F)


def cap_price(
    cap: Cap,
    curve: DiscountCurve,
    vol,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Curve-aware cap/floor price as a smile-aware strip of optionlets.

    Each constituent optionlet is priced at the vol read from the source at its
    own ``(strike, expiry_i, forward=F_i)`` -- i.e. the caplet-stripping the
    per-expiry smile supplies, anchored to that period's own projected forward
    -- then summed via the ``DiscountCurve`` cap pricer (which accepts a
    per-caplet vol vector and the same forward vector).

    Args:
        cap: Cap/floor contract.
        curve: Discount curve.
        vol: A vol source (``OptionletVolSurface``/``ConstantVol``/callable) or
            a bare scalar (flat lognormal, reproducing the flat-vol cap price).
        forward_curve: Optional forwarding curve.

    Returns:
        Cap or floor price.
    """
    src = _as_vol_source(vol)
    T = year_fraction(curve.reference_date, cap.fixing_dates, cap.day_count)
    F = cap_forwards(cap, curve, forward_curve)
    vols = jax.vmap(lambda t, f: src(cap.strike, t, forward=f))(T, F)
    if src.is_normal:
        return cap_price_bachelier(cap, curve, vols, forward_curve, forwards=F)
    return cap_price_black76(cap, curve, vols, forward_curve, forwards=F)


# ── CurveGraph wrappers ───────────────────────────────────────────────

def swaption_price_from_graph(
    swaption: Swaption,
    graph: CurveGraph,
    discount_id: str,
    vol,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """:func:`swaption_price` resolved on a :class:`CurveGraph`."""
    discount_curve = graph[discount_id]
    forward_curve = None if forward_id is None else graph[forward_id]
    return swaption_price(swaption, discount_curve, vol, forward_curve)


def caplet_price_from_graph(
    caplet: Caplet,
    graph: CurveGraph,
    discount_id: str,
    vol,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """:func:`caplet_price` resolved on a :class:`CurveGraph`."""
    discount_curve = graph[discount_id]
    forward_curve = None if forward_id is None else graph[forward_id]
    return caplet_price(caplet, discount_curve, vol, forward_curve)


def cap_price_from_graph(
    cap: Cap,
    graph: CurveGraph,
    discount_id: str,
    vol,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """:func:`cap_price` resolved on a :class:`CurveGraph`."""
    discount_curve = graph[discount_id]
    forward_curve = None if forward_id is None else graph[forward_id]
    return cap_price(cap, discount_curve, vol, forward_curve)
