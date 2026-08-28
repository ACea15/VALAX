"""Curve-aware, smile-aware rates pricing via a callable vol source.

Unifies flat volatility and full SABR smiles behind one entry point per rates
instrument family. A *vol source* is any callable that returns an implied vol at
the instrument's coordinate and exposes an ``is_normal`` quoting flag --
satisfied by :class:`~valax.surfaces.swaption_cube.SwaptionCube`,
:class:`~valax.surfaces.optionlet_surface.OptionletVolSurface`, and
:class:`~valax.surfaces.constant.ConstantVol` (a bare scalar is auto-wrapped).

Each pricer computes the instrument's coordinate (time-to-expiry ``T`` and, for
swaptions, the swap ``tenor``), reads the vol from the source there, and routes
to the Black-76 or Bachelier core per the source's convention -- reusing the
``DiscountCurve``-based pricers for the forward / annuity / discounting mechanics.
So passing a flat scalar reproduces the existing pricers bit-for-bit, while
passing a smile object makes the price strike/expiry(/tenor)-aware.

This module sits *above* both the analytic pricers and the surfaces package.
``ConstantVol`` is imported from its standalone submodule to keep the
surfaces -> pricing dependency acyclic.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.rates import Cap, Caplet, Swaption
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import year_fraction
from valax.pricing.analytic.swaptions import (
    swaption_price_black76,
    swaption_price_bachelier,
)
from valax.pricing.analytic.caplets import (
    caplet_price_black76,
    caplet_price_bachelier,
    cap_price_black76,
    cap_price_bachelier,
)
from valax.surfaces.constant import ConstantVol


def _as_vol_source(vol):
    """Normalize a scalar-or-callable into a vol source.

    A vol source is a callable exposing an ``is_normal`` attribute (e.g.
    ``SwaptionCube``, ``OptionletVolSurface``, ``ConstantVol``). A bare scalar
    is wrapped in a lognormal :class:`ConstantVol`; pass ``ConstantVol(v,
    is_normal=True)`` explicitly for a flat normal vol.
    """
    if callable(vol) and hasattr(vol, "is_normal"):
        return vol
    return ConstantVol(jnp.asarray(vol))


# ── Swaptions (swaption cube) ─────────────────────────────────────────

def swaption_price(
    swaption: Swaption,
    curve: DiscountCurve,
    vol,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Curve-aware European swaption price from a vol source.

    Reads the implied vol at ``(strike, expiry, tenor)`` -- where ``tenor`` is
    the underlying swap length -- and prices via Black-76 or Bachelier per the
    source's ``is_normal`` flag.

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
    v = src(swaption.strike, T, tenor)
    if src.is_normal:
        return swaption_price_bachelier(swaption, curve, v, forward_curve)
    return swaption_price_black76(swaption, curve, v, forward_curve)


# ── Caplets / floorlets (optionlet surface) ───────────────────────────

def caplet_price(
    caplet: Caplet,
    curve: DiscountCurve,
    vol,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Curve-aware caplet/floorlet price from a vol source.

    Reads the implied vol at ``(strike, expiry)`` and prices via Black-76 or
    Bachelier per the source's ``is_normal`` flag.

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
    v = src(caplet.strike, T)
    if src.is_normal:
        return caplet_price_bachelier(caplet, curve, v, forward_curve)
    return caplet_price_black76(caplet, curve, v, forward_curve)


def cap_price(
    cap: Cap,
    curve: DiscountCurve,
    vol,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    """Curve-aware cap/floor price as a smile-aware strip of optionlets.

    Each constituent optionlet is priced at the vol read from the source at its
    own ``(strike, expiry_i)`` -- i.e. the caplet-stripping the per-expiry smile
    supplies -- then summed via the ``DiscountCurve`` cap pricer (which accepts a
    per-caplet vol vector).

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
    vols = jax.vmap(lambda t: src(cap.strike, t))(T)
    if src.is_normal:
        return cap_price_bachelier(cap, curve, vols, forward_curve)
    return cap_price_black76(cap, curve, vols, forward_curve)


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
