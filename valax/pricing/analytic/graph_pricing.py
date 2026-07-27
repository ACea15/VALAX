"""Optional :class:`~valax.curves.graph.CurveGraph` entry points for the
analytic rates pricers.

Every function here is a **thin wrapper**: it resolves curve ids on the
graph and delegates to the corresponding ``DiscountCurve``-based pricer.
No numeric logic lives in this module.

    price = swap_price_from_graph(swap, graph, "USD.SOFR.OIS", "USD.LIBOR.3M")

is exactly

    price = swap_price(swap, graph["USD.SOFR.OIS"], graph["USD.LIBOR.3M"])

Passing ``forward_id=None`` (the default) reproduces the single-curve
behaviour of the underlying pricer bit-for-bit, so existing workflows are
unaffected.  Curve ids are plain Python strings (static at trace time), so
the wrappers are as jittable as the pricers they delegate to.

The ``DiscountCurve``-based signatures remain the primary API; use these
wrappers when your curves already live in a :class:`CurveGraph` built by
:func:`valax.curves.bootstrap_curve_graph`.
"""

from jaxtyping import Float
from jax import Array

from valax.curves.graph import CurveGraph
from valax.curves.discount import DiscountCurve
from valax.instruments.bonds import FloatingRateBond
from valax.instruments.rates import (
    Cap,
    Caplet,
    CMSCapFloor,
    CMSSwap,
    InterestRateSwap,
    OISSwap,
    RangeAccrual,
    Swaption,
    TotalReturnSwap,
)
from valax.pricing.analytic.caplets import (
    cap_price_bachelier,
    cap_price_black76,
    caplet_price_bachelier,
    caplet_price_black76,
)
from valax.pricing.analytic.floating import (
    floating_rate_bond_price,
    ois_swap_price,
    ois_swap_rate,
)
from valax.pricing.analytic.rates_exotics import (
    cms_cap_floor_price_black76,
    cms_swap_price,
    range_accrual_price_black76,
    total_return_swap_price,
)
from valax.pricing.analytic.swaptions import (
    swap_price,
    swap_rate,
    swaption_price_bachelier,
    swaption_price_black76,
)


def _resolve(
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None,
) -> tuple[DiscountCurve, DiscountCurve | None]:
    """Look up the discount and (optional) forward curve on the graph."""
    discount_curve = graph[discount_id]
    forward_curve = None if forward_id is None else graph[forward_id]
    return discount_curve, forward_curve


# ── Swaps and swaptions ───────────────────────────────────────────────

def swap_price_from_graph(
    swap: InterestRateSwap,
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """:func:`~valax.pricing.analytic.swaptions.swap_price` on a curve graph.

    Args:
        swap: Swap contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the floating-leg projection curve.
            ``None`` (default) reproduces the single-curve behaviour.

    Returns:
        Swap NPV.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return swap_price(swap, discount_curve, forward_curve)


def swap_rate_from_graph(
    swap: InterestRateSwap,
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """:func:`~valax.pricing.analytic.swaptions.swap_rate` on a curve graph.

    Args:
        swap: Swap contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the floating-leg projection curve.

    Returns:
        Par swap rate (annualized).
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return swap_rate(swap, discount_curve, forward_curve)


def swaption_price_black76_from_graph(
    swaption: Swaption,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Black-76 swaption price on a curve graph.

    Args:
        swaption: Swaption contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Black (lognormal) swaption implied volatility.
        forward_id: Optional id of the floating-leg projection curve.

    Returns:
        Payer or receiver swaption price.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return swaption_price_black76(swaption, discount_curve, vol, forward_curve)


def swaption_price_bachelier_from_graph(
    swaption: Swaption,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Bachelier swaption price on a curve graph.

    Args:
        swaption: Swaption contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Normal (Bachelier) swaption volatility.
        forward_id: Optional id of the floating-leg projection curve.

    Returns:
        Payer or receiver swaption price.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return swaption_price_bachelier(swaption, discount_curve, vol, forward_curve)


# ── Floating-rate instruments ─────────────────────────────────────────

def floating_rate_bond_price_from_graph(
    bond: FloatingRateBond,
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """FRN price on a curve graph.

    Args:
        bond: Floating rate note contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the coupon projection curve.

    Returns:
        Present value at the discount curve's reference date.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return floating_rate_bond_price(bond, discount_curve, forward_curve)


def ois_swap_price_from_graph(
    swap: OISSwap,
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """OIS swap NPV on a curve graph.

    Args:
        swap: OIS swap contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the floating-leg projection curve.

    Returns:
        Swap NPV.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return ois_swap_price(swap, discount_curve, forward_curve)


def ois_swap_rate_from_graph(
    swap: OISSwap,
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Par OIS rate on a curve graph.

    Args:
        swap: OIS swap contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the floating-leg projection curve.

    Returns:
        Par swap rate (annualized).
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return ois_swap_rate(swap, discount_curve, forward_curve)


# ── Caplets, caps, and floors ─────────────────────────────────────────

def caplet_price_black76_from_graph(
    caplet: Caplet,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Black-76 caplet/floorlet price on a curve graph.

    Args:
        caplet: Caplet/floorlet contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Black (lognormal) implied volatility of the forward rate.
        forward_id: Optional id of the projection curve.

    Returns:
        Caplet or floorlet price.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return caplet_price_black76(caplet, discount_curve, vol, forward_curve)


def caplet_price_bachelier_from_graph(
    caplet: Caplet,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Bachelier caplet/floorlet price on a curve graph.

    Args:
        caplet: Caplet/floorlet contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Normal (Bachelier) volatility of the forward rate.
        forward_id: Optional id of the projection curve.

    Returns:
        Caplet or floorlet price.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return caplet_price_bachelier(caplet, discount_curve, vol, forward_curve)


def cap_price_black76_from_graph(
    cap: Cap,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Black-76 cap/floor price on a curve graph.

    Args:
        cap: Cap/floor contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Flat Black volatility (scalar) or per-caplet vols (shape n).
        forward_id: Optional id of the projection curve.

    Returns:
        Cap or floor price.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return cap_price_black76(cap, discount_curve, vol, forward_curve)


def cap_price_bachelier_from_graph(
    cap: Cap,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Bachelier cap/floor price on a curve graph.

    Args:
        cap: Cap/floor contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Flat normal volatility (scalar) or per-caplet vols (shape n).
        forward_id: Optional id of the projection curve.

    Returns:
        Cap or floor price.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return cap_price_bachelier(cap, discount_curve, vol, forward_curve)


# ── Rates exotics ─────────────────────────────────────────────────────

def total_return_swap_price_from_graph(
    swap: TotalReturnSwap,
    graph: CurveGraph,
    discount_id: str,
    unrealized_return: Float[Array, ""] = None,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """TRS NPV on a curve graph.

    Args:
        swap: Total return swap contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        unrealized_return: Optional fractional return of the reference
            asset since the last reset date.
        forward_id: Optional id of the funding-leg projection curve.

    Returns:
        Swap NPV.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return total_return_swap_price(
        swap, discount_curve, unrealized_return, forward_curve
    )


def cms_swap_price_from_graph(
    swap: CMSSwap,
    graph: CurveGraph,
    discount_id: str,
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """CMS swap NPV on a curve graph (no convexity adjustment).

    Args:
        swap: CMS swap contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        forward_id: Optional id of the projection curve for the
            underlying swap rates.

    Returns:
        Swap NPV.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return cms_swap_price(swap, discount_curve, forward_curve)


def cms_cap_floor_price_black76_from_graph(
    cap: CMSCapFloor,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Black-76 CMS cap/floor price on a curve graph (no convexity adj.).

    Args:
        cap: CMS cap or floor contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Black-76 volatility of the CMS rate, scalar or per-period.
        forward_id: Optional id of the projection curve for the
            underlying swap rates.

    Returns:
        Cap or floor NPV.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return cms_cap_floor_price_black76(cap, discount_curve, vol, forward_curve)


def range_accrual_price_black76_from_graph(
    accrual: RangeAccrual,
    graph: CurveGraph,
    discount_id: str,
    vol: Float[Array, ""],
    forward_id: str | None = None,
) -> Float[Array, ""]:
    """Digital-replication range accrual price on a curve graph.

    Args:
        accrual: Range accrual contract.
        graph: Curve graph holding the required curves.
        discount_id: Id of the discounting curve on the graph.
        vol: Black-76 volatility of the reference rate.
        forward_id: Optional id of the reference-rate projection curve.

    Returns:
        NPV of the range accrual coupons.
    """
    discount_curve, forward_curve = _resolve(graph, discount_id, forward_id)
    return range_accrual_price_black76(accrual, discount_curve, vol, forward_curve)
