"""Analytical pricers for exotic rates products.

Contents
--------
- ``cross_currency_swap_price`` / ``cross_currency_basis_spread``
  Two-curve telescoping pricer for XCCY basis swaps.
- ``total_return_swap_price``
  Single-curve reduction of a TRS under self-financing asset assumption.
- ``cms_swap_price``
  CMS swap priced off a synthetic annual underlying swap (no convexity
  adjustment).
- ``cms_cap_floor_price_black76``
  Black-76 on the unadjusted forward CMS rate (no convexity adjustment).
- ``range_accrual_price_black76``
  Digital-replication range accrual: per-period snapshot probability of
  the reference rate lying in ``[lower_barrier, upper_barrier]``.

All pricers are pure functions and differentiable via ``jax.grad``.  The
CMS and range-accrual pricers ship **without** a convexity/timing
adjustment — the true CMS rate differs from the forward swap rate by a
Hagan-replication or SABR-integration correction that is a legitimate
larger piece of work.  Users who need the adjustment should compose it
externally.

References:
    Hagan (2003), "Convexity Conundrums: Pricing CMS Swaps, Caps, and
        Floors".
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 13 (CMS) and
        ch. 16 (range accruals).
    Henrard (2014), *Interest Rate Modelling in the Multi-Curve
        Framework*, ch. 9 (cross-currency).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.rates import (
    CrossCurrencySwap,
    TotalReturnSwap,
    CMSSwap,
    CMSCapFloor,
    RangeAccrual,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction

# Reuse the fixed-leg annuity helper from the swaptions module rather
# than duplicating it (same reasoning as `floating.py`).
from valax.pricing.analytic.swaptions import _annuity


# ─────────────────────────────────────────────────────────────────────
# Cross-currency basis swap
# ─────────────────────────────────────────────────────────────────────

def cross_currency_swap_price(
    swap: CrossCurrencySwap,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    spot: Float[Array, ""],
) -> Float[Array, ""]:
    """NPV (in domestic currency) of a cross-currency basis swap.

    The receiver convention is **receive domestic leg** (domestic
    floating index + basis spread) **and pay foreign leg** (foreign
    floating index).  NPV is quoted in domestic currency using
    ``spot`` — the number of domestic currency units per one unit of
    foreign currency — to convert the foreign leg.

    **Single-curve-per-currency** projection: each currency's floating
    leg is forecasted off its own discount curve via the telescoping
    identity

    .. math::

        \\sum_i F_i \\tau_i\\, DF_c(T_i) = DF_c(T_0) - DF_c(T_n).

    With ``exchange_notional=True`` the initial and final notional
    exchanges cancel exactly against the telescoped float coupons in
    each currency, collapsing the NPV to the PV of the basis spread
    on the domestic leg:

    .. math::

        \\text{NPV}
        = N_d \\cdot s \\cdot A_d

    where :math:`A_d` is the fixed-leg annuity on the domestic curve.
    For ``exchange_notional=False`` the notional-exchange terms are
    omitted and the float legs do not cancel.

    Args:
        swap: Cross-currency swap contract.
        domestic_curve: Discount / projection curve in the domestic
            currency.
        foreign_curve: Discount / projection curve in the foreign
            currency.
        spot: Spot FX rate, domestic per foreign (``FOR/DOM``
            quotation).

    Returns:
        Swap NPV in domestic currency.  A positive value means the
        receive-domestic side is in-the-money.
    """
    A_d = _annuity(
        swap.start_date, swap.payment_dates, domestic_curve, swap.day_count
    )

    df_d_start = domestic_curve(swap.start_date)
    df_d_end = domestic_curve(swap.maturity_date)
    df_f_start = foreign_curve(swap.start_date)
    df_f_end = foreign_curve(swap.maturity_date)

    # Domestic leg: receive floating + basis spread coupons.
    dom_float_pv = swap.domestic_notional * (df_d_start - df_d_end)
    spread_pv = swap.domestic_notional * swap.basis_spread * A_d
    dom_leg_pv = dom_float_pv + spread_pv

    # Foreign leg: pay floating coupons, converted to domestic at spot.
    for_leg_pv = spot * swap.foreign_notional * (df_f_start - df_f_end)

    npv = dom_leg_pv - for_leg_pv

    if swap.exchange_notional:
        # Initial exchange: pay N_d in dom, receive N_f in foreign.
        initial = (
            -swap.domestic_notional * df_d_start
            + spot * swap.foreign_notional * df_f_start
        )
        # Final exchange: receive N_d in dom, pay N_f in foreign.
        final = (
            swap.domestic_notional * df_d_end
            - spot * swap.foreign_notional * df_f_end
        )
        npv = npv + initial + final

    return npv


def cross_currency_basis_spread(
    swap: CrossCurrencySwap,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    spot: Float[Array, ""],
) -> Float[Array, ""]:
    """Par basis spread: :math:`s^\\ast` that zeroes the XCCY NPV.

    Uses the linearity of the NPV in the basis spread:

    .. math::

        \\text{NPV}(s) = \\text{NPV}(0) + s \\cdot N_d \\cdot A_d
        \\quad\\Longrightarrow\\quad
        s^\\ast = -\\frac{\\text{NPV}(0)}{N_d \\cdot A_d}.

    Args:
        swap: Cross-currency swap contract (the ``basis_spread`` field
            is ignored; the solver returns the par spread that makes
            the NPV zero).
        domestic_curve: Domestic discount curve.
        foreign_curve: Foreign discount curve.
        spot: Spot FX rate (domestic per foreign).

    Returns:
        Par basis spread (decimal, annualized).
    """
    # NPV at zero spread — rebuild the swap with s=0 by pytree replacement.
    zero_spread = CrossCurrencySwap(
        start_date=swap.start_date,
        payment_dates=swap.payment_dates,
        maturity_date=swap.maturity_date,
        domestic_notional=swap.domestic_notional,
        foreign_notional=swap.foreign_notional,
        basis_spread=jnp.zeros_like(swap.basis_spread),
        exchange_notional=swap.exchange_notional,
        currency_pair=swap.currency_pair,
        day_count=swap.day_count,
    )
    base_npv = cross_currency_swap_price(
        zero_spread, domestic_curve, foreign_curve, spot
    )
    A_d = _annuity(
        swap.start_date, swap.payment_dates, domestic_curve, swap.day_count
    )
    return -base_npv / (swap.domestic_notional * A_d)


# ─────────────────────────────────────────────────────────────────────
# Total return swap
# ─────────────────────────────────────────────────────────────────────

def total_return_swap_price(
    swap: TotalReturnSwap,
    curve: DiscountCurve,
    unrealized_return: Float[Array, ""] = None,
) -> Float[Array, ""]:
    """NPV of a total return swap under the self-financing assumption.

    A TRS exchanges the **total return** (price change plus income)
    of a reference asset against a funding leg paying a floating rate
    plus ``funding_spread``.  Under the standard assumption that the
    reference asset is **self-financing** — i.e. its expected return
    under the risk-neutral measure equals the short rate used to
    discount — the total-return leg and the floating-rate portion of
    the funding leg telescope to the same present value and cancel.
    The TRS then collapses to the PV of the funding spread on the
    notional annuity:

    .. math::

        \\text{NPV}_{\\text{TR receiver}} = -N \\cdot s \\cdot A

    (the TR receiver *pays* the funding spread, so the sign is
    negative).  This is exact at a reset date, when the reference
    asset's current level equals the previous reset level.

    To mark a live position between reset dates, pass the accrued-
    but-unsettled fractional return on the reference asset since the
    last reset in ``unrealized_return``; this adds
    :math:`N \\cdot u \\cdot DF(T_1)` to the receiver's PV, where
    :math:`T_1` is the next reset (first payment) date.

    .. note::

       The :class:`TotalReturnSwap` pytree does not carry a reference
       asset identifier, so richer modelling (dividend projection,
       non-self-financing assets, credit adjustments) is not in
       scope for this pricer.  See the "TRS pricing" discussion in
       the fixed-income guide for context.

    Args:
        swap: Total return swap contract.
        curve: Discount / projection curve (single-curve assumption).
        unrealized_return: Optional scalar fractional return of the
            reference asset since the last reset date, e.g. ``0.02``
            for a 2 % unrealized gain.  Defaults to zero.

    Returns:
        Swap NPV.  Sign follows ``is_total_return_receiver``.
    """
    A = _annuity(swap.start_date, swap.payment_dates, curve, swap.day_count)
    spread_pv = -swap.notional * swap.funding_spread * A

    if unrealized_return is None:
        unrealized_return = jnp.array(0.0)

    next_payment = swap.payment_dates[0]
    accrued_pv = swap.notional * unrealized_return * curve(next_payment)

    receiver_pv = spread_pv + accrued_pv

    if swap.is_total_return_receiver:
        return receiver_pv
    return -receiver_pv


# ─────────────────────────────────────────────────────────────────────
# CMS rate helper
# ─────────────────────────────────────────────────────────────────────

def _cms_forward_rates(
    fixing_dates: Int[Array, " n"],
    cms_tenor: int,
    curve: DiscountCurve,
) -> Float[Array, " n"]:
    """Per-fixing forward CMS rate (no convexity adjustment).

    Builds a synthetic annual schedule ``[t_i + 1Y, …, t_i + N*Y]`` for
    each fixing date ``t_i`` (with ``N`` = ``cms_tenor``, 365 days per
    year) and returns

    .. math::

        S_i = \\frac{DF(t_i) - DF(t_i + N\\!\\cdot\\!Y)}
                   {\\sum_{k=1}^N DF(t_i + k\\!\\cdot\\!Y)}

    — the forward par swap rate for that underlying annual swap.  This
    matches what a single-curve Black-76 caplet-style pricer uses as the
    martingale CMS rate; it deliberately omits the Hagan convexity
    adjustment.
    """
    year_days = jnp.int32(365)
    periods = jnp.arange(1, cms_tenor + 1, dtype=jnp.int32)

    # (n_fixings, cms_tenor) grid of payment dates for each underlying swap.
    ann_dates = fixing_dates[:, None] + periods[None, :] * year_days

    maturity_dates = fixing_dates + cms_tenor * year_days

    dfs_ann = curve(ann_dates)            # (n_fixings, cms_tenor)
    annuity = jnp.sum(dfs_ann, axis=1)    # (n_fixings,) — tau = 1 year each

    df_start = curve(fixing_dates)
    df_end = curve(maturity_dates)
    return (df_start - df_end) / annuity


# ─────────────────────────────────────────────────────────────────────
# CMS swap
# ─────────────────────────────────────────────────────────────────────

def cms_swap_price(
    swap: CMSSwap,
    curve: DiscountCurve,
) -> Float[Array, ""]:
    """NPV of a CMS swap (constant-maturity-swap fixed-vs-CMS exchange).

    At each payment date :math:`t_i` the CMS leg pays the forward par
    swap rate :math:`S_i` of a synthetic annual ``cms_tenor``-year
    swap starting at :math:`t_i`, computed with :func:`_cms_forward_rates`
    above.  The fixed leg is the usual annuity.  Net sign follows
    ``pay_fixed``:

    .. math::

        \\text{NPV}_{\\text{payer}} = N \\sum_i S_i \\tau_i DF(t_i)
                                    - N \\cdot K \\cdot A.

    .. warning::

       **No convexity adjustment.**  The true expected CMS rate under
       each payment measure differs from the forward par swap rate by
       a convexity term that depends on the swap-rate volatility.
       This pricer uses the forward directly, which is a standard
       *baseline* but is not market-accurate for flows on CMS rates.
       Hagan-replication / SABR-integration is a legitimate larger
       piece of work tracked in the roadmap.

    Args:
        swap: CMS swap contract.
        curve: Discount / projection curve (single-curve assumption).

    Returns:
        Swap NPV.
    """
    cms_rates = _cms_forward_rates(swap.payment_dates, swap.cms_tenor, curve)

    # CMS leg: per-period tau against the swap's own day count, DF at payment.
    payment_starts = jnp.concatenate(
        [swap.start_date[None], swap.payment_dates[:-1]]
    )
    tau = year_fraction(
        payment_starts, swap.payment_dates, swap.day_count
    )
    df_pay = curve(swap.payment_dates)
    cms_leg_pv = swap.notional * jnp.sum(cms_rates * tau * df_pay)

    # Fixed leg via the standard annuity helper.
    A = _annuity(swap.start_date, swap.payment_dates, curve, swap.day_count)
    fixed_leg_pv = swap.notional * swap.fixed_rate * A

    payer_pv = cms_leg_pv - fixed_leg_pv
    if swap.pay_fixed:
        return payer_pv
    return -payer_pv


# ─────────────────────────────────────────────────────────────────────
# CMS cap / floor
# ─────────────────────────────────────────────────────────────────────

def cms_cap_floor_price_black76(
    cap: CMSCapFloor,
    curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-76 price of a CMS cap or floor (no convexity adjustment).

    Each CMS caplet is priced with Black-76 on the forward CMS rate
    :math:`F_i` computed by :func:`_cms_forward_rates`:

    .. math::

        \\text{caplet}_i
          = N \\cdot \\tau_i \\cdot DF(t_i) \\cdot
            \\left[F_i\\, \\Phi(d_1) - K\\, \\Phi(d_2)\\right]

    with :math:`d_1 = (\\ln(F_i/K) + \\tfrac12 \\sigma^2 T_i) / (\\sigma\\sqrt{T_i})`
    and :math:`d_2 = d_1 - \\sigma\\sqrt{T_i}`.  Floors are obtained via
    put-call parity:
    :math:`\\text{floorlet}_i = \\text{caplet}_i - N \\tau_i DF(t_i) (F_i - K)`.

    Because :class:`CMSCapFloor` does not carry an explicit accrual
    start, accrual periods are assumed **uniform**: each period's
    length equals the gap between consecutive ``payment_dates``, and
    the first period mirrors the second period's length.  Requires
    ``n >= 2`` payment dates.

    ``vol`` can be a scalar (flat Black vol) or a 1-D array of shape
    ``(n,)`` for a per-caplet vol term structure.

    .. warning::

       Same caveat as :func:`cms_swap_price` — the forward CMS rate is
       **not** convexity-adjusted, so this pricer is a baseline rather
       than a market-accurate pricer.

    Args:
        cap: CMS cap or floor contract.
        curve: Discount curve.
        vol: Black-76 volatility of the CMS rate, scalar or per-period.

    Returns:
        Cap or floor NPV.
    """
    cms_rates = _cms_forward_rates(cap.payment_dates, cap.cms_tenor, curve)

    # Uniform accrual periods: mirror period-1 length onto period-0 so
    # the first period does not span the whole distance from the
    # reference date to the first payment.
    diffs = jnp.diff(cap.payment_dates)
    tau_days = jnp.concatenate([diffs[:1], diffs])
    payment_starts = cap.payment_dates - tau_days
    tau = year_fraction(payment_starts, cap.payment_dates, cap.day_count)
    df_pay = curve(cap.payment_dates)

    # Time from valuation to each CMS fixing (assumed at accrual start).
    T = year_fraction(curve.reference_date, payment_starts, cap.day_count)
    # Guard T<=0 for a period that has already started.
    T_safe = jnp.maximum(T, 1e-10)
    sqrt_T = jnp.sqrt(T_safe)

    K = cap.strike
    F = cms_rates
    d1 = (jnp.log(F / K) + 0.5 * vol**2 * T_safe) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    caplet_pvs = cap.notional * tau * df_pay * (F * Phi(d1) - K * Phi(d2))

    if cap.is_cap:
        return jnp.sum(caplet_pvs)
    # Floor via put-call parity on each period.
    floorlet_pvs = caplet_pvs - cap.notional * tau * df_pay * (F - K)
    return jnp.sum(floorlet_pvs)


# ─────────────────────────────────────────────────────────────────────
# Range accrual
# ─────────────────────────────────────────────────────────────────────

def range_accrual_price_black76(
    accrual: RangeAccrual,
    curve: DiscountCurve,
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Digital-replication price of a range accrual note on a rate index.

    For each accrual period :math:`[t_{i-1}, t_i]` the coupon is
    :math:`N R \\tau_i \\cdot (\\text{days in range} / \\text{days total})`,
    where "in range" means the reference rate is inside
    ``[lower_barrier, upper_barrier]``.  This pricer replaces the true
    per-day monitoring with the **snapshot probability** under
    Black-76 that the forward rate for the period,

    .. math::

        F_i = \\frac{1}{\\tau_i}\\left(\\frac{DF(t_{i-1})}{DF(t_i)} - 1\\right),

    lies in the range at the fixing time:

    .. math::

        \\mathbb{P}(L < F_i < U)
          = \\Phi(-d_{2,U}) - \\Phi(-d_{2,L})

    with :math:`d_{2,X} = (\\ln(F_i/X) - \\tfrac12 \\sigma^2 T_i)
    / (\\sigma\\sqrt{T_i})`.  The per-period PV is then
    :math:`N R \\tau_i \\mathbb{P}(L < F_i < U) DF(t_i)`, summed over
    all periods.

    Because :class:`RangeAccrual` does not carry an explicit accrual
    start, periods are assumed **uniform**: each period's length
    equals the gap between consecutive ``payment_dates``, and the
    first period mirrors the second period's length.  Requires
    ``n >= 2`` payment dates.

    .. warning::

       This is a **snapshot approximation**: it replaces the expected
       day-by-day count with a single-time probability at the start of
       each period.  For short accrual periods and stable curves the
       error is small, but true per-day pricing requires path
       simulation.  Only the ``reference_index="rate"`` case is
       supported here; CMS and FX range accruals use different
       projection logic.

    Args:
        accrual: Range accrual contract.
        curve: Discount / projection curve.
        vol: Black-76 volatility of the reference rate.  Scalar or
            per-period shape ``(n,)``.

    Returns:
        NPV of the range accrual coupons.  Redemption of principal is
        **not** included (add it separately if the note repays par).
    """
    # Uniform accrual periods: mirror period-1 length onto period-0 so
    # the first period does not span the whole distance from the
    # reference date to the first payment.
    diffs = jnp.diff(accrual.payment_dates)
    tau_days = jnp.concatenate([diffs[:1], diffs])
    payment_starts = accrual.payment_dates - tau_days
    tau = year_fraction(
        payment_starts, accrual.payment_dates, accrual.day_count
    )
    df_start = curve(payment_starts)
    df_end = curve(accrual.payment_dates)
    F = (df_start / df_end - 1.0) / tau

    # Time from valuation to the start of each period.
    T = year_fraction(
        curve.reference_date, payment_starts, accrual.day_count
    )
    # Guard for a period that has already started (T <= 0).
    T_safe = jnp.maximum(T, 1e-10)
    sqrt_T = jnp.sqrt(T_safe)
    sigma_sqrt_T = vol * sqrt_T

    L = accrual.lower_barrier
    U = accrual.upper_barrier

    d2_L = (jnp.log(F / L) - 0.5 * vol**2 * T_safe) / sigma_sqrt_T
    d2_U = (jnp.log(F / U) - 0.5 * vol**2 * T_safe) / sigma_sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    prob_in_range = Phi(-d2_U) - Phi(-d2_L)
    # Clip to [0, 1] to absorb any tiny numerical noise for degenerate
    # periods (T≈0, narrow ranges, etc.).
    prob_in_range = jnp.clip(prob_in_range, 0.0, 1.0)

    coupon_pv = (
        accrual.notional
        * accrual.coupon_rate
        * tau
        * prob_in_range
        * df_end
    )
    return jnp.sum(coupon_pv)
