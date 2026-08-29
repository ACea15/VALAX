"""Centralised QuantLib adapter utilities for parametric sweep tests.

All conversion from sampled markets (dicts of JAX scalars) into
QuantLib's object graph lives here.  Tests should never construct
`ql.Date`, `ql.YieldTermStructureHandle`, or
`ql.BlackScholesMertonProcess` directly — the adapter centralises
every convention translation so a QL-side bug is fixable in one
place and a VALAX-side bug appears in the diff of production code.

Design rules
------------
1. **Integer-day expiry alignment.** QuantLib expiries are integer
   ordinal dates; VALAX expiries are continuous year-fractions.  Each
   adapter rounds the sampled expiry to the nearest integer day and
   returns an ``effective_market`` dict with the snapped expiry, so
   both engines use bit-identical inputs.  Tests must consume the
   ``effective_market``, not the raw input.
2. **No business calendars.**  VALAX uses ordinal dates with no
   business-day logic; the adapter uses ``ql.NullCalendar`` everywhere
   to match.
3. **Act/365 throughout.**  Both sides use ``ql.Actual365Fixed()`` so
   day-count conventions cannot be a source of disagreement.
4. **Stable evaluation date.**  ``ql.Settings.instance().evaluationDate``
   is reset by every adapter call to a documented default; tests must
   not depend on whatever date was set by a previous test.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import QuantLib as ql


# Default evaluation date matching VALAX's synthetic-config default
# (2026-01-01 ordinal).  Kept as a module constant so a test can
# pass an override when it needs to.
DEFAULT_QL_DATE = ql.Date(1, 1, 2026)


def reset_evaluation_date(d: ql.Date = DEFAULT_QL_DATE) -> ql.Date:
    """Set the global QL evaluation date and return it."""
    ql.Settings.instance().evaluationDate = d
    return d


def snap_expiry_to_days(t_years: float, day_count: int = 365) -> tuple[int, float]:
    """Round a year-fraction expiry to the nearest integer day.

    Returns ``(days, t_effective)`` where ``t_effective = days / day_count``.
    Tests use ``t_effective`` on both engines so the comparison is
    apples-to-apples.

    Minimum of 1 day enforced — a zero-day expiry would zero-out the
    BS variance and is not a meaningful test case.
    """
    days = max(1, int(round(float(t_years) * day_count)))
    return days, days / day_count


def market_to_ql_bsm(
    market: dict[str, Any],
    *,
    today: ql.Date = DEFAULT_QL_DATE,
    is_call: bool = True,
) -> tuple[ql.VanillaOption, ql.BlackScholesMertonProcess, dict[str, Any]]:
    """Build a QL European option + analytic BSM engine from a sampled market.

    Args:
        market: Dict produced by ``valax.market.sample_scalar_market``
            with keys ``spot, vol, rate, dividend, expiry, strike``
            as JAX scalars.
        today: QL evaluation date.  Defaults to ``DEFAULT_QL_DATE``.
        is_call: ``True`` for calls, ``False`` for puts.

    Returns:
        Triple ``(ql_option, ql_process, effective_market)``:

        - ``ql_option`` is a fully-configured ``ql.VanillaOption`` with
          ``AnalyticEuropeanEngine`` attached.
        - ``ql_process`` is the underlying ``BlackScholesMertonProcess``
          (returned so callers can build additional QL engines on the
          same flat-vol surface, e.g. an implied-vol solver).
        - ``effective_market`` is a copy of ``market`` with the
          ``expiry`` field snapped to an integer-day-aligned value.
          Callers **must** use this for the VALAX side of the
          comparison; using the original ``market`` will introduce a
          systematic discretisation gap.
    """
    reset_evaluation_date(today)

    spot = float(market["spot"])
    vol = float(market["vol"])
    rate = float(market["rate"])
    div = float(market["dividend"])
    strike = float(market["strike"])
    days, t_eff = snap_expiry_to_days(float(market["expiry"]))

    dc = ql.Actual365Fixed()
    cal = ql.NullCalendar()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, div, dc))
    vol_h = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, cal, vol, dc)
    )
    process = ql.BlackScholesMertonProcess(spot_h, div_h, rate_h, vol_h)

    maturity = today + ql.Period(days, ql.Days)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if is_call else ql.Option.Put, strike
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    effective_market = dict(market)
    effective_market["expiry"] = jnp.array(t_eff)
    return option, process, effective_market


def market_to_ql_heston_process(
    market: dict[str, Any],
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    *,
    today: ql.Date = DEFAULT_QL_DATE,
) -> tuple[ql.HestonProcess, dict[str, Any]]:
    """Build a QL HestonProcess and return the effective-market dict.

    Args:
        market: Sampled market dict (``spot, rate, dividend, expiry``).
            ``vol`` is ignored — Heston has its own variance.
        v0, kappa, theta, xi, rho: Heston parameters (see
            :class:`valax.models.HestonModel`).
        today: QL evaluation date.

    Returns:
        ``(process, effective_market)`` — see :func:`market_to_ql_bsm`
        for the effective-market contract.
    """
    reset_evaluation_date(today)

    spot = float(market["spot"])
    rate = float(market["rate"])
    div = float(market["dividend"])
    _, t_eff = snap_expiry_to_days(float(market["expiry"]))

    dc = ql.Actual365Fixed()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_h = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    div_h = ql.YieldTermStructureHandle(ql.FlatForward(today, div, dc))
    process = ql.HestonProcess(
        rate_h, div_h, spot_h,
        float(v0), float(kappa), float(theta), float(xi), float(rho),
    )

    effective_market = dict(market)
    effective_market["expiry"] = jnp.array(t_eff)
    return process, effective_market


def ql_flat_curve(
    rate: float,
    today: ql.Date = DEFAULT_QL_DATE,
    day_count: ql.DayCounter | None = None,
) -> ql.YieldTermStructureHandle:
    """Build a QL flat-rate ``YieldTermStructureHandle``."""
    reset_evaluation_date(today)
    dc = day_count or ql.Actual365Fixed()
    return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))


def ql_dates_from_year_offsets(
    years: list[float] | tuple[float, ...],
    today: ql.Date = DEFAULT_QL_DATE,
) -> list[ql.Date]:
    """Convert a list of year-offsets to a list of integer-day ``ql.Date``."""
    return [today + ql.Period(int(round(y * 365)), ql.Days) for y in years]


def ql_annual_ibor_index(
    rate: float,
    today: ql.Date = DEFAULT_QL_DATE,
) -> tuple[ql.IborIndex, ql.YieldTermStructureHandle]:
    """Build the annual, Act/365, ``NullCalendar`` flat Ibor index.

    This is the single-curve index used by the caplet/CMS fixtures below; it
    matches VALAX's annual synthetic-swap convention (one period per year,
    Act/365, no business-day logic).

    Args:
        rate: Flat continuously-compounded rate for the discount curve.
        today: QL evaluation date.

    Returns:
        Tuple ``(ibor_index, discount_handle)``.
    """
    reset_evaluation_date(today)
    dc = ql.Actual365Fixed()
    disc = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))
    idx = ql.IborIndex(
        "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(),
        ql.NullCalendar(), ql.Unadjusted, False, dc, disc,
    )
    return idx, disc


def ql_optionlet_vol_surface(
    expiry_years: list[float] | tuple[float, ...],
    strikes: list[float] | tuple[float, ...],
    vol_grid,
    rate: float,
    today: ql.Date = DEFAULT_QL_DATE,
) -> dict[str, Any]:
    """Build a QL optionlet (caplet) vol structure directly from a vol grid.

    Wraps ``ql.StrippedOptionlet`` + ``ql.StrippedOptionletAdapter`` around a
    pre-computed ``expiry x strike`` grid of caplet Black vols (e.g. a VALAX
    :class:`~valax.surfaces.optionlet_surface.OptionletVolSurface` sampled at
    its calibration nodes).  At the node ``(expiry, strike)`` points the adapter
    returns the input vol exactly, so a cap priced with
    ``ql.BlackCapFloorEngine`` reading this structure matches a VALAX cap
    priced off the same surface with no interpolation gap — provided the cap's
    strike and caplet expiries coincide with grid nodes.

    Args:
        expiry_years: Optionlet expiries (year fractions), one per grid row.
        strikes: Strikes (grid columns).
        vol_grid: ``(n_expiries, n_strikes)`` array-like of Black caplet vols.
        rate: Flat discount rate.
        today: QL evaluation date.

    Returns:
        Dict with ``adapter`` (``StrippedOptionletAdapter``), ``handle``
        (``OptionletVolatilityStructureHandle``), ``index`` (the annual Ibor
        index) and ``discount`` (yield-curve handle).
    """
    idx, disc = ql_annual_ibor_index(rate, today)
    strikes = [float(k) for k in strikes]
    opt_dates = [
        today + ql.Period(int(round(float(t) * 365)), ql.Days)
        for t in expiry_years
    ]
    vol_quotes = [
        [ql.QuoteHandle(ql.SimpleQuote(float(vol_grid[i][j])))
         for j in range(len(strikes))]
        for i in range(len(opt_dates))
    ]
    stripped = ql.StrippedOptionlet(
        0, ql.NullCalendar(), ql.Unadjusted, idx,
        opt_dates, strikes, vol_quotes, ql.Actual365Fixed(),
    )
    adapter = ql.StrippedOptionletAdapter(stripped)
    handle = ql.OptionletVolatilityStructureHandle(adapter)
    return {
        "adapter": adapter, "handle": handle,
        "index": idx, "discount": disc,
    }


def ql_stripped_optionlet_surface(
    cap_tenor_years: list[float] | tuple[float, ...],
    cap_strikes: list[float] | tuple[float, ...],
    cap_vol_grid,
    rate: float,
    switch_strike: float,
    today: ql.Date = DEFAULT_QL_DATE,
) -> dict[str, Any]:
    """Strip caplet vols from cap (term) vols via ``ql.OptionletStripper1``.

    Builds a ``ql.CapFloorTermVolSurface`` from a ``tenor x strike`` grid of
    flat cap vols and inverts it into a caplet (optionlet) vol structure with
    ``ql.OptionletStripper1`` + ``ql.StrippedOptionletAdapter`` — the market
    "bootstrap the caplet smile from cap quotes" path.

    Args:
        cap_tenor_years: Cap maturities in years (surface rows).
        cap_strikes: Strikes (surface columns).
        cap_vol_grid: ``(n_tenors, n_strikes)`` array-like of flat cap vols.
        rate: Flat discount rate.
        switch_strike: Reference strike QL uses to switch between capfloor
            parity instruments during stripping.
        today: QL evaluation date.

    Returns:
        Dict with ``adapter``, ``handle``, ``stripper`` (the
        ``OptionletStripper1``), ``index`` and ``discount``.
    """
    idx, disc = ql_annual_ibor_index(rate, today)
    cap_strikes = [float(k) for k in cap_strikes]
    tenors = [ql.Period(int(round(float(t))), ql.Years) for t in cap_tenor_years]
    m = ql.Matrix(len(tenors), len(cap_strikes))
    for i in range(len(tenors)):
        for j in range(len(cap_strikes)):
            m[i][j] = float(cap_vol_grid[i][j])
    term_surface = ql.CapFloorTermVolSurface(
        0, ql.NullCalendar(), ql.Unadjusted, tenors, cap_strikes, m,
        ql.Actual365Fixed(),
    )
    stripper = ql.OptionletStripper1(
        term_surface, idx, float(switch_strike), 1e-6, 100, disc,
    )
    adapter = ql.StrippedOptionletAdapter(stripper)
    handle = ql.OptionletVolatilityStructureHandle(adapter)
    return {
        "adapter": adapter, "handle": handle, "stripper": stripper,
        "index": idx, "discount": disc,
    }


def ql_cms_convexity_setup(
    rate: float,
    tenor_years: int,
    expiry_years: float,
    flat_vol: float,
    today: ql.Date = DEFAULT_QL_DATE,
) -> dict[str, Any]:
    """Build a QL CMS coupon and read Hagan analytic/numeric convexity.

    Constructs an annual, Act/365 ``ql.SwapIndex`` (matching VALAX's annual
    synthetic swap), a flat ``ql.ConstantSwaptionVolatility`` and a
    near-natural CMS coupon (payment one day after the fixing, so the
    fixing→payment delay is negligible and the coupon approximates the natural
    CMS swaplet VALAX models).  Both ``ql.AnalyticHaganPricer`` and
    ``ql.NumericHaganPricer`` (``GFunctionStandard``, zero mean reversion) are
    evaluated.

    Args:
        rate: Flat discount rate.
        tenor_years: Underlying-swap tenor in years.
        expiry_years: Time to the CMS fixing in years.
        flat_vol: Flat Black swaption volatility.
        today: QL evaluation date.

    Returns:
        Dict with ``forward`` (forward swap rate), ``analytic_adj`` and
        ``numeric_adj`` (convexity adjustments), and ``analytic_rate`` /
        ``numeric_rate`` (adjusted CMS rates).
    """
    idx, disc = ql_annual_ibor_index(rate, today)
    dc = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    swvolh = ql.SwaptionVolatilityStructureHandle(
        ql.ConstantSwaptionVolatility(
            today, cal, ql.Unadjusted,
            ql.QuoteHandle(ql.SimpleQuote(float(flat_vol))), dc,
        )
    )
    mean_rev = ql.QuoteHandle(ql.SimpleQuote(0.0))
    swap_index = ql.SwapIndex(
        "FlatCMS", ql.Period(int(tenor_years), ql.Years), 0, ql.EURCurrency(),
        cal, ql.Period(ql.Annual), ql.Unadjusted, dc, idx,
    )
    start = today + ql.Period(int(round(float(expiry_years) * 365)), ql.Days)
    end = start + ql.Period(1, ql.Days)   # minimal delay ⇒ natural swaplet
    forward = float(swap_index.fixing(start))
    coupon = ql.CmsCoupon(end, 1_000_000.0, start, end, 0, swap_index)

    coupon.setPricer(
        ql.AnalyticHaganPricer(swvolh, ql.GFunctionFactory.Standard, mean_rev)
    )
    analytic_adj = float(coupon.convexityAdjustment())
    analytic_rate = float(coupon.rate())

    coupon.setPricer(
        ql.NumericHaganPricer(swvolh, ql.GFunctionFactory.Standard, mean_rev)
    )
    numeric_adj = float(coupon.convexityAdjustment())
    numeric_rate = float(coupon.rate())

    return {
        "forward": forward,
        "analytic_adj": analytic_adj, "analytic_rate": analytic_rate,
        "numeric_adj": numeric_adj, "numeric_rate": numeric_rate,
    }


__all__ = [
    "DEFAULT_QL_DATE",
    "reset_evaluation_date",
    "snap_expiry_to_days",
    "market_to_ql_bsm",
    "market_to_ql_heston_process",
    "ql_flat_curve",
    "ql_dates_from_year_offsets",
    "ql_annual_ibor_index",
    "ql_optionlet_vol_surface",
    "ql_stripped_optionlet_surface",
    "ql_cms_convexity_setup",
]
