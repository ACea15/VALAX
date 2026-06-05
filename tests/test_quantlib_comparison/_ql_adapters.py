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


__all__ = [
    "DEFAULT_QL_DATE",
    "reset_evaluation_date",
    "snap_expiry_to_days",
    "market_to_ql_bsm",
    "market_to_ql_heston_process",
    "ql_flat_curve",
    "ql_dates_from_year_offsets",
]
