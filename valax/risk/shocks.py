"""Shock application: apply scenarios to market data.

Core operation for VaR and stress testing. All functions are pure,
JIT-compatible, and differentiable — so ``jax.grad`` through a shocked
repricing gives sensitivities to the shock magnitudes themselves.
"""

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.market.data import MarketData
from valax.market.scenario import MarketScenario


# ── Curve shocks ─────────────────────────────────────────────────────


def bump_curve_zero_rates(
    curve: DiscountCurve,
    rate_bumps: Float[Array, " n_pillars"],
) -> DiscountCurve:
    """Apply additive zero-rate bumps to each pillar of a discount curve.

    Given continuously-compounded zero rates ``r_i`` at year fractions
    ``t_i``, bumping by ``dr_i`` gives::

        new_df[i] = exp(-(r_i + dr_i) * t_i) = old_df[i] * exp(-dr_i * t_i)

    The returned curve has the same pillar structure; its log-linear
    interpolation naturally produces smooth shocked rates between pillars.

    Follows the same construction pattern as ``key_rate_durations`` in
    ``valax/pricing/analytic/bonds.py``.
    """
    pillar_times = year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )
    adjustment = jnp.exp(-rate_bumps * pillar_times)
    new_dfs = curve.discount_factors * adjustment
    return DiscountCurve(
        pillar_dates=curve.pillar_dates,
        discount_factors=new_dfs,
        reference_date=curve.reference_date,
        day_count=curve.day_count,
    )


def parallel_shift(
    curve: DiscountCurve,
    bump: Float[Array, ""],
) -> DiscountCurve:
    """Parallel shift: bump all pillar zero rates by the same amount."""
    n = curve.pillar_dates.shape[0]
    return bump_curve_zero_rates(curve, jnp.full(n, bump))


def key_rate_bump(
    curve: DiscountCurve,
    pillar_index: int,
    bump: Float[Array, ""],
) -> DiscountCurve:
    """Bump a single pillar's zero rate."""
    n = curve.pillar_dates.shape[0]
    bumps = jnp.zeros(n).at[pillar_index].set(bump)
    return bump_curve_zero_rates(curve, bumps)


# ── Full scenario application ────────────────────────────────────────


def apply_scenario(
    base: MarketData,
    scenario: MarketScenario,
) -> MarketData:
    """Apply a scenario (risk factor deltas) to a base market state.

    Returns a new MarketData with shocked values. Fully differentiable
    with respect to both the base market data and the shock magnitudes.

    Spot shocks are multiplicative when ``scenario.multiplicative`` is True
    (``new = old * (1 + shock)``), otherwise additive (``new = old + shock``).
    Vol, dividend, and rate shocks are always additive.
    """
    if scenario.multiplicative:
        new_spots = base.spots * (1.0 + scenario.spot_shocks)
    else:
        new_spots = base.spots + scenario.spot_shocks

    new_vols = base.vols + scenario.vol_shocks
    new_dividends = base.dividends + scenario.dividend_shocks
    new_curve = bump_curve_zero_rates(base.discount_curve, scenario.rate_shocks)

    return MarketData(
        spots=new_spots,
        vols=new_vols,
        dividends=new_dividends,
        discount_curve=new_curve,
    )
