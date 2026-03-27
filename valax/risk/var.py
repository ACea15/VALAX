"""Value-at-Risk and Expected Shortfall via scenario repricing.

The core loop is ``jax.vmap`` over scenarios: each iteration applies one
``MarketScenario`` to the base ``MarketData``, reprices the portfolio,
and returns a scalar P&L.  The resulting P&L vector feeds into standard
risk measures.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

import equinox as eqx

from valax.market.data import MarketData
from valax.market.scenario import MarketScenario, ScenarioSet
from valax.risk.shocks import apply_scenario


def reprice_under_scenario(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenario: MarketScenario,
) -> Float[Array, ""]:
    """Reprice a portfolio under a single scenario.

    Applies the scenario to the base market, extracts flat market
    arguments (spot, vol, rate, dividend) per instrument, prices
    via ``jax.vmap``, and sums to get portfolio value.

    The ``pricing_fn`` must have signature
    ``(instrument, spot, vol, rate, dividend) -> price``.
    Rate is extracted as the mean zero rate implied by the discount
    curve (a rough scalar proxy for equity-style pricing functions).
    """
    shocked = apply_scenario(base, scenario)
    curve = shocked.discount_curve

    # Extract a scalar short rate from the curve for equity pricing.
    # Use the first non-reference pillar's zero rate as a proxy.
    pillar_times = (
        (curve.pillar_dates - curve.reference_date).astype(jnp.float64) / 365.0
    )
    log_dfs = jnp.log(curve.discount_factors)
    # Avoid division by zero at t=0 by using the second pillar
    safe_times = jnp.where(pillar_times > 0, pillar_times, 1.0)
    zero_rates = -log_dfs / safe_times
    # Use the shortest-maturity positive-time rate
    short_rate = zero_rates[1] if curve.pillar_dates.shape[0] > 1 else zero_rates[0]

    n = shocked.spots.shape[0]
    rates = jnp.full(n, short_rate)

    prices = jax.vmap(pricing_fn)(instruments, shocked.spots, shocked.vols, rates, shocked.dividends)
    return jnp.sum(prices)


def portfolio_pnl(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenarios: ScenarioSet,
) -> Float[Array, " n_scenarios"]:
    """Compute P&L for each scenario via vmap.

    ``pnl[i] = portfolio_value(scenario_i) - portfolio_value(base)``

    Args:
        pricing_fn: Scalar pricing function with signature
            ``(instrument, spot, vol, rate, dividend) -> price``.
        instruments: Batched instrument pytree (leading batch dim).
        base: Base market state.
        scenarios: Batched scenarios with leading ``n_scenarios`` axis.

    Returns:
        P&L vector of shape ``(n_scenarios,)``.
    """
    # Base value (zero scenario)
    n_assets = base.spots.shape[0]
    n_pillars = base.discount_curve.pillar_dates.shape[0]
    zero = MarketScenario(
        spot_shocks=jnp.zeros(n_assets),
        vol_shocks=jnp.zeros(n_assets),
        rate_shocks=jnp.zeros(n_pillars),
        dividend_shocks=jnp.zeros(n_assets),
    )
    base_value = reprice_under_scenario(pricing_fn, instruments, base, zero)

    def _single_pnl(scenario: MarketScenario) -> Float[Array, ""]:
        return reprice_under_scenario(pricing_fn, instruments, base, scenario) - base_value

    return jax.vmap(_single_pnl)(scenarios)


def value_at_risk(
    pnl: Float[Array, " n_scenarios"],
    confidence: float = 0.99,
) -> Float[Array, ""]:
    """Value-at-Risk: negative of the ``(1 - confidence)`` quantile of P&L.

    A positive VaR indicates a loss threshold: there is a
    ``(1 - confidence)`` probability of losing more than VaR.
    """
    q = (1.0 - confidence) * 100.0
    return -jnp.percentile(pnl, q)


def expected_shortfall(
    pnl: Float[Array, " n_scenarios"],
    confidence: float = 0.99,
) -> Float[Array, ""]:
    """Expected Shortfall (CVaR): mean of losses beyond VaR.

    ``ES = -E[P&L | P&L <= -VaR]``
    """
    var = value_at_risk(pnl, confidence)
    tail_mask = pnl <= -var
    n_tail = jnp.sum(tail_mask)
    tail_sum = jnp.sum(jnp.where(tail_mask, pnl, 0.0))
    return -tail_sum / jnp.maximum(n_tail, 1.0)
