"""Value-at-Risk, Expected Shortfall, parametric VaR, and P&L attribution.

The full-revaluation VaR loop uses ``jax.vmap`` over scenarios: each
iteration applies one ``MarketScenario`` to the base ``MarketData``,
reprices the portfolio, and returns a scalar P&L.

Parametric VaR uses autodiff Greeks to approximate P&L without
repricing — faster for large portfolios with many scenarios.

P&L attribution decomposes an observed P&L into risk factor contributions
(delta, gamma, vega, theta, cross-gamma, unexplained) using a Taylor
expansion with autodiff-computed sensitivities.

Pricing function conventions
----------------------------
The ``pricing_fn`` passed to repricing functions must have the signature::

    pricing_fn(instrument, market: MarketData) -> Float[Array, ""]

This works for any instrument type — equity options, bonds, swaps, caps,
swaptions — because the pricing function receives the full market state
(including the complete discount curve) and extracts what it needs.

For equity-style pricing functions with signature
``(instrument, spot, vol, rate, dividend) -> price``, use
``wrap_equity_pricing_fn`` to adapt them.
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
from valax.dates.daycounts import year_fraction


# ── Pricing function adapter ─────────────────────────────────────────


def wrap_equity_pricing_fn(fn: Callable) -> Callable:
    """Wrap an equity-style pricing function for use with the risk engine.

    Adapts a function with signature
    ``(instrument, spot, vol, rate, dividend) -> price``
    to the market-data-aware signature
    ``(instrument, market: MarketData) -> price``.

    The scalar risk-free rate is extracted as the shortest-maturity
    zero rate from the discount curve.
    """

    def wrapped(instrument, market: MarketData) -> Float[Array, ""]:
        rate = _extract_short_rate(market.discount_curve)
        return fn(instrument, market.spots, market.vols, rate, market.dividends)

    return wrapped


def _extract_short_rate(curve):
    """Extract a scalar short rate from a DiscountCurve."""
    pillar_times = (
        (curve.pillar_dates - curve.reference_date).astype(jnp.float64) / 365.0
    )
    log_dfs = jnp.log(curve.discount_factors)
    safe_times = jnp.where(pillar_times > 0, pillar_times, 1.0)
    zero_rates = -log_dfs / safe_times
    return zero_rates[1] if curve.pillar_dates.shape[0] > 1 else zero_rates[0]


# ── Full-revaluation repricing ───────────────────────────────────────


def reprice_under_scenario(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenario: MarketScenario,
) -> Float[Array, ""]:
    """Reprice a portfolio under a single scenario.

    Applies the scenario to the base market state, then prices each
    instrument via ``jax.vmap`` and sums to get the portfolio value.

    The ``pricing_fn`` must have signature
    ``(instrument, market: MarketData) -> price``.
    It receives the full shocked ``MarketData`` including the complete
    discount curve, so it works for any instrument type.

    For equity-style functions, wrap them first::

        from valax.risk.var import wrap_equity_pricing_fn
        market_fn = wrap_equity_pricing_fn(black_scholes_price)

    Args:
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree (leading batch dim).
        base: Base market state.
        scenario: Single scenario to apply.

    Returns:
        Portfolio value (sum of individual prices).
    """
    shocked = apply_scenario(base, scenario)

    # vmap over instruments AND per-asset market data (spot, vol, dividend),
    # while sharing the discount curve across all instruments.
    def _price_one(inst, spot, vol, div):
        per_inst_market = MarketData(
            spots=spot,
            vols=vol,
            dividends=div,
            discount_curve=shocked.discount_curve,
        )
        return pricing_fn(inst, per_inst_market)

    prices = jax.vmap(_price_one)(
        instruments, shocked.spots, shocked.vols, shocked.dividends,
    )
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
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree (leading batch dim).
        base: Base market state.
        scenarios: Batched scenarios with leading ``n_scenarios`` axis.

    Returns:
        P&L vector of shape ``(n_scenarios,)``.
    """
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


# ── Parametric VaR ───────────────────────────────────────────────────


def parametric_var(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    cov: Float[Array, "n_factors n_factors"],
    confidence: float = 0.99,
) -> Float[Array, ""]:
    """Delta-normal VaR using autodiff sensitivities.

    Computes the portfolio's sensitivity (gradient) to each risk factor
    via ``jax.grad``, then uses the covariance matrix to estimate the
    portfolio variance without any repricing::

        sigma_P^2 = delta^T @ cov @ delta
        VaR = z_alpha * sigma_P

    where ``z_alpha`` is the normal quantile at the given confidence level.

    This is a first-order (delta) approximation — it assumes linear
    P&L. Fast and accurate for well-hedged portfolios; less accurate
    for highly convex positions (use full-revaluation VaR for those).

    Risk factor ordering in the covariance matrix must match::

        [spot_0..spot_n, vol_0..vol_n, rate_0..rate_p, div_0..div_n]

    Args:
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree (leading batch dim).
        base: Base market state.
        cov: Covariance matrix of risk factor changes.
        confidence: VaR confidence level (e.g., 0.99).

    Returns:
        Parametric VaR (positive = loss threshold).
    """
    def portfolio_value(spots, vols, dfs, dividends):
        curve = eqx.tree_at(
            lambda c: c.discount_factors, base.discount_curve, dfs,
        )

        def _price_one(inst, spot, vol, div):
            per_inst_market = MarketData(
                spots=spot, vols=vol, dividends=div, discount_curve=curve,
            )
            return pricing_fn(inst, per_inst_market)

        return jnp.sum(jax.vmap(_price_one)(instruments, spots, vols, dividends))

    # Compute gradients w.r.t. each risk factor group
    grad_fn = jax.grad(portfolio_value, argnums=(0, 1, 2, 3))
    g_spots, g_vols, g_dfs, g_divs = grad_fn(
        base.spots, base.vols,
        base.discount_curve.discount_factors, base.dividends,
    )

    # Convert DF sensitivities to zero-rate sensitivities:
    # dP/dr_i = dP/dDF_i * dDF_i/dr_i = dP/dDF_i * (-t_i * DF_i)
    pillar_times = year_fraction(
        base.discount_curve.reference_date,
        base.discount_curve.pillar_dates,
        base.discount_curve.day_count,
    )
    g_rates = g_dfs * (-pillar_times * base.discount_curve.discount_factors)

    # Stack into a single delta vector matching covariance column ordering
    delta = jnp.concatenate([g_spots, g_vols, g_rates, g_divs])

    # Portfolio variance and VaR
    portfolio_var = delta @ cov @ delta
    portfolio_std = jnp.sqrt(jnp.maximum(portfolio_var, 0.0))

    z_alpha = -jax.scipy.stats.norm.ppf(1.0 - confidence)
    return z_alpha * portfolio_std


# ── P&L Attribution ──────────────────────────────────────────────────


def pnl_attribution(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenario: MarketScenario,
) -> dict[str, Float[Array, ""]]:
    """Decompose scenario P&L into risk factor contributions.

    Uses a second-order Taylor expansion of the portfolio value:

    .. math::

        \\Delta P \\approx \\underbrace{\\sum_i \\delta_i \\Delta x_i}_{\\text{delta}}
        + \\underbrace{\\frac{1}{2} \\sum_i \\gamma_{ii} \\Delta x_i^2}_{\\text{gamma}}
        + \\underbrace{\\text{remainder}}_{\\text{unexplained}}

    The decomposition is:

    - **delta_spot**: P&L from spot moves (first order)
    - **delta_vol**: P&L from vol moves (vega, first order)
    - **delta_rate**: P&L from rate moves (rho/DV01, first order)
    - **gamma_spot**: P&L from spot convexity (second order)
    - **total_first_order**: sum of all delta terms
    - **total_second_order**: total_first_order + gamma_spot
    - **actual**: true P&L from full repricing
    - **unexplained**: actual - total_second_order

    All sensitivities are computed via autodiff — no finite differences.

    Args:
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree (leading batch dim).
        base: Base market state.
        scenario: Single scenario whose P&L to decompose.

    Returns:
        Dict with attribution components.
    """
    def portfolio_value_from_factors(spots, vols, dfs, dividends):
        curve = eqx.tree_at(
            lambda c: c.discount_factors, base.discount_curve, dfs,
        )

        def _price_one(inst, spot, vol, div):
            per_inst_market = MarketData(
                spots=spot, vols=vol, dividends=div, discount_curve=curve,
            )
            return pricing_fn(inst, per_inst_market)

        return jnp.sum(jax.vmap(_price_one)(instruments, spots, vols, dividends))

    base_spots = base.spots
    base_vols = base.vols
    base_dfs = base.discount_curve.discount_factors
    base_divs = base.dividends

    # First-order sensitivities
    grad_fn = jax.grad(portfolio_value_from_factors, argnums=(0, 1, 2, 3))
    g_spots, g_vols, g_dfs, g_divs = grad_fn(
        base_spots, base_vols, base_dfs, base_divs,
    )

    # Second-order: spot gamma (diagonal of Hessian w.r.t. spots)
    def pv_spots(s):
        return portfolio_value_from_factors(s, base_vols, base_dfs, base_divs)

    spot_hessian_diag = jax.grad(lambda s: jnp.sum(jax.grad(pv_spots)(s) * s))(base_spots)
    # The above computes d/ds_i (dP/ds_i * s_i) which is not quite right.
    # Use the proper diagonal: d^2P/ds_i^2
    spot_gamma = jnp.diag(jax.hessian(pv_spots)(base_spots))

    # Convert DF sensitivities to rate-shock sensitivities
    pillar_times = year_fraction(
        base.discount_curve.reference_date,
        base.discount_curve.pillar_dates,
        base.discount_curve.day_count,
    )
    # dP/dr_i = dP/dDF_i * (-t_i * DF_i)
    g_rates = g_dfs * (-pillar_times * base_dfs)

    # Compute the shocked market to get actual deltas
    shocked = apply_scenario(base, scenario)
    d_spots = shocked.spots - base_spots
    d_vols = shocked.vols - base_vols
    d_divs = shocked.dividends - base_divs

    # Rate shocks are already in zero-rate space
    d_rates = scenario.rate_shocks

    # Attribution components
    delta_spot = jnp.sum(g_spots * d_spots)
    delta_vol = jnp.sum(g_vols * d_vols)
    delta_rate = jnp.sum(g_rates * d_rates)
    delta_div = jnp.sum(g_divs * d_divs)
    gamma_spot = 0.5 * jnp.sum(spot_gamma * d_spots**2)

    total_first_order = delta_spot + delta_vol + delta_rate + delta_div
    total_second_order = total_first_order + gamma_spot

    # Actual P&L via full repricing
    n_assets = base.spots.shape[0]
    n_pillars = base.discount_curve.pillar_dates.shape[0]
    zero = MarketScenario(
        spot_shocks=jnp.zeros(n_assets),
        vol_shocks=jnp.zeros(n_assets),
        rate_shocks=jnp.zeros(n_pillars),
        dividend_shocks=jnp.zeros(n_assets),
    )
    base_val = reprice_under_scenario(pricing_fn, instruments, base, zero)
    shocked_val = reprice_under_scenario(pricing_fn, instruments, base, scenario)
    actual = shocked_val - base_val

    return {
        "delta_spot": delta_spot,
        "delta_vol": delta_vol,
        "delta_rate": delta_rate,
        "delta_div": delta_div,
        "gamma_spot": gamma_spot,
        "total_first_order": total_first_order,
        "total_second_order": total_second_order,
        "actual": actual,
        "unexplained": actual - total_second_order,
    }


# ── Risk measures ────────────────────────────────────────────────────


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
