"""Bucketed sensitivities and waterfall P&L explain (flat ``MarketData`` path).

**Bucketed sensitivities** are the first- and second-order Greeks of a
portfolio at the *base* market, one entry per risk-factor bucket (asset
for spot / vol / dividend, curve pillar for rates).  They are what FRTB
SA and ISDA SIMM call *sensitivities*.  Desks often call the
first-order tenor vectors a "DV01 ladder" or "vega ladder"; VALAX
reserves *ladder* for revaluation at shifted levels (a one-dimensional
revaluation grid) — see :doc:`/theory/risk-measures` § 7.4.

The **waterfall** is the second-order Taylor P&L explain built from
them, one term per Greek::

    1  Delta (spot)    Σ  δ_S  · ΔS
    2  Delta (vol)     Σ  δ_σ  · Δσ       (= vega)
    3  Delta (rate)    Σ  δ_r  · Δr       (= rho / DV01)
    4  Delta (div)     Σ  δ_q  · Δq
    5  Gamma (spot)    ½ Σ  γ_SS · ΔS²
    6  Gamma (rate)    ½ Σ  γ_rr · Δr²
    7  Vanna           Σ  γ_Sσ · ΔS · Δσ
    8  Volga           ½ Σ  γ_σσ · Δσ²
    9  Cross (S×r)     Σ  γ_Sr · ΔS · Δr
    10 Cross (σ×r)     Σ  γ_σr · Δσ · Δr
    ─────────────────────────────────────────────
    Predicted  =  sum of all terms
    Unexplained = Actual − Predicted

All sensitivities are computed via ``jax.jacobian`` (first order) and
``jax.hessian`` (second order) in at most two autodiff passes —
compared with the ``2N`` bump-and-reprice evaluations a traditional
system needs for ``N`` risk factors, and ``N²`` for the cross-gamma
matrix.

Pricing function conventions
----------------------------
Same as :mod:`valax.risk.var`: the ``pricing_fn`` must accept
``(instrument, market: MarketData) -> Float[Array, ""]``.

The keyed counterpart on a ``MarketState`` — one block per named curve,
surface and asset — is :mod:`valax.risk.state_sensitivities`.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

import equinox as eqx

from valax.market.data import MarketData
from valax.market.scenario import MarketScenario
from valax.risk.shocks import apply_scenario
from valax.risk.var import reprice_under_scenario
from valax.dates.daycounts import year_fraction
from valax.risk._deprecation import renamed_kwarg


# ── Bucketed sensitivities pytree ────────────────────────────────────


class FlatSensitivities(eqx.Module):
    """Bucketed first- and second-order sensitivities for a portfolio.

    Every field is a JAX array whose length matches the corresponding
    risk factor dimension (``n_assets`` for spot/vol/div, ``n_pillars``
    for rates).  Second-order fields are the **diagonals** of the
    Hessian blocks; cross-gamma fields are the off-diagonal blocks
    contracted to vectors via the relevant factor pairing.

    Attributes:
        delta_spot: ∂V/∂S per asset — shape ``(n_assets,)``.
        delta_vol: ∂V/∂σ per asset (vega) — shape ``(n_assets,)``.
        delta_rate: ∂V/∂r per pillar (key-rate DV01s) — shape ``(n_pillars,)``.
        delta_div: ∂V/∂q per asset — shape ``(n_assets,)``.
        gamma_spot: ∂²V/∂S² diagonal — shape ``(n_assets,)``.
        gamma_rate: ∂²V/∂r² diagonal — shape ``(n_pillars,)``.
        volga: ∂²V/∂σ² diagonal — shape ``(n_assets,)``.
        vanna: ∂²V/∂S∂σ diagonal — shape ``(n_assets,)``.
        cross_spot_rate: ∂²V/∂S∂r matrix — shape ``(n_assets, n_pillars)``.
        cross_vol_rate: ∂²V/∂σ∂r matrix — shape ``(n_assets, n_pillars)``.
    """

    # First order
    delta_spot: Float[Array, " n_assets"]
    delta_vol: Float[Array, " n_assets"]
    delta_rate: Float[Array, " n_pillars"]
    delta_div: Float[Array, " n_assets"]

    # Second order — diagonals
    gamma_spot: Float[Array, " n_assets"]
    gamma_rate: Float[Array, " n_pillars"]
    volga: Float[Array, " n_assets"]
    vanna: Float[Array, " n_assets"]

    # Second order — cross blocks
    cross_spot_rate: Float[Array, "n_assets n_pillars"]
    cross_vol_rate: Float[Array, "n_assets n_pillars"]


# ── Waterfall P&L pytree ─────────────────────────────────────────────


class WaterfallPnL(eqx.Module):
    """Term-by-term P&L explain from bucketed sensitivities.

    Each field is a scalar representing the P&L contribution from that
    Taylor term.  ``predicted`` is the sum of all terms.  When ``actual`` and
    ``unexplained`` are populated (by :func:`waterfall_pnl_report`),
    ``unexplained = actual - predicted``.

    Attributes:
        delta_spot: First-order spot P&L.
        delta_vol: First-order vol P&L (vega P&L).
        delta_rate: First-order rate P&L (rho / DV01 P&L).
        delta_div: First-order dividend P&L.
        gamma_spot: Second-order spot P&L (gamma P&L).
        gamma_rate: Second-order rate P&L (rate convexity).
        vanna_pnl: Cross spot × vol P&L.
        volga_pnl: Second-order vol P&L (vega convexity).
        cross_spot_rate_pnl: Cross spot × rate P&L.
        cross_vol_rate_pnl: Cross vol × rate P&L.
        total_first_order: Sum of all first-order terms.
        total_second_order: Sum of all second-order terms.
        predicted: Sum of all terms (first + second order).
        actual: True P&L from full repricing (NaN if not computed).
        unexplained: actual − predicted (NaN if not computed).
    """

    delta_spot: Float[Array, ""]
    delta_vol: Float[Array, ""]
    delta_rate: Float[Array, ""]
    delta_div: Float[Array, ""]
    gamma_spot: Float[Array, ""]
    gamma_rate: Float[Array, ""]
    vanna_pnl: Float[Array, ""]
    volga_pnl: Float[Array, ""]
    cross_spot_rate_pnl: Float[Array, ""]
    cross_vol_rate_pnl: Float[Array, ""]
    total_first_order: Float[Array, ""]
    total_second_order: Float[Array, ""]
    predicted: Float[Array, ""]
    actual: Float[Array, ""]
    unexplained: Float[Array, ""]


# ── Sensitivity computation ──────────────────────────────────────────


def compute_flat_sensitivities(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
) -> FlatSensitivities:
    """Compute bucketed first- and second-order sensitivities for a portfolio.

    Uses ``jax.grad`` for the first-order terms (one reverse-mode pass)
    and ``jax.hessian`` for the second-order terms (one additional pass).
    This gives **all** bucketed first- and second-order sensitivities
    simultaneously — no bump-and-reprice.

    Rate sensitivities are expressed in **zero-rate space**: the raw
    DF-space gradient is converted via the chain rule
    ``∂V/∂r_i = ∂V/∂DF_i · (−t_i · DF_i)``.

    Args:
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree (leading batch dim).
        base: Base market state.

    Returns:
        A :class:`FlatSensitivities` with all bucketed sensitivities.
    """
    n_assets = base.spots.shape[0]
    n_pillars = base.discount_curve.pillar_dates.shape[0]

    pillar_times = year_fraction(
        base.discount_curve.reference_date,
        base.discount_curve.pillar_dates,
        base.discount_curve.day_count,
    )

    # ── Portfolio value as a function of flat risk factor arrays ──────

    def pv(spots, vols, dfs, dividends):
        curve = eqx.tree_at(
            lambda c: c.discount_factors, base.discount_curve, dfs,
        )

        def _price_one(inst, spot, vol, div):
            mkt = MarketData(
                spots=spot, vols=vol, dividends=div, discount_curve=curve,
            )
            return pricing_fn(inst, mkt)

        return jnp.sum(jax.vmap(_price_one)(
            instruments, spots, vols, dividends,
        ))

    base_spots = base.spots
    base_vols = base.vols
    base_dfs = base.discount_curve.discount_factors
    base_divs = base.dividends

    # ── First order (Jacobian) ───────────────────────────────────────

    grad_fn = jax.grad(pv, argnums=(0, 1, 2, 3))
    g_spots, g_vols, g_dfs, g_divs = grad_fn(
        base_spots, base_vols, base_dfs, base_divs,
    )

    # Convert DF sensitivities → zero-rate sensitivities
    g_rates = g_dfs * (-pillar_times * base_dfs)

    # ── Second order (Hessian) ───────────────────────────────────────
    #
    # The full Hessian over (spots, vols, dfs, divs) is a block matrix.
    # We compute the blocks we need individually for efficiency.

    # Spot gamma: ∂²V/∂S²  (diagonal)
    def pv_spots(s):
        return pv(s, base_vols, base_dfs, base_divs)

    gamma_spot = jnp.diag(jax.hessian(pv_spots)(base_spots))

    # Vol gamma (volga): ∂²V/∂σ²  (diagonal)
    def pv_vols(v):
        return pv(base_spots, v, base_dfs, base_divs)

    volga = jnp.diag(jax.hessian(pv_vols)(base_vols))

    # Rate gamma: ∂²V/∂DF²  → convert to ∂²V/∂r²
    # For the diagonal in rate space we need the chain rule:
    # ∂²V/∂r_i² ≈ (∂²V/∂DF_i²) · (t_i · DF_i)² + (∂V/∂DF_i) · (t_i² · DF_i)
    def pv_dfs(d):
        return pv(base_spots, base_vols, d, base_divs)

    hess_dfs = jax.hessian(pv_dfs)(base_dfs)
    gamma_rate_raw = jnp.diag(hess_dfs)
    df_to_rate = pillar_times * base_dfs
    gamma_rate = gamma_rate_raw * df_to_rate**2 + g_dfs * (pillar_times**2 * base_dfs)

    # Vanna: ∂²V/∂S∂σ  (diagonal — asset i's spot × asset i's vol)
    def pv_spot_vol(s, v):
        return pv(s, v, base_dfs, base_divs)

    hess_sv = jax.hessian(pv_spot_vol, argnums=(0, 1))(base_spots, base_vols)
    # hess_sv is ((H_ss, H_sv), (H_vs, H_vv)); we want H_sv diagonal
    vanna = jnp.diag(hess_sv[0][1])

    # Cross spot × rate: ∂²V/∂S∂DF → convert to ∂²V/∂S∂r
    def pv_spot_df(s, d):
        return pv(s, base_vols, d, base_divs)

    hess_sd = jax.hessian(pv_spot_df, argnums=(0, 1))(base_spots, base_dfs)
    # Shape: (n_assets, n_pillars); convert DF dimension to rate dimension
    cross_spot_rate = hess_sd[0][1] * (-df_to_rate[None, :])

    # Cross vol × rate: ∂²V/∂σ∂DF → convert to ∂²V/∂σ∂r
    def pv_vol_df(v, d):
        return pv(base_spots, v, d, base_divs)

    hess_vd = jax.hessian(pv_vol_df, argnums=(0, 1))(base_vols, base_dfs)
    cross_vol_rate = hess_vd[0][1] * (-df_to_rate[None, :])

    return FlatSensitivities(
        delta_spot=g_spots,
        delta_vol=g_vols,
        delta_rate=g_rates,
        delta_div=g_divs,
        gamma_spot=gamma_spot,
        gamma_rate=gamma_rate,
        volga=volga,
        vanna=vanna,
        cross_spot_rate=cross_spot_rate,
        cross_vol_rate=cross_vol_rate,
    )


# ── Waterfall P&L decomposition ─────────────────────────────────────


@renamed_kwarg("ladder", "sensitivities")
def waterfall_pnl(
    sensitivities: FlatSensitivities,
    scenario: MarketScenario,
    base: MarketData,
) -> WaterfallPnL:
    """Decompose a scenario into term-by-term P&L from precomputed sensitivities.

    This is pure arithmetic on the sensitivity arrays — no repricing.
    The ``actual`` and ``unexplained`` fields are set to NaN; use
    :func:`waterfall_pnl_report` to include full-repricing comparison.

    Args:
        sensitivities: Precomputed bucketed sensitivities (the deprecated
            keyword ``ladder=`` is still accepted, with a warning).
        scenario: Risk factor shocks to decompose.
        base: Base market state (needed to compute spot deltas when
            scenario uses multiplicative shocks).

    Returns:
        A :class:`WaterfallPnL` with all term contributions.
    """
    # Risk factor changes
    if scenario.multiplicative:
        d_spots = base.spots * scenario.spot_shocks
    else:
        d_spots = scenario.spot_shocks

    d_vols = scenario.vol_shocks
    d_rates = scenario.rate_shocks
    d_divs = scenario.dividend_shocks

    # Terms 1–4: First-order (delta) terms
    r_delta_spot = jnp.sum(sensitivities.delta_spot * d_spots)
    r_delta_vol = jnp.sum(sensitivities.delta_vol * d_vols)
    r_delta_rate = jnp.sum(sensitivities.delta_rate * d_rates)
    r_delta_div = jnp.sum(sensitivities.delta_div * d_divs)

    # Terms 5–6: Second-order diagonal terms
    r_gamma_spot = 0.5 * jnp.sum(sensitivities.gamma_spot * d_spots**2)
    r_gamma_rate = 0.5 * jnp.sum(sensitivities.gamma_rate * d_rates**2)

    # Terms 7–8: Volatility second-order
    r_vanna = jnp.sum(sensitivities.vanna * d_spots * d_vols)
    r_volga = 0.5 * jnp.sum(sensitivities.volga * d_vols**2)

    # Terms 9–10: Cross terms with rates
    r_cross_sr = jnp.sum(sensitivities.cross_spot_rate * d_spots[:, None] * d_rates[None, :])
    r_cross_vr = jnp.sum(sensitivities.cross_vol_rate * d_vols[:, None] * d_rates[None, :])

    total_1st = r_delta_spot + r_delta_vol + r_delta_rate + r_delta_div
    total_2nd = r_gamma_spot + r_gamma_rate + r_vanna + r_volga + r_cross_sr + r_cross_vr
    predicted = total_1st + total_2nd

    nan = jnp.array(jnp.nan)

    return WaterfallPnL(
        delta_spot=r_delta_spot,
        delta_vol=r_delta_vol,
        delta_rate=r_delta_rate,
        delta_div=r_delta_div,
        gamma_spot=r_gamma_spot,
        gamma_rate=r_gamma_rate,
        vanna_pnl=r_vanna,
        volga_pnl=r_volga,
        cross_spot_rate_pnl=r_cross_sr,
        cross_vol_rate_pnl=r_cross_vr,
        total_first_order=total_1st,
        total_second_order=total_2nd,
        predicted=predicted,
        actual=nan,
        unexplained=nan,
    )


@renamed_kwarg("ladder", "sensitivities")
def waterfall_pnl_report(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenario: MarketScenario,
    sensitivities: FlatSensitivities | None = None,
) -> WaterfallPnL:
    """Full waterfall P&L report with actual repricing and unexplained.

    If ``sensitivities`` is not provided, they are computed on the fly.
    For repeated decomposition across many scenarios, precompute them
    once with :func:`compute_flat_sensitivities` and pass them in.

    Args:
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree.
        base: Base market state.
        scenario: Scenario to decompose.
        sensitivities: Optional precomputed sensitivities (the deprecated
            keyword ``ladder=`` is still accepted, with a warning).

    Returns:
        A :class:`WaterfallPnL` with ``actual`` and ``unexplained`` populated.
    """
    if sensitivities is None:
        sensitivities = compute_flat_sensitivities(pricing_fn, instruments, base)

    wf = waterfall_pnl(sensitivities, scenario, base)

    # Full repricing for actual P&L
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

    return WaterfallPnL(
        delta_spot=wf.delta_spot,
        delta_vol=wf.delta_vol,
        delta_rate=wf.delta_rate,
        delta_div=wf.delta_div,
        gamma_spot=wf.gamma_spot,
        gamma_rate=wf.gamma_rate,
        vanna_pnl=wf.vanna_pnl,
        volga_pnl=wf.volga_pnl,
        cross_spot_rate_pnl=wf.cross_spot_rate_pnl,
        cross_vol_rate_pnl=wf.cross_vol_rate_pnl,
        total_first_order=wf.total_first_order,
        total_second_order=wf.total_second_order,
        predicted=wf.predicted,
        actual=actual,
        unexplained=actual - wf.predicted,
    )
