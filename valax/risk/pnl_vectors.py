"""P&L vectors: hypothetical, risk-theoretical, unexplained.

Every risk measure in VALAX — VaR, ES, backtests, FRTB PLA — reduces to a
**P&L vector** of length ``n_scenarios``: one entry per scenario or per
historical observation day.  Three flavours are kept distinct in the FRTB
sense:

- **APL** (actual P&L): from the books; not produced here.
- **HPL** (hypothetical P&L): full-revaluation of today's portfolio under
  tomorrow's market data.  Computed by :func:`hypothetical_pnl_vector`,
  which is an alias for :func:`valax.risk.var.portfolio_pnl`.
- **RTPL** (risk-theoretical P&L): the risk engine's predicted P&L using
  the precomputed bucketed sensitivities.  Computed by
  :func:`risk_theoretical_pnl_vector` as a pure batched contraction of the
  sensitivities with each scenario's shock vectors — no repricing.

For an ``n_scenarios`` × ``F`` problem the cost split is::

    Sensitivities     O(F^2)            * one pricing
    RTPL vector       O(n_scenarios * F^2)  (pure arithmetic)
    HPL  vector       O(n_scenarios)    * one pricing

so RTPL is typically two to three orders of magnitude cheaper than HPL
on option-heavy portfolios where pricing cost dominates.

See :func:`valax.risk.sensitivities.waterfall_pnl` for the per-scenario
waterfall used as the per-scenario building block.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

import equinox as eqx

from valax.market.data import MarketData
from valax.market.scenario import ScenarioSet
from valax.risk._deprecation import renamed_kwarg
from valax.risk.sensitivities import FlatSensitivities
from valax.risk.var import portfolio_pnl


# ── Risk-theoretical P&L vector ──────────────────────────────────────


@renamed_kwarg("ladder", "sensitivities")
def risk_theoretical_pnl_vector(
    sensitivities: FlatSensitivities,
    scenarios: ScenarioSet,
    base: MarketData,
) -> Float[Array, " n_scenarios"]:
    """Predicted P&L for each scenario from precomputed sensitivities.

    Implements the 10-term waterfall decomposition of
    :func:`valax.risk.sensitivities.waterfall_pnl` simultaneously across all
    scenarios via batched array contractions::

        1       Σ_a  δ_S[a]  · ΔS[s,a]
        2       Σ_a  δ_σ[a]  · Δσ[s,a]
        3       Σ_p  δ_r[p]  · Δr[s,p]
        4       Σ_a  δ_q[a]  · Δq[s,a]
        5       ½ Σ_a γ_S[a]  · ΔS[s,a]²
        6       ½ Σ_p γ_r[p]  · Δr[s,p]²
        7       Σ_a vanna[a] · ΔS[s,a] · Δσ[s,a]
        8       ½ Σ_a volga[a]· Δσ[s,a]²
        9       Σ_{a,p} γ_{S,r}[a,p] · ΔS[s,a] · Δr[s,p]
        10      Σ_{a,p} γ_{σ,r}[a,p] · Δσ[s,a] · Δr[s,p]

    Honours ``scenarios.multiplicative``: when True the per-scenario spot
    delta is ``base.spots * spot_shocks`` (returns), otherwise the spot
    shocks are used directly (absolute price changes).

    Args:
        sensitivities: Precomputed bucketed sensitivities from
            :func:`valax.risk.sensitivities.compute_flat_sensitivities`
            (the deprecated keyword ``ladder=`` is still accepted, with a
            warning).
        scenarios: Batched scenarios with leading ``n_scenarios`` axis.
        base: Base market state (only used when scenarios are
            multiplicative, to convert returns to absolute spot deltas).

    Returns:
        Predicted P&L vector of shape ``(n_scenarios,)``.
    """
    if scenarios.multiplicative:
        d_spots = base.spots[None, :] * scenarios.spot_shocks
    else:
        d_spots = scenarios.spot_shocks
    d_vols = scenarios.vol_shocks
    d_rates = scenarios.rate_shocks
    d_divs = scenarios.dividend_shocks

    # First-order terms (matrix-vector contractions per scenario)
    r1 = d_spots @ sensitivities.delta_spot
    r2 = d_vols @ sensitivities.delta_vol
    r3 = d_rates @ sensitivities.delta_rate
    r4 = d_divs @ sensitivities.delta_div

    # Second-order diagonal terms
    r5 = 0.5 * (d_spots ** 2) @ sensitivities.gamma_spot
    r6 = 0.5 * (d_rates ** 2) @ sensitivities.gamma_rate

    # Vol second-order terms
    r7 = (d_spots * d_vols) @ sensitivities.vanna
    r8 = 0.5 * (d_vols ** 2) @ sensitivities.volga

    # Cross blocks: contract over (a, p) for each scenario s
    r9 = jnp.einsum("sa,sp,ap->s", d_spots, d_rates, sensitivities.cross_spot_rate)
    r10 = jnp.einsum("sa,sp,ap->s", d_vols, d_rates, sensitivities.cross_vol_rate)

    return r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10


# ── Hypothetical P&L vector (alias for portfolio_pnl) ────────────────


def hypothetical_pnl_vector(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenarios: ScenarioSet,
) -> Float[Array, " n_scenarios"]:
    """Full-revaluation P&L for each scenario (HPL series).

    Identical to :func:`valax.risk.var.portfolio_pnl` — provided under
    this name for symmetry with :func:`risk_theoretical_pnl_vector` in
    P&L-explain and FRTB PLA workflows.
    """
    return portfolio_pnl(pricing_fn, instruments, base, scenarios)


# ── Combined explained/unexplained vector report ─────────────────────


@renamed_kwarg("ladder", "sensitivities")
def explained_unexplained_vector(
    pricing_fn: Callable,
    instruments: eqx.Module,
    base: MarketData,
    scenarios: ScenarioSet,
    sensitivities: FlatSensitivities | None = None,
) -> dict[str, Float[Array, " n_scenarios"]]:
    """Compute RTPL, HPL and the per-scenario unexplained residual.

    Useful as a one-shot diagnostic when validating bucketed sensitivities
    against full revaluation across a scenario set.  If ``sensitivities``
    is not provided they are computed on the fly.

    Args:
        pricing_fn: ``(instrument, MarketData) -> price``.
        instruments: Batched instrument pytree.
        base: Base market state.
        scenarios: Batched scenarios.
        sensitivities: Optional precomputed sensitivities; if ``None``,
            computed here (deprecated keyword: ``ladder=``).

    Returns:
        Dict with three ``(n_scenarios,)`` vectors:

        - ``"rtpl"``: sensitivity-predicted P&L.
        - ``"hpl"``: full-revaluation P&L.
        - ``"unexplained"``: ``hpl - rtpl``.
    """
    if sensitivities is None:
        # Local import to avoid a circular module-load order.
        from valax.risk.sensitivities import compute_flat_sensitivities
        sensitivities = compute_flat_sensitivities(pricing_fn, instruments, base)

    rtpl = risk_theoretical_pnl_vector(sensitivities, scenarios, base)
    hpl = hypothetical_pnl_vector(pricing_fn, instruments, base, scenarios)
    return {
        "rtpl": rtpl,
        "hpl": hpl,
        "unexplained": hpl - rtpl,
    }
