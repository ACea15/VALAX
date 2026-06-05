"""Risk measurement: scenario generation, shock application, VaR, ladders.

Modules
-------
- :mod:`valax.risk.shocks`        — apply risk-factor shocks to ``MarketData``.
- :mod:`valax.risk.scenarios`     — generate parametric / historical / stress scenarios.
- :mod:`valax.risk.var`           — VaR, ES, parametric VaR, scalar P&L attribution.
- :mod:`valax.risk.ladders`       — sensitivity ladders and waterfall P&L.
- :mod:`valax.risk.pnl_vectors`   — HPL / RTPL P&L vectors over scenario sets.
- :mod:`valax.risk.backtesting`   — Kupiec / Christoffersen / FRTB PLA tests.
"""

from valax.risk.shocks import (
    apply_scenario,
    bump_curve_zero_rates,
    parallel_shift,
    key_rate_bump,
    # Multi-curve basis shocks
    bump_discount_curve,
    bump_forward_curve,
    parallel_basis_shift,
    # Credit shocks
    bump_hazard_rates,
    parallel_credit_spread_shift,
    key_rate_hazard_bump,
)
from valax.risk.scenarios import (
    parametric_scenarios,
    historical_scenarios,
    stress_scenario,
    steepener,
    flattener,
    butterfly,
)
from valax.risk.var import (
    wrap_equity_pricing_fn,
    reprice_under_scenario,
    portfolio_pnl,
    parametric_var,
    pnl_attribution,
    value_at_risk,
    expected_shortfall,
)
from valax.risk.ladders import (
    SensitivityLadder,
    WaterfallPnL,
    compute_ladder,
    waterfall_pnl,
    waterfall_pnl_report,
)
from valax.risk.pnl_vectors import (
    risk_theoretical_pnl_vector,
    hypothetical_pnl_vector,
    explained_unexplained_vector,
)
from valax.risk.backtesting import (
    var_breaches,
    kupiec_pof,
    christoffersen_independence,
    christoffersen_conditional_coverage,
    basel_traffic_light,
    pla_spearman,
    ks_statistic,
    pla_ks,
    pla_traffic_light,
)
from valax.risk.bucketing import (
    BucketMap,
    BucketedLadder,
    aggregate,
    pushforward_scenario,
    aggregate_covariance,
    aggregate_matrix,
    pushforward_sensitivities,
    pullback_shocks,
    reparameterize_covariance,
    jacobian_from_fn,
    tenor_bucket_map,
    equal_weight_bucket_map,
    level_slope_curvature_jacobian,
    pca_jacobian,
    bucket_sensitivity_ladder,
)
