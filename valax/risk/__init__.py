"""Risk measurement: scenarios, shocks, sensitivities, P&L explain, VaR.

Modules
-------
- :mod:`valax.risk.shocks`        — apply risk-factor shocks to ``MarketData``.
- :mod:`valax.risk.state_shocks`  — apply *keyed* shocks to ``MarketState``.
- :mod:`valax.risk.scenarios`     — generate parametric / historical / stress scenarios,
  over the legacy flat layout or the identifier-keyed, per-curve
  :class:`~valax.risk.scenarios.KeyedFactorLayout`.
- :mod:`valax.risk.var`           — VaR, ES, parametric VaR, scalar P&L attribution.
- :mod:`valax.risk.state_var`     — group repricing / P&L / gradients through ``price``.
- :mod:`valax.risk.sensitivities`       — bucketed sensitivities and waterfall
  P&L on the flat ``MarketData`` view.
- :mod:`valax.risk.state_sensitivities` — *keyed* sensitivities / waterfall /
  VaR on a ``MarketState``, one block per named curve, surface and asset.
- :mod:`valax.risk.pnl_vectors`   — HPL / RTPL P&L vectors over scenario sets.
- :mod:`valax.risk.backtesting`   — Kupiec / Christoffersen / FRTB PLA tests.

Terminology: *sensitivities* are base-point bucketed Greeks (the FRTB SA /
ISDA SIMM sense).  The former ``*Ladder`` names (``SensitivityLadder``,
``KeyedLadder``, ...) remain importable with a ``DeprecationWarning``.
"""

from valax.risk.shocks import (
    apply_scenario,
    bump_curve_zero_rates,
    parallel_shift,
    key_rate_bump,
    pca_curve_shock,
    # Multi-curve basis shocks
    bump_discount_curve,
    bump_forward_curve,
    parallel_basis_shift,
    # Curve-graph shocks
    bump_graph_curve,
    parallel_graph_shift,
    key_rate_graph_bump,
    # Credit shocks
    bump_hazard_rates,
    parallel_credit_spread_shift,
    key_rate_hazard_bump,
)
from valax.risk.state_shocks import (
    shift_surface,
    shock,
)
from valax.risk.scenarios import (
    parametric_scenarios,
    historical_scenarios,
    stress_scenario,
    steepener,
    flattener,
    butterfly,
    # Curve-shape algebra shared by both layouts
    CURVE_SHAPES,
    curve_shape_bumps,
    # Keyed, per-curve factor layout
    FACTOR_KINDS,
    FACTOR_CATEGORIES,
    FactorBlock,
    KeyedFactorLayout,
    as_basis_layout,
    basis_factor_prefix,
    block_covariance,
    historical_block_covariance,
    psd_factor,
    # Keyed generators
    keyed_parametric_scenarios,
    keyed_historical_scenarios,
    keyed_stress_scenario,
    keyed_curve_stress,
    keyed_steepener,
    keyed_flattener,
    keyed_butterfly,
    keyed_basis_shift,
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
from valax.risk.state_var import (
    group_size,
    group_prices,
    group_pv,
    group_reprice,
    group_scenario_pnl,
    group_market_gradient,
    group_model_gradient,
)
from valax.risk.sensitivities import (
    FlatSensitivities,
    WaterfallPnL,
    compute_flat_sensitivities,
    waterfall_pnl,
    waterfall_pnl_report,
)
from valax.risk.state_sensitivities import (
    BASIS_POINT,
    KeyedSensitivities,
    KeyedWaterfall,
    keyed_sensitivities,
    keyed_waterfall_pnl,
    keyed_waterfall_report,
    keyed_parametric_var,
    keyed_component_var,
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
    BucketedSensitivities,
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
    bucket_sensitivities,
)
from valax.risk.factors import (
    RatesFactorModel,
    fit_rates_pca,
    zero_rate_returns_from_snapshots,
)
from valax.risk._deprecation import RENAMED_SYMBOLS as _RENAMED_SYMBOLS
from valax.risk._deprecation import deprecated_module_getattr as _deprecated_getattr
from valax.risk.oas import (
    callable_bond_oas,
    bond_z_spread,
    price_under_spread,
    effective_duration,
    effective_convexity,
    z_spread_duration,
    z_spread_dv01,
    z_spread_convexity,
    continuous_to_compounded_spread,
    compounded_to_continuous_spread,
)

# Deprecated ``*Ladder`` aliases, served with a ``DeprecationWarning``.
__getattr__ = _deprecated_getattr(__name__, globals(), _RENAMED_SYMBOLS)
