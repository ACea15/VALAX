"""Market data containers, scenarios, and synthetic-market generators."""

from valax.market.data import MarketData
from valax.market.scenario import (
    MarketScenario,
    ScenarioSet,
    stack_scenarios,
    zero_scenario,
)

# Synthetic-market generators (Stage 1–6 of the end-to-end workflow).
# Re-exported here so user code can simply ``from valax.market import
# sample_market_data`` without reaching into the submodule path.
from valax.market.synthetic import (
    # Config / seeds
    SeedRegistry,
    SyntheticMarketConfig,
    default_config,
    # Curves
    flat_discount_curve,
    sample_discount_curve,
    sample_flat_curve,
    sample_nss_curve,
    # Correlations
    block_correlation,
    sample_correlation,
    sample_correlation_from_config,
    # Snapshots
    sample_market_data,
    sample_market_with_correlation,
    sample_scalar_market,
    # Model params
    sample_bs_params,
    sample_heston_params,
    sample_hull_white_params,
    sample_multi_asset_gbm_params,
    sample_sabr_params,
    # Observations
    synthesize_curve_quotes,
    synthesize_price_strip,
    synthesize_sabr_smile,
    # Portfolio
    OptionPortfolioSpec,
    SwapPortfolioSpec,
    sample_option_portfolio,
    sample_swap_portfolio,
    # Paths + scenarios
    evolve_market,
    sample_scenario_set,
    # Arbitrage
    ArbDiagnosis,
    inject_basket_variance_violation,
    inject_butterfly_arb,
    inject_calendar_arb,
    inject_inconsistent_bootstrap_quotes,
    inject_negative_density,
    inject_non_convex_smile,
    inject_non_psd_correlation,
    inject_pcp_violation,
)
