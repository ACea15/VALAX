"""Synthetic market data generators.

Public surface (re-exported from :mod:`valax.market`):

Configuration & seeds
    :class:`SyntheticMarketConfig`, :func:`default_config`,
    :class:`SeedRegistry`.

Stage 1 — Ground-truth world
    :func:`flat_discount_curve`, :func:`sample_flat_curve`,
    :func:`sample_nss_curve`, :func:`sample_discount_curve`,
    :func:`sample_correlation`, :func:`block_correlation`,
    :func:`sample_correlation_from_config`,
    :func:`sample_scalar_market`,
    :func:`sample_market_data`, :func:`sample_market_with_correlation`,
    :func:`sample_bs_params`, :func:`sample_heston_params`,
    :func:`sample_sabr_params`, :func:`sample_hull_white_params`,
    :func:`sample_multi_asset_gbm_params`.

Stage 2 — Observation layer
    :func:`synthesize_sabr_smile`, :func:`synthesize_price_strip`,
    :func:`synthesize_curve_quotes`.

Stage 4 — Portfolio
    :class:`OptionPortfolioSpec`, :class:`SwapPortfolioSpec`,
    :func:`sample_option_portfolio`, :func:`sample_swap_portfolio`.

Time evolution & risk
    :func:`evolve_market`, :func:`sample_scenario_set`.

Arbitrage stress tests
    :class:`ArbDiagnosis`, :func:`inject_non_psd_correlation`,
    :func:`inject_butterfly_arb`, :func:`inject_non_convex_smile`,
    :func:`inject_calendar_arb`, :func:`inject_pcp_violation`,
    :func:`inject_negative_density`,
    :func:`inject_inconsistent_bootstrap_quotes`,
    :func:`inject_basket_variance_violation`.
"""

from valax.market.synthetic.arbitrage import (
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
from valax.market.synthetic.config import (
    SyntheticMarketConfig,
    default_config,
)
from valax.market.synthetic.correlations import (
    block_correlation,
    sample_correlation,
    sample_correlation_from_config,
)
from valax.market.synthetic.curves import (
    flat_discount_curve,
    sample_discount_curve,
    sample_flat_curve,
    sample_nss_curve,
)
from valax.market.synthetic.model_params import (
    sample_bs_params,
    sample_heston_params,
    sample_hull_white_params,
    sample_multi_asset_gbm_params,
    sample_sabr_params,
)
from valax.market.synthetic.observations import (
    synthesize_curve_quotes,
    synthesize_price_strip,
    synthesize_sabr_smile,
)
from valax.market.synthetic.paths import evolve_market
from valax.market.synthetic.portfolio import (
    OptionPortfolioSpec,
    SwapPortfolioSpec,
    sample_option_portfolio,
    sample_swap_portfolio,
)
from valax.market.synthetic.scalars import sample_scalar_market
from valax.market.synthetic.scenarios import sample_scenario_set
from valax.market.synthetic.seeds import SeedRegistry
from valax.market.synthetic.snapshots import (
    sample_market_data,
    sample_market_with_correlation,
)


__all__ = [
    # Config / seeds
    "SyntheticMarketConfig",
    "default_config",
    "SeedRegistry",
    # Curves
    "flat_discount_curve",
    "sample_flat_curve",
    "sample_nss_curve",
    "sample_discount_curve",
    # Correlations
    "sample_correlation",
    "block_correlation",
    "sample_correlation_from_config",
    # Snapshots
    "sample_scalar_market",
    "sample_market_data",
    "sample_market_with_correlation",
    # Model params
    "sample_bs_params",
    "sample_heston_params",
    "sample_sabr_params",
    "sample_hull_white_params",
    "sample_multi_asset_gbm_params",
    # Observations
    "synthesize_sabr_smile",
    "synthesize_price_strip",
    "synthesize_curve_quotes",
    # Portfolio
    "OptionPortfolioSpec",
    "SwapPortfolioSpec",
    "sample_option_portfolio",
    "sample_swap_portfolio",
    # Paths + scenarios
    "evolve_market",
    "sample_scenario_set",
    # Arbitrage
    "ArbDiagnosis",
    "inject_non_psd_correlation",
    "inject_butterfly_arb",
    "inject_non_convex_smile",
    "inject_calendar_arb",
    "inject_pcp_violation",
    "inject_negative_density",
    "inject_inconsistent_bootstrap_quotes",
    "inject_basket_variance_violation",
]
