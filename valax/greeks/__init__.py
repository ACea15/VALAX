"""Greek computation via automatic differentiation.

Two layers coexist:

* :func:`greeks` / :func:`greek` — the classic scalar-argument API over an
  analytic pricing function ``fn(instrument, spot, vol, rate, dividend)``.
* :mod:`valax.greeks.market_greeks` — the unified layer over the ontology's
  ``price(instrument, model, market)`` (P4), giving market / model / contract
  greeks for every method and asset class from one entry point.
* :mod:`valax.greeks.adjoint` / :mod:`valax.greeks.vol_adjoint` — quote-space
  (hedge-space) risk: the same gradient pushed one calibration further back, to
  the curve quotes and the volatility quotes a desk actually trades.
* :mod:`valax.greeks.quote_risk` — quote risk through a whole calibration
  chain (curves → vols → model), by an explicit adjoint sweep and,
  independently, by autodiff through the solves.
"""

from valax.greeks.autodiff import greeks, greek
from valax.greeks.adjoint import (
    ILL_CONDITIONED_THRESHOLD,
    CalibrationLevel,
    CurveAdjoint,
    LevelAdjoint,
    condition_alert,
    curve_adjoint,
    curve_log_df_cotangent,
    curve_quote_deltas,
    level_adjoint,
    level_condition_number,
    quote_deltas_from_cotangent,
)
from valax.greeks.fixed_inputs import (
    FixedInput,
    FixedInputAdjoint,
    FixedInputs,
    fixed_input_adjoints,
    fixed_inputs_from_paths,
)
from valax.greeks.vol_adjoint import (
    SABRSliceInputs,
    SliceVolAdjoint,
    cube_node_cotangent,
    cube_node_level,
    optionlet_slice_level,
    sabr_model_from_theta,
    sabr_slice_level,
    sabr_theta,
    sabr_theta_names,
    vol_quote_vegas,
)
from valax.greeks.quote_risk import (
    QuoteRisk,
    QuoteRiskReconciliation,
    chain_adjoint,
    quote_risk_adjoint,
    quote_risk_autodiff,
    reconcile_quote_risk,
)
from valax.greeks.market_greeks import (
    MarketRisk,
    market_risk,
    bucketed_curve_deltas,
    curve_pillar_deltas,
    dual_delta,
    market_greeks,
    market_gradient,
    model_gradient,
    spot_delta,
    theta,
)

__all__ = [
    "greeks",
    "greek",
    "market_gradient",
    "model_gradient",
    "spot_delta",
    "curve_pillar_deltas",
    "bucketed_curve_deltas",
    "dual_delta",
    "theta",
    "market_greeks",
    "MarketRisk",
    "market_risk",
    # Quote-space (hedge-space) risk via the implicit function theorem
    "CalibrationLevel",
    "LevelAdjoint",
    "level_adjoint",
    "ILL_CONDITIONED_THRESHOLD",
    "level_condition_number",
    "condition_alert",
    "FixedInput",
    "FixedInputs",
    "FixedInputAdjoint",
    "fixed_input_adjoints",
    "fixed_inputs_from_paths",
    "CurveAdjoint",
    "curve_adjoint",
    "curve_log_df_cotangent",
    "quote_deltas_from_cotangent",
    "curve_quote_deltas",
    # Vol-quote (vega-space) risk — the same step one layer up
    "SABRSliceInputs",
    "SliceVolAdjoint",
    "sabr_theta",
    "sabr_theta_names",
    "sabr_model_from_theta",
    "sabr_slice_level",
    "vol_quote_vegas",
    "cube_node_level",
    "cube_node_cotangent",
    "optionlet_slice_level",
    # Quote risk through a whole calibration chain — two independent paths
    "QuoteRisk",
    "QuoteRiskReconciliation",
    "chain_adjoint",
    "quote_risk_adjoint",
    "quote_risk_autodiff",
    "reconcile_quote_risk",
]
