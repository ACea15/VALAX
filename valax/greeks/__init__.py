"""Greek computation via automatic differentiation.

Two layers coexist:

* :func:`greeks` / :func:`greek` — the classic scalar-argument API over an
  analytic pricing function ``fn(instrument, spot, vol, rate, dividend)``.
* :mod:`valax.greeks.market_greeks` — the unified layer over the ontology's
  ``price(instrument, model, market)`` (P4), giving market / model / contract
  greeks for every method and asset class from one entry point.
"""

from valax.greeks.autodiff import greeks, greek
from valax.greeks.market_greeks import (
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
]
