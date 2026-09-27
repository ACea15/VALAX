"""Core types, the calibrated market container, and the unified pricer.

The pricing ontology (``PRICING_ONTOLOGY.md``) names
four nouns and six verbs.  This package owns the two that every other
package depends on: the **`MarketState`** noun and the **`price`** verb,
plus the `resolve` binding that connects an instrument's static market
selectors to the market it is priced against.

Nothing here re-implements a pricer: :func:`price` dispatches to the
existing analytic / lattice / PDE / MC implementations.
"""

from valax.core.types import Scalar, Vec, Mat
from valax.core.market_state import (
    DEFAULT_ID,
    MarketState,
    validate_surface_id,
)
from valax.core.resolve import (
    ResolvedMarket,
    resolve,
    resolve_curves,
    resolve_spot,
    resolve_surface,
)
from valax.core.pricing import (
    METHODS,
    PricingResult,
    auto_method,
    available_methods,
    price,
    register_pricer,
    registered_pricers,
    set_auto_method,
)

__all__ = [
    "Scalar",
    "Vec",
    "Mat",
    "DEFAULT_ID",
    "MarketState",
    "validate_surface_id",
    "ResolvedMarket",
    "resolve",
    "resolve_curves",
    "resolve_spot",
    "resolve_surface",
    "METHODS",
    "PricingResult",
    "auto_method",
    "available_methods",
    "price",
    "register_pricer",
    "registered_pricers",
    "set_auto_method",
]
