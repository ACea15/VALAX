"""Monte Carlo pricing engine.

Two layers of API are provided:

1. **Unified dispatcher** — :func:`mc_price_dispatch` looks up an
   ``(instrument, model)`` recipe and runs the full pipeline (paths +
   payoff + discounting) in one call. This is the preferred API for
   end users.

2. **Low-level building blocks** — path generators
   (:func:`generate_gbm_paths`, :func:`generate_heston_paths`, ...)
   and payoff functions (:func:`european_payoff`, :func:`cap_mc_payoff`,
   ...). Call these directly when you need fine control or are
   writing a new recipe.

The older :func:`mc_price` / :func:`mc_price_with_stderr` in
:mod:`valax.pricing.mc.engine` remain for backward compatibility
but the dispatcher is the preferred entry point going forward.
"""

# Unified dispatcher (preferred API)
from valax.pricing.mc.dispatch import (
    MCConfig,
    MCResult,
    mc_price_dispatch,
    register,
    registered_recipes,
)

# Importing recipes populates the dispatcher registry.
# This must happen before any caller uses mc_price_dispatch.
from valax.pricing.mc import recipes  # noqa: F401  (import-for-side-effects)

# Legacy entry points (unchanged)
from valax.pricing.mc.engine import mc_price, mc_price_with_stderr

# Path generators
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.sabr_paths import generate_sabr_paths
from valax.pricing.mc.lmm_paths import LMMPathResult, generate_lmm_paths

# Payoff functions
from valax.pricing.mc.payoffs import (
    asian_option_payoff,
    asian_payoff,
    barrier_payoff,
    equity_barrier_payoff,
    european_payoff,
    lookback_payoff,
    variance_swap_payoff,
)
from valax.pricing.mc.rate_payoffs import (
    cap_mc_payoff,
    caplet_mc_payoff,
    swaption_mc_payoff,
)

# Bermudan (Longstaff-Schwartz)
from valax.pricing.mc.bermudan import LSMConfig, bermudan_swaption_lsm


__all__ = [
    # Unified dispatcher
    "MCConfig",
    "MCResult",
    "mc_price_dispatch",
    "register",
    "registered_recipes",
    # Legacy
    "mc_price",
    "mc_price_with_stderr",
    # Path generators
    "generate_gbm_paths",
    "generate_heston_paths",
    "generate_sabr_paths",
    "generate_lmm_paths",
    "LMMPathResult",
    # Payoffs
    "european_payoff",
    "asian_payoff",
    "asian_option_payoff",
    "barrier_payoff",
    "equity_barrier_payoff",
    "lookback_payoff",
    "variance_swap_payoff",
    "caplet_mc_payoff",
    "cap_mc_payoff",
    "swaption_mc_payoff",
    # Bermudan
    "LSMConfig",
    "bermudan_swaption_lsm",
]
