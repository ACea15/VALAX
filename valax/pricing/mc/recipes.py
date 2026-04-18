"""Built-in Monte Carlo recipes.

Each recipe is a thin wrapper that:

1. Calls the appropriate path generator for the given model.
2. Calls the appropriate payoff function for the given instrument.
3. Applies discounting and returns an :class:`MCResult`.

Recipes are registered with :func:`valax.pricing.mc.dispatch.register` at
import time. Importing this module (which ``valax.pricing.mc.__init__``
does automatically) populates the dispatcher registry.

Coverage
--------

Equity (single asset):

    +-------------------------+----------------------+----------------------+
    | Instrument              | BlackScholesModel    | HestonModel          |
    +=========================+======================+======================+
    | EuropeanOption          | ✓                    | ✓                    |
    | AsianOption             | ✓                    | ✓                    |
    | EquityBarrierOption     | ✓                    | ✓                    |
    | LookbackOption          | ✓                    | ✓                    |
    | VarianceSwap            | ✓                    | ✓                    |
    +-------------------------+----------------------+----------------------+

Rates (LMM):

    +-------------------------+----------------------+
    | Instrument              | LMMModel             |
    +=========================+======================+
    | Caplet                  | ✓                    |
    | Cap                     | ✓                    |
    | Swaption (European)     | ✓                    |
    | BermudanSwaption        | ✓                    |
    +-------------------------+----------------------+

Recipes not yet registered
--------------------------

- Correlated multi-asset GBM (for SpreadOption, WorstOfBasket, QuantoOption).
- Hull-White short-rate MC (for bond / callable / puttable MC pricing).
- Jarrow-Yildirim inflation MC (for YYIS / inflation caps with convexity).
- Autocallable / path-dependent structured-product engine.
- American / Bermudan equity via LSM on GBM/Heston paths (LSM engine already
  exists for LMM — just needs lifting).

See :doc:`/guide/monte-carlo` and the roadmap for tracking.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.instruments.options import (
    AsianOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    VarianceSwap,
)
from valax.instruments.rates import BermudanSwaption, Cap, Caplet, Swaption
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.lmm import LMMModel
from valax.pricing.mc.bermudan import LSMConfig, bermudan_swaption_lsm
from valax.pricing.mc.dispatch import (
    MCConfig,
    MCResult,
    discounted_mean_and_stderr,
    register,
)
from valax.pricing.mc.lmm_paths import generate_lmm_paths
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.payoffs import (
    asian_option_payoff,
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


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _equity_paths(
    model,
    spot: Float[Array, ""],
    T: Float[Array, ""],
    config: MCConfig,
    key: jax.Array,
) -> tuple[Float[Array, "n_paths n_steps_plus1"], Float[Array, ""]]:
    """Generate paths for a single-asset equity model and return (paths, rate).

    Branches on model type to pick the right generator. The returned
    ``rate`` is the risk-free rate used for discounting.
    """
    if isinstance(model, HestonModel):
        paths, _ = generate_heston_paths(
            model, spot, T, config.n_steps, config.n_paths, key,
        )
    else:
        # BlackScholesModel (or any GBM-like model)
        paths = generate_gbm_paths(
            model, spot, T, config.n_steps, config.n_paths, key,
        )
    return paths, model.rate


def _equity_recipe(
    payoff_fn,
    instrument,
    model,
    config: MCConfig,
    key: jax.Array,
    spot: Float[Array, ""],
) -> MCResult:
    """Generic equity MC recipe: paths → payoff → discount.

    The payoff signature is ``payoff_fn(paths, instrument) -> cashflows``.
    """
    T = instrument.expiry
    paths, rate = _equity_paths(model, spot, T, config, key)
    cashflows = payoff_fn(paths, instrument)
    df = jnp.exp(-rate * T)
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


# ─────────────────────────────────────────────────────────────────────
# Equity recipes (BlackScholesModel and HestonModel share the payoffs)
# ─────────────────────────────────────────────────────────────────────


@register(EuropeanOption, BlackScholesModel)
def _european_bsm(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """European option under GBM.

    Required market args:
        spot: Current spot price.
    """
    return _equity_recipe(european_payoff, instrument, model, config, key, spot)


@register(EuropeanOption, HestonModel)
def _european_heston(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """European option under Heston stochastic volatility.

    Required market args:
        spot: Current spot price.
    """
    return _equity_recipe(european_payoff, instrument, model, config, key, spot)


@register(AsianOption, BlackScholesModel)
def _asian_bsm(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Arithmetic/geometric Asian option under GBM."""
    return _equity_recipe(asian_option_payoff, instrument, model, config, key, spot)


@register(AsianOption, HestonModel)
def _asian_heston(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Arithmetic/geometric Asian option under Heston."""
    return _equity_recipe(asian_option_payoff, instrument, model, config, key, spot)


@register(EquityBarrierOption, BlackScholesModel)
def _barrier_bsm(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Knock-in/out equity barrier option under GBM.

    The instrument carries ``smoothing`` on itself; use a positive value
    for pathwise-differentiable Greeks.
    """
    return _equity_recipe(equity_barrier_payoff, instrument, model, config, key, spot)


@register(EquityBarrierOption, HestonModel)
def _barrier_heston(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Knock-in/out equity barrier option under Heston."""
    return _equity_recipe(equity_barrier_payoff, instrument, model, config, key, spot)


@register(LookbackOption, BlackScholesModel)
def _lookback_bsm(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Floating- or fixed-strike lookback under GBM."""
    return _equity_recipe(lookback_payoff, instrument, model, config, key, spot)


@register(LookbackOption, HestonModel)
def _lookback_heston(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Floating- or fixed-strike lookback under Heston."""
    return _equity_recipe(lookback_payoff, instrument, model, config, key, spot)


@register(VarianceSwap, BlackScholesModel)
def _varswap_bsm(
    *,
    instrument,
    model,
    config,
    key,
    spot: Float[Array, ""],
    annual_factor: Float[Array, ""] = jnp.array(252.0),
    **kwargs,
) -> MCResult:
    """Variance swap under GBM.

    Realized variance is computed from path log-returns. By default the
    observation frequency is taken to be ``annual_factor = 252`` (daily).
    """
    T = instrument.expiry
    paths, rate = _equity_paths(model, spot, T, config, key)
    cashflows = variance_swap_payoff(paths, instrument, annual_factor)
    df = jnp.exp(-rate * T)
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(VarianceSwap, HestonModel)
def _varswap_heston(
    *,
    instrument,
    model,
    config,
    key,
    spot: Float[Array, ""],
    annual_factor: Float[Array, ""] = jnp.array(252.0),
    **kwargs,
) -> MCResult:
    """Variance swap under Heston."""
    T = instrument.expiry
    paths, rate = _equity_paths(model, spot, T, config, key)
    cashflows = variance_swap_payoff(paths, instrument, annual_factor)
    df = jnp.exp(-rate * T)
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


# ─────────────────────────────────────────────────────────────────────
# Rates recipes (LMM)
#
# LMM-based recipes need the instrument's payoff period mapped to the
# LMM tenor structure. The caller passes ``forward_index`` (caplet) or
# ``forward_indices`` + ``taus`` (cap, swaption, Bermudan) via
# ``market_args``. Automatic date-to-index resolution is a planned
# follow-up.
# ─────────────────────────────────────────────────────────────────────


@register(Caplet, LMMModel)
def _caplet_lmm(
    *,
    instrument,
    model,
    config,
    key,
    forward_index: int,
    tau: Float[Array, ""],
    n_steps_per_period: int = 20,
    **kwargs,
) -> MCResult:
    """Caplet / floorlet under the LIBOR Market Model.

    Required market args:
        forward_index: Index of the forward rate in the LMM tenor
            structure that corresponds to the caplet's accrual period.
        tau: Accrual fraction for the caplet period.

    Optional:
        n_steps_per_period: Number of Euler steps between consecutive
            tenor dates.  Default 20.
    """
    result = generate_lmm_paths(
        model,
        n_steps_per_period=n_steps_per_period,
        n_paths=config.n_paths,
        key=key,
    )
    cashflows = caplet_mc_payoff(result, instrument, forward_index, tau)
    # rate_payoffs return cashflows already discounted to 0 via path DFs.
    price = jnp.mean(cashflows)
    stderr = jnp.std(cashflows) / jnp.sqrt(
        jnp.array(config.n_paths, dtype=cashflows.dtype),
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(Cap, LMMModel)
def _cap_lmm(
    *,
    instrument,
    model,
    config,
    key,
    forward_indices: Int[Array, " n_caplets"],
    taus: Float[Array, " n_caplets"],
    n_steps_per_period: int = 20,
    **kwargs,
) -> MCResult:
    """Cap / floor strip under the LIBOR Market Model."""
    result = generate_lmm_paths(
        model,
        n_steps_per_period=n_steps_per_period,
        n_paths=config.n_paths,
        key=key,
    )
    cashflows = cap_mc_payoff(result, instrument, forward_indices, taus)
    price = jnp.mean(cashflows)
    stderr = jnp.std(cashflows) / jnp.sqrt(
        jnp.array(config.n_paths, dtype=cashflows.dtype),
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(Swaption, LMMModel)
def _swaption_lmm(
    *,
    instrument,
    model,
    config,
    key,
    forward_indices: Int[Array, " n_periods"],
    taus: Float[Array, " n_periods"],
    n_steps_per_period: int = 20,
    **kwargs,
) -> MCResult:
    """European swaption under LMM.

    Required market args:
        forward_indices: Indices of the forwards spanning the underlying
            swap in the LMM tenor structure.
        taus: Accrual fractions for each swap period.
    """
    result = generate_lmm_paths(
        model,
        n_steps_per_period=n_steps_per_period,
        n_paths=config.n_paths,
        key=key,
    )
    cashflows = swaption_mc_payoff(result, instrument, forward_indices, taus)
    price = jnp.mean(cashflows)
    stderr = jnp.std(cashflows) / jnp.sqrt(
        jnp.array(config.n_paths, dtype=cashflows.dtype),
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(BermudanSwaption, LMMModel)
def _bermudan_lmm(
    *,
    instrument,
    model,
    config,
    key,
    exercise_indices: Int[Array, " n_exercise"],
    taus: Float[Array, " N"],
    lsm_config: LSMConfig | None = None,
    n_steps_per_period: int = 20,
    **kwargs,
) -> MCResult:
    """Bermudan swaption via Longstaff-Schwartz on LMM paths.

    Required market args:
        exercise_indices: Tenor indices of the Bermudan exercise dates.
        taus: Accrual fractions for each forward period.

    Optional:
        lsm_config: :class:`LSMConfig` for the regression (default
            cubic polynomial basis).

    Notes:
        Standard-error estimation is set to ``0.0`` — the LSM
        continuation-value regression makes a path-wise variance
        estimate unreliable. Run multiple independent simulations and
        compute across-seed dispersion for a practical uncertainty
        bound.
    """
    cfg = lsm_config if lsm_config is not None else LSMConfig()
    result = generate_lmm_paths(
        model,
        n_steps_per_period=n_steps_per_period,
        n_paths=config.n_paths,
        key=key,
    )
    price = bermudan_swaption_lsm(result, instrument, exercise_indices, taus, cfg)
    return MCResult(
        price=price,
        stderr=jnp.array(0.0, dtype=price.dtype),
        n_paths=config.n_paths,
    )
