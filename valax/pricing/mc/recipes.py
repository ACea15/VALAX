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

from valax.dates.daycounts import year_fraction
from valax.instruments.bonds import CallableBond, FixedRateBond, FloatingRateBond, PuttableBond
from valax.instruments.options import (
    AsianOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    SpreadOption,
    VarianceSwap,
    WorstOfBasketOption,
)
from valax.instruments.rates import (
    BermudanSwaption,
    Cap,
    Caplet,
    CMSSpreadSwap,
    Swaption,
)
from valax.models.black_scholes import BlackScholesModel
from valax.models.heston import HestonModel
from valax.models.hull_white import HullWhiteModel, hw_bond_price
from valax.models.g2pp import G2PPModel, g2pp_bond_price
from valax.models.lmm import LMMModel
from valax.models.local_vol import LocalVolModel
from valax.models.multi_asset import MultiAssetGBMModel
from valax.models.slv import SLVModel
from valax.pricing.mc.hull_white_paths import HullWhitePathResult, generate_hull_white_paths
from valax.pricing.mc.g2pp_paths import G2PPPathResult, generate_g2pp_paths
from valax.pricing.mc.bermudan import LSMConfig, bermudan_swaption_lsm
from valax.pricing.mc.dispatch import (
    MCConfig,
    MCResult,
    discounted_mean_and_stderr,
    register,
)
from valax.pricing.mc.lmm_paths import generate_lmm_paths
from valax.pricing.mc.local_vol_paths import generate_local_vol_paths
from valax.pricing.mc.multi_asset_paths import generate_correlated_gbm_paths
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.slv_paths import generate_slv_paths
from valax.pricing.mc.payoffs import (
    asian_option_payoff,
    equity_barrier_payoff,
    european_payoff,
    jax_sigmoid,
    lookback_payoff,
    spread_option_mc_payoff,
    variance_swap_payoff,
    worst_of_basket_payoff,
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
    *,
    lv_scheme: str = "midpoint_euler",
    slv_scheme: str = "midpoint_euler",
) -> tuple[Float[Array, "n_paths n_steps_plus1"], Float[Array, ""]]:
    """Generate paths for a single-asset equity model and return (paths, rate).

    Branches on model type to pick the right generator. The returned
    ``rate`` is the risk-free rate used for discounting.

    Args:
        lv_scheme: Forwarded to ``generate_local_vol_paths`` when ``model``
            is a ``LocalVolModel``. Ignored otherwise. See
            :func:`valax.pricing.mc.generate_local_vol_paths` for accepted
            values. Default ``"midpoint_euler"``.
        slv_scheme: Forwarded to ``generate_slv_paths`` when ``model``
            is an ``SLVModel``. Ignored otherwise. See
            :func:`valax.pricing.mc.generate_slv_paths` for accepted
            values. Default ``"midpoint_euler"``.
    """
    if isinstance(model, HestonModel):
        paths, _ = generate_heston_paths(
            model, spot, T, config.n_steps, config.n_paths, key,
        )
    elif isinstance(model, SLVModel):
        paths, _ = generate_slv_paths(
            model, spot, T, config.n_steps, config.n_paths, key,
            scheme=slv_scheme,
        )
    elif isinstance(model, LocalVolModel):
        paths = generate_local_vol_paths(
            model, spot, T, config.n_steps, config.n_paths, key,
            scheme=lv_scheme,
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
    *,
    lv_scheme: str = "midpoint_euler",
    slv_scheme: str = "midpoint_euler",
) -> MCResult:
    """Generic equity MC recipe: paths → payoff → discount.

    The payoff signature is ``payoff_fn(paths, instrument) -> cashflows``.

    Args:
        lv_scheme: Forwarded to ``_equity_paths`` (only consumed when
            ``model`` is a ``LocalVolModel``).
        slv_scheme: Forwarded to ``_equity_paths`` (only consumed when
            ``model`` is an ``SLVModel``).
    """
    T = instrument.expiry
    paths, rate = _equity_paths(
        model, spot, T, config, key,
        lv_scheme=lv_scheme,
        slv_scheme=slv_scheme,
    )
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


@register(EuropeanOption, LocalVolModel)
def _european_localvol(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """European option under Dupire local volatility.

    Required market args:
        spot: Current spot price.

    Optional market args:
        lv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"`` — see
            :func:`valax.pricing.mc.generate_local_vol_paths`. Milstein
            is opt-in for path-dependent payoffs where strong-order
            convergence matters.
    """
    return _equity_recipe(
        european_payoff, instrument, model, config, key, spot,
        lv_scheme=kwargs.get("lv_scheme", "midpoint_euler"),
    )


@register(EuropeanOption, SLVModel)
def _european_slv(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """European option under Stochastic-Local Volatility.

    Required market args:
        spot: Current spot price.

    Optional market args:
        slv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"`` —
            see :func:`valax.pricing.mc.generate_slv_paths`.
    """
    return _equity_recipe(
        european_payoff, instrument, model, config, key, spot,
        slv_scheme=kwargs.get("slv_scheme", "midpoint_euler"),
    )


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


@register(AsianOption, LocalVolModel)
def _asian_localvol(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Arithmetic/geometric Asian option under Dupire local volatility.

    Optional market args:
        lv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    return _equity_recipe(
        asian_option_payoff, instrument, model, config, key, spot,
        lv_scheme=kwargs.get("lv_scheme", "midpoint_euler"),
    )


@register(AsianOption, SLVModel)
def _asian_slv(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Arithmetic/geometric Asian option under Stochastic-Local Volatility.

    Optional market args:
        slv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    return _equity_recipe(
        asian_option_payoff, instrument, model, config, key, spot,
        slv_scheme=kwargs.get("slv_scheme", "midpoint_euler"),
    )


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


@register(EquityBarrierOption, LocalVolModel)
def _barrier_localvol(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Knock-in/out equity barrier option under Dupire local volatility.

    Barrier options are the canonical exotic where local vol differs
    materially from constant-vol BSM — the vol smile at the barrier
    matters.

    Optional market args:
        lv_scheme: ``"milstein"`` (default) or ``"midpoint_euler"``.
    """
    return _equity_recipe(
        equity_barrier_payoff, instrument, model, config, key, spot,
        lv_scheme=kwargs.get("lv_scheme", "midpoint_euler"),
    )


@register(EquityBarrierOption, SLVModel)
def _barrier_slv(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Knock-in/out equity barrier option under Stochastic-Local Volatility.

    SLV is the workhorse model for barriers when both the smile *and*
    the forward-skew dynamics matter (e.g., reverse knock-outs near
    the barrier). For pure-vanilla repricing LV and SLV agree by
    construction; the SLV/LV difference appears in path-dependent
    payoffs.

    Optional market args:
        slv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    return _equity_recipe(
        equity_barrier_payoff, instrument, model, config, key, spot,
        slv_scheme=kwargs.get("slv_scheme", "midpoint_euler"),
    )


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


@register(LookbackOption, LocalVolModel)
def _lookback_localvol(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Floating- or fixed-strike lookback under Dupire local volatility.

    Optional market args:
        lv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    return _equity_recipe(
        lookback_payoff, instrument, model, config, key, spot,
        lv_scheme=kwargs.get("lv_scheme", "midpoint_euler"),
    )


@register(LookbackOption, SLVModel)
def _lookback_slv(
    *, instrument, model, config, key, spot: Float[Array, ""], **kwargs,
) -> MCResult:
    """Floating- or fixed-strike lookback under Stochastic-Local Volatility.

    Optional market args:
        slv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    return _equity_recipe(
        lookback_payoff, instrument, model, config, key, spot,
        slv_scheme=kwargs.get("slv_scheme", "midpoint_euler"),
    )


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


@register(VarianceSwap, LocalVolModel)
def _varswap_localvol(
    *,
    instrument,
    model,
    config,
    key,
    spot: Float[Array, ""],
    annual_factor: Float[Array, ""] = jnp.array(252.0),
    **kwargs,
) -> MCResult:
    """Variance swap under Dupire local volatility.

    Optional market args:
        lv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    T = instrument.expiry
    paths, rate = _equity_paths(
        model, spot, T, config, key,
        lv_scheme=kwargs.get("lv_scheme", "midpoint_euler"),
    )
    cashflows = variance_swap_payoff(paths, instrument, annual_factor)
    df = jnp.exp(-rate * T)
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(VarianceSwap, SLVModel)
def _varswap_slv(
    *,
    instrument,
    model,
    config,
    key,
    spot: Float[Array, ""],
    annual_factor: Float[Array, ""] = jnp.array(252.0),
    **kwargs,
) -> MCResult:
    """Variance swap under Stochastic-Local Volatility.

    Realised variance is computed from path log-returns. By default the
    observation frequency is taken to be ``annual_factor = 252`` (daily).

    Optional market args:
        slv_scheme: ``"midpoint_euler"`` (default) or ``"milstein"``.
    """
    T = instrument.expiry
    paths, rate = _equity_paths(
        model, spot, T, config, key,
        slv_scheme=kwargs.get("slv_scheme", "midpoint_euler"),
    )
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


# ─────────────────────────────────────────────────────────────────────
# Multi-asset equity recipes (MultiAssetGBMModel)
#
# These unlock spread options and worst-of-basket options. The caller
# passes `spots` as a length-n_assets array; the recipe uses the model's
# per-asset vols, dividends, and correlation to generate correlated
# paths.
# ─────────────────────────────────────────────────────────────────────


@register(SpreadOption, MultiAssetGBMModel)
def _spread_option_multi_asset(
    *,
    instrument,
    model,
    config,
    key,
    spots: Float[Array, " n_assets"],
    asset1_index: int = 0,
    asset2_index: int = 1,
    **kwargs,
) -> MCResult:
    """Spread option under correlated multi-asset GBM.

    Required market args:
        spots: Initial spot prices, shape ``(n_assets,)``. ``n_assets``
            must be at least 2.

    Optional:
        asset1_index: Column of ``spots`` / ``paths`` for :math:`S_1`
            (default 0).
        asset2_index: Column for :math:`S_2` (default 1).

    The recipe validates the model's correlation matrix shape against
    ``len(spots)``; mismatches raise a clear ValueError at trace time.
    """
    if spots.shape[0] < 2:
        raise ValueError(
            f"SpreadOption recipe needs at least 2 assets; got "
            f"spots.shape={spots.shape}.",
        )
    if model.correlation.shape != (spots.shape[0], spots.shape[0]):
        raise ValueError(
            f"model.correlation has shape {model.correlation.shape} but "
            f"spots has {spots.shape[0]} assets. These must match.",
        )
    T = instrument.expiry
    paths = generate_correlated_gbm_paths(
        model, spots, T, config.n_steps, config.n_paths, key,
    )
    cashflows = spread_option_mc_payoff(
        paths, instrument, asset1_index=asset1_index, asset2_index=asset2_index,
    )
    df = jnp.exp(-model.rate * T)
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(WorstOfBasketOption, MultiAssetGBMModel)
def _worst_of_basket_multi_asset(
    *,
    instrument,
    model,
    config,
    key,
    spots: Float[Array, " n_assets"],
    **kwargs,
) -> MCResult:
    """Worst-of basket option under correlated multi-asset GBM.

    Required market args:
        spots: Initial spot prices, shape ``(n_assets,)``. Must match
            ``instrument.n_assets`` and the size of
            ``model.correlation``.
    """
    n = spots.shape[0]
    if n != instrument.n_assets:
        raise ValueError(
            f"spots has {n} assets but instrument.n_assets="
            f"{instrument.n_assets}; these must match.",
        )
    if model.correlation.shape != (n, n):
        raise ValueError(
            f"model.correlation has shape {model.correlation.shape} but "
            f"spots has {n} assets.",
        )
    T = instrument.expiry
    paths = generate_correlated_gbm_paths(
        model, spots, T, config.n_steps, config.n_paths, key,
    )
    cashflows = worst_of_basket_payoff(paths, instrument, spots)
    df = jnp.exp(-model.rate * T)
    price, stderr = discounted_mean_and_stderr(cashflows, df, config.n_paths)
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


# ─────────────────────────────────────────────────────────────────────
# Rates recipes (Hull-White short-rate MC)
#
# These recipes simulate the short rate with the exact conditional
# distribution (no Euler discretisation bias) and discount each cash
# flow with the money-market numeraire accumulated along the path.
#
# API contract for callers:
#   Required market args:
#     n_steps (int): Number of time steps for path generation.
#     T       (float): Horizon in year fractions (must cover bond maturity).
#   Optional:
#     n_steps defaults to 100 when not supplied.
# ─────────────────────────────────────────────────────────────────────


def _hw_step_index(
    times: Float[Array, " n_times"],
    T: Float[Array, ""],
    n_steps: int,
) -> Int[Array, " n_times"]:
    """Snap year-fraction times to the nearest simulation step index.

    Kept as a traced gather (rather than a Python ``int(round(...))``) so the
    recipes stay composable with ``jax.jit`` / ``jax.grad`` when the instrument
    is passed as a traced pytree.

    Args:
        times: Event times in year fractions.
        T: Simulation horizon used to generate the paths.
        n_steps: Number of simulation steps (static).

    Returns:
        Step indices in ``[0, n_steps]``.
    """
    raw = jnp.round(times * (n_steps / T)).astype(jnp.int32)
    return jnp.clip(raw, 0, n_steps)


def _hw_path_sdf(
    result: HullWhitePathResult,
    times: Float[Array, " n_times"],
    T: Float[Array, ""],
    n_steps: int,
) -> Float[Array, "n_paths n_times"]:
    """Path-wise stochastic discount factors at a set of event times.

    Args:
        result: Output of :func:`generate_hull_white_paths`.
        times: Event times in year fractions.
        T: Simulation horizon used to generate the paths.
        n_steps: Number of simulation steps (static).

    Returns:
        Discount factor :math:`\\hat{D}(0, t_j)` for each path and event time.
    """
    idx = _hw_step_index(times, T, n_steps)
    return jnp.exp(jnp.take(result.log_discount_factors, idx, axis=1))


def _hw_coupon_schedule(
    instrument: FixedRateBond | CallableBond | PuttableBond,
    ref: Int[Array, ""],
) -> tuple[Float[Array, " n_pay"], Float[Array, " n_pay"]]:
    """Cash-flow times and amounts for a fixed-coupon bond.

    Args:
        instrument: Bond carrying ``payment_dates``, ``coupon_rate``,
            ``face_value`` and a static ``frequency``.
        ref: Curve reference date (ordinal).

    Returns:
        ``(times, amounts)`` — year fractions from ``ref`` to each payment
        date, and the corresponding cash flow (principal folded into the
        final coupon).
    """
    times = year_fraction(ref, instrument.payment_dates, instrument.day_count)
    coupon = instrument.face_value * instrument.coupon_rate / instrument.frequency
    n_pay = instrument.payment_dates.shape[0]
    amounts = jnp.full((n_pay,), coupon).at[-1].add(instrument.face_value)
    return times, amounts


@register(FixedRateBond, HullWhiteModel)
def _fixed_bond_hw(
    *,
    instrument: FixedRateBond,
    model: HullWhiteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    **kwargs,
) -> MCResult:
    """Fixed-rate bond price under Hull-White short-rate MC.

    The bond's coupon and principal cash flows are discounted with the
    money-market numeraire accumulated along each simulated path.  No
    embedded optionality: use :func:`_callable_bond_hw` for callable bonds.

    Required market args:
        (none beyond instrument and model)

    Optional market args:
        T: Horizon in year fractions.  Defaults to the year fraction of
            the last payment date.
        n_steps: Number of time steps (default 100).
    """
    ref = model.initial_curve.reference_date
    cf_times, cf_amounts = _hw_coupon_schedule(instrument, ref)

    if T is None:
        T = cf_times[-1]

    result = generate_hull_white_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    sdf = _hw_path_sdf(result, cf_times, T, n_steps)     # (n_paths, n_pay)
    pv = jnp.sum(cf_amounts[None, :] * sdf, axis=1)      # (n_paths,)

    # Discounting is already path-wise inside `pv`, so the deterministic
    # factor is unity; the helper supplies the shared mean/stderr convention.
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(FloatingRateBond, HullWhiteModel)
def _floating_bond_hw(
    *,
    instrument: FloatingRateBond,
    model: HullWhiteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    **kwargs,
) -> MCResult:
    """Floating-rate bond (FRN) price under Hull-White short-rate MC.

    Under the risk-neutral measure a par-at-reset FRN prices to par
    at each reset date.  For seasoned FRNs or non-zero spread, the
    floating leg is replicated as:

    .. math::

        V = \\text{face} \\cdot \\hat{D}(0, T_0) + s \\sum_i \\tau_i \\hat{D}(0, T_i) + \\text{face} \\cdot \\hat{D}(0, T_N)

    where :math:`\\hat{D}` is the path-wise stochastic discount factor.

    Optional market args:
        T: Horizon in year fractions (defaults to maturity).
        n_steps: Number of time steps (default 100).
    """
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count
    face = instrument.face_value

    cf_times = year_fraction(ref, instrument.payment_dates, day_count)
    prev_dates = jnp.concatenate(
        [instrument.settlement_date[None], instrument.payment_dates[:-1]]
    )
    taus = year_fraction(prev_dates, instrument.payment_dates, day_count)
    spread_amounts = instrument.spread * face * taus          # (n_pay,)

    t_settle = year_fraction(ref, instrument.settlement_date, day_count)
    if T is None:
        T = cf_times[-1]

    result = generate_hull_white_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )

    # Floating-leg replication.  The floating coupons between settlement and
    # maturity are worth face * (D(settle) - D(T_N)); adding the principal
    # redemption face * D(T_N) cancels the second term exactly, leaving the
    # familiar "an FRN prices to par at reset" identity:
    #
    #     PV = face * D(0, t_settle)  +  spread coupons
    settle_sdf = _hw_path_sdf(result, t_settle[None], T, n_steps)[:, 0]
    spread_sdf = _hw_path_sdf(result, cf_times, T, n_steps)

    pv = face * settle_sdf + jnp.sum(spread_amounts[None, :] * spread_sdf, axis=1)

    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


def _hw_exercisable_bond_pv(
    model: HullWhiteModel,
    result: HullWhitePathResult,
    cf_times: Float[Array, " n_pay"],
    cf_amounts: Float[Array, " n_pay"],
    ex_times: Float[Array, " n_ex"],
    ex_prices: Float[Array, " n_ex"],
    T: Float[Array, ""],
    n_steps: int,
    is_call: bool,
    smoothing: Float[Array, ""] | None = None,
) -> Float[Array, " n_paths"]:
    """Path-wise PV of a fixed-coupon bond with a Bermudan call or put.

    At each exercise date the continuation value is the analytic Hull-White
    affine PV of the *remaining* cash flows conditional on that path's short
    rate.  Because Hull-White zero-coupon bonds are affine this continuation
    value is exact for the underlying bullet bond, so no regression
    (Longstaff-Schwartz) basis is required.

    Two conventions are baked in, both matching
    :func:`valax.pricing.lattice.hull_white_tree.callable_bond_price`:

    - **Ex-coupon exercise.** The strike is quoted ex-coupon, so a holder
      exercised on a coupon date still receives that date's coupon.  The
      continuation value likewise excludes it.
    - **Myopic policy.** The continuation value is that of the *bullet*
      remainder and ignores the option value of later exercise dates.  With a
      single exercise date this is exact; with several it is a valid but
      suboptimal adapted policy, so for a callable bond (where the issuer
      minimises) it is an upper bound on the true price.

    The exercise indicator is smoothed with a sigmoid so pathwise Greeks stay
    well-defined, following ``valax/pricing/mc/payoffs.py``.

    Args:
        model: Hull-White model.
        result: Simulated short-rate paths.
        cf_times: Coupon/principal payment times (year fractions, ascending).
        cf_amounts: Cash flow amounts, principal folded into the final coupon.
        ex_times: Exercise dates (year fractions, **ascending**).
        ex_prices: Exercise (strike) amounts in currency units.
        T: Simulation horizon.
        n_steps: Number of simulation steps (static).
        is_call: ``True`` for an issuer call, ``False`` for a holder put.
        smoothing: Sigmoid width in currency units.  Defaults to 0.1 % of
            each exercise price.

    Returns:
        Per-path present value.
    """
    n_ex = ex_times.shape[0]
    ex_sdf = _hw_path_sdf(result, ex_times, T, n_steps)        # (n_paths, n_ex)
    cf_sdf = _hw_path_sdf(result, cf_times, T, n_steps)        # (n_paths, n_pay)
    r_ex = jnp.take(
        result.short_rates, _hw_step_index(ex_times, T, n_steps), axis=1
    )                                                          # (n_paths, n_ex)

    width = 1e-3 * ex_prices if smoothing is None else smoothing

    # Continuation value at each exercise date (loop is over a static count).
    conts = []
    for k in range(n_ex):
        t_k = ex_times[k]
        # Strictly-later cash flows only: the coupon falling on t_k is paid
        # regardless of exercise and so is excluded from the comparison.
        later = (cf_times > t_k).astype(cf_amounts.dtype)      # (n_pay,)
        zcb = hw_bond_price(
            model, r_ex[:, k][:, None], t_k, cf_times[None, :]
        )                                                      # (n_paths, n_pay)
        conts.append(jnp.sum((later * cf_amounts)[None, :] * zcb, axis=1))
    cont = jnp.stack(conts, axis=1)                            # (n_paths, n_ex)

    # Issuer calls when continuation exceeds the call price; holder puts when
    # continuation falls below the put price.
    moneyness = cont - ex_prices[None, :]
    if not is_call:
        moneyness = -moneyness
    exercise = jax_sigmoid(moneyness / width[None, :])          # (n_paths, n_ex)

    # Probability-weighted survival *entering* each exercise date.  Relies on
    # `ex_times` being ascending, as the instrument schedules guarantee.
    ones = jnp.ones((exercise.shape[0], 1), dtype=exercise.dtype)
    alive_before = jnp.concatenate(
        [ones, jnp.cumprod(1.0 - exercise, axis=1)[:, :-1]], axis=1
    )
    pv_exercise = jnp.sum(alive_before * exercise * ex_prices[None, :] * ex_sdf, axis=1)

    # A coupon survives every exercise date STRICTLY before it (ties are paid,
    # per the ex-coupon convention above).
    strictly_before = (ex_times[None, :] < cf_times[:, None]).astype(exercise.dtype)
    survival = jnp.prod(
        1.0 - exercise[:, None, :] * strictly_before[None, :, :], axis=2
    )                                                          # (n_paths, n_pay)
    pv_coupons = jnp.sum(survival * cf_amounts[None, :] * cf_sdf, axis=1)

    return pv_exercise + pv_coupons


@register(CallableBond, HullWhiteModel)
def _callable_bond_hw(
    *,
    instrument: CallableBond,
    model: HullWhiteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    smoothing: Float[Array, ""] | None = None,
    **kwargs,
) -> MCResult:
    """Callable bond price under Hull-White short-rate MC.

    At each call date the issuer's decision compares the analytic Hull-White
    affine PV of the remaining cash flows, conditional on that path's short
    rate, against the call price.  See :func:`_hw_exercisable_bond_pv` for the
    ex-coupon and myopic-policy conventions, which match the trinomial-tree
    pricer in :mod:`valax.pricing.lattice.hull_white_tree`.

    Optional market args:
        T: Horizon in year fractions (defaults to bond maturity).
        n_steps: Number of time steps (default 100).
        smoothing: Sigmoid width for the exercise indicator, in currency
            units.  Defaults to 0.1 % of each call price.
    """
    ref = model.initial_curve.reference_date
    cf_times, cf_amounts = _hw_coupon_schedule(instrument, ref)

    call_times = year_fraction(ref, instrument.call_dates, instrument.day_count)
    call_prices = instrument.call_prices * instrument.face_value

    if T is None:
        T = cf_times[-1]

    result = generate_hull_white_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    pv = _hw_exercisable_bond_pv(
        model, result, cf_times, cf_amounts, call_times, call_prices,
        T, n_steps, is_call=True, smoothing=smoothing,
    )
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(PuttableBond, HullWhiteModel)
def _puttable_bond_hw(
    *,
    instrument: PuttableBond,
    model: HullWhiteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    smoothing: Float[Array, ""] | None = None,
    **kwargs,
) -> MCResult:
    """Puttable bond price under Hull-White short-rate MC.

    Symmetric to the callable recipe: at each put date the holder exercises
    when the analytic Hull-White continuation value falls below the put price.
    See :func:`_hw_exercisable_bond_pv` for the shared conventions.

    Optional market args:
        T: Horizon in year fractions (defaults to bond maturity).
        n_steps: Number of time steps (default 100).
        smoothing: Sigmoid width for the exercise indicator, in currency
            units.  Defaults to 0.1 % of each put price.
    """
    ref = model.initial_curve.reference_date
    cf_times, cf_amounts = _hw_coupon_schedule(instrument, ref)

    put_times = year_fraction(ref, instrument.put_dates, instrument.day_count)
    put_prices = instrument.put_prices * instrument.face_value

    if T is None:
        T = cf_times[-1]

    result = generate_hull_white_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    pv = _hw_exercisable_bond_pv(
        model, result, cf_times, cf_amounts, put_times, put_prices,
        T, n_steps, is_call=False, smoothing=smoothing,
    )
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


# ─────────────────────────────────────────────────────────────────────
# Rates recipes (G2++ two-factor short-rate MC)
#
# Same contract as the Hull-White recipes: the exact two-factor conditional
# scheme (no Euler bias) is used, and every cash flow is discounted with the
# money-market numeraire accumulated along each path.
#
#   Optional market args:
#     T       (float): Horizon in year fractions (defaults per instrument).
#     n_steps (int)  : Number of time steps (default 100).
# ─────────────────────────────────────────────────────────────────────


def _g2pp_path_sdf(
    result: G2PPPathResult,
    times: Float[Array, " n_times"],
    T: Float[Array, ""],
    n_steps: int,
) -> Float[Array, "n_paths n_times"]:
    """Path-wise stochastic discount factors at a set of event times.

    Mirrors :func:`_hw_path_sdf` for :class:`G2PPPathResult`.

    Args:
        result: Output of :func:`generate_g2pp_paths`.
        times: Event times in year fractions.
        T: Simulation horizon used to generate the paths.
        n_steps: Number of simulation steps (static).

    Returns:
        Discount factor for each path and event time.
    """
    idx = _hw_step_index(times, T, n_steps)
    return jnp.exp(jnp.take(result.log_discount_factors, idx, axis=1))


@register(FixedRateBond, G2PPModel)
def _fixed_bond_g2pp(
    *,
    instrument: FixedRateBond,
    model: G2PPModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    **kwargs,
) -> MCResult:
    """Fixed-rate bond price under G2++ two-factor short-rate MC.

    Optional market args:
        T: Horizon in year fractions (defaults to the last payment date).
        n_steps: Number of time steps (default 100).
    """
    ref = model.initial_curve.reference_date
    cf_times, cf_amounts = _hw_coupon_schedule(instrument, ref)

    if T is None:
        T = cf_times[-1]

    result = generate_g2pp_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    sdf = _g2pp_path_sdf(result, cf_times, T, n_steps)
    pv = jnp.sum(cf_amounts[None, :] * sdf, axis=1)
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(FloatingRateBond, G2PPModel)
def _floating_bond_g2pp(
    *,
    instrument: FloatingRateBond,
    model: G2PPModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    **kwargs,
) -> MCResult:
    """Floating-rate note price under G2++ two-factor short-rate MC.

    Uses the same par-at-reset floating-leg replication as the Hull-White
    recipe: the floating coupons plus principal collapse to
    ``face * D(0, t_settle)`` and the fixed spread is discounted explicitly.

    Optional market args:
        T: Horizon in year fractions (defaults to maturity).
        n_steps: Number of time steps (default 100).
    """
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count
    face = instrument.face_value

    cf_times = year_fraction(ref, instrument.payment_dates, day_count)
    prev_dates = jnp.concatenate(
        [instrument.settlement_date[None], instrument.payment_dates[:-1]]
    )
    taus = year_fraction(prev_dates, instrument.payment_dates, day_count)
    spread_amounts = instrument.spread * face * taus

    t_settle = year_fraction(ref, instrument.settlement_date, day_count)
    if T is None:
        T = cf_times[-1]

    result = generate_g2pp_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    settle_sdf = _g2pp_path_sdf(result, t_settle[None], T, n_steps)[:, 0]
    spread_sdf = _g2pp_path_sdf(result, cf_times, T, n_steps)

    pv = face * settle_sdf + jnp.sum(spread_amounts[None, :] * spread_sdf, axis=1)
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(Swaption, G2PPModel)
def _swaption_g2pp(
    *,
    instrument: Swaption,
    model: G2PPModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    **kwargs,
) -> MCResult:
    """European swaption price under G2++ two-factor short-rate MC.

    At expiry the swap value on unit notional is
    :math:`1 - \\sum_i c_i P(T, t_i \\mid x, y)` (payer) computed from the
    analytic affine ZCB conditional on each path's factors, floored at zero and
    discounted by the path-wise money-market numeraire.  This provides the
    MC <-> analytic triangulation for :func:`g2pp_swaption_price`.

    Optional market args:
        T: Horizon in year fractions (defaults to the option expiry).
        n_steps: Number of time steps (default 100).
    """
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count

    expiry_time = year_fraction(ref, instrument.expiry_date, day_count)
    cashflow_times = year_fraction(ref, instrument.fixed_dates, day_count)

    starts = jnp.concatenate(
        [instrument.expiry_date[None], instrument.fixed_dates[:-1]]
    )
    taus = year_fraction(starts, instrument.fixed_dates, day_count)
    cashflows = instrument.strike * taus
    cashflows = cashflows.at[-1].add(1.0)

    if T is None:
        T = expiry_time

    result = generate_g2pp_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    idx = _hw_step_index(expiry_time[None], T, n_steps)[0]
    x_ex = result.factor_x[:, idx]                       # (n_paths,)
    y_ex = result.factor_y[:, idx]
    sdf_ex = jnp.exp(result.log_discount_factors[:, idx])

    # Coupon bond value per path from the analytic affine ZCB.
    zcb = g2pp_bond_price(
        model, x_ex[:, None], y_ex[:, None], expiry_time, cashflow_times[None, :]
    )                                                    # (n_paths, n_fixed)
    coupon_bond = jnp.sum(cashflows[None, :] * zcb, axis=1)

    swap_value = 1.0 - coupon_bond            # payer swap value at expiry
    if not instrument.is_payer:
        swap_value = -swap_value
    payoff = jnp.maximum(swap_value, 0.0)

    pv = instrument.notional * payoff * sdf_ex
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


def _g2pp_cms_rate(
    model: G2PPModel,
    x: Float[Array, " n_paths"],
    y: Float[Array, " n_paths"],
    t_fix: Float[Array, ""],
    tenor: int,
) -> Float[Array, " n_paths"]:
    """Par swap rate of an annual ``tenor``-year swap fixed at ``t_fix``.

    Computed per path from the analytic G2++ affine ZCB conditional on that
    path's factors:

    .. math::

        S = \\frac{1 - P(t_\\text{fix}, t_\\text{fix} + \\text{tenor})}
                 {\\sum_{j=1}^{\\text{tenor}} P(t_\\text{fix}, t_\\text{fix} + j)}

    (annual accruals, unit year fraction per period).

    Args:
        model: G2++ model.
        x: First factor at the fixing time, one per path.
        y: Second factor at the fixing time, one per path.
        t_fix: Fixing time in year fractions.
        tenor: Underlying swap tenor in whole years (static).

    Returns:
        Per-path forward par swap rate.
    """
    js = jnp.arange(1, tenor + 1, dtype=jnp.float64)
    maturities = t_fix + js                                   # (tenor,)
    zcb = g2pp_bond_price(
        model, x[:, None], y[:, None], t_fix, maturities[None, :]
    )                                                         # (n_paths, tenor)
    annuity = jnp.sum(zcb, axis=1)
    return (1.0 - zcb[:, -1]) / annuity


@register(CMSSpreadSwap, G2PPModel)
def _cms_spread_swap_g2pp(
    *,
    instrument: CMSSpreadSwap,
    model: G2PPModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int = 100,
    **kwargs,
) -> MCResult:
    """CMS-spread swap (steepener / flattener) under G2++ two-factor MC.

    At each period's accrual start both CMS rates are computed analytically
    from the path's factors, their spread net of the fixed strike accrues over
    the period, and the coupon is discounted by the path-wise money-market
    numeraire.  This is the decorrelation-sensitive payoff that motivates the
    second factor.

    Optional market args:
        T: Horizon in year fractions (defaults to the last payment date).
        n_steps: Number of time steps (default 100).
    """
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count

    pay_times = year_fraction(ref, instrument.payment_dates, day_count)
    starts = jnp.concatenate(
        [instrument.start_date[None], instrument.payment_dates[:-1]]
    )
    fix_times = year_fraction(ref, starts, day_count)
    taus = year_fraction(starts, instrument.payment_dates, day_count)

    if T is None:
        T = pay_times[-1]

    result = generate_g2pp_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )

    sign = 1.0 if instrument.pay_fixed else -1.0
    n_periods = instrument.payment_dates.shape[0]

    pv = jnp.zeros(config.n_paths, dtype=jnp.float64)
    for i in range(n_periods):
        fix_idx = _hw_step_index(fix_times[i][None], T, n_steps)[0]
        x_fix = result.factor_x[:, fix_idx]
        y_fix = result.factor_y[:, fix_idx]

        s_long = _g2pp_cms_rate(
            model, x_fix, y_fix, fix_times[i], instrument.cms_tenor_long
        )
        s_short = _g2pp_cms_rate(
            model, x_fix, y_fix, fix_times[i], instrument.cms_tenor_short
        )
        coupon = sign * (s_long - s_short - instrument.fixed_rate) * taus[i]

        pay_sdf = _g2pp_path_sdf(result, pay_times[i][None], T, n_steps)[:, 0]
        pv = pv + instrument.notional * coupon * pay_sdf

    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)
