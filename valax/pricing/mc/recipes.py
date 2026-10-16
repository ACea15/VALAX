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

from valax.dates.accrual import fixed_coupon_amounts, floating_accrual_periods
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
from valax.models.cheyette import (
    CheyetteModel,
    CheyetteModel2F,
    cheyette_bond_price,
    cheyette_bond_price_2f,
)
from valax.models.lmm import LMMModel
from valax.models.local_vol import LocalVolModel
from valax.models.multi_asset import MultiAssetGBMModel
from valax.models.slv import SLVModel
from valax.pricing.mc.hull_white_paths import HullWhitePathResult, generate_hull_white_paths
from valax.pricing.mc.g2pp_paths import G2PPPathResult, generate_g2pp_paths
from valax.pricing.mc.cheyette_paths import (
    CheyettePathResult,
    Cheyette2FPathResult,
    generate_cheyette_paths,
    generate_cheyette_paths_2f,
)
from valax.pricing.mc.bermudan import LSMConfig, bermudan_swaption_lsm
from valax.pricing.mc.dispatch import (
    MCConfig,
    MCResult,
    discounted_mean_and_stderr,
    register,
)
from valax.pricing.multicurve import (
    apply_swap_basis,
    model_swap_leg_basis,
    model_tenor_basis,
    require_single_curve,
    with_discount_curve,
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
#     n_steps defaults to config.n_steps (MCConfig) when not supplied;
#     before D15 it defaulted to 100 and MCConfig.n_steps was ignored.
#
# Multi-curve: the Hull-White and G2++ rates recipes below additionally take
# `discount_curve` and (where a floating leg is projected) `forward_curve`,
# supplied automatically by valax.core.pricing.price from the instrument's
# discount_id / forward_id selectors.  Hull-White and G2++ are exact-fit
# models, so honouring a discount curve means re-fitting the model to it
# (valax.pricing.multicurve.with_discount_curve); projection is handled by the
# frozen tenor-basis overlay in the same module.  Both default to None, which
# is the single-curve case and leaves prices unchanged bit-for-bit.
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
            ``face_value`` and optional ``accrual_start_dates``.
        ref: Curve reference date (ordinal).

    Returns:
        ``(times, amounts)`` — year fractions from ``ref`` to each payment
        date, and the corresponding cash flow (principal folded into the
        final coupon).
    """
    times = year_fraction(ref, instrument.payment_dates, instrument.day_count)
    amounts = fixed_coupon_amounts(instrument).at[-1].add(instrument.face_value)
    return times, amounts


@register(FixedRateBond, HullWhiteModel)
def _fixed_bond_hw(
    *,
    instrument: FixedRateBond,
    model: HullWhiteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    discount_curve=None,
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
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        discount_curve: Curve to discount on; re-fits the model to it
            (:func:`~valax.pricing.multicurve.with_discount_curve`).
            Defaults to the model's own ``initial_curve``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
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
    n_steps: int | None = None,
    discount_curve=None,
    forward_curve=None,
    **kwargs,
) -> MCResult:
    """Floating-rate bond (FRN) price under Hull-White short-rate MC.

    Under the risk-neutral measure a par-at-reset FRN prices to par
    at each reset date.  For seasoned FRNs or non-zero spread, the
    floating leg is replicated as:

    .. math::

        V = \\text{face} \\cdot \\hat{D}(0, T_0) + s \\sum_i \\tau_i \\hat{D}(0, T_i) + \\text{face} \\cdot \\hat{D}(0, T_N)

    where :math:`\\hat{D}` is the path-wise stochastic discount factor.

    **Dual curve.** When the coupons project off a curve distinct from the
    discount curve, that telescoping collapse no longer happens.  Under the
    frozen tenor basis :math:`b_i` of :mod:`valax.pricing.multicurve` the
    accrued floating coupon is
    :math:`\\tau_i L_i = b_i / P(T_{i-1}, T_i) - 1`, whose path-wise
    unbiased estimator is
    :math:`b_i \\hat{D}(0, T_{i-1}) - \\hat{D}(0, T_i)` — the martingale
    identity :math:`E_{T_{i-1}}[\\hat{D}(T_{i-1}, T_i)] = P(T_{i-1}, T_i)`.
    With every :math:`b_i = 1` the sum telescopes back to the expression
    above, so the single-curve path is left literally unchanged.

    Optional market args:
        T: Horizon in year fractions (defaults to maturity).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        discount_curve: Curve to discount on; re-fits the model to it.
        forward_curve: Projection curve for the floating coupons. ``None``
            (the default) projects off the discount curve.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count
    face = instrument.face_value

    cf_times = year_fraction(ref, instrument.payment_dates, day_count)
    # One accrual convention for every FRN engine: the coupon accrues over
    # [accrual_start, payment_date) and the projected rate is the forward
    # over that same period. Previously this recipe accrued from the
    # previous payment date (and the first period from settlement) while
    # the analytic pricer accrued from the fixing date, so the two
    # disagreed whenever those dates differed.
    prev_dates, accrual_end = floating_accrual_periods(instrument)
    taus = year_fraction(prev_dates, accrual_end, day_count)
    spread_amounts = instrument.spread * face * taus          # (n_pay,)
    basis = model_tenor_basis(
        model, forward_curve, prev_dates, instrument.payment_dates
    )

    t_settle = year_fraction(ref, instrument.settlement_date, day_count)
    prev_times = year_fraction(ref, prev_dates, day_count)
    if T is None:
        T = cf_times[-1]

    result = generate_hull_white_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )

    spread_sdf = _hw_path_sdf(result, cf_times, T, n_steps)

    if basis is None:
        # Floating-leg replication.  The floating coupons between settlement
        # and maturity are worth face * (D(settle) - D(T_N)); adding the
        # principal redemption face * D(T_N) cancels the second term exactly,
        # leaving the familiar "an FRN prices to par at reset" identity:
        #
        #     PV = face * D(0, t_settle)  +  spread coupons
        settle_sdf = _hw_path_sdf(result, t_settle[None], T, n_steps)[:, 0]
        pv = face * settle_sdf
    else:
        reset_sdf = _hw_path_sdf(result, prev_times, T, n_steps)
        floating = jnp.sum(basis[None, :] * reset_sdf - spread_sdf, axis=1)
        pv = face * (floating + spread_sdf[:, -1])

    pv = pv + jnp.sum(spread_amounts[None, :] * spread_sdf, axis=1)

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
    n_steps: int | None = None,
    smoothing: Float[Array, ""] | None = None,
    discount_curve=None,
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
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        smoothing: Sigmoid width for the exercise indicator, in currency
            units.  Defaults to 0.1 % of each call price.
        discount_curve: Curve to discount on; re-fits the model to it.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
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
    n_steps: int | None = None,
    smoothing: Float[Array, ""] | None = None,
    discount_curve=None,
    **kwargs,
) -> MCResult:
    """Puttable bond price under Hull-White short-rate MC.

    Symmetric to the callable recipe: at each put date the holder exercises
    when the analytic Hull-White continuation value falls below the put price.
    See :func:`_hw_exercisable_bond_pv` for the shared conventions.

    Optional market args:
        T: Horizon in year fractions (defaults to bond maturity).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        smoothing: Sigmoid width for the exercise indicator, in currency
            units.  Defaults to 0.1 % of each put price.
        discount_curve: Curve to discount on; re-fits the model to it.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
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
    n_steps: int | None = None,
    discount_curve=None,
    **kwargs,
) -> MCResult:
    """Fixed-rate bond price under G2++ two-factor short-rate MC.

    Optional market args:
        T: Horizon in year fractions (defaults to the last payment date).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        discount_curve: Curve to discount on; re-fits the model to it.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
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
    n_steps: int | None = None,
    discount_curve=None,
    forward_curve=None,
    **kwargs,
) -> MCResult:
    """Floating-rate note price under G2++ two-factor short-rate MC.

    Uses the same par-at-reset floating-leg replication as the Hull-White
    recipe: the floating coupons plus principal collapse to
    ``face * D(0, t_settle)`` and the fixed spread is discounted explicitly.
    The dual-curve form (see :func:`_floating_bond_hw`) is identical.

    Optional market args:
        T: Horizon in year fractions (defaults to maturity).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        discount_curve: Curve to discount on; re-fits the model to it.
        forward_curve: Projection curve for the floating coupons. ``None``
            (the default) projects off the discount curve.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count
    face = instrument.face_value

    cf_times = year_fraction(ref, instrument.payment_dates, day_count)
    # One accrual convention for every FRN engine: the coupon accrues over
    # [accrual_start, payment_date) and the projected rate is the forward
    # over that same period. Previously this recipe accrued from the
    # previous payment date (and the first period from settlement) while
    # the analytic pricer accrued from the fixing date, so the two
    # disagreed whenever those dates differed.
    prev_dates, accrual_end = floating_accrual_periods(instrument)
    taus = year_fraction(prev_dates, accrual_end, day_count)
    spread_amounts = instrument.spread * face * taus
    basis = model_tenor_basis(
        model, forward_curve, prev_dates, instrument.payment_dates
    )

    t_settle = year_fraction(ref, instrument.settlement_date, day_count)
    prev_times = year_fraction(ref, prev_dates, day_count)
    if T is None:
        T = cf_times[-1]

    result = generate_g2pp_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    spread_sdf = _g2pp_path_sdf(result, cf_times, T, n_steps)

    if basis is None:
        settle_sdf = _g2pp_path_sdf(result, t_settle[None], T, n_steps)[:, 0]
        pv = face * settle_sdf
    else:
        reset_sdf = _g2pp_path_sdf(result, prev_times, T, n_steps)
        floating = jnp.sum(basis[None, :] * reset_sdf - spread_sdf, axis=1)
        pv = face * (floating + spread_sdf[:, -1])

    pv = pv + jnp.sum(spread_amounts[None, :] * spread_sdf, axis=1)
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
    n_steps: int | None = None,
    discount_curve=None,
    forward_curve=None,
    **kwargs,
) -> MCResult:
    """European swaption price under G2++ two-factor short-rate MC.

    At expiry the swap value on unit notional is
    :math:`1 - \\sum_i c_i P(T, t_i \\mid x, y)` (payer) computed from the
    analytic affine ZCB conditional on each path's factors, floored at zero and
    discounted by the path-wise money-market numeraire.  This provides the
    MC <-> analytic triangulation for :func:`g2pp_swaption_price`.

    With a distinct projection curve the same expression holds with the
    constant and the cash-flow vector adjusted by the frozen tenor basis
    (:func:`~valax.pricing.multicurve.apply_swap_basis`).

    Optional market args:
        T: Horizon in year fractions (defaults to the option expiry).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        discount_curve: Curve the swap discounts on; re-fits the model to it.
        forward_curve: Projection curve for the floating leg. ``None`` (the
            default) projects off the discount curve.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
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
    float_start, cashflows = apply_swap_basis(
        cashflows,
        model_swap_leg_basis(
            model,
            forward_curve,
            instrument.expiry_date,
            instrument.fixed_dates,
        ),
    )

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

    swap_value = float_start - coupon_bond    # payer swap value at expiry
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
    n_steps: int | None = None,
    discount_curve=None,
    forward_curve=None,
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
        n_steps: Number of time steps; defaults to ``config.n_steps``.
        discount_curve: Curve the coupons discount on; re-fits the model.
        forward_curve: **Not supported yet.** The CMS rates are rebuilt from
            the model's own bond strip, so a distinct projection curve would
            be silently ignored; a distinct curve is rejected instead.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    model = with_discount_curve(model, discount_curve)
    require_single_curve(
        model.initial_curve, forward_curve, "The G2++ CMS-spread MC recipe"
    )
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


# ─────────────────────────────────────────────────────────────────────
# Rates recipes (Cheyette one-factor quasi-Gaussian short-rate MC)
#
# The Cheyette model adds a volatility *skew* to the rate dynamics via a
# state-dependent local volatility.  The MC scheme is the integrating-factor
# scan in valax.pricing.mc.cheyette_paths (no exact Gaussian sampling, since
# the state is non-Gaussian), with the deterministic forward part of the
# discount integrated exactly.  Every cash flow is discounted with the
# money-market numeraire accumulated along each path.
#
#   Optional market args:
#     T       (float): Horizon in year fractions (defaults per instrument).
#     n_steps (int)  : Number of time steps (default 100).
# ─────────────────────────────────────────────────────────────────────


def _cheyette_path_sdf(
    result: CheyettePathResult,
    times: Float[Array, " n_times"],
    T: Float[Array, ""],
    n_steps: int,
) -> Float[Array, "n_paths n_times"]:
    """Path-wise stochastic discount factors at a set of event times.

    Mirrors :func:`_g2pp_path_sdf` for :class:`CheyettePathResult`.

    Args:
        result: Output of :func:`generate_cheyette_paths`.
        times: Event times in year fractions.
        T: Simulation horizon used to generate the paths.
        n_steps: Number of simulation steps (static).

    Returns:
        Discount factor for each path and event time.
    """
    idx = _hw_step_index(times, T, n_steps)
    return jnp.exp(jnp.take(result.log_discount_factors, idx, axis=1))


@register(FixedRateBond, CheyetteModel)
def _fixed_bond_cheyette(
    *,
    instrument: FixedRateBond,
    model: CheyetteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """Fixed-rate bond price under one-factor Cheyette short-rate MC.

    Optional market args:
        T: Horizon in year fractions (defaults to the last payment date).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    ref = model.initial_curve.reference_date
    cf_times, cf_amounts = _hw_coupon_schedule(instrument, ref)

    if T is None:
        T = cf_times[-1]

    result = generate_cheyette_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    sdf = _cheyette_path_sdf(result, cf_times, T, n_steps)
    pv = jnp.sum(cf_amounts[None, :] * sdf, axis=1)
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(FloatingRateBond, CheyetteModel)
def _floating_bond_cheyette(
    *,
    instrument: FloatingRateBond,
    model: CheyetteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """Floating-rate note price under one-factor Cheyette short-rate MC.

    Uses the same par-at-reset floating-leg replication as the Hull-White and
    G2++ recipes: the floating coupons plus principal collapse to
    ``face * D(0, t_settle)`` and the fixed spread is discounted explicitly.

    Optional market args:
        T: Horizon in year fractions (defaults to maturity).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count
    face = instrument.face_value

    cf_times = year_fraction(ref, instrument.payment_dates, day_count)
    # One accrual convention for every FRN engine: the coupon accrues over
    # [accrual_start, payment_date) and the projected rate is the forward
    # over that same period. Previously this recipe accrued from the
    # previous payment date (and the first period from settlement) while
    # the analytic pricer accrued from the fixing date, so the two
    # disagreed whenever those dates differed.
    prev_dates, accrual_end = floating_accrual_periods(instrument)
    taus = year_fraction(prev_dates, accrual_end, day_count)
    spread_amounts = instrument.spread * face * taus

    t_settle = year_fraction(ref, instrument.settlement_date, day_count)
    if T is None:
        T = cf_times[-1]

    result = generate_cheyette_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    settle_sdf = _cheyette_path_sdf(result, t_settle[None], T, n_steps)[:, 0]
    spread_sdf = _cheyette_path_sdf(result, cf_times, T, n_steps)

    pv = face * settle_sdf + jnp.sum(spread_amounts[None, :] * spread_sdf, axis=1)
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


def _cheyette_forward_rate(
    model: CheyetteModel,
    result: CheyettePathResult,
    t_fix: Float[Array, ""],
    t_start: Float[Array, ""],
    t_end: Float[Array, ""],
    tau: Float[Array, ""],
    T: Float[Array, ""],
    n_steps: int,
) -> Float[Array, " n_paths"]:
    """Per-path simply-compounded forward rate over ``[t_start, t_end]``.

    Observed at ``t_fix`` from the analytic Cheyette affine ZCB conditional on
    each path's state:

    .. math::

        F = \\frac{1}{\\tau}\\Bigl(\\frac{P(t_\\text{fix}, t_\\text{start})}
                                       {P(t_\\text{fix}, t_\\text{end})} - 1\\Bigr).

    Args:
        model: One-factor Cheyette model.
        result: Simulated Cheyette paths.
        t_fix: Fixing time in year fractions.
        t_start: Accrual start time in year fractions.
        t_end: Accrual end time in year fractions.
        tau: Accrual fraction of the period.
        T: Simulation horizon.
        n_steps: Number of simulation steps (static).

    Returns:
        Per-path forward rate.
    """
    idx = _hw_step_index(t_fix[None], T, n_steps)[0]
    x_fix = result.factor_x[:, idx]
    y_fix = result.aux_y[:, idx]
    p_start = cheyette_bond_price(model, x_fix, y_fix, t_fix, t_start)
    p_end = cheyette_bond_price(model, x_fix, y_fix, t_fix, t_end)
    return (p_start / p_end - 1.0) / tau


@register(Caplet, CheyetteModel)
def _caplet_cheyette(
    *,
    instrument: Caplet,
    model: CheyetteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """Caplet / floorlet under one-factor Cheyette short-rate MC.

    The forward rate is observed at the fixing date from the analytic affine
    ZCB conditional on each path's state, and the intrinsic value is discounted
    by the path-wise money-market numeraire to the payment (end) date.  This is
    the skew-sensitive vanilla that motivates the Cheyette local volatility.

    Optional market args:
        T: Horizon in year fractions (defaults to the accrual end date).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count

    t_fix = year_fraction(ref, instrument.fixing_date, day_count)
    t_start = year_fraction(ref, instrument.start_date, day_count)
    t_end = year_fraction(ref, instrument.end_date, day_count)
    tau = year_fraction(instrument.start_date, instrument.end_date, day_count)

    if T is None:
        T = t_end

    result = generate_cheyette_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    fwd = _cheyette_forward_rate(
        model, result, t_fix, t_start, t_end, tau, T, n_steps
    )
    if instrument.is_cap:
        intrinsic = jnp.maximum(fwd - instrument.strike, 0.0)
    else:
        intrinsic = jnp.maximum(instrument.strike - fwd, 0.0)

    pay_sdf = _cheyette_path_sdf(result, t_end[None], T, n_steps)[:, 0]
    pv = instrument.notional * tau * intrinsic * pay_sdf
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(Cap, CheyetteModel)
def _cap_cheyette(
    *,
    instrument: Cap,
    model: CheyetteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """Cap / floor strip under one-factor Cheyette short-rate MC.

    Sums the discounted caplet/floorlet payoffs across the payment schedule,
    each priced from the analytic affine ZCB conditional on the path state at
    its fixing date.

    Optional market args:
        T: Horizon in year fractions (defaults to the last accrual end date).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count

    t_fix = year_fraction(ref, instrument.fixing_dates, day_count)
    t_start = year_fraction(ref, instrument.start_dates, day_count)
    t_end = year_fraction(ref, instrument.end_dates, day_count)
    taus = year_fraction(instrument.start_dates, instrument.end_dates, day_count)

    if T is None:
        T = t_end[-1]

    result = generate_cheyette_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )

    n_periods = instrument.fixing_dates.shape[0]
    pv = jnp.zeros(config.n_paths, dtype=jnp.float64)
    for i in range(n_periods):
        fwd = _cheyette_forward_rate(
            model, result, t_fix[i], t_start[i], t_end[i], taus[i], T, n_steps
        )
        if instrument.is_cap:
            intrinsic = jnp.maximum(fwd - instrument.strike, 0.0)
        else:
            intrinsic = jnp.maximum(instrument.strike - fwd, 0.0)
        pay_sdf = _cheyette_path_sdf(result, t_end[i][None], T, n_steps)[:, 0]
        pv = pv + instrument.notional * taus[i] * intrinsic * pay_sdf

    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(Swaption, CheyetteModel)
def _swaption_cheyette(
    *,
    instrument: Swaption,
    model: CheyetteModel,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """European swaption price under one-factor Cheyette short-rate MC.

    At expiry the swap value on unit notional is
    :math:`1 - \\sum_i c_i P(T, t_i \\mid x, y)` (payer) from the analytic affine
    ZCB conditional on each path's state, floored at zero and discounted by the
    path-wise money-market numeraire.  With a non-zero skew this carries the
    swaption smile that the Gaussian models cannot.

    Optional market args:
        T: Horizon in year fractions (defaults to the option expiry).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
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

    result = generate_cheyette_paths(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    idx = _hw_step_index(expiry_time[None], T, n_steps)[0]
    x_ex = result.factor_x[:, idx]
    y_ex = result.aux_y[:, idx]
    sdf_ex = jnp.exp(result.log_discount_factors[:, idx])

    zcb = cheyette_bond_price(
        model, x_ex[:, None], y_ex[:, None], expiry_time, cashflow_times[None, :]
    )
    coupon_bond = jnp.sum(cashflows[None, :] * zcb, axis=1)

    swap_value = 1.0 - coupon_bond
    if not instrument.is_payer:
        swap_value = -swap_value
    payoff = jnp.maximum(swap_value, 0.0)

    pv = instrument.notional * payoff * sdf_ex
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


# ─────────────────────────────────────────────────────────────────────
# Rates recipes (Cheyette two-factor quasi-Gaussian short-rate MC)
#
# The second factor adds decorrelation on top of the skew, so the headline
# recipe is the decorrelation-sensitive CMS-spread swap; the bonds provide the
# option-free martingale calibration of the scheme.
# ─────────────────────────────────────────────────────────────────────


def _cheyette2f_path_sdf(
    result: Cheyette2FPathResult,
    times: Float[Array, " n_times"],
    T: Float[Array, ""],
    n_steps: int,
) -> Float[Array, "n_paths n_times"]:
    """Path-wise stochastic discount factors for :class:`Cheyette2FPathResult`."""
    idx = _hw_step_index(times, T, n_steps)
    return jnp.exp(jnp.take(result.log_discount_factors, idx, axis=1))


@register(FixedRateBond, CheyetteModel2F)
def _fixed_bond_cheyette2f(
    *,
    instrument: FixedRateBond,
    model: CheyetteModel2F,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """Fixed-rate bond price under two-factor Cheyette short-rate MC.

    Optional market args:
        T: Horizon in year fractions (defaults to the last payment date).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    ref = model.initial_curve.reference_date
    cf_times, cf_amounts = _hw_coupon_schedule(instrument, ref)

    if T is None:
        T = cf_times[-1]

    result = generate_cheyette_paths_2f(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    sdf = _cheyette2f_path_sdf(result, cf_times, T, n_steps)
    pv = jnp.sum(cf_amounts[None, :] * sdf, axis=1)
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


@register(FloatingRateBond, CheyetteModel2F)
def _floating_bond_cheyette2f(
    *,
    instrument: FloatingRateBond,
    model: CheyetteModel2F,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """Floating-rate note price under two-factor Cheyette short-rate MC.

    Optional market args:
        T: Horizon in year fractions (defaults to maturity).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
    ref = model.initial_curve.reference_date
    day_count = instrument.day_count
    face = instrument.face_value

    cf_times = year_fraction(ref, instrument.payment_dates, day_count)
    # One accrual convention for every FRN engine: the coupon accrues over
    # [accrual_start, payment_date) and the projected rate is the forward
    # over that same period. Previously this recipe accrued from the
    # previous payment date (and the first period from settlement) while
    # the analytic pricer accrued from the fixing date, so the two
    # disagreed whenever those dates differed.
    prev_dates, accrual_end = floating_accrual_periods(instrument)
    taus = year_fraction(prev_dates, accrual_end, day_count)
    spread_amounts = instrument.spread * face * taus

    t_settle = year_fraction(ref, instrument.settlement_date, day_count)
    if T is None:
        T = cf_times[-1]

    result = generate_cheyette_paths_2f(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )
    settle_sdf = _cheyette2f_path_sdf(result, t_settle[None], T, n_steps)[:, 0]
    spread_sdf = _cheyette2f_path_sdf(result, cf_times, T, n_steps)

    pv = face * settle_sdf + jnp.sum(spread_amounts[None, :] * spread_sdf, axis=1)
    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)


def _cheyette2f_cms_rate(
    model: CheyetteModel2F,
    result: Cheyette2FPathResult,
    idx: Int[Array, ""],
    t_fix: Float[Array, ""],
    tenor: int,
) -> Float[Array, " n_paths"]:
    """Par swap rate of an annual ``tenor``-year swap fixed at ``t_fix``.

    Computed per path from the analytic two-factor Cheyette affine ZCB
    conditional on that path's state at step ``idx``:

    .. math::

        S = \\frac{1 - P(t_\\text{fix}, t_\\text{fix} + \\text{tenor})}
                 {\\sum_{j=1}^{\\text{tenor}} P(t_\\text{fix}, t_\\text{fix} + j)}.

    Args:
        model: Two-factor Cheyette model.
        result: Simulated two-factor Cheyette paths.
        idx: Time-grid step index of the fixing.
        t_fix: Fixing time in year fractions.
        tenor: Underlying swap tenor in whole years (static).

    Returns:
        Per-path forward par swap rate.
    """
    x1 = result.factor_x1[:, idx][:, None]
    x2 = result.factor_x2[:, idx][:, None]
    y11 = result.aux_y11[:, idx][:, None]
    y22 = result.aux_y22[:, idx][:, None]
    y12 = result.aux_y12[:, idx][:, None]
    js = jnp.arange(1, tenor + 1, dtype=jnp.float64)
    maturities = t_fix + js
    zcb = cheyette_bond_price_2f(
        model, x1, x2, y11, y22, y12, t_fix, maturities[None, :]
    )
    annuity = jnp.sum(zcb, axis=1)
    return (1.0 - zcb[:, -1]) / annuity


@register(CMSSpreadSwap, CheyetteModel2F)
def _cms_spread_swap_cheyette2f(
    *,
    instrument: CMSSpreadSwap,
    model: CheyetteModel2F,
    config: MCConfig,
    key: jax.Array,
    T: float | None = None,
    n_steps: int | None = None,
    **kwargs,
) -> MCResult:
    """CMS-spread swap (steepener / flattener) under two-factor Cheyette MC.

    At each period's accrual start both CMS rates are computed analytically from
    the path's factors, their spread net of the fixed strike accrues over the
    period, and the coupon is discounted by the path-wise money-market
    numeraire.  This is the decorrelation-sensitive payoff that motivates the
    second factor — now on a skew-capable quasi-Gaussian base.

    Optional market args:
        T: Horizon in year fractions (defaults to the last payment date).
        n_steps: Number of time steps; defaults to ``config.n_steps``.
    """
    n_steps = config.n_steps if n_steps is None else n_steps
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

    result = generate_cheyette_paths_2f(
        model, T=T, n_steps=n_steps, n_paths=config.n_paths, key=key,
    )

    sign = 1.0 if instrument.pay_fixed else -1.0
    n_periods = instrument.payment_dates.shape[0]

    pv = jnp.zeros(config.n_paths, dtype=jnp.float64)
    for i in range(n_periods):
        fix_idx = _hw_step_index(fix_times[i][None], T, n_steps)[0]
        s_long = _cheyette2f_cms_rate(
            model, result, fix_idx, fix_times[i], instrument.cms_tenor_long
        )
        s_short = _cheyette2f_cms_rate(
            model, result, fix_idx, fix_times[i], instrument.cms_tenor_short
        )
        coupon = sign * (s_long - s_short - instrument.fixed_rate) * taus[i]
        pay_sdf = _cheyette2f_path_sdf(result, pay_times[i][None], T, n_steps)[:, 0]
        pv = pv + instrument.notional * coupon * pay_sdf

    price, stderr = discounted_mean_and_stderr(
        pv, jnp.ones((), dtype=pv.dtype), config.n_paths
    )
    return MCResult(price=price, stderr=stderr, n_paths=config.n_paths)
