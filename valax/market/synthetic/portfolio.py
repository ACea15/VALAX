"""Synthetic portfolio samplers (stacked instrument pytrees).

Produces *stacked* pytrees of instruments suitable for
``valax.portfolio.batch.batch_price`` / ``batch_greeks``, which
expect each leaf to carry a leading batch dimension.

Currently supported:

- :func:`sample_option_portfolio` — stacked ``EuropeanOption`` over a
  configurable cross-product of moneyness × expiry × asset, with
  put/call mix.
- :func:`sample_swap_portfolio` — stacked
  :class:`~valax.instruments.rates.InterestRateSwap` with random
  tenors and notionals.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market.data import MarketData

from valax.market.synthetic.seeds import SeedRegistry


# ── Option portfolio ──────────────────────────────────────────────


@dataclass(frozen=True)
class OptionPortfolioSpec:
    """Config for :func:`sample_option_portfolio`."""

    n_per_asset: int = 6
    expiry_range: tuple[float, float] = (0.1, 2.0)
    moneyness_range: tuple[float, float] = (0.8, 1.2)
    call_probability: float = 0.5


def _stacked_european_options(
    strikes: Float[Array, " n"],
    expiries: Float[Array, " n"],
    is_call_mask: Array,
) -> EuropeanOption:
    """Build a stacked EuropeanOption pytree from parallel arrays.

    Because ``EuropeanOption.is_call`` is a *static* field, a stacked
    pytree with mixed call/put flags is not directly representable.
    We therefore use a uniform call/put flag — the workaround is to
    pass calls and puts as two separate stacks.

    This helper constructs a single stack with the *majority* flag and
    returns it; the caller is responsible for splitting the strikes
    array if it wants both flags.  See the public
    :func:`sample_option_portfolio` for a stack-by-flag convenience.
    """
    is_call_majority = bool(jnp.sum(is_call_mask) >= is_call_mask.shape[0] / 2)
    return EuropeanOption(
        strike=strikes,
        expiry=expiries,
        is_call=is_call_majority,
    )


def sample_option_portfolio(
    registry: SeedRegistry,
    md: MarketData,
    spec: OptionPortfolioSpec = OptionPortfolioSpec(),
) -> dict[str, tuple[EuropeanOption, Float[Array, " n"]]]:
    """Sample a portfolio of European options across the assets of ``md``.

    Because ``EuropeanOption.is_call`` is a static field, the returned
    portfolio is split into two stacks (one for calls, one for puts).
    Each stack maps strike / expiry vectors to the appropriate per-leg
    market vectors ready for :func:`valax.portfolio.batch.batch_price`.

    Args:
        registry: Seed registry.
        md: Reference market snapshot.  Spots and per-asset vols /
            dividends are read here; the returned vectors are
            replicated to match the leg count.
        spec: Portfolio specification (see :class:`OptionPortfolioSpec`).

    Returns:
        Dict with two keys:

        - ``"calls"``: ``(stacked_european_calls, asset_index_vector)``
        - ``"puts"``:  ``(stacked_european_puts,  asset_index_vector)``

        Each ``asset_index_vector`` is an ``Int`` array of the same
        length as the stack, giving the asset index each leg references.
        Callers can use it to gather ``md.spots[idx]``, ``md.vols[idx]``,
        ``md.dividends[idx]`` for ``batch_price``.

    Stream prefix: ``synthetic.portfolio.options.*``.
    """
    n_assets = md.spots.shape[0]
    n_per_asset = spec.n_per_asset
    total = n_assets * n_per_asset

    # Asset indices (repeat each asset n_per_asset times).
    asset_idx = jnp.repeat(jnp.arange(n_assets, dtype=jnp.int32), n_per_asset)

    # Moneyness uniform per leg.
    moneyness = jax.random.uniform(
        registry.key("synthetic.portfolio.options.moneyness"),
        shape=(total,), dtype=jnp.float64,
        minval=spec.moneyness_range[0], maxval=spec.moneyness_range[1],
    )
    spots_per_leg = md.spots[asset_idx]
    strikes = spots_per_leg * moneyness

    expiries = jax.random.uniform(
        registry.key("synthetic.portfolio.options.expiry"),
        shape=(total,), dtype=jnp.float64,
        minval=spec.expiry_range[0], maxval=spec.expiry_range[1],
    )

    # Bernoulli call/put per leg.
    u = jax.random.uniform(
        registry.key("synthetic.portfolio.options.is_call"),
        shape=(total,), dtype=jnp.float64,
    )
    is_call_mask = u < spec.call_probability

    call_slice = is_call_mask
    put_slice = jnp.logical_not(is_call_mask)

    call_strikes = strikes[call_slice]
    call_expiries = expiries[call_slice]
    call_assets = asset_idx[call_slice]
    put_strikes = strikes[put_slice]
    put_expiries = expiries[put_slice]
    put_assets = asset_idx[put_slice]

    calls = EuropeanOption(
        strike=call_strikes, expiry=call_expiries, is_call=True,
    )
    puts = EuropeanOption(
        strike=put_strikes, expiry=put_expiries, is_call=False,
    )
    return {
        "calls": (calls, call_assets),
        "puts": (puts, put_assets),
    }


# ── Swap portfolio ────────────────────────────────────────────────


@dataclass(frozen=True)
class SwapPortfolioSpec:
    """Config for :func:`sample_swap_portfolio`."""

    n_swaps: int = 5
    tenor_range_years: tuple[float, float] = (1.0, 10.0)
    fixed_rate_range: tuple[float, float] = (0.005, 0.06)
    notional_range: tuple[float, float] = (1e6, 1e8)
    pay_freq_per_year: int = 2  # semi-annual fixed coupons


def sample_swap_portfolio(
    registry: SeedRegistry,
    curve: DiscountCurve,
    spec: SwapPortfolioSpec = SwapPortfolioSpec(),
) -> list[InterestRateSwap]:
    """Sample a list of :class:`InterestRateSwap` instruments.

    Notes:
        Returned as a *list* (not a stacked pytree) because every
        swap has a different number of fixed coupons, so the
        ``fixed_dates`` arrays cannot be stacked into a single
        rectangular leaf.  Pricers should iterate the list.

    Stream prefix: ``synthetic.portfolio.swaps.*``.
    """
    ref = int(curve.reference_date)
    n = spec.n_swaps

    tenors = jax.random.uniform(
        registry.key("synthetic.portfolio.swaps.tenor"),
        shape=(n,), dtype=jnp.float64,
        minval=spec.tenor_range_years[0], maxval=spec.tenor_range_years[1],
    )
    fixed_rates = jax.random.uniform(
        registry.key("synthetic.portfolio.swaps.rate"),
        shape=(n,), dtype=jnp.float64,
        minval=spec.fixed_rate_range[0], maxval=spec.fixed_rate_range[1],
    )
    notionals = jax.random.uniform(
        registry.key("synthetic.portfolio.swaps.notional"),
        shape=(n,), dtype=jnp.float64,
        minval=spec.notional_range[0], maxval=spec.notional_range[1],
    )
    pay_fixed = jax.random.bernoulli(
        registry.key("synthetic.portfolio.swaps.pay_fixed"),
        p=0.5, shape=(n,),
    )

    swaps: list[InterestRateSwap] = []
    days_per_period = int(round(365.0 / spec.pay_freq_per_year))
    start_date = jnp.int32(ref)
    for i in range(n):
        tenor_years = float(tenors[i])
        n_periods = max(1, int(round(tenor_years * spec.pay_freq_per_year)))
        fixed_dates = jnp.int32(
            ref + days_per_period * jnp.arange(1, n_periods + 1, dtype=jnp.int32)
        )
        swaps.append(
            InterestRateSwap(
                start_date=start_date,
                fixed_dates=fixed_dates,
                fixed_rate=fixed_rates[i],
                notional=notionals[i],
                pay_fixed=bool(pay_fixed[i]),
            )
        )
    return swaps


__all__ = [
    "OptionPortfolioSpec",
    "SwapPortfolioSpec",
    "sample_option_portfolio",
    "sample_swap_portfolio",
]
