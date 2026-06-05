"""Example 07 — Generate synthetic market data and price a small basket.

Demonstrates the Stage-1 generators (snapshot + correlation), the
Stage-4 portfolio sampler, and the Stage-5 batched analytic pricer.
No external data required — only ``valax``.

Run::

    python examples/07_synthetic_market.py
"""

from __future__ import annotations

import jax.numpy as jnp

import valax
from valax.market.synthetic import (
    OptionPortfolioSpec,
    SeedRegistry,
    SyntheticMarketConfig,
    evolve_market,
    sample_market_with_correlation,
    sample_option_portfolio,
)
from valax.portfolio.batch import batch_price
from valax.pricing.analytic.black_scholes import black_scholes_price


def main() -> None:
    registry = SeedRegistry(
        master_seed=20260101, library_version=valax.__version__
    )
    cfg = SyntheticMarketConfig(n_assets=4)

    # --- Stage 1 : snapshot + correlation ---
    md, corr = sample_market_with_correlation(registry, cfg)
    print("== Stage 1 — synthetic snapshot ==")
    print(f"  spots:     {md.spots}")
    print(f"  vols:      {md.vols}")
    print(f"  dividends: {md.dividends}")
    print(f"  corr min eig: "
          f"{float(jnp.min(jnp.linalg.eigvalsh(corr))):.4f}")
    print(f"  curve has {md.discount_curve.pillar_dates.shape[0]} pillars")

    # --- Stage 4 : option portfolio ---
    spec = OptionPortfolioSpec(n_per_asset=5, call_probability=0.7)
    port = sample_option_portfolio(registry, md, spec)
    calls, call_idx = port["calls"]
    puts, put_idx = port["puts"]
    print(f"\n== Stage 4 — portfolio ==")
    print(f"  calls: {calls.strike.shape[0]} legs")
    print(f"  puts : {puts.strike.shape[0]} legs")

    # --- Stage 5 : batched analytic pricing ---
    # Flat rate read from the curve.
    longest_expiry_date = (
        md.discount_curve.reference_date
        + jnp.int32(int(2 * 365))
    )
    df = md.discount_curve(longest_expiry_date)
    flat_rate = -jnp.log(df) / 2.0

    def price_stack(stack, idx):
        n = stack.strike.shape[0]
        if n == 0:
            return jnp.zeros(0)
        spots = md.spots[idx]
        vols = md.vols[idx]
        rates = jnp.full((n,), float(flat_rate))
        divs = md.dividends[idx]
        return batch_price(
            black_scholes_price, stack, spots, vols, rates, divs,
        )

    call_prices = price_stack(calls, call_idx)
    put_prices = price_stack(puts, put_idx)
    print(f"\n== Stage 5 — analytic prices ==")
    print(f"  total call PV: {float(jnp.sum(call_prices)):.4f}")
    print(f"  total put  PV: {float(jnp.sum(put_prices)):.4f}")

    # --- Stage 6 : evolve a small tape ---
    dates = md.discount_curve.reference_date + jnp.array(
        [0, 30, 90, 180, 365], dtype=jnp.int32,
    )
    tape = evolve_market(registry, md, dates, corr)
    print(f"\n== Stage 6 — tape ({dates.shape[0]} dates) ==")
    for i in range(dates.shape[0]):
        print(f"  t{i}: spots = {tape.spots[i]}")


if __name__ == "__main__":
    main()
