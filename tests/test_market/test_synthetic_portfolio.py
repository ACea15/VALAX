"""Tests for synthetic portfolio samplers."""

import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.instruments.rates import InterestRateSwap
from valax.market.synthetic import (
    OptionPortfolioSpec,
    SwapPortfolioSpec,
    sample_market_data,
    sample_option_portfolio,
    sample_swap_portfolio,
)


class TestSampleOptionPortfolio:
    def test_returns_calls_and_puts(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        port = sample_option_portfolio(
            seed_registry, md,
            OptionPortfolioSpec(n_per_asset=6, call_probability=0.5),
        )
        assert set(port) == {"calls", "puts"}
        for tag in ("calls", "puts"):
            stack, idx = port[tag]
            assert isinstance(stack, EuropeanOption)
            assert stack.strike.shape == stack.expiry.shape
            assert stack.strike.shape == idx.shape

    def test_total_legs_matches_spec(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        spec = OptionPortfolioSpec(n_per_asset=5)
        port = sample_option_portfolio(seed_registry, md, spec)
        total = port["calls"][0].strike.shape[0] + port["puts"][0].strike.shape[0]
        assert total == default_synth_cfg.n_assets * spec.n_per_asset

    def test_strikes_positive(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        port = sample_option_portfolio(seed_registry, md)
        for tag in ("calls", "puts"):
            stack, _ = port[tag]
            if stack.strike.shape[0] > 0:
                assert bool(jnp.all(stack.strike > 0))
                assert bool(jnp.all(stack.expiry > 0))

    def test_calls_only_when_p1(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        port = sample_option_portfolio(
            seed_registry, md,
            OptionPortfolioSpec(n_per_asset=4, call_probability=1.0),
        )
        assert port["puts"][0].strike.shape[0] == 0
        assert port["calls"][0].strike.shape[0] > 0

    def test_batch_pricer_compatible(self, seed_registry, default_synth_cfg):
        """Stack should plug into batch_price without surgery."""
        from valax.portfolio.batch import batch_price
        from valax.pricing.analytic.black_scholes import black_scholes_price

        md = sample_market_data(seed_registry, default_synth_cfg)
        port = sample_option_portfolio(
            seed_registry, md,
            OptionPortfolioSpec(n_per_asset=4, call_probability=1.0),
        )
        calls, idx = port["calls"]
        spots = md.spots[idx]
        vols = md.vols[idx]
        divs = md.dividends[idx]
        # Use a flat rate read from the curve at the longest expiry.
        rates = jnp.full_like(spots, 0.03)
        prices = batch_price(
            black_scholes_price, calls, spots, vols, rates, divs,
        )
        assert prices.shape == calls.strike.shape
        assert bool(jnp.all(prices >= 0))


class TestSampleSwapPortfolio:
    def test_returns_list_of_swaps(self, seed_registry, default_synth_cfg):
        md = sample_market_data(seed_registry, default_synth_cfg)
        swaps = sample_swap_portfolio(
            seed_registry, md.discount_curve,
            SwapPortfolioSpec(n_swaps=4),
        )
        assert len(swaps) == 4
        for s in swaps:
            assert isinstance(s, InterestRateSwap)
            assert s.fixed_dates.shape[0] >= 1
            assert float(s.fixed_rate) > 0
            assert float(s.notional) > 0
