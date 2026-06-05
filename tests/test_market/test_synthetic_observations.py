"""Tests for the synthetic observation layer (noisy quote synthesis)."""

import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.market.synthetic import (
    sample_sabr_params,
    synthesize_curve_quotes,
    synthesize_price_strip,
    synthesize_sabr_smile,
)
from valax.pricing.analytic.black_scholes import black_scholes_price


class TestSynthesizeSABRSmile:
    def test_zero_noise_is_clean(self, seed_registry, default_synth_cfg):
        sabr = sample_sabr_params(seed_registry, default_synth_cfg)
        strikes = jnp.linspace(80.0, 120.0, 9)
        clean = synthesize_sabr_smile(
            seed_registry, sabr, jnp.array(100.0), jnp.array(1.0),
            strikes, vol_bp_noise=0.0,
        )
        # Calling twice with vol_bp_noise=0 returns the same numbers.
        clean2 = synthesize_sabr_smile(
            seed_registry, sabr, jnp.array(100.0), jnp.array(1.0),
            strikes, vol_bp_noise=0.0,
        )
        assert jnp.allclose(clean, clean2)

    def test_noise_level_is_in_bp(self, seed_registry, default_synth_cfg):
        sabr = sample_sabr_params(seed_registry, default_synth_cfg)
        strikes = jnp.linspace(80.0, 120.0, 101)
        clean = synthesize_sabr_smile(
            seed_registry, sabr, jnp.array(100.0), jnp.array(1.0),
            strikes, vol_bp_noise=0.0,
        )
        noisy = synthesize_sabr_smile(
            seed_registry, sabr, jnp.array(100.0), jnp.array(1.0),
            strikes, vol_bp_noise=20.0,  # 20 bp
        )
        rms = float(jnp.sqrt(jnp.mean((noisy - clean) ** 2)))
        # 20 bp = 2e-3 in vol; allow a generous range to account for
        # finite-sample variation.
        assert 1e-3 < rms < 4e-3


class TestSynthesizePriceStrip:
    def test_zero_noise_matches_pricer(self, seed_registry):
        strikes = jnp.array([90.0, 100.0, 110.0])
        T = jnp.array(1.0)
        spot = jnp.array(100.0); vol = jnp.array(0.20)
        r = jnp.array(0.04); q = jnp.array(0.01)

        def args_for(K):
            opt = EuropeanOption(strike=K, expiry=T, is_call=True)
            return (opt, spot, vol, r, q)

        prices = synthesize_price_strip(
            seed_registry, black_scholes_price, args_for, strikes,
            price_rel_noise=0.0,
        )
        # Direct call should match.
        for i, K in enumerate(strikes):
            expected = black_scholes_price(*args_for(K))
            assert float(prices[i]) == pytest.approx(float(expected))


class TestSynthesizeCurveQuotes:
    def test_zero_noise_is_identity(self, seed_registry):
        par = jnp.array([0.01, 0.02, 0.03])
        out = synthesize_curve_quotes(seed_registry, par, bp_noise=0.0)
        assert jnp.array_equal(out, par)

    def test_noise_magnitude_in_bp(self, seed_registry):
        par = jnp.zeros(200)
        noisy = synthesize_curve_quotes(seed_registry, par, bp_noise=5.0)
        std = float(jnp.std(noisy))
        # 5 bp = 5e-4; finite-sample tolerance.
        assert 3e-4 < std < 8e-4
