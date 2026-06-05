"""Sanity tests for the arbitrage *injection* functions.

These tests only verify that the injectors produce objects whose
arbitrage diagnostic is empirically verifiable (e.g., an injected
non-PSD matrix really does have a negative eigenvalue).  They do
*not* check how the rest of the library reacts to those broken
objects — that is the job of ``test_arbitrage_handling.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from valax.market.synthetic import (
    inject_basket_variance_violation,
    inject_butterfly_arb,
    inject_calendar_arb,
    inject_inconsistent_bootstrap_quotes,
    inject_negative_density,
    inject_non_convex_smile,
    inject_non_psd_correlation,
    inject_pcp_violation,
    sample_correlation,
)


pytestmark = pytest.mark.arbitrage


class TestNonPSDCorrelation:
    def test_injection_produces_negative_eigenvalue(self, seed_registry):
        c = sample_correlation(seed_registry, n=6)
        bad, diag = inject_non_psd_correlation(c, eps=0.5)
        min_eig = float(jnp.min(jnp.linalg.eigvalsh(bad)))
        assert min_eig < 0
        assert diag.kind == "non_psd_correlation"
        assert diag.magnitude < 0

    def test_diagonal_remains_unit(self, seed_registry):
        c = sample_correlation(seed_registry, n=4)
        bad, _ = inject_non_psd_correlation(c, eps=0.3)
        assert bool(jnp.allclose(jnp.diag(bad), 1.0))


class TestSmileInjections:
    def test_butterfly_bumps_one_strike(self):
        vols = jnp.full((5,), 0.2)
        strikes = jnp.linspace(80.0, 120.0, 5)
        bad, diag = inject_butterfly_arb(strikes, vols, k_index=2, bump=-0.05)
        assert float(bad[2]) == pytest.approx(0.15)
        # All other strikes are untouched.
        for i in [0, 1, 3, 4]:
            assert float(bad[i]) == pytest.approx(0.2)
        assert diag.kind == "butterfly"

    def test_non_convex_bumps_up(self):
        vols = jnp.full((5,), 0.2)
        strikes = jnp.linspace(80.0, 120.0, 5)
        bad, diag = inject_non_convex_smile(strikes, vols, k_index=2, bump=0.1)
        assert float(bad[2]) == pytest.approx(0.3)
        assert diag.kind == "non_convex_smile"

    def test_calendar_swaps_variances(self):
        w = jnp.array([0.04, 0.08, 0.12, 0.16])
        bad, diag = inject_calendar_arb(w, i=1, j=3)
        # After swap, position 1 has the value that was at position 3.
        assert float(bad[1]) == pytest.approx(0.16)
        assert float(bad[3]) == pytest.approx(0.08)
        # Now monotonicity is broken between positions 1 and 2.
        assert bool(jnp.any(bad[1:] < bad[:-1]))
        assert diag.kind == "calendar"


class TestPriceStripInjections:
    def test_pcp_violation_inflates_calls_only(self):
        calls = jnp.array([10.0, 5.0, 1.0])
        puts = jnp.array([1.0, 5.0, 10.0])
        (bad_calls, untouched), diag = inject_pcp_violation(calls, puts, bp=100.0)
        # 100 bp = 1% inflation on each call.
        assert jnp.allclose(bad_calls, calls * 1.01)
        assert jnp.array_equal(untouched, puts)
        assert diag.kind == "put_call_parity"

    def test_negative_density_spike(self):
        prices = jnp.array([20.0, 10.0, 5.0, 2.0])
        strikes = jnp.linspace(80.0, 110.0, 4)
        bad, diag = inject_negative_density(strikes, prices, k_index=1, bump=0.5)
        assert float(bad[1]) == pytest.approx(15.0)
        assert diag.kind == "negative_density"


class TestQuoteInjections:
    def test_inconsistent_bootstrap_offset(self):
        quotes = jnp.array([0.01, 0.02, 0.03])
        bad, diag = inject_inconsistent_bootstrap_quotes(
            quotes, bp_offset=25.0, index=1
        )
        assert float(bad[1]) == pytest.approx(0.0225)
        assert diag.kind == "inconsistent_quotes"


class TestBasketVariance:
    def test_off_diagonal_overshoots(self, seed_registry):
        c = sample_correlation(seed_registry, n=4)
        bad, diag = inject_basket_variance_violation(c, 0, 1, new_value=1.5)
        assert float(bad[0, 1]) == 1.5
        assert float(bad[1, 0]) == 1.5
        assert diag.kind == "basket_variance"

    def test_rejects_diagonal_index(self, seed_registry):
        c = sample_correlation(seed_registry, n=4)
        with pytest.raises(ValueError, match="i and j must differ"):
            inject_basket_variance_violation(c, 2, 2)
