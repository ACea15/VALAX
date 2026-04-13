"""Tests for spread option pricing: Margrabe and Kirk's approximation."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import SpreadOption
from valax.pricing.analytic.spread import margrabe_price, kirk_price, spread_option_price


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def exchange_option():
    """K=0 exchange option (Margrabe case)."""
    return SpreadOption(
        expiry=jnp.array(1.0), strike=jnp.array(0.0), notional=jnp.array(1.0),
    )


@pytest.fixture
def spread_call():
    return SpreadOption(
        expiry=jnp.array(1.0), strike=jnp.array(5.0), notional=jnp.array(1.0),
    )


@pytest.fixture
def spread_put():
    return SpreadOption(
        expiry=jnp.array(1.0), strike=jnp.array(5.0), notional=jnp.array(1.0),
        is_call=False,
    )


@pytest.fixture
def market():
    return dict(
        s1=jnp.array(100.0), s2=jnp.array(95.0),
        vol1=jnp.array(0.20), vol2=jnp.array(0.25),
        rho=jnp.array(0.50), rate=jnp.array(0.05),
    )


# ── Margrabe ─────────────────────────────────────────────────────────

class TestMargrabe:
    def test_positive_for_atm(self, exchange_option):
        """ATM exchange option has positive value from volatility."""
        p = margrabe_price(
            exchange_option, jnp.array(100.0), jnp.array(100.0),
            jnp.array(0.20), jnp.array(0.25), jnp.array(0.50),
        )
        assert float(p) > 0.0

    def test_deep_itm(self, exchange_option):
        """S1 >> S2 → call ≈ S1 - S2."""
        p = margrabe_price(
            exchange_option, jnp.array(200.0), jnp.array(100.0),
            jnp.array(0.20), jnp.array(0.25), jnp.array(0.50),
        )
        assert float(p) == pytest.approx(100.0, rel=0.01)

    def test_put_call_parity(self, exchange_option):
        """C - P = S1 - S2 for K=0 (Margrabe parity, q1=q2=0)."""
        s1, s2 = jnp.array(100.0), jnp.array(95.0)
        args = (jnp.array(0.20), jnp.array(0.25), jnp.array(0.50))
        c = margrabe_price(exchange_option, s1, s2, *args)
        p = margrabe_price(
            SpreadOption(expiry=jnp.array(1.0), strike=jnp.array(0.0),
                         notional=jnp.array(1.0), is_call=False),
            s1, s2, *args,
        )
        assert float(c - p) == pytest.approx(float(s1 - s2), abs=1e-10)

    def test_increases_with_vol(self, exchange_option):
        """Higher spread vol → higher option value."""
        # Fix vol2=0.25, rho=0, increase vol1
        p_low = margrabe_price(
            exchange_option, jnp.array(100.0), jnp.array(100.0),
            jnp.array(0.10), jnp.array(0.10), jnp.array(0.0),
        )
        p_high = margrabe_price(
            exchange_option, jnp.array(100.0), jnp.array(100.0),
            jnp.array(0.30), jnp.array(0.30), jnp.array(0.0),
        )
        assert float(p_high) > float(p_low)

    def test_higher_corr_lower_price(self, exchange_option):
        """Higher correlation → lower spread vol → lower option value."""
        p_low_rho = margrabe_price(
            exchange_option, jnp.array(100.0), jnp.array(100.0),
            jnp.array(0.20), jnp.array(0.20), jnp.array(0.0),
        )
        p_high_rho = margrabe_price(
            exchange_option, jnp.array(100.0), jnp.array(100.0),
            jnp.array(0.20), jnp.array(0.20), jnp.array(0.90),
        )
        assert float(p_high_rho) < float(p_low_rho)

    def test_jit_compatible(self, exchange_option):
        args = (jnp.array(100.0), jnp.array(95.0),
                jnp.array(0.20), jnp.array(0.25), jnp.array(0.50))
        eager = margrabe_price(exchange_option, *args)
        jitted = jax.jit(margrabe_price)(exchange_option, *args)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_s1_positive(self, exchange_option):
        """Delta wrt S1 > 0 for a call."""
        def price_s1(s):
            return margrabe_price(
                exchange_option, s, jnp.array(100.0),
                jnp.array(0.20), jnp.array(0.25), jnp.array(0.50),
            )
        delta = jax.grad(price_s1)(jnp.array(100.0))
        assert 0.0 < float(delta) < 1.0


# ── Kirk ─────────────────────────────────────────────────────────────

class TestKirk:
    def test_agrees_with_margrabe_at_k0(self, exchange_option):
        """Kirk at K=0 should match Margrabe exactly."""
        args = (jnp.array(100.0), jnp.array(100.0),
                jnp.array(0.20), jnp.array(0.25), jnp.array(0.50))
        m = margrabe_price(exchange_option, *args)
        k = kirk_price(exchange_option, *args, jnp.array(0.05))
        assert float(k) == pytest.approx(float(m), rel=1e-6)

    def test_put_call_parity(self, spread_call, spread_put, market):
        """C - P = disc * (F1 - F2 - K)."""
        c = kirk_price(spread_call, **market)
        p = kirk_price(spread_put, **market)
        T = float(spread_call.expiry)
        r = float(market['rate'])
        F1 = float(market['s1']) * jnp.exp(r * T)
        F2 = float(market['s2']) * jnp.exp(r * T)
        K = float(spread_call.strike)
        expected = float(jnp.exp(-r * T)) * (F1 - F2 - K)
        assert float(c - p) == pytest.approx(expected, rel=1e-10)

    def test_call_positive(self, spread_call, market):
        p = kirk_price(spread_call, **market)
        assert float(p) > 0.0

    def test_put_positive(self, spread_put, market):
        p = kirk_price(spread_put, **market)
        assert float(p) > 0.0

    def test_higher_corr_lower_call_price(self, spread_call, market):
        """Higher correlation → smaller spread vol → cheaper option."""
        p_low = kirk_price(spread_call, market['s1'], market['s2'],
                           market['vol1'], market['vol2'], jnp.array(0.0),
                           market['rate'])
        p_high = kirk_price(spread_call, market['s1'], market['s2'],
                            market['vol1'], market['vol2'], jnp.array(0.90),
                            market['rate'])
        assert float(p_high) < float(p_low)

    def test_jit_compatible(self, spread_call, market):
        eager = kirk_price(spread_call, **market)
        jitted = jax.jit(kirk_price)(spread_call, **market)
        assert float(jitted) == pytest.approx(float(eager), rel=1e-10)

    def test_grad_wrt_s1_positive(self, spread_call, market):
        def price_s1(s):
            return kirk_price(spread_call, s, market['s2'], market['vol1'],
                              market['vol2'], market['rho'], market['rate'])
        delta = jax.grad(price_s1)(market['s1'])
        assert float(delta) > 0.0

    def test_grad_wrt_rho(self, spread_call, market):
        """Spread call vega-rho: dC/d(rho) < 0."""
        def price_rho(r):
            return kirk_price(spread_call, market['s1'], market['s2'],
                              market['vol1'], market['vol2'], r, market['rate'])
        drho = jax.grad(price_rho)(market['rho'])
        assert float(drho) < 0.0

    def test_vmap_across_s1(self, spread_call, market):
        s1s = jnp.linspace(80.0, 120.0, 9)
        def price_one(s):
            return kirk_price(spread_call, s, market['s2'], market['vol1'],
                              market['vol2'], market['rho'], market['rate'])
        batch = jax.vmap(price_one)(s1s)
        assert batch.shape == (9,)
        # Monotonically increasing in S1 for a call
        assert jnp.all(jnp.diff(batch) > 0.0)


# ── Convenience wrapper ──────────────────────────────────────────────

class TestSpreadOptionPrice:
    def test_dispatches_to_kirk(self, spread_call, market):
        p1 = spread_option_price(spread_call, **market)
        p2 = kirk_price(spread_call, **market)
        assert float(p1) == pytest.approx(float(p2), abs=1e-12)
