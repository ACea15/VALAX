"""Comprehensive tests for autodiff Greeks against analytical closed-forms.

All analytical Greeks are derived independently from the pricing functions
to ensure the autodiff machinery is correct, not just self-consistent.

Black-Scholes analytical Greeks reference:
    S = spot, K = strike, T = expiry, σ = vol, r = rate, q = dividend
    d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T
    df = exp(-rT), qf = exp(-qT)
    n(x) = standard normal PDF, N(x) = standard normal CDF

    Call:
        delta   = qf * N(d1)
        gamma   = qf * n(d1) / (S * σ * √T)
        vega    = S * qf * n(d1) * √T
        theta   = -(S * qf * n(d1) * σ) / (2√T) - r*K*df*N(d2) + q*S*qf*N(d1)
                  (per-year theta; we test per-day = theta/365)
        rho     = K * T * df * N(d2)
        div_rho = -S * T * qf * N(d1)
        vanna   = -qf * n(d1) * d2 / σ
        volga   = S * qf * n(d1) * √T * d1 * d2 / σ

    Put: use put-call symmetry for delta, rho, div_rho; gamma/vega/vanna/volga same.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import pytest

from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.black76 import black76_price
from valax.pricing.analytic.bachelier import bachelier_price
from valax.greeks.autodiff import greek, greeks


# ── Helpers ───────────────────────────────────────────────────────────

def _bs_params(S, K, T, sigma, r, q):
    """Compute d1, d2, df, qf, n(d1), N(d1), N(d2)."""
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r * T)
    qf = jnp.exp(-q * T)
    nd1 = stats.norm.pdf(d1)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    return d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T


def _b76_params(F, K, T, sigma, r):
    """Compute d1, d2, df for Black-76."""
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = jnp.exp(-r * T)
    nd1 = stats.norm.pdf(d1)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    return d1, d2, df, nd1, Nd1, Nd2, sqrt_T


# ── Test parameters ──────────────────────────────────────────────────

# Multiple parameter sets to exercise different moneyness/vol regimes
BS_PARAMS = [
    # (S, K, T, sigma, r, q, is_call) — label
    (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, True),    # ATM call
    (100.0, 100.0, 1.0, 0.20, 0.05, 0.02, False),   # ATM put
    (100.0, 110.0, 0.5, 0.30, 0.03, 0.01, True),    # OTM call, high vol
    (100.0, 90.0, 0.5, 0.30, 0.03, 0.01, False),    # OTM put, high vol
    (100.0, 80.0, 2.0, 0.15, 0.08, 0.00, True),     # deep ITM call, long dated
    (50.0, 55.0, 0.25, 0.40, 0.01, 0.03, True),     # OTM call, short dated, high div
]

TOL = 1e-10  # float64 tolerance


# ── Black-Scholes Greeks ─────────────────────────────────────────────

class TestBSAnalyticalGreeks:
    """Test every BS Greek against its closed-form."""

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_delta(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        if is_call:
            analytical = qf * Nd1
        else:
            analytical = -qf * stats.norm.cdf(-d1)

        autodiff = greek(black_scholes_price, "delta", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_gamma(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        # Gamma is the same for calls and puts
        analytical = qf * nd1 / (S * sigma * sqrt_T)

        autodiff = greek(black_scholes_price, "gamma", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_vega(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        # Vega is the same for calls and puts
        analytical = S * qf * nd1 * sqrt_T

        autodiff = greek(black_scholes_price, "vega", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_rho(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        if is_call:
            analytical = K * T * df * Nd2
        else:
            analytical = -K * T * df * stats.norm.cdf(-d2)

        autodiff = greek(black_scholes_price, "rho", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_dividend_rho(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        if is_call:
            analytical = -S * T * qf * Nd1
        else:
            analytical = S * T * qf * stats.norm.cdf(-d1)

        autodiff = greek(black_scholes_price, "dividend_rho", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_vanna(self, S, K, T, sigma, r, q, is_call):
        """Vanna = d(delta)/d(vol) = -qf * n(d1) * d2 / sigma. Same for call/put."""
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        analytical = -qf * nd1 * d2 / sigma

        autodiff = greek(black_scholes_price, "vanna", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_volga(self, S, K, T, sigma, r, q, is_call):
        """Volga = d(vega)/d(vol) = S * qf * n(d1) * sqrt(T) * d1 * d2 / sigma."""
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        analytical = S * qf * nd1 * sqrt_T * d1 * d2 / sigma

        autodiff = greek(black_scholes_price, "volga", option,
                         jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))
        assert abs(float(autodiff) - float(analytical)) < TOL


# ── Black-Scholes theta (bump-based, looser tolerance) ───────────────

class TestBSTheta:
    """Theta uses finite difference, so we compare against the analytical formula
    and accept a wider tolerance proportional to dt."""

    @pytest.mark.parametrize("S,K,T,sigma,r,q,is_call", BS_PARAMS)
    def test_theta(self, S, K, T, sigma, r, q, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, qf, nd1, Nd1, Nd2, sqrt_T = _bs_params(S, K, T, sigma, r, q)

        # Analytical theta (per year)
        common = -(S * sigma * qf * nd1) / (2.0 * sqrt_T)
        if is_call:
            analytical_per_year = common - r * K * df * Nd2 + q * S * qf * Nd1
        else:
            analytical_per_year = common + r * K * df * stats.norm.cdf(-d2) - q * S * qf * stats.norm.cdf(-d1)

        # _theta bumps by dt=1/365 and divides by dt, so it returns per-year theta
        autodiff_theta = greek(black_scholes_price, "theta", option,
                               jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        # Bump theta has O(dt) error; use relative tolerance
        err = abs(float(autodiff_theta) - float(analytical_per_year))
        scale = max(abs(float(analytical_per_year)), 1e-4)
        assert err / scale < 5e-3, f"theta error {err} vs analytical {float(analytical_per_year)}"


# ── Black-76 Greeks ──────────────────────────────────────────────────

B76_PARAMS = [
    # (F, K, T, sigma, r, is_call)
    (100.0, 100.0, 0.5, 0.25, 0.03, True),    # ATM call
    (100.0, 100.0, 0.5, 0.25, 0.03, False),   # ATM put
    (100.0, 110.0, 1.0, 0.30, 0.05, True),    # OTM call
    (100.0, 90.0, 1.0, 0.30, 0.05, False),    # OTM put
]


class TestB76AnalyticalGreeks:
    """Black-76 Greeks: delta = df*N(d1), gamma = df*n(d1)/(F*σ√T), vega = df*F*n(d1)*√T."""

    @pytest.mark.parametrize("F,K,T,sigma,r,is_call", B76_PARAMS)
    def test_delta(self, F, K, T, sigma, r, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, nd1, Nd1, Nd2, sqrt_T = _b76_params(F, K, T, sigma, r)

        if is_call:
            analytical = df * Nd1
        else:
            analytical = -df * stats.norm.cdf(-d1)

        autodiff = greek(black76_price, "delta", option,
                         jnp.array(F), jnp.array(sigma), jnp.array(r))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("F,K,T,sigma,r,is_call", B76_PARAMS)
    def test_gamma(self, F, K, T, sigma, r, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, nd1, Nd1, Nd2, sqrt_T = _b76_params(F, K, T, sigma, r)

        analytical = df * nd1 / (F * sigma * sqrt_T)

        autodiff = greek(black76_price, "gamma", option,
                         jnp.array(F), jnp.array(sigma), jnp.array(r))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("F,K,T,sigma,r,is_call", B76_PARAMS)
    def test_vega(self, F, K, T, sigma, r, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        d1, d2, df, nd1, Nd1, Nd2, sqrt_T = _b76_params(F, K, T, sigma, r)

        analytical = df * F * nd1 * sqrt_T

        autodiff = greek(black76_price, "vega", option,
                         jnp.array(F), jnp.array(sigma), jnp.array(r))
        assert abs(float(autodiff) - float(analytical)) < TOL


# ── Bachelier Greeks ─────────────────────────────────────────────────

BACH_PARAMS = [
    # (F, K, T, sigma_n, r, is_call)
    (100.0, 100.0, 1.0, 20.0, 0.02, True),    # ATM call
    (100.0, 100.0, 1.0, 20.0, 0.02, False),   # ATM put
    (100.0, 110.0, 0.5, 15.0, 0.03, True),    # OTM call
    (100.0, 90.0, 0.5, 15.0, 0.03, False),    # OTM put
]


class TestBachelierAnalyticalGreeks:
    """Bachelier Greeks:
        d = (F - K) / (σ√T)
        Call delta = df * N(d)
        Gamma = df * n(d) / (σ√T)
        Vega = df * √T * n(d)
    """

    @pytest.mark.parametrize("F,K,T,sigma,r,is_call", BACH_PARAMS)
    def test_delta(self, F, K, T, sigma, r, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        sqrt_T = jnp.sqrt(T)
        d = (F - K) / (sigma * sqrt_T)
        df = jnp.exp(-r * T)

        if is_call:
            analytical = df * stats.norm.cdf(d)
        else:
            analytical = -df * stats.norm.cdf(-d)

        autodiff = greek(bachelier_price, "delta", option,
                         jnp.array(F), jnp.array(sigma), jnp.array(r))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("F,K,T,sigma,r,is_call", BACH_PARAMS)
    def test_gamma(self, F, K, T, sigma, r, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        sqrt_T = jnp.sqrt(T)
        d = (F - K) / (sigma * sqrt_T)
        df = jnp.exp(-r * T)

        # Gamma same for call/put
        analytical = df * stats.norm.pdf(d) / (sigma * sqrt_T)

        autodiff = greek(bachelier_price, "gamma", option,
                         jnp.array(F), jnp.array(sigma), jnp.array(r))
        assert abs(float(autodiff) - float(analytical)) < TOL

    @pytest.mark.parametrize("F,K,T,sigma,r,is_call", BACH_PARAMS)
    def test_vega(self, F, K, T, sigma, r, is_call):
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=is_call)
        sqrt_T = jnp.sqrt(T)
        d = (F - K) / (sigma * sqrt_T)
        df = jnp.exp(-r * T)

        # Vega same for call/put
        analytical = df * sqrt_T * stats.norm.pdf(d)

        autodiff = greek(bachelier_price, "vega", option,
                         jnp.array(F), jnp.array(sigma), jnp.array(r))
        assert abs(float(autodiff) - float(analytical)) < TOL


# ── Cross-model consistency ──────────────────────────────────────────

class TestCrossModelConsistency:
    """Verify relationships that must hold across models."""

    def test_bs_put_call_delta_relation(self):
        """delta_call - delta_put = exp(-qT) for BS."""
        S, K, T, sigma, r, q = 100.0, 105.0, 1.0, 0.25, 0.05, 0.02
        call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)
        args = (jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        dc = greek(black_scholes_price, "delta", call, *args)
        dp = greek(black_scholes_price, "delta", put, *args)
        assert abs(float(dc - dp) - float(jnp.exp(-q * T))) < TOL

    def test_bs_call_put_gamma_equal(self):
        """Gamma is the same for call and put."""
        S, K, T, sigma, r, q = 100.0, 95.0, 0.5, 0.30, 0.04, 0.01
        call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)
        args = (jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        gc = greek(black_scholes_price, "gamma", call, *args)
        gp = greek(black_scholes_price, "gamma", put, *args)
        assert abs(float(gc) - float(gp)) < TOL

    def test_bs_call_put_vega_equal(self):
        """Vega is the same for call and put."""
        S, K, T, sigma, r, q = 100.0, 105.0, 1.0, 0.20, 0.05, 0.02
        call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)
        args = (jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        vc = greek(black_scholes_price, "vega", call, *args)
        vp = greek(black_scholes_price, "vega", put, *args)
        assert abs(float(vc) - float(vp)) < TOL

    def test_bs_call_put_vanna_equal(self):
        """Vanna is the same for call and put."""
        S, K, T, sigma, r, q = 100.0, 100.0, 1.0, 0.20, 0.05, 0.02
        call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)
        args = (jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        vc = greek(black_scholes_price, "vanna", call, *args)
        vp = greek(black_scholes_price, "vanna", put, *args)
        assert abs(float(vc) - float(vp)) < TOL

    def test_greeks_dict_consistency(self):
        """greeks() dict values match individual greek() calls."""
        S, K, T, sigma, r, q = 100.0, 100.0, 1.0, 0.20, 0.05, 0.02
        option = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        args = (jnp.array(S), jnp.array(sigma), jnp.array(r), jnp.array(q))

        result = greeks(black_scholes_price, option, *args)

        for name in ["delta", "gamma", "vega", "vanna", "volga", "rho", "dividend_rho", "theta"]:
            individual = greek(black_scholes_price, name, option, *args)
            assert abs(float(result[name]) - float(individual)) < TOL, \
                f"{name}: dict={float(result[name])}, individual={float(individual)}"

    def test_b76_put_call_delta_relation(self):
        """delta_call - delta_put = df for Black-76."""
        F, K, T, sigma, r = 100.0, 105.0, 1.0, 0.25, 0.03
        call = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=True)
        put = EuropeanOption(strike=jnp.array(K), expiry=jnp.array(T), is_call=False)
        args = (jnp.array(F), jnp.array(sigma), jnp.array(r))

        dc = greek(black76_price, "delta", call, *args)
        dp = greek(black76_price, "delta", put, *args)
        assert abs(float(dc - dp) - float(jnp.exp(-r * T))) < TOL
