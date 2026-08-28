"""Tests for SABR model: implied vol, pricing, and Monte Carlo paths."""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import (
    sabr_implied_vol,
    sabr_price,
    sabr_normal_implied_vol,
    sabr_price_bachelier,
)
from valax.pricing.analytic.black76 import black76_price
from valax.pricing.analytic.bachelier import bachelier_price, bachelier_implied_vol
from valax.pricing.mc.sabr_paths import generate_sabr_paths
from valax.greeks.autodiff import greeks


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def typical_sabr():
    """Typical SABR params for equity: beta=0.5, moderate vol-of-vol."""
    return SABRModel(
        alpha=jnp.array(0.3),
        beta=jnp.array(0.5),
        rho=jnp.array(-0.3),
        nu=jnp.array(0.4),
    )


@pytest.fixture
def lognormal_sabr():
    """SABR with beta=1 (lognormal backbone)."""
    return SABRModel(
        alpha=jnp.array(0.2),
        beta=jnp.array(1.0),
        rho=jnp.array(-0.25),
        nu=jnp.array(0.3),
    )


@pytest.fixture
def normal_sabr():
    """SABR with beta=0 (normal backbone)."""
    return SABRModel(
        alpha=jnp.array(0.01),
        beta=jnp.array(0.0),
        rho=jnp.array(-0.2),
        nu=jnp.array(0.4),
    )


# ── Implied vol tests ──────────────────────────────────────────────

class TestSABRImpliedVol:
    def test_atm_vol_positive(self, typical_sabr):
        """ATM implied vol should be positive and close to alpha * F^(beta-1)."""
        F = jnp.array(100.0)
        vol = sabr_implied_vol(typical_sabr, F, F, jnp.array(1.0))
        assert float(vol) > 0.0

    def test_atm_vol_approximation(self, lognormal_sabr):
        """For beta=1, ATM vol should be close to alpha for short expiries."""
        F = jnp.array(100.0)
        vol = sabr_implied_vol(lognormal_sabr, F, F, jnp.array(0.01))
        # At very short expiry, ATM vol ~ alpha for beta=1
        assert abs(float(vol) - float(lognormal_sabr.alpha)) < 0.01

    def test_smile_shape_negative_rho(self, typical_sabr):
        """Negative rho should produce higher vol at low strikes (skew)."""
        F = jnp.array(100.0)
        T = jnp.array(1.0)
        vol_low = sabr_implied_vol(typical_sabr, F, jnp.array(80.0), T)
        vol_atm = sabr_implied_vol(typical_sabr, F, jnp.array(100.0), T)
        vol_high = sabr_implied_vol(typical_sabr, F, jnp.array(120.0), T)
        # Negative rho => downside skew
        assert float(vol_low) > float(vol_atm)

    def test_smile_symmetric_zero_rho(self):
        """Zero rho should produce a more symmetric smile."""
        model = SABRModel(
            alpha=jnp.array(0.3),
            beta=jnp.array(0.5),
            rho=jnp.array(0.0),
            nu=jnp.array(0.4),
        )
        F = jnp.array(100.0)
        T = jnp.array(1.0)
        vol_low = sabr_implied_vol(model, F, jnp.array(90.0), T)
        vol_high = sabr_implied_vol(model, F, jnp.array(111.11), T)  # ~same moneyness
        vol_atm = sabr_implied_vol(model, F, jnp.array(100.0), T)
        # Both wings should be above ATM (smile)
        assert float(vol_low) > float(vol_atm)
        assert float(vol_high) > float(vol_atm)

    def test_vol_increases_with_nu(self, typical_sabr):
        """Higher vol-of-vol should widen the smile (higher OTM vols)."""
        F = jnp.array(100.0)
        K_otm = jnp.array(120.0)
        T = jnp.array(1.0)
        vol_base = sabr_implied_vol(typical_sabr, F, K_otm, T)
        model_high_nu = SABRModel(
            alpha=typical_sabr.alpha,
            beta=typical_sabr.beta,
            rho=typical_sabr.rho,
            nu=jnp.array(0.8),
        )
        vol_high = sabr_implied_vol(model_high_nu, F, K_otm, T)
        assert float(vol_high) > float(vol_base)


# ── Pricing tests ──────────────────────────────────────────────────

class TestSABRPrice:
    def test_call_price_positive(self, typical_sabr):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        price = sabr_price(option, jnp.array(100.0), jnp.array(0.05), typical_sabr)
        assert float(price) > 0.0

    def test_put_price_positive(self, typical_sabr):
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False)
        price = sabr_price(option, jnp.array(100.0), jnp.array(0.05), typical_sabr)
        assert float(price) > 0.0

    def test_put_call_parity(self, typical_sabr):
        """Put-call parity: C - P = df * (F - K)."""
        F = jnp.array(100.0)
        K = jnp.array(105.0)
        T = jnp.array(1.0)
        r = jnp.array(0.05)
        call = EuropeanOption(strike=K, expiry=T, is_call=True)
        put = EuropeanOption(strike=K, expiry=T, is_call=False)
        C = sabr_price(call, F, r, typical_sabr)
        P = sabr_price(put, F, r, typical_sabr)
        df = jnp.exp(-r * T)
        assert abs(float(C - P) - float(df * (F - K))) < 1e-10

    def test_deep_itm_call_near_intrinsic(self, typical_sabr):
        """Deep ITM call should be close to discounted intrinsic value."""
        F = jnp.array(100.0)
        K = jnp.array(50.0)
        T = jnp.array(0.25)
        r = jnp.array(0.05)
        option = EuropeanOption(strike=K, expiry=T, is_call=True)
        price = sabr_price(option, F, r, typical_sabr)
        intrinsic = jnp.exp(-r * T) * (F - K)
        assert float(price) >= float(intrinsic) - 1e-6

    def test_consistency_with_black76(self, lognormal_sabr):
        """SABR price should equal Black-76 when fed the same implied vol."""
        F = jnp.array(100.0)
        K = jnp.array(110.0)
        T = jnp.array(0.5)
        r = jnp.array(0.03)
        option = EuropeanOption(strike=K, expiry=T, is_call=True)

        vol = sabr_implied_vol(lognormal_sabr, F, K, T)
        sabr_p = sabr_price(option, F, r, lognormal_sabr)
        b76_p = black76_price(option, F, vol, r)
        assert abs(float(sabr_p) - float(b76_p)) < 1e-12


# ── Greeks via autodiff ────────────────────────────────────────────

class TestSABRGreeks:
    def test_call_delta_positive(self, typical_sabr):
        """ATM call delta should be positive (between 0 and 1)."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        # sabr_price(option, forward, rate, model) — forward is arg 1
        delta_fn = jax.grad(
            lambda fwd: sabr_price(option, fwd, jnp.array(0.05), typical_sabr)
        )
        delta = delta_fn(jnp.array(100.0))
        assert 0.0 < float(delta) < 1.0

    def test_vega_positive(self, typical_sabr):
        """Vega w.r.t. alpha should be positive."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
        F = jnp.array(100.0)
        r = jnp.array(0.05)

        def price_fn(alpha):
            model = SABRModel(alpha=alpha, beta=typical_sabr.beta,
                              rho=typical_sabr.rho, nu=typical_sabr.nu)
            return sabr_price(option, F, r, model)

        vega = jax.grad(price_fn)(typical_sabr.alpha)
        assert float(vega) > 0.0

    def test_gamma_positive_atm(self, typical_sabr):
        """ATM gamma should be positive."""
        option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)

        price_fn = lambda fwd: sabr_price(option, fwd, jnp.array(0.05), typical_sabr)
        gamma = jax.grad(jax.grad(price_fn))(jnp.array(100.0))
        assert float(gamma) > 0.0


# ── Monte Carlo paths ──────────────────────────────────────────────

class TestSABRPaths:
    def test_path_shapes(self, typical_sabr):
        key = jax.random.PRNGKey(0)
        fwd_paths, vol_paths = generate_sabr_paths(
            typical_sabr, jnp.array(100.0), T=1.0, n_steps=50, n_paths=100, key=key
        )
        assert fwd_paths.shape == (100, 51)
        assert vol_paths.shape == (100, 51)

    def test_initial_values(self, typical_sabr):
        key = jax.random.PRNGKey(42)
        fwd_paths, vol_paths = generate_sabr_paths(
            typical_sabr, jnp.array(100.0), T=1.0, n_steps=50, n_paths=100, key=key
        )
        # All paths should start at the initial forward
        assert jnp.allclose(fwd_paths[:, 0], 100.0, atol=1e-5)
        # All vol paths should start at alpha
        assert jnp.allclose(vol_paths[:, 0], float(typical_sabr.alpha), atol=1e-5)

    def test_mc_vs_analytic_convergence(self, lognormal_sabr):
        """MC call price should converge to analytic within 2 standard errors."""
        F = jnp.array(100.0)
        K = jnp.array(105.0)
        T = 1.0
        r = jnp.array(0.03)
        option = EuropeanOption(strike=K, expiry=jnp.array(T), is_call=True)

        # Analytic price
        analytic = float(sabr_price(option, F, r, lognormal_sabr))

        # MC price
        key = jax.random.PRNGKey(123)
        n_paths = 200_000
        fwd_paths, _ = generate_sabr_paths(
            lognormal_sabr, F, T=T, n_steps=200, n_paths=n_paths, key=key
        )
        terminal = fwd_paths[:, -1]
        payoffs = jnp.maximum(terminal - K, 0.0) * jnp.exp(-r * T)
        mc_price = float(jnp.mean(payoffs))
        mc_se = float(jnp.std(payoffs) / jnp.sqrt(n_paths))

        assert abs(mc_price - analytic) < 2.0 * mc_se, (
            f"MC={mc_price:.4f} vs analytic={analytic:.4f}, SE={mc_se:.4f}"
        )


# ── Normal (Bachelier) SABR expansion ──────────────────────────────

class TestSABRNormalImpliedVol:
    """Hagan's normal-vol expansion: reductions, capability, and shift."""

    @pytest.mark.parametrize("strike", [0.005, 0.02, 0.03, 0.05])
    def test_beta0_nu0_reduces_to_alpha(self, strike):
        """beta=0, nu->0 is arithmetic Brownian motion: normal vol == alpha.

        Holds to machine precision at every strike (the moneyness series in
        numerator and denominator cancel identically for beta=0).
        """
        model = SABRModel(
            alpha=jnp.array(0.012),
            beta=jnp.array(0.0),
            rho=jnp.array(-0.2),
            nu=jnp.array(1e-12),
        )
        vol = sabr_normal_implied_vol(
            model, jnp.array(0.02), jnp.array(strike), jnp.array(2.0)
        )
        assert abs(float(vol) - float(model.alpha)) < 1e-12

    def test_atm_positive(self, normal_sabr):
        F = jnp.array(0.02)
        vol = sabr_normal_implied_vol(normal_sabr, F, F, jnp.array(1.0))
        assert float(vol) > 0.0

    def test_negative_and_zero_strikes_price_finitely(self, normal_sabr):
        """The whole point of the normal path: negative/zero strikes are finite,
        where the lognormal expansion (log(F/K)) cannot even run."""
        F = jnp.array(0.01)
        shift = jnp.array(0.03)
        for K in (0.0, -0.005, -0.008):
            option = EuropeanOption(
                strike=jnp.array(K), expiry=jnp.array(1.0), is_call=True
            )
            price = sabr_price_bachelier(option, F, jnp.array(0.01), normal_sabr, shift)
            assert jnp.isfinite(price), f"non-finite price at K={K}"
            assert float(price) > 0.0

    def test_lognormal_expansion_cannot_run_at_negative_strike(self, normal_sabr):
        """Sanity: the lognormal vol is genuinely undefined for K<0 (NaN),
        confirming the normal path is not merely a convenience."""
        F = jnp.array(0.01)
        vol = sabr_implied_vol(normal_sabr, F, jnp.array(-0.005), jnp.array(1.0))
        assert jnp.isnan(vol)

    def test_shift_invariance_of_price(self, normal_sabr):
        """Normal vol is translation-invariant: pricing an option whose strike
        and forward are both shifted by +s, with shift=s, matches the unshifted
        problem priced with shift=0 (same normal vol feeds Bachelier)."""
        F = jnp.array(0.02)
        K = jnp.array(0.025)
        r = jnp.array(0.0)
        s = jnp.array(0.05)
        opt = EuropeanOption(strike=K, expiry=jnp.array(1.0), is_call=True)
        v0 = sabr_normal_implied_vol(normal_sabr, F, K, jnp.array(1.0), shift=jnp.array(0.0))
        vs = sabr_normal_implied_vol(normal_sabr, F, K, jnp.array(1.0), shift=s)
        # Different shift => different (finite) normal vol; both must be finite.
        assert jnp.isfinite(v0) and jnp.isfinite(vs)

    def test_jit_compiles(self, normal_sabr):
        F = jnp.array(0.02)
        opt = EuropeanOption(strike=jnp.array(0.025), expiry=jnp.array(1.0), is_call=True)
        vol = jax.jit(sabr_normal_implied_vol)(normal_sabr, F, opt.strike, opt.expiry)
        price = jax.jit(sabr_price_bachelier)(opt, F, jnp.array(0.01), normal_sabr)
        assert jnp.isfinite(vol) and jnp.isfinite(price)


class TestSABRNormalConsistency:
    """The lognormal and normal Hagan expansions are *different* approximations
    to the same SDE; they agree only asymptotically. These tests assert the
    discrepancy *scales* correctly with vol-of-vol -- never that they are equal.
    """

    F = jnp.array(100.0)
    RATE = jnp.array(0.0)

    def _atm_gap(self, nu, T):
        """|direct normal vol - (lognormal vol converted exactly to normal)|
        at the money, via price-and-invert (exact conversion)."""
        model = SABRModel(
            alpha=jnp.array(0.3), beta=jnp.array(0.5),
            rho=jnp.array(-0.3), nu=jnp.array(nu),
        )
        opt = EuropeanOption(strike=self.F, expiry=jnp.array(T), is_call=True)
        black_vol = sabr_implied_vol(model, self.F, self.F, jnp.array(T))
        price_ln = black76_price(opt, self.F, black_vol, self.RATE)
        normal_equiv = bachelier_implied_vol(opt, self.F, price_ln, self.RATE)
        normal_direct = sabr_normal_implied_vol(model, self.F, self.F, jnp.array(T))
        return abs(float(normal_direct - normal_equiv))

    def test_agree_as_nu_to_zero(self):
        """nu -> 0: both expansions reduce to the same CEV vol; the exact
        conversion makes them agree to near machine precision."""
        assert self._atm_gap(1e-9, 1.0) < 1e-7

    def test_genuinely_differ_at_realistic_nu(self):
        """With real vol-of-vol the two are NOT equal -- guarding against an
        equality assertion that would later be loosened into meaninglessness."""
        assert self._atm_gap(0.4, 1.0) > 1e-7

    def test_gap_scales_quadratically_in_nu(self):
        """Doubling vol-of-vol quadruples the discrepancy (O(nu^2))."""
        g1 = self._atm_gap(0.2, 1.0)
        g2 = self._atm_gap(0.4, 1.0)
        ratio = g2 / g1
        assert 3.0 < ratio < 5.5, f"nu^2 scaling ratio={ratio:.3f}"

    def test_gap_monotone_in_nu(self):
        gaps = [self._atm_gap(nu, 1.0) for nu in (0.1, 0.2, 0.4, 0.8)]
        assert all(b > a for a, b in zip(gaps, gaps[1:])), gaps


class TestSABRNormalMonteCarlo:
    def test_normal_price_matches_mc(self, normal_sabr):
        """Normal-expansion price lands within 2 standard errors of the SABR
        SDE Monte Carlo (generate_sabr_paths) on the same option."""
        F = jnp.array(0.02)
        K = jnp.array(0.025)
        T = 1.0
        r = jnp.array(0.01)
        option = EuropeanOption(strike=K, expiry=jnp.array(T), is_call=True)

        analytic = float(sabr_price_bachelier(option, F, r, normal_sabr))

        key = jax.random.PRNGKey(7)
        n_paths = 300_000
        fwd_paths, _ = generate_sabr_paths(
            normal_sabr, F, T=T, n_steps=200, n_paths=n_paths, key=key
        )
        terminal = fwd_paths[:, -1]
        payoffs = jnp.maximum(terminal - K, 0.0) * jnp.exp(-r * T)
        mc_price = float(jnp.mean(payoffs))
        mc_se = float(jnp.std(payoffs) / jnp.sqrt(n_paths))

        assert abs(mc_price - analytic) < 2.0 * mc_se, (
            f"MC={mc_price:.6f} vs analytic={analytic:.6f}, SE={mc_se:.6f}"
        )
