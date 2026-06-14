"""Unit tests for the Fang-Oosterlee COS Heston pricer.

The QuantLib agreement test lives in
``tests/test_quantlib_comparison/test_heston_ql.py``; the QE-MC vs COS
cross-check lives there as well. This file covers self-contained
correctness properties:

* Sanity (positive, finite prices).
* Put-call parity to floating-point precision.
* BSM-near-limit: with ``xi`` tiny and ``theta = v0``, COS collapses to
  Black-Scholes-Merton.
* Truncation convergence: increasing ``N`` reduces the residual w.r.t.
  a high-N reference.
* Autodiff Greeks vs central finite differences for every Heston model
  parameter.
* Golden price for a canonical scenario (locks the numerical contract).
"""

import jax
import jax.numpy as jnp
import pytest

from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel
from valax.pricing.analytic.heston import heston_cos_price
from valax.pricing.analytic.black_scholes import black_scholes_price

from tests.golden._helpers import assert_matches_golden


jax.config.update("jax_enable_x64", True)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def realistic_heston():
    """A realistic Heston parameter set (negative skew, vol-of-vol)."""
    return HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(2.0),
        theta=jnp.array(0.04),
        xi=jnp.array(0.5),
        rho=jnp.array(-0.7),
        rate=jnp.array(0.03),
        dividend=jnp.array(0.0),
    )


@pytest.fixture
def atm_call():
    return EuropeanOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True,
    )


@pytest.fixture
def atm_put():
    return EuropeanOption(
        strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=False,
    )


# ── Sanity ────────────────────────────────────────────────────────────

class TestSanity:
    def test_call_price_positive_and_finite(self, realistic_heston, atm_call):
        price = heston_cos_price(
            atm_call, jnp.array(100.0),
            realistic_heston.rate, realistic_heston.dividend,
            realistic_heston,
        )
        assert jnp.isfinite(price)
        assert float(price) > 0.0

    def test_put_price_positive_and_finite(self, realistic_heston, atm_put):
        price = heston_cos_price(
            atm_put, jnp.array(100.0),
            realistic_heston.rate, realistic_heston.dividend,
            realistic_heston,
        )
        assert jnp.isfinite(price)
        assert float(price) > 0.0

    def test_deep_itm_call_above_intrinsic(self, realistic_heston):
        opt = EuropeanOption(
            strike=jnp.array(50.0), expiry=jnp.array(1.0), is_call=True,
        )
        price = heston_cos_price(
            opt, jnp.array(100.0),
            realistic_heston.rate, realistic_heston.dividend,
            realistic_heston,
        )
        T = float(opt.expiry)
        r = float(realistic_heston.rate)
        q = float(realistic_heston.dividend)
        intrinsic = 100.0 * jnp.exp(-q * T) - 50.0 * jnp.exp(-r * T)
        assert float(price) >= float(intrinsic) - 1e-8

    def test_deep_otm_call_below_spot(self, realistic_heston):
        opt = EuropeanOption(
            strike=jnp.array(200.0), expiry=jnp.array(1.0), is_call=True,
        )
        price = heston_cos_price(
            opt, jnp.array(100.0),
            realistic_heston.rate, realistic_heston.dividend,
            realistic_heston,
        )
        assert 0.0 < float(price) < 100.0


# ── Put-call parity ──────────────────────────────────────────────────

class TestPutCallParity:
    @pytest.mark.parametrize("moneyness", [0.7, 0.9, 1.0, 1.1, 1.3])
    def test_parity_holds_to_floating_point(
        self, realistic_heston, moneyness,
    ):
        spot = jnp.array(100.0)
        strike = spot * moneyness
        T = jnp.array(1.0)
        r = realistic_heston.rate
        q = realistic_heston.dividend

        # COS uses the put payoff directly (not parity), so the parity
        # residual stacks the truncation errors on both sides.  The
        # default L=12 is tuned for moneyness ~0.85-1.15; at deep ITM
        # the Heston left tail under-captures and gives a ~1e-6 floor.
        # Bump L=18, N=256 to push tail-truncation below 1e-10 for the
        # parity check (this is a method-correctness test, not a
        # performance test).
        call = heston_cos_price(
            EuropeanOption(strike=strike, expiry=T, is_call=True),
            spot, r, q, realistic_heston,
            N=256, L=18.0,
        )
        put = heston_cos_price(
            EuropeanOption(strike=strike, expiry=T, is_call=False),
            spot, r, q, realistic_heston,
            N=256, L=18.0,
        )
        forward_diff = spot * jnp.exp(-q * T) - strike * jnp.exp(-r * T)

        assert abs(float(call - put - forward_diff)) < 1e-9


# ── BSM-near-limit ───────────────────────────────────────────────────

class TestBSMNearLimit:
    """With ``xi`` small and ``theta = v0``, the variance process is
    nearly constant and Heston collapses to BSM with ``sigma = sqrt(v0)``.

    The error is O(xi^2) — at ``xi = 0.01`` the residual is well below
    1e-3 across the strike range.
    """

    @pytest.mark.parametrize("moneyness", [0.9, 1.0, 1.1])
    @pytest.mark.parametrize("is_call", [True, False])
    def test_collapses_to_bsm(self, moneyness, is_call):
        v0 = 0.04
        model = HestonModel(
            v0=jnp.array(v0),
            kappa=jnp.array(50.0),     # strong mean reversion
            theta=jnp.array(v0),       # long-run = initial
            xi=jnp.array(0.01),        # tiny vol of vol
            rho=jnp.array(0.0),
            rate=jnp.array(0.05),
            dividend=jnp.array(0.02),
        )
        spot = jnp.array(100.0)
        strike = spot * moneyness
        T = jnp.array(1.0)
        opt = EuropeanOption(strike=strike, expiry=T, is_call=is_call)

        cos_price = heston_cos_price(
            opt, spot, model.rate, model.dividend, model,
        )
        bsm = black_scholes_price(
            opt, spot, jnp.sqrt(jnp.array(v0)),
            model.rate, model.dividend,
        )
        assert abs(float(cos_price) - float(bsm)) < 1e-3


# ── Truncation convergence ───────────────────────────────────────────

class TestTruncationConvergence:
    def test_increasing_N_reduces_error(self, realistic_heston, atm_call):
        spot = jnp.array(100.0)
        ref = heston_cos_price(
            atm_call, spot,
            realistic_heston.rate, realistic_heston.dividend,
            realistic_heston,
            N=512, L=14.0,
        )

        prev_err = float("inf")
        for N in (32, 64, 128, 256):
            price = heston_cos_price(
                atm_call, spot,
                realistic_heston.rate, realistic_heston.dividend,
                realistic_heston,
                N=N,
            )
            err = abs(float(price - ref))
            assert err < prev_err, (
                f"Error not decreasing at N={N}: {err} >= {prev_err}"
            )
            prev_err = err

        # N=256 should already be at floating-point noise vs N=512.
        assert prev_err < 1e-10


# ── Autodiff vs finite differences ───────────────────────────────────

def _fd_grad(f, x, h):
    """Central finite difference: (f(x+h) - f(x-h)) / (2h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestAutodiffVsFiniteDifference:
    @pytest.mark.parametrize(
        "param,h",
        [
            ("v0",    1e-5),
            ("kappa", 1e-4),
            ("theta", 1e-5),
            ("xi",    1e-5),
            ("rho",   1e-4),
        ],
    )
    def test_grad_matches_central_fd(self, realistic_heston, atm_call, param, h):
        spot = jnp.array(100.0)
        r = realistic_heston.rate
        q = realistic_heston.dividend

        def price_with(value):
            kwargs = {
                "v0": realistic_heston.v0,
                "kappa": realistic_heston.kappa,
                "theta": realistic_heston.theta,
                "xi": realistic_heston.xi,
                "rho": realistic_heston.rho,
                "rate": r,
                "dividend": q,
            }
            kwargs[param] = value
            return heston_cos_price(
                atm_call, spot, r, q, HestonModel(**kwargs),
            )

        x0 = getattr(realistic_heston, param)
        ad = float(jax.grad(price_with)(x0))
        fd = float(_fd_grad(price_with, x0, jnp.array(h)))

        if abs(fd) < 1e-6:
            assert abs(ad - fd) < 1e-6, (
                f"{param}: AD={ad} vs FD={fd} (both near zero)"
            )
        else:
            rel = abs(ad - fd) / abs(fd)
            assert rel < 1e-4, (
                f"{param}: AD={ad} vs FD={fd}  (rel={rel:.2e})"
            )


# ── Golden ───────────────────────────────────────────────────────────

class TestGolden:
    def test_canonical_price_grid(self):
        """Lock the COS price across a canonical (model, market, strikes) scenario.

        Bump ``version=`` and regen with ``REGEN_GOLDEN=1`` when the
        numerical contract changes intentionally.
        """
        model = HestonModel(
            v0=jnp.array(0.04),
            kappa=jnp.array(2.0),
            theta=jnp.array(0.04),
            xi=jnp.array(0.5),
            rho=jnp.array(-0.7),
            rate=jnp.array(0.03),
            dividend=jnp.array(0.0),
        )
        spot = jnp.array(100.0)
        strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
        T = jnp.array(1.0)

        prices = jax.vmap(
            lambda K: heston_cos_price(
                EuropeanOption(strike=K, expiry=T, is_call=True),
                spot, model.rate, model.dividend, model,
            )
        )(strikes)

        assert_matches_golden(
            "heston_cos_canonical_call_grid",
            prices,
            version=1,
            atol=1e-12,
            rtol=1e-12,
        )


# ── JIT and vmap ─────────────────────────────────────────────────────

class TestJitAndVmap:
    def test_jit_compiles(self, realistic_heston, atm_call):
        spot = jnp.array(100.0)
        jitted = jax.jit(heston_cos_price)
        price = jitted(
            atm_call, spot,
            realistic_heston.rate, realistic_heston.dividend,
            realistic_heston,
        )
        assert jnp.isfinite(price)

    def test_vmap_over_strikes(self, realistic_heston):
        spot = jnp.array(100.0)
        T = jnp.array(1.0)
        strikes = jnp.linspace(80.0, 120.0, 9)

        prices = jax.vmap(
            lambda K: heston_cos_price(
                EuropeanOption(strike=K, expiry=T, is_call=True),
                spot, realistic_heston.rate, realistic_heston.dividend,
                realistic_heston,
            )
        )(strikes)

        assert prices.shape == (9,)
        # Call price is monotonically decreasing in strike.
        diffs = jnp.diff(prices)
        assert jnp.all(diffs <= 0.0)
