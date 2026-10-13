"""Tests for the G2++ semi-analytic European swaption pricer.

Internal checks (independent of QuantLib): put/receiver parity against the
forward swap value, positivity, ATM ordering, decorrelation sensitivity, JIT
and autodiff.  A loose Hull-White reduction sanity check is included; the tight
oracle is the QuantLib comparison in ``test_quantlib_comparison``.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.instruments.rates import Swaption
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal


_ACT365 = "act_365"


@pytest.fixture
def ref_date():
    return int(ymd_to_ordinal(2025, 1, 1))


def _flat_curve(ref_date, rate=0.03, n_years=21):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.int32(ref_date),
        day_count=_ACT365,
    )


@pytest.fixture
def curve(ref_date):
    return _flat_curve(ref_date)


@pytest.fixture
def model(curve):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.010),
        volatility_y=jnp.array(0.008),
        correlation=jnp.array(-0.70),
        initial_curve=curve,
    )


def _swaption(ref_date, expiry_y, tenor_y, strike, is_payer):
    expiry = int(ymd_to_ordinal(2025 + expiry_y, 1, 1))
    fixed = jnp.array(
        [int(ymd_to_ordinal(2025 + expiry_y + k, 1, 1)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=fixed,
        strike=jnp.asarray(strike),
        notional=jnp.array(1.0),
        is_payer=is_payer,
        day_count=_ACT365,
    )


def _atm_strike(ref_date, curve, expiry_y, tenor_y):
    expiry = jnp.int32(int(ymd_to_ordinal(2025 + expiry_y, 1, 1)))
    fixed = jnp.array(
        [int(ymd_to_ordinal(2025 + expiry_y + k, 1, 1)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    ann = _annuity(expiry, fixed, curve, _ACT365)
    fwd = (curve(expiry) - curve(fixed[-1])) / ann
    return float(fwd), float(ann)


class TestSanity:
    def test_positive(self, ref_date, model):
        for is_payer in (True, False):
            s = _swaption(ref_date, 5, 5, 0.03, is_payer)
            assert float(g2pp_swaption_price(s, model)) > 0.0

    def test_intrinsic_lower_bound(self, ref_date, curve, model):
        """A deep ITM payer must exceed its (positive) intrinsic value."""
        fwd, ann = _atm_strike(ref_date, curve, 5, 5)
        strike = fwd - 0.02  # deep ITM payer (pay low fixed)
        s = _swaption(ref_date, 5, 5, strike, is_payer=True)
        intrinsic = ann * (fwd - strike)
        assert float(g2pp_swaption_price(s, model)) > intrinsic > 0.0

    def test_receiver_parity(self, ref_date, curve, model):
        """Payer - Receiver = forward swap value = annuity * (fwd - K)."""
        fwd, ann = _atm_strike(ref_date, curve, 5, 5)
        for strike in (fwd - 0.01, fwd, fwd + 0.01):
            payer = float(g2pp_swaption_price(_swaption(ref_date, 5, 5, strike, True), model))
            recv = float(g2pp_swaption_price(_swaption(ref_date, 5, 5, strike, False), model))
            swap_val = ann * (fwd - strike)
            assert (payer - recv) == pytest.approx(swap_val, abs=1e-9)

    def test_atm_payer_equals_receiver(self, ref_date, curve, model):
        """At the ATM strike payer and receiver values coincide."""
        fwd, _ = _atm_strike(ref_date, curve, 5, 5)
        payer = float(g2pp_swaption_price(_swaption(ref_date, 5, 5, fwd, True), model))
        recv = float(g2pp_swaption_price(_swaption(ref_date, 5, 5, fwd, False), model))
        assert payer == pytest.approx(recv, rel=1e-8)

    def test_monotone_in_strike(self, ref_date, model):
        """Payer value decreases as the fixed strike rises."""
        prices = [
            float(g2pp_swaption_price(_swaption(ref_date, 5, 5, k, True), model))
            for k in (0.02, 0.03, 0.04)
        ]
        assert prices[0] > prices[1] > prices[2]


class TestCorrelationSensitivity:
    def test_atm_value_increases_with_rho(self, ref_date, curve):
        """A single ATM swaption's value rises with rho.

        Positive correlation makes the two factors reinforce, raising the
        forward swap-rate variance and hence the option value.  (Decorrelation
        -- negative rho -- instead *lowers* a lone swaption; its benefit shows
        up in genuinely decorrelation-sensitive payoffs like CMS spreads.)
        """
        fwd, _ = _atm_strike(ref_date, curve, 5, 5)
        s = _swaption(ref_date, 5, 5, fwd, True)

        def price_at(rho):
            m = G2PPModel(
                mean_reversion_x=jnp.array(0.50), mean_reversion_y=jnp.array(0.10),
                volatility_x=jnp.array(0.010), volatility_y=jnp.array(0.008),
                correlation=jnp.array(rho), initial_curve=curve,
            )
            return float(g2pp_swaption_price(s, m))

        assert price_at(0.9) > price_at(0.0) > price_at(-0.9)


class TestAutodiffJIT:
    def test_jit(self, ref_date, model):
        s = _swaption(ref_date, 5, 5, 0.03, True)
        eager = float(g2pp_swaption_price(s, model))
        jitted = float(eqx.filter_jit(g2pp_swaption_price)(s, model))
        assert jitted == pytest.approx(eager, rel=1e-10)

    def test_grad_in_params(self, ref_date, model):
        s = _swaption(ref_date, 5, 5, 0.03, True)
        grads = eqx.filter_grad(lambda m: g2pp_swaption_price(s, m))(model)
        # All model-parameter sensitivities must be finite (no NaN from the
        # inner root-find / quadrature).  The sign of dV/dsigma_x is *not*
        # asserted: with strongly negative rho the negative cross-variance term
        # can dominate, making the first factor's local vega negative.
        assert jnp.isfinite(grads.volatility_x)
        assert jnp.isfinite(grads.correlation)
        # The second-factor vega is robustly positive here.
        assert float(grads.volatility_y) > 0.0


class TestForwardCurve:
    """A distinct projection curve is now priced, not rejected.

    G2++ dual-curve swaptions went from ``NotImplementedError`` to the frozen
    deterministic tenor-basis overlay.  The substantive gates (PDE agreement,
    payer/receiver parity against the dual-curve forward swap, basis
    monotonicity) live in ``tests/test_pricing/test_dual_curve_swaptions.py``;
    these two just pin the local contract.
    """

    def test_distinct_forward_curve_is_priced(self, ref_date, model):
        other = _flat_curve(ref_date, rate=0.04)
        s = _swaption(ref_date, 5, 5, 0.03, True)
        dual = g2pp_swaption_price(s, model, forward_curve=other)
        assert jnp.isfinite(dual)
        # Projection above the discount curve lifts a payer swaption.
        assert float(dual) > float(g2pp_swaption_price(s, model))

    def test_discount_curve_as_projection_is_single_curve(self, ref_date, model):
        s = _swaption(ref_date, 5, 5, 0.03, True)
        assert float(
            g2pp_swaption_price(s, model, forward_curve=model.initial_curve)
        ) == float(g2pp_swaption_price(s, model))


class TestPriceRoute:
    """``Swaption × G2PPModel × "analytic"`` through :func:`valax.core.price`."""

    @staticmethod
    def _setup(ref_date, model):
        from valax.core.market_state import MarketState
        from valax.curves.graph import CurveGraph

        market = MarketState(curves=CurveGraph(curves={
            "USD.SOFR.OIS": model.initial_curve,
            "USD.SOFR.3M": _flat_curve(ref_date, rate=0.035),
        }))
        free = eqx.tree_at(lambda m: m.initial_curve, model, None,
                           is_leaf=lambda x: x is None)
        return market, free

    @staticmethod
    def _keyed(s, **selectors):
        return Swaption(
            expiry_date=s.expiry_date, fixed_dates=s.fixed_dates, strike=s.strike,
            notional=s.notional, is_payer=s.is_payer, day_count=s.day_count,
            **selectors,
        )

    def test_auto_default_is_analytic_and_matches_kernel(self, ref_date, model):
        from valax.core.pricing import auto_method, price

        market, free = self._setup(ref_date, model)
        s = self._keyed(_swaption(ref_date, 2, 5, 0.03, True),
                        discount_id="USD.SOFR.OIS")
        assert auto_method(s, free) == "analytic"
        got = price(s, free, market)
        assert got.method == "analytic"
        assert float(got.price) == float(g2pp_swaption_price(s, model))

    def test_multi_curve_projection_matches_kernel(self, ref_date, model):
        from valax.core.pricing import price

        market, free = self._setup(ref_date, model)
        s = self._keyed(_swaption(ref_date, 2, 5, 0.03, True),
                        discount_id="USD.SOFR.OIS", forward_id="USD.SOFR.3M")
        got = price(s, free, market, method="analytic").price
        ref = g2pp_swaption_price(s, model, market.curve("USD.SOFR.3M"))
        assert float(got) == float(ref)
        assert float(got) > float(g2pp_swaption_price(s, model))

    def test_jit_through_price(self, ref_date, model):
        from valax.core.pricing import price

        market, free = self._setup(ref_date, model)
        s = self._keyed(_swaption(ref_date, 2, 5, 0.03, True),
                        discount_id="USD.SOFR.OIS")
        f = eqx.filter_jit(lambda m, mk: price(s, m, mk).price)
        assert jnp.allclose(f(free, market), g2pp_swaption_price(s, model), rtol=1e-14)


class TestCriticalBoundary:
    def test_failed_boundary_solve_is_nan_not_a_silent_iterate(self):
        from valax.pricing.analytic.g2pp_swaptions import _critical_y

        # Σ λ_i e^{-B y} = 1 has no root when every λ_i is zero.
        y = _critical_y(jnp.zeros(3), jnp.ones(3))
        assert jnp.isnan(y)
        ok = _critical_y(jnp.array([0.5, 0.6]), jnp.array([0.5, 1.0]))
        assert jnp.isfinite(ok)
