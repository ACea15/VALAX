"""Hull-White European swaptions via Jamshidian decomposition.

Structural properties that must hold for *any* correct implementation:

- the critical rate really does price the coupon bond at par;
- payer minus receiver equals the forward swap PV (put-call parity), which for
  an exact-fitted model is a pure curve quantity, independent of `a` and
  `sigma`;
- the price is monotone in volatility and collapses to intrinsic as vol -> 0;
- deep out-of-the-money options are worthless.

The QuantLib cross-check lives in
``tests/test_quantlib_comparison/test_hull_white_swaptions_ql.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel, hw_bond_price
from valax.pricing.analytic.hull_white_swaptions import (
    hw_critical_rate,
    hw_swaption_price,
    hw_zcb_option_price,
)
from valax.pricing.analytic.swaptions import _annuity

_DAY_COUNT = "act_365"
_NOTIONAL = 1_000_000.0


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref_date() -> int:
    return int(ymd_to_ordinal(2026, 1, 1))


def _flat_curve(ref: int, rate: float, n_years: int) -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, n_years + 2)]
    pillars = jnp.array(
        [ref + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(rate) * times),
        reference_date=jnp.int32(ref),
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def curve(ref_date) -> DiscountCurve:
    return _flat_curve(ref_date, 0.04, 20)


@pytest.fixture(scope="module")
def model(curve) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.asarray(0.10),
        volatility=jnp.asarray(0.010),
        initial_curve=curve,
    )


def _make_swaption(
    ref: int, curve: DiscountCurve, expiry_y: int, tenor_y: int,
    strike_offset: float = 0.0, is_payer: bool = True,
) -> Swaption:
    """ATM-plus-offset swaption on an annual fixed leg."""
    expiry_ord = ref + expiry_y * 365
    fixed = jnp.array(
        [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )
    ann = _annuity(jnp.int32(expiry_ord), fixed, curve, _DAY_COUNT)
    fwd = (curve(jnp.int32(expiry_ord)) - curve(fixed[-1])) / ann
    return Swaption(
        expiry_date=jnp.int32(expiry_ord),
        fixed_dates=fixed,
        strike=fwd + strike_offset,
        notional=jnp.asarray(_NOTIONAL),
        is_payer=is_payer,
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def atm_payer(ref_date, curve) -> Swaption:
    return _make_swaption(ref_date, curve, 1, 5)


# ── Jamshidian internals ──────────────────────────────────────────────

class TestCriticalRate:
    def test_prices_coupon_bond_at_par(self, model):
        """By construction, sum_i c_i P(T0, T_i, r*) == 1."""
        expiry = jnp.asarray(1.0)
        cf_times = jnp.array([2.0, 3.0, 4.0, 5.0, 6.0])
        cashflows = jnp.array([0.04, 0.04, 0.04, 0.04, 1.04])

        r_star = hw_critical_rate(model, expiry, cf_times, cashflows)
        bond = jnp.sum(cashflows * hw_bond_price(model, r_star, expiry, cf_times))
        assert float(bond) == pytest.approx(1.0, abs=1e-10)

    def test_independent_of_initial_guess(self, model):
        """Newton converges to the same unique root from either side."""
        expiry = jnp.asarray(2.0)
        cf_times = jnp.array([3.0, 4.0, 5.0])
        cashflows = jnp.array([0.05, 0.05, 1.05])

        base = hw_critical_rate(model, expiry, cf_times, cashflows)
        for guess in [-0.02, 0.0, 0.15]:
            other = hw_critical_rate(
                model, expiry, cf_times, cashflows,
                initial_guess=jnp.asarray(guess),
            )
            assert float(other) == pytest.approx(float(base), abs=1e-10)


class TestZCBOption:
    def test_call_put_parity(self, model):
        """ZBC - ZBP == P(0,S) - X*P(0,T), the forward-contract value."""
        from valax.models.hull_white import hw_market_df

        T = jnp.asarray(1.0)
        S = jnp.array([2.0, 3.0, 5.0])
        X = jnp.array([0.95, 0.90, 0.82])

        call = hw_zcb_option_price(model, T, S, X, is_call=True)
        put = hw_zcb_option_price(model, T, S, X, is_call=False)
        expected = hw_market_df(model, S) - X * hw_market_df(model, T)
        assert jnp.allclose(call - put, expected, atol=1e-12)

    def test_prices_are_non_negative(self, model):
        T = jnp.asarray(2.0)
        S = jnp.array([3.0, 4.0, 7.0])
        X = jnp.array([0.99, 0.60, 0.30])
        for is_call in (True, False):
            px = hw_zcb_option_price(model, T, S, X, is_call=is_call)
            assert jnp.all(px >= -1e-14)


# ── Swaption-level structure ──────────────────────────────────────────

class TestSwaptionStructure:
    @pytest.mark.parametrize("offset", [-0.015, -0.005, 0.0, 0.005, 0.02])
    def test_payer_receiver_parity(self, ref_date, curve, model, offset):
        r"""payer - receiver == notional * annuity * (S - K).

        The forward swap PV is a pure curve quantity, so this also confirms the
        model is exact-fitted: no choice of ``a`` or ``sigma`` may move it.
        """
        payer = _make_swaption(ref_date, curve, 2, 5, offset, is_payer=True)
        receiver = _make_swaption(ref_date, curve, 2, 5, offset, is_payer=False)

        diff = float(hw_swaption_price(payer, model)) - float(
            hw_swaption_price(receiver, model)
        )
        ann = float(_annuity(payer.expiry_date, payer.fixed_dates, curve, _DAY_COUNT))
        fwd = (
            float(curve(payer.expiry_date)) - float(curve(payer.fixed_dates[-1]))
        ) / ann
        expected = _NOTIONAL * ann * (fwd - float(payer.strike))
        assert diff == pytest.approx(expected, rel=1e-9, abs=1e-6)

    def test_parity_is_model_independent(self, ref_date, curve):
        """The parity gap must not move with (a, sigma)."""
        payer = _make_swaption(ref_date, curve, 2, 5, 0.01, is_payer=True)
        receiver = _make_swaption(ref_date, curve, 2, 5, 0.01, is_payer=False)

        gaps = []
        for a, sigma in [(0.02, 0.005), (0.10, 0.010), (0.35, 0.020)]:
            m = HullWhiteModel(
                mean_reversion=jnp.asarray(a),
                volatility=jnp.asarray(sigma),
                initial_curve=curve,
            )
            gaps.append(
                float(hw_swaption_price(payer, m))
                - float(hw_swaption_price(receiver, m))
            )
        assert max(gaps) - min(gaps) < 1e-6

    def test_atm_payer_equals_atm_receiver(self, atm_payer, ref_date, curve, model):
        """At the forward swap rate the parity term vanishes."""
        receiver = _make_swaption(ref_date, curve, 1, 5, 0.0, is_payer=False)
        assert float(hw_swaption_price(atm_payer, model)) == pytest.approx(
            float(hw_swaption_price(receiver, model)), rel=1e-9
        )

    def test_monotone_increasing_in_volatility(self, atm_payer, curve):
        prices = []
        for sigma in [0.002, 0.005, 0.010, 0.020, 0.040]:
            m = HullWhiteModel(
                mean_reversion=jnp.asarray(0.10),
                volatility=jnp.asarray(sigma),
                initial_curve=curve,
            )
            prices.append(float(hw_swaption_price(atm_payer, m)))
        assert all(b > a for a, b in zip(prices, prices[1:]))

    def test_zero_vol_limit_is_intrinsic(self, ref_date, curve):
        """As sigma -> 0 the option collapses to the forward swap's intrinsic."""
        m = HullWhiteModel(
            mean_reversion=jnp.asarray(0.10),
            volatility=jnp.asarray(1e-10),
            initial_curve=curve,
        )
        # In-the-money payer: strike well below the forward swap rate.
        payer = _make_swaption(ref_date, curve, 2, 5, -0.01, is_payer=True)
        ann = float(_annuity(payer.expiry_date, payer.fixed_dates, curve, _DAY_COUNT))
        fwd = (
            float(curve(payer.expiry_date)) - float(curve(payer.fixed_dates[-1]))
        ) / ann
        intrinsic = _NOTIONAL * ann * (fwd - float(payer.strike))
        assert float(hw_swaption_price(payer, m)) == pytest.approx(
            intrinsic, rel=1e-6
        )

    def test_deep_otm_is_worthless(self, ref_date, curve, model):
        payer = _make_swaption(ref_date, curve, 1, 5, 0.50, is_payer=True)
        assert float(hw_swaption_price(payer, model)) == pytest.approx(0.0, abs=1e-6)

    def test_price_is_non_negative(self, ref_date, curve, model):
        for offset in [-0.05, -0.01, 0.0, 0.01, 0.05]:
            for payer in (True, False):
                sw = _make_swaption(ref_date, curve, 2, 5, offset, is_payer=payer)
                assert float(hw_swaption_price(sw, model)) >= -1e-9


# ── JAX transforms ────────────────────────────────────────────────────

class TestJITAndGrad:
    def test_filter_jit_smoke(self, atm_payer, model):
        jitted = float(eqx.filter_jit(hw_swaption_price)(atm_payer, model))
        assert jitted == pytest.approx(
            float(hw_swaption_price(atm_payer, model)), rel=1e-12
        )

    @pytest.mark.parametrize("field", ["volatility", "mean_reversion"])
    def test_autodiff_matches_finite_difference(self, atm_payer, model, field):
        """Implicit differentiation through the Newton solve must be exact."""
        grads = eqx.filter_grad(lambda m: hw_swaption_price(atm_payer, m))(model)
        analytic = float(getattr(grads, field))

        where = lambda m: getattr(m, field)  # noqa: E731
        base = getattr(model, field)
        h = 1e-6
        up = eqx.tree_at(where, model, base + h)
        down = eqx.tree_at(where, model, base - h)
        fd = (
            float(hw_swaption_price(atm_payer, up))
            - float(hw_swaption_price(atm_payer, down))
        ) / (2.0 * h)
        assert analytic == pytest.approx(fd, rel=1e-4)

    def test_vega_is_positive(self, atm_payer, model):
        grads = eqx.filter_grad(lambda m: hw_swaption_price(atm_payer, m))(model)
        assert float(grads.volatility) > 0.0

    def test_grad_wrt_strike_is_negative_for_payer(self, atm_payer, model):
        """A payer swaption is a decreasing function of its strike."""
        g = eqx.filter_grad(
            lambda sw: hw_swaption_price(sw, model)
        )(atm_payer)
        assert float(g.strike) < 0.0

    def test_vmap_over_volatility(self, atm_payer, curve):
        """Batched pricing across a vol grid."""
        def price(sigma):
            m = HullWhiteModel(
                mean_reversion=jnp.asarray(0.10),
                volatility=sigma,
                initial_curve=curve,
            )
            return hw_swaption_price(atm_payer, m)

        sigmas = jnp.array([0.005, 0.010, 0.020])
        batched = jax.vmap(price)(sigmas)
        assert batched.shape == (3,)
        for i, s in enumerate(sigmas):
            assert float(batched[i]) == pytest.approx(float(price(s)), rel=1e-10)
