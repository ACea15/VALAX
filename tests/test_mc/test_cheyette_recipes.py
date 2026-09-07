"""Tests for the Cheyette Monte-Carlo pricing recipes (via ``mc_price_dispatch``).

Each registered ``(instrument, CheyetteModel*)`` recipe is exercised against a
model-independent or reduction oracle:

- Fixed-rate bonds reprice the initial curve (both 1F and 2F).
- Floating-rate notes price to par at reset.
- Caplet minus floorlet obeys the model-free cap/floor parity.
- The one-factor swaption reduces to the analytic Hull-White price at zero skew,
  and moves with the skew (its raison d'etre).
- The two-factor CMS-spread steepener and flattener sum to zero and move with
  the correlation (the decorrelation signature).
"""

import jax
import jax.numpy as jnp
import pytest

from valax.models.cheyette import CheyetteModel, CheyetteModel2F, cheyette_market_df
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.mc import mc_price_dispatch, MCConfig
from valax.instruments.bonds import FixedRateBond, FloatingRateBond
from valax.instruments.rates import Caplet, Swaption, CMSSpreadSwap
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction


_ACT365 = "act_365"


@pytest.fixture(scope="module")
def ref_date():
    return int(ymd_to_ordinal(2025, 1, 1))


def _steep_curve(ref, n_years=21):
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(ref)).astype(jnp.float64) / 365.0
    zero = 0.02 + 0.03 * (1.0 - jnp.exp(-0.3 * times))
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=jnp.int32(ref),
        day_count=_ACT365,
    )


@pytest.fixture
def curve(ref_date):
    return _steep_curve(ref_date)


@pytest.fixture
def model(curve):
    return CheyetteModel(
        mean_reversion=jnp.array(0.10),
        volatility=jnp.array(0.010),
        skew=jnp.array(-3.0),
        initial_curve=curve,
    )


@pytest.fixture
def model_2f(curve):
    return CheyetteModel2F(
        mean_reversion_1=jnp.array(0.50),
        mean_reversion_2=jnp.array(0.10),
        volatility_1=jnp.array(0.010),
        volatility_2=jnp.array(0.008),
        skew_1=jnp.array(-2.0),
        skew_2=jnp.array(1.5),
        correlation=jnp.array(-0.70),
        initial_curve=curve,
    )


def _ords(years):
    return jnp.array([int(ymd_to_ordinal(y, 1, 1)) for y in years], dtype=jnp.int32)


# ── Fixed-rate bond: reprices the curve ──────────────────────────────

class TestFixedBond:
    def test_reprices_curve_1f(self, model, ref_date):
        pay = _ords(range(2026, 2031))
        bond = FixedRateBond(
            payment_dates=pay, settlement_date=jnp.array(ref_date),
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
            frequency=1, day_count=_ACT365,
        )
        res = mc_price_dispatch(
            bond, model, MCConfig(n_paths=150_000, n_steps=60),
            jax.random.PRNGKey(0), n_steps=60,
        )
        cft = year_fraction(ref_date, pay, _ACT365)
        amt = jnp.full((5,), 100.0 * 0.04).at[-1].add(100.0)
        analytic = float(jnp.sum(amt * cheyette_market_df(model, cft)))
        assert abs(float(res.price) - analytic) < 3.0 * float(res.stderr) + 1e-2

    def test_reprices_curve_2f(self, model_2f, ref_date):
        pay = _ords(range(2026, 2031))
        bond = FixedRateBond(
            payment_dates=pay, settlement_date=jnp.array(ref_date),
            coupon_rate=jnp.array(0.04), face_value=jnp.array(100.0),
            frequency=1, day_count=_ACT365,
        )
        res = mc_price_dispatch(
            bond, model_2f, MCConfig(n_paths=150_000, n_steps=60),
            jax.random.PRNGKey(1), n_steps=60,
        )
        cft = year_fraction(ref_date, pay, _ACT365)
        amt = jnp.full((5,), 100.0 * 0.04).at[-1].add(100.0)
        analytic = float(jnp.sum(amt * cheyette_market_df(model_2f, cft)))
        assert abs(float(res.price) - analytic) < 3.0 * float(res.stderr) + 1e-2


# ── Floating-rate note: par at reset ─────────────────────────────────

class TestFloatingBond:
    def test_par_at_reset_1f(self, model, ref_date):
        pay = _ords(range(2026, 2031))
        fixings = jnp.concatenate([jnp.array(ref_date)[None], pay[:-1]])
        frn = FloatingRateBond(
            payment_dates=pay, fixing_dates=fixings,
            settlement_date=jnp.array(ref_date),
            spread=jnp.array(0.0), face_value=jnp.array(100.0),
            frequency=1, day_count=_ACT365,
        )
        res = mc_price_dispatch(
            frn, model, MCConfig(n_paths=100_000, n_steps=60),
            jax.random.PRNGKey(2), n_steps=60,
        )
        assert abs(float(res.price) - 100.0) < 3.0 * float(res.stderr) + 1e-6


# ── Caplet: model-free cap/floor parity ──────────────────────────────

class TestCapletParity:
    def test_caplet_minus_floorlet(self, model, ref_date):
        fix = int(ymd_to_ordinal(2027, 1, 1))
        start = fix
        end = int(ymd_to_ordinal(2028, 1, 1))
        strike = jnp.array(0.045)
        notional = jnp.array(100.0)
        common = dict(
            fixing_date=jnp.array(fix), start_date=jnp.array(start),
            end_date=jnp.array(end), strike=strike, notional=notional,
            day_count=_ACT365,
        )
        cap = Caplet(is_cap=True, **common)
        flr = Caplet(is_cap=False, **common)
        cfg = MCConfig(n_paths=200_000, n_steps=60)
        rc = mc_price_dispatch(cap, model, cfg, jax.random.PRNGKey(3), n_steps=60)
        rf = mc_price_dispatch(flr, model, cfg, jax.random.PRNGKey(3), n_steps=60)

        tau = float(year_fraction(jnp.array(start), jnp.array(end), _ACT365))
        p_start = float(cheyette_market_df(model, year_fraction(ref_date, jnp.array(start), _ACT365)))
        p_end = float(cheyette_market_df(model, year_fraction(ref_date, jnp.array(end), _ACT365)))
        # caplet - floorlet = notional * [(P0_start - P0_end) - tau*K*P0_end].
        parity = float(notional) * ((p_start - p_end) - tau * float(strike) * p_end)
        diff = float(rc.price) - float(rf.price)
        stderr = float(rc.stderr) + float(rf.stderr)
        assert abs(diff - parity) < 3.0 * stderr + 1e-4

    def test_caplet_positive(self, model):
        cap = Caplet(
            fixing_date=jnp.array(int(ymd_to_ordinal(2027, 1, 1))),
            start_date=jnp.array(int(ymd_to_ordinal(2027, 1, 1))),
            end_date=jnp.array(int(ymd_to_ordinal(2028, 1, 1))),
            strike=jnp.array(0.045), notional=jnp.array(100.0),
            is_cap=True, day_count=_ACT365,
        )
        res = mc_price_dispatch(cap, model, MCConfig(n_paths=50_000, n_steps=40), jax.random.PRNGKey(4), n_steps=40)
        assert float(res.price) > 0.0


# ── Swaption: HW reduction and skew signature ────────────────────────

def _atm_swaption(ref_date, curve):
    expiry = int(ymd_to_ordinal(2027, 1, 1))
    fixed = _ords(range(2028, 2033))
    # ATM strike = forward swap rate from the curve.
    et = year_fraction(ref_date, jnp.array(expiry), _ACT365)
    ft = year_fraction(ref_date, fixed, _ACT365)
    starts = jnp.concatenate([jnp.array(expiry)[None], fixed[:-1]])
    taus = year_fraction(starts, fixed, _ACT365)
    p_exp = jnp.exp(jnp.interp(et, jnp.concatenate([jnp.zeros(1), (curve.pillar_dates - ref_date).astype(jnp.float64) / 365.0]),
                               jnp.concatenate([jnp.zeros(1), jnp.log(curve.discount_factors)])))
    p_fix = jnp.exp(jnp.interp(ft, jnp.concatenate([jnp.zeros(1), (curve.pillar_dates - ref_date).astype(jnp.float64) / 365.0]),
                               jnp.concatenate([jnp.zeros(1), jnp.log(curve.discount_factors)])))
    annuity = jnp.sum(taus * p_fix)
    fwd = (p_exp - p_fix[-1]) / annuity
    return Swaption(
        expiry_date=jnp.array(expiry), fixed_dates=fixed, strike=fwd,
        notional=jnp.array(100.0), is_payer=True, day_count=_ACT365,
    )


class TestSwaption:
    def test_reduces_to_hull_white_at_zero_skew(self, curve, ref_date):
        ch = CheyetteModel(
            mean_reversion=jnp.array(0.10), volatility=jnp.array(0.010),
            skew=jnp.array(0.0), initial_curve=curve,
        )
        hw = HullWhiteModel(
            mean_reversion=jnp.array(0.10), volatility=jnp.array(0.010),
            initial_curve=curve,
        )
        sw = _atm_swaption(ref_date, curve)
        res = mc_price_dispatch(
            sw, ch, MCConfig(n_paths=300_000, n_steps=48),
            jax.random.PRNGKey(5), n_steps=48,
        )
        analytic = float(hw_swaption_price(sw, hw))
        assert abs(float(res.price) - analytic) < 3.5 * float(res.stderr) + 5e-3

    def test_skew_moves_price_monotonically(self, curve, ref_date):
        """The skew moves the swaption price monotonically and measurably — the
        smile that the Gaussian model cannot produce.

        Common random numbers (a shared PRNG key) are used across the skew
        values, so the *difference* between prices is a very low-variance paired
        estimator: the tiny per-path noise cancels and the residual ordering is
        the deterministic skew sensitivity.
        """
        sw = _atm_swaption(ref_date, curve)
        cfg = MCConfig(n_paths=300_000, n_steps=48)
        key = jax.random.PRNGKey(6)
        prices = []
        for skew in [-12.0, 0.0, 12.0]:
            m = CheyetteModel(
                mean_reversion=jnp.array(0.10), volatility=jnp.array(0.012),
                skew=jnp.array(skew), initial_curve=curve,
            )
            prices.append(float(mc_price_dispatch(sw, m, cfg, key, n_steps=48).price))
        monotone = (prices[0] < prices[1] < prices[2]) or (prices[0] > prices[1] > prices[2])
        spread = max(prices) - min(prices)
        assert monotone
        assert spread > 1e-3


# ── CMS-spread swap: decorrelation signature ─────────────────────────

class TestCMSSpread:
    def _cms(self, pay_fixed):
        return CMSSpreadSwap(
            start_date=jnp.array(int(ymd_to_ordinal(2026, 1, 1))),
            payment_dates=_ords(range(2027, 2032)),
            fixed_rate=jnp.array(0.0), notional=jnp.array(100.0),
            cms_tenor_long=10, cms_tenor_short=2, pay_fixed=pay_fixed,
            day_count=_ACT365,
        )

    def test_steepener_plus_flattener_zero(self, model_2f):
        cfg = MCConfig(n_paths=80_000, n_steps=60)
        steep = mc_price_dispatch(self._cms(True), model_2f, cfg, jax.random.PRNGKey(7), n_steps=60)
        flat = mc_price_dispatch(self._cms(False), model_2f, cfg, jax.random.PRNGKey(7), n_steps=60)
        assert abs(float(steep.price) + float(flat.price)) < 1e-6

    def test_monotone_in_correlation(self, curve):
        cfg = MCConfig(n_paths=120_000, n_steps=60)
        cms = self._cms(True)
        prices = []
        for rho in [-0.9, -0.3, 0.5]:
            m = CheyetteModel2F(
                mean_reversion_1=jnp.array(0.50), mean_reversion_2=jnp.array(0.10),
                volatility_1=jnp.array(0.010), volatility_2=jnp.array(0.008),
                skew_1=jnp.array(-2.0), skew_2=jnp.array(1.5),
                correlation=jnp.array(rho), initial_curve=curve,
            )
            r = mc_price_dispatch(cms, m, cfg, jax.random.PRNGKey(8), n_steps=60)
            prices.append(float(r.price))
        # A steepener's PV varies monotonically with the driving correlation.
        assert (prices[0] < prices[1] < prices[2]) or (prices[0] > prices[1] > prices[2])
