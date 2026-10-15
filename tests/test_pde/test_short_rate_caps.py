r"""Tests for the short-rate caplet / cap / floor PDE recipes and the
two-factor callable/puttable additions.

The validation follows the same triangulation discipline as the rest of
``test_pde``:

- **Caplet vs analytic.** A Hull-White caplet is :math:`N(1+K\tau)` puts on the
  accrual-end zero-coupon bond, so it has the closed form
  :func:`~valax.pricing.analytic.hull_white_swaptions.hw_zcb_option_price`; the
  PDE must reproduce it tightly.
- **Put-call parity (model-free).** ``caplet - floorlet`` equals
  :math:`N(P(0,T_0) - (1+K\tau)P(0,T_1))` off the initial curve alone, for
  every model — a discretisation-only check that needs no oracle.
- **Reduction.** At :math:`\beta = 0` the frozen-``y`` Cheyette caplet reduces
  to the Hull-White caplet on the same mesh.
- **PDE vs MC.** The Cheyette caplet PDE agrees with the exact-``y`` Cheyette
  Monte-Carlo engine within Monte-Carlo error.
- **Cap convergence.** The single-sweep cap strip converges to the analytic
  caplet sum under time refinement (the mid-sweep kink injections are only
  Rannacher-damped at the horizon).
- **2F callable/puttable.** Ordering ``callable < bullet < puttable`` and an
  unreachable option recovering the bullet to machine precision.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.instruments.rates import Cap, Caplet
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_zcb_option_price
from valax.pricing.mc import MCConfig
from valax.pricing.pde import (
    PDEConfig,
    PDEConfig2D,
    Scheme,
    pde_price_dispatch,
    registered_recipes,
)
from valax.pricing.reconcile import price_all_engines

_ACT365 = "act_365"
_REF = int(ymd_to_ordinal(2025, 1, 1))
_YF = lambda a, b: year_fraction(jnp.int32(int(a)), jnp.int32(int(b)), _ACT365)

C1 = PDEConfig(n_spot=301, n_time=200, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
C2 = PDEConfig2D(n_x=81, n_y=81, n_time=100, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def flat_curve():
    pillars = jnp.array(
        [_REF + int(round(y * 365)) for y in range(11)], dtype=jnp.int32
    )
    times = (pillars - _REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.03 * times),
        reference_date=jnp.int32(_REF),
        day_count=_ACT365,
    )


@pytest.fixture
def hw(flat_curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.10), volatility=jnp.array(0.01),
        initial_curve=flat_curve,
    )


@pytest.fixture
def g2(flat_curve):
    return G2PPModel(
        mean_reversion_x=jnp.array(0.5), mean_reversion_y=jnp.array(0.1),
        volatility_x=jnp.array(0.01), volatility_ratio=jnp.array(0.8),
        correlation=jnp.array(-0.7), initial_curve=flat_curve,
    )


def _cheyette_1f(flat_curve, beta=0.0, kappa=0.10, sigma=0.01):
    return CheyetteModel(
        mean_reversion=jnp.array(kappa), volatility=jnp.array(sigma),
        skew=jnp.array(beta), initial_curve=flat_curve,
    )


@pytest.fixture
def cheyette_2f(flat_curve):
    return CheyetteModel2F(
        mean_reversion_1=jnp.array(0.5), mean_reversion_2=jnp.array(0.1),
        volatility_1=jnp.array(0.01), volatility_ratio=jnp.array(0.008) / jnp.array(0.01),
        skew_1=jnp.array(0.1), skew_2=jnp.array(0.05),
        correlation=jnp.array(-0.7), initial_curve=flat_curve,
    )


def _caplet(strike=0.03, is_cap=True, fix=(2027, 1, 1), end=(2027, 7, 1)):
    f = int(ymd_to_ordinal(*fix))
    e = int(ymd_to_ordinal(*end))
    return Caplet(
        fixing_date=jnp.int32(f), start_date=jnp.int32(f), end_date=jnp.int32(e),
        strike=jnp.array(strike), notional=jnp.array(1.0e6),
        is_cap=is_cap, day_count=_ACT365,
    )


def _semiannual_cap(strike=0.03, is_cap=True, n=6):
    fixings = jnp.array(
        [int(ymd_to_ordinal(2026 + i // 2, 1 if i % 2 == 0 else 7, 1))
         for i in range(n)],
        dtype=jnp.int32,
    )
    ends = jnp.array(
        [int(ymd_to_ordinal(2026 + (i + 1) // 2, 7 if i % 2 == 0 else 1, 1))
         for i in range(n)],
        dtype=jnp.int32,
    )
    return Cap(
        fixing_dates=fixings, start_dates=fixings, end_dates=ends,
        strike=jnp.array(strike), notional=jnp.array(1.0e6),
        is_cap=is_cap, day_count=_ACT365,
    )


def _hw_caplet_analytic(caplet, model):
    """Closed-form Hull-White caplet/floorlet via the ZCB option formula."""
    T = _YF(_REF, caplet.fixing_date)
    S = _YF(_REF, caplet.end_date)
    tau = _YF(caplet.start_date, caplet.end_date)
    strike_bond = 1.0 / (1.0 + caplet.strike * tau)
    zb = hw_zcb_option_price(
        model, T, jnp.array([S]), jnp.array([strike_bond]),
        is_call=not caplet.is_cap,
    )[0]
    return float(caplet.notional * (1.0 + caplet.strike * tau) * zb)


# ── Registration ─────────────────────────────────────────────────────


class TestRegistration:
    @pytest.mark.parametrize("instrument", ["Caplet", "Cap"])
    @pytest.mark.parametrize(
        "model",
        ["HullWhiteModel", "G2PPModel", "CheyetteModel", "CheyetteModel2F"],
    )
    def test_recipe_registered(self, instrument, model):
        assert (instrument, model) in registered_recipes()

    @pytest.mark.parametrize("model", ["G2PPModel", "CheyetteModel2F"])
    @pytest.mark.parametrize("instrument", ["CallableBond", "PuttableBond"])
    def test_two_factor_callable_registered(self, instrument, model):
        assert (instrument, model) in registered_recipes()


# ── Caplet vs the closed form ────────────────────────────────────────


class TestCapletVsAnalytic:
    @pytest.mark.parametrize("is_cap", [True, False])
    @pytest.mark.parametrize("strike", [0.02, 0.03, 0.04])
    def test_hw_caplet_matches_zcb_option(self, hw, is_cap, strike):
        caplet = _caplet(strike=strike, is_cap=is_cap)
        pde = float(pde_price_dispatch(caplet, hw, C1).price)
        analytic = _hw_caplet_analytic(caplet, hw)
        assert pde == pytest.approx(analytic, rel=2e-3)

    def test_hw_caplet_converges_to_analytic(self, hw):
        caplet = _caplet()
        analytic = _hw_caplet_analytic(caplet, hw)
        coarse = abs(
            float(pde_price_dispatch(
                caplet, hw, PDEConfig(n_spot=101, n_time=60, spot_range=6.0)
            ).price) - analytic
        )
        fine = abs(float(pde_price_dispatch(caplet, hw, C1).price) - analytic)
        assert fine < coarse


# ── Put-call parity (model-free, discretisation-only) ────────────────


class TestCapletFloorletParity:
    @pytest.mark.parametrize("model_name", ["hw", "g2", "chey"])
    def test_parity_matches_curve(self, request, flat_curve, model_name):
        models = {
            "hw": (request.getfixturevalue("hw"), C1),
            "g2": (request.getfixturevalue("g2"), C2),
            "chey": (_cheyette_1f(flat_curve, beta=0.1), C1),
        }
        model, cfg = models[model_name]
        caplet = _caplet(strike=0.035, is_cap=True)
        floorlet = _caplet(strike=0.035, is_cap=False)
        cpl = float(pde_price_dispatch(caplet, model, cfg).price)
        flr = float(pde_price_dispatch(floorlet, model, cfg).price)

        tau = float(_YF(caplet.start_date, caplet.end_date))
        p_fix = float(flat_curve(caplet.fixing_date))
        p_end = float(flat_curve(caplet.end_date))
        expected = 1.0e6 * (p_fix - (1.0 + 0.035 * tau) * p_end)
        assert (cpl - flr) == pytest.approx(expected, abs=1.0, rel=2e-3)


# ── Reduction: beta = 0 Cheyette caplet -> Hull-White caplet ─────────


class TestCheyetteReducesToHullWhite:
    @pytest.mark.parametrize("is_cap", [True, False])
    def test_beta_zero_caplet_matches_hw(self, flat_curve, hw, is_cap):
        chey = _cheyette_1f(flat_curve, beta=0.0)  # same kappa, sigma as hw
        caplet = _caplet(is_cap=is_cap)
        hw_px = float(pde_price_dispatch(caplet, hw, C1).price)
        chey_px = float(pde_price_dispatch(caplet, chey, C1).price)
        assert chey_px == pytest.approx(hw_px, rel=1e-6)

    def test_skew_moves_the_caplet(self, flat_curve):
        """A non-zero skew must change the caplet price (the beta signal)."""
        caplet = _caplet(strike=0.04)
        flat = float(pde_price_dispatch(
            caplet, _cheyette_1f(flat_curve, beta=0.0), C1
        ).price)
        skewed = float(pde_price_dispatch(
            caplet, _cheyette_1f(flat_curve, beta=0.3), C1
        ).price)
        assert abs(skewed - flat) > 1e-3 * flat


# ── PDE vs Monte Carlo (Cheyette) ────────────────────────────────────


class TestCapletVsMonteCarlo:
    def test_cheyette_caplet_pde_vs_mc(self, flat_curve):
        chey = _cheyette_1f(flat_curve, beta=0.1)
        caplet = _caplet(strike=0.03)
        table = price_all_engines(
            caplet, chey,
            pde_config=C1,
            mc_config=MCConfig(n_paths=60_000, n_steps=100),
            mc_key=jax.random.PRNGKey(0),
        )
        # PDE (frozen-y) vs exact-y MC: within MC noise plus the tiny frozen-y
        # bias. Three standard errors gives headroom against seed variation.
        diff = abs(table.price("pde") - table.price("mc"))
        assert diff < 3.0 * table.quotes["mc"].stderr


# ── Cap strip ────────────────────────────────────────────────────────


class TestCapStrip:
    def _analytic_cap(self, cap, model):
        total = 0.0
        n = cap.fixing_dates.shape[0]
        for i in range(n):
            tau = float(_YF(cap.start_dates[i], cap.end_dates[i]))
            T = _YF(_REF, cap.fixing_dates[i])
            S = _YF(_REF, cap.end_dates[i])
            strike_bond = 1.0 / (1.0 + float(cap.strike) * tau)
            zb = hw_zcb_option_price(
                model, T, jnp.array([S]), jnp.array([strike_bond]),
                is_call=not cap.is_cap,
            )[0]
            total += float(cap.notional * (1.0 + float(cap.strike) * tau) * zb)
        return total

    def test_cap_converges_to_analytic_sum(self, hw):
        cap = _semiannual_cap(strike=0.03, is_cap=True)
        analytic = self._analytic_cap(cap, hw)
        errs = []
        for nt in (200, 400, 800):
            cfg = PDEConfig(n_spot=401, n_time=nt, spot_range=6.0)
            errs.append(abs(float(pde_price_dispatch(cap, hw, cfg).price) - analytic))
        assert errs[-1] < errs[0]
        assert errs[-1] / max(analytic, 1.0) < 5e-3

    def test_floor_is_positive_and_distinct(self, hw):
        floor = _semiannual_cap(strike=0.03, is_cap=False)
        cap = _semiannual_cap(strike=0.03, is_cap=True)
        cfg = PDEConfig(n_spot=401, n_time=400, spot_range=6.0)
        floor_px = float(pde_price_dispatch(floor, hw, cfg).price)
        cap_px = float(pde_price_dispatch(cap, hw, cfg).price)
        assert floor_px > 0.0
        assert cap_px > floor_px  # curve at 3% above a 3% strike floor->ITM cap


# ── Two-factor callable / puttable ───────────────────────────────────


def _bond(cls, dates_key, dates, price):
    kw = dict(
        payment_dates=generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2).payment_dates,
        settlement_date=jnp.int32(_REF), coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0), frequency=2,
    )
    kw[dates_key] = dates
    kw[dates_key.replace("dates", "prices")] = jnp.full(dates.shape[0], price)
    return cls(**kw)


class TestTwoFactorCallablePuttable:
    @pytest.fixture
    def bullet(self):
        return FixedRateBond(
            payment_dates=generate_schedule(2025, 1, 1, 2030, 1, 1, frequency=2).payment_dates,
            settlement_date=jnp.int32(_REF), coupon_rate=jnp.array(0.04),
            face_value=jnp.array(100.0), frequency=2,
        )

    @pytest.fixture
    def dates(self):
        return jnp.array(
            [int(ymd_to_ordinal(y, 1, 1)) for y in (2027, 2028, 2029)],
            dtype=jnp.int32,
        )

    @pytest.mark.parametrize("model_name", ["g2", "chey2f"])
    def test_ordering(self, request, bullet, dates, model_name):
        model = (
            request.getfixturevalue("g2") if model_name == "g2"
            else request.getfixturevalue("cheyette_2f")
        )
        callable_b = _bond(CallableBond, "call_dates", dates, 1.0)
        puttable_b = _bond(PuttableBond, "put_dates", dates, 1.0)
        c = float(pde_price_dispatch(callable_b, model, C2).price)
        b = float(pde_price_dispatch(bullet, model, C2).price)
        p = float(pde_price_dispatch(puttable_b, model, C2).price)
        assert c < b < p

    @pytest.mark.parametrize("model_name", ["g2", "chey2f"])
    def test_unreachable_recovers_bullet(self, request, bullet, dates, model_name):
        model = (
            request.getfixturevalue("g2") if model_name == "g2"
            else request.getfixturevalue("cheyette_2f")
        )
        never_call = _bond(CallableBond, "call_dates", dates, 10.0)
        never_put = _bond(PuttableBond, "put_dates", dates, 0.0)
        b = float(pde_price_dispatch(bullet, model, C2).price)
        assert float(pde_price_dispatch(never_call, model, C2).price) == pytest.approx(b, abs=1e-9)
        assert float(pde_price_dispatch(never_put, model, C2).price) == pytest.approx(b, abs=1e-9)


# ── JIT / autodiff ───────────────────────────────────────────────────


class TestJitAndGreeks:
    def test_caplet_filter_jit(self, hw):
        caplet = _caplet()

        @eqx.filter_jit
        def price(m):
            return pde_price_dispatch(caplet, m, C1).price

        assert jnp.isfinite(price(hw))

    def test_caplet_vega_positive(self, flat_curve):
        caplet = _caplet(strike=0.03)

        def price(sigma):
            m = _cheyette_1f(flat_curve, beta=0.1, sigma=sigma)
            return pde_price_dispatch(caplet, m, C1).price

        grad = float(jax.grad(price)(jnp.array(0.01)))
        assert jnp.isfinite(grad)
        assert grad > 0.0

    def test_caplet_skew_grad_is_finite(self, flat_curve):
        """Autodiff must flow through the caplet PDE in the skew (calibration)."""
        caplet = _caplet(strike=0.04)

        def price(beta):
            m = CheyetteModel(
                mean_reversion=jnp.array(0.10), volatility=jnp.array(0.01),
                skew=beta, initial_curve=flat_curve,
            )
            return pde_price_dispatch(caplet, m, C1).price

        assert jnp.isfinite(float(jax.grad(price)(jnp.array(0.1))))
