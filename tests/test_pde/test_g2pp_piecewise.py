"""G2++ 2-D ADI PDE with piecewise-constant σ(t) and ratio r(t).

Each backward step diffuses at the exact step averages of σ², η², ρση (a
per-step operator assembled inside the ADI loop) and discounts by the exact
step integral of φ. Oracles:

* the analytic Brigo-Mercurio price with the same σ(t), η(t) — converging
  under refinement (measured relative errors on the 3Y×5Y ATM payer:
  61²×50 −3.2e-3, 111²×90 −1.1e-3, 161²×140 −4.3e-4);
* the curve itself, for an option-free bond (exact-fit discipline);
* the constant-operator path, for two equal buckets;
* co-terminal European bounds for the Bermudan.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.accrual import accrual_periods
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.bonds import FixedRateBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import PDEConfig2D, Scheme, pde_price_dispatch

_ACT365 = "act_365"
REF = int(ymd_to_ordinal(2026, 1, 1))
VOLS = (0.012, 0.008, 0.015)
RATIOS = (0.8, 0.6, 0.9)
TIMES = (1.0, 2.5)

COARSE = PDEConfig2D(n_x=61, n_y=61, n_time=50, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)
MODERATE = PDEConfig2D(n_x=111, n_y=111, n_time=90, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)
FINE = PDEConfig2D(n_x=161, n_y=161, n_time=140, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)


def _curve():
    years = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0)
    zeros = (0.020, 0.021, 0.022, 0.025, 0.027, 0.031, 0.034, 0.038, 0.040, 0.041)
    pillars = jnp.array([REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-jnp.asarray(zeros) * times),
        reference_date=jnp.int32(REF),
        day_count=_ACT365,
    )


CURVE = _curve()


def _model(vols=VOLS, ratios=RATIOS, times=TIMES):
    return G2PPModel(
        mean_reversion_x=jnp.asarray(0.5),
        mean_reversion_y=jnp.asarray(0.1),
        volatility_x=jnp.asarray(vols),
        volatility_ratio=jnp.asarray(ratios),
        correlation=jnp.asarray(-0.7),
        initial_curve=CURVE,
        volatility_times=jnp.asarray(times),
    )


def _fixed(start_y, n):
    return jnp.array([REF + 365 * (start_y + k) for k in range(1, n + 1)], dtype=jnp.int32)


def _atm(expiry_ord, fixed):
    ann = _annuity(jnp.int32(expiry_ord), fixed, CURVE, _ACT365)
    return float((CURVE(jnp.int32(expiry_ord)) - CURVE(fixed[-1])) / ann)


def _swaption(expiry_y, tenor_y, moneyness=0.0, is_payer=True, strike=None):
    expiry = REF + 365 * expiry_y
    fixed = _fixed(expiry_y, tenor_y)
    k = _atm(expiry, fixed) + moneyness if strike is None else strike
    return Swaption(
        expiry_date=jnp.int32(expiry), fixed_dates=fixed, strike=jnp.asarray(k),
        notional=jnp.asarray(1.0), is_payer=is_payer, day_count=_ACT365,
    )


def _pde(inst, model, config=MODERATE):
    return float(pde_price_dispatch(inst, model, config).price)


class TestEuropeanVsAnalytic:
    def test_converges_to_analytic(self):
        """Errors shrink under refinement; the finest is within 1e-3."""
        s = _swaption(3, 5)
        model = _model()
        analytic = float(g2pp_swaption_price(s, model))
        errs = [abs(_pde(s, model, cfg) / analytic - 1.0) for cfg in (COARSE, MODERATE, FINE)]
        assert errs[0] > errs[1] > errs[2]
        assert errs[2] < 1e-3

    @pytest.mark.parametrize("expiry_y,tenor_y,moneyness,is_payer", [
        (3, 5, 0.0, False), (2, 4, 0.01, True), (4, 3, -0.01, False), (1, 6, 0.0, True),
    ])
    def test_moderate_grid(self, expiry_y, tenor_y, moneyness, is_payer):
        s = _swaption(expiry_y, tenor_y, moneyness, is_payer)
        model = _model()
        analytic = float(g2pp_swaption_price(s, model))
        assert _pde(s, model) == pytest.approx(analytic, rel=3e-3, abs=2e-5)


class TestExactFitAndConstantCase:
    def test_bond_reprices_curve(self):
        fixed = _fixed(0, 8)
        bond = FixedRateBond(
            payment_dates=fixed, settlement_date=jnp.int32(REF),
            coupon_rate=jnp.asarray(0.035), face_value=jnp.asarray(1.0),
            frequency=1, day_count=_ACT365,
        )
        starts, _ = accrual_periods(fixed, None, 1)
        taus = year_fraction(starts, fixed, _ACT365)
        dfs = CURVE(fixed)
        analytic = float(jnp.sum(0.035 * taus * dfs) + dfs[-1])
        assert _pde(bond, _model()) == pytest.approx(analytic, abs=1e-4)

    def test_equal_buckets_match_constant_operator(self):
        """Two equal buckets take the per-step path; the constant model the
        single-operator path. Same scheme, so they agree to rounding."""
        const = _model(vols=(0.011,), ratios=(0.7,), times=())
        split = _model(vols=(0.011, 0.011, 0.011), ratios=(0.7, 0.7, 0.7), times=TIMES)
        s = _swaption(3, 5)
        assert _pde(s, split, COARSE) == pytest.approx(_pde(s, const, COARSE), rel=1e-11)


class TestBermudan:
    def _coterminal(self):
        exercise_y = (1, 2, 3, 4)
        fixed = _fixed(1, 5)  # years 2..6
        strike = _atm(REF + 365, fixed)
        berm = BermudanSwaption(
            exercise_dates=jnp.array([REF + 365 * y for y in exercise_y], dtype=jnp.int32),
            fixed_dates=fixed, strike=jnp.asarray(strike), notional=jnp.asarray(1.0),
            is_payer=True, day_count=_ACT365,
        )
        europeans = [
            Swaption(
                expiry_date=jnp.int32(REF + 365 * y), fixed_dates=fixed[i:],
                strike=jnp.asarray(strike), notional=jnp.asarray(1.0),
                is_payer=True, day_count=_ACT365,
            )
            for i, y in enumerate(exercise_y)
        ]
        return berm, europeans

    def test_dominates_coterminal_europeans(self):
        model = _model()
        berm, europeans = self._coterminal()
        b = _pde(berm, model)
        pde_euro = [_pde(e, model) for e in europeans]
        analytic_euro = [float(g2pp_swaption_price(e, model)) for e in europeans]
        assert b >= max(pde_euro) - 1e-9
        assert b <= sum(pde_euro) + 1e-9
        # Against the exact Europeans, up to the grid's European error.
        assert b >= max(analytic_euro) * (1.0 - 3e-3)

    def test_single_exercise_is_european(self):
        model = _model()
        s = _swaption(3, 5)
        berm = BermudanSwaption(
            exercise_dates=s.expiry_date[None], fixed_dates=s.fixed_dates,
            strike=s.strike, notional=s.notional, is_payer=True, day_count=_ACT365,
        )
        assert _pde(berm, model, COARSE) == pytest.approx(_pde(s, model, COARSE), rel=1e-12)


class TestDifferentiationAndJIT:
    def test_filter_jit_smoke(self):
        s = _swaption(3, 5)

        @eqx.filter_jit
        def f(m):
            return pde_price_dispatch(s, m, COARSE).price

        model = _model()
        assert float(f(model)) == pytest.approx(_pde(s, model, COARSE), rel=1e-12)

    def test_bucket_vega_matches_fd(self):
        """Autodiff through the per-step operators vs FD of the same grid."""
        s = _swaption(3, 5)
        model = _model()

        def f(vols):
            return pde_price_dispatch(
                s, eqx.tree_at(lambda m: m.volatility_x, model, vols), COARSE
            ).price

        v0 = model.volatility_x
        grad = np.asarray(jax.grad(f)(v0))
        h = 1e-6
        for i in range(v0.shape[0]):
            e = jnp.zeros(v0.shape[0]).at[i].set(h)
            fd = (float(f(v0 + e)) - float(f(v0 - e))) / (2 * h)
            assert grad[i] == pytest.approx(fd, rel=1e-4)
