r"""Tests for Cheyette calibration on top of the PDE.

Synthetic round-trip: generate target prices with a known model *through the
same PDE engine*, then recover the parameters. Because the PDE is exact-fitted to
the curve, the fit isolates the volatility level (:math:`\sigma`) and the skew
(:math:`\beta`), which is what a strike slice of swaptions identifies. Meshes are
deliberately coarse — the calibration loop reprices the whole instrument set at
every iteration — since the target is generated on the *same* mesh, so the
recovery is a clean inverse problem rather than a grid-convergence test.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.calibration.cheyette import (
    calibrate_cheyette,
    calibrate_cheyette_2f,
    cheyette_pde_prices,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.pricing.analytic.swaptions import _annuity
from valax.pricing.pde import PDEConfig, PDEConfig2D, Scheme

_ACT365 = "act_365"

# Coarse meshes keep PDE-in-the-loop calibration fast; targets are generated on
# the same mesh, so the round-trip is exact up to optimiser tolerance.
CFG_FROZEN_1F = PDEConfig(n_spot=81, n_time=36, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON)
CFG_EXACT_1F = PDEConfig2D(n_x=61, n_y=41, n_time=40, x_range=6.0, y_max=3.0)
CFG_FROZEN_2F = PDEConfig2D(n_x=61, n_y=61, n_time=32, x_range=6.0, scheme=Scheme.CRAIG_SNEYD)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ref_date():
    return int(ymd_to_ordinal(2026, 1, 1))


@pytest.fixture(scope="module")
def curve(ref_date):
    years = [0.0] + [float(k) for k in range(1, 22)]
    pillars = jnp.array(
        [ref_date + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref_date).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.03 * times),
        reference_date=jnp.int32(ref_date),
        day_count=_ACT365,
    )


def _fixed_dates(start_year, tenor_y):
    return jnp.array(
        [int(ymd_to_ordinal(start_year + k, 1, 1)) for k in range(1, tenor_y + 1)],
        dtype=jnp.int32,
    )


def _swaption(expiry_year, tenor_y, strike):
    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(expiry_year, 1, 1))),
        fixed_dates=_fixed_dates(expiry_year, tenor_y),
        strike=jnp.asarray(strike),
        notional=jnp.array(1.0),
        is_payer=True,
        day_count=_ACT365,
    )


def _atm(curve, expiry_year, tenor_y):
    expiry = jnp.int32(int(ymd_to_ordinal(expiry_year, 1, 1)))
    fixed = _fixed_dates(expiry_year, tenor_y)
    ann = _annuity(expiry, fixed, curve, _ACT365)
    return float((curve(expiry) - curve(fixed[-1])) / ann)


@pytest.fixture(scope="module")
def strike_slice(curve):
    """A strike slice at one expiry/tenor — where the skew shows."""
    k = _atm(curve, 2030, 5)
    return [
        _swaption(2030, 5, k - 0.006),
        _swaption(2030, 5, k - 0.003),
        _swaption(2030, 5, k),
        _swaption(2030, 5, k + 0.003),
        _swaption(2030, 5, k + 0.006),
    ]


# ── One factor, frozen-y ─────────────────────────────────────────────


class TestRoundTrip1F:
    def test_recovers_vol_and_skew_pinned_kappa(self, curve, strike_slice):
        true = CheyetteModel(
            mean_reversion=jnp.array(0.05),
            volatility=jnp.array(0.012),
            skew=jnp.array(0.25),
            initial_curve=curve,
        )
        market = cheyette_pde_prices(true, strike_slice, "frozen", CFG_FROZEN_1F)
        guess = CheyetteModel(
            mean_reversion=jnp.array(0.05),
            volatility=jnp.array(0.008),
            skew=jnp.array(0.0),
            initial_curve=curve,
        )
        fit, _ = calibrate_cheyette(
            strike_slice, market, curve,
            initial_guess=guess,
            fixed_params={"mean_reversion": 0.05},
            engine="frozen", config=CFG_FROZEN_1F, max_steps=200,
        )
        assert float(fit.volatility) == pytest.approx(0.012, rel=1e-2)
        assert float(fit.skew) == pytest.approx(0.25, abs=1e-2)

    def test_full_fit_reprices(self, curve, strike_slice):
        true = CheyetteModel(
            mean_reversion=jnp.array(0.05),
            volatility=jnp.array(0.011),
            skew=jnp.array(0.2),
            initial_curve=curve,
        )
        market = cheyette_pde_prices(true, strike_slice, "frozen", CFG_FROZEN_1F)
        fit, _ = calibrate_cheyette(
            strike_slice, market, curve,
            fixed_params={"mean_reversion": 0.05},
            engine="frozen", config=CFG_FROZEN_1F, max_steps=200,
        )
        reprice = cheyette_pde_prices(fit, strike_slice, "frozen", CFG_FROZEN_1F)
        rel = jnp.abs(reprice - market) / jnp.abs(market)
        assert float(jnp.max(rel)) < 1e-3


class TestRoundTrip1FExactEngine:
    def test_recovers_vol_and_skew_bias_free(self, curve):
        k = _atm(curve, 2029, 4)
        insts = [_swaption(2029, 4, k - 0.004), _swaption(2029, 4, k + 0.004)]
        true = CheyetteModel(
            mean_reversion=jnp.array(0.05),
            volatility=jnp.array(0.011),
            skew=jnp.array(0.2),
            initial_curve=curve,
        )
        market = cheyette_pde_prices(true, insts, "exact", CFG_EXACT_1F)
        fit, _ = calibrate_cheyette(
            insts, market, curve,
            fixed_params={"mean_reversion": 0.05},
            engine="exact", config=CFG_EXACT_1F, max_steps=150,
        )
        assert float(fit.volatility) == pytest.approx(0.011, rel=2e-2)
        assert float(fit.skew) == pytest.approx(0.2, abs=2e-2)


# ── Two factors, frozen-y ────────────────────────────────────────────


class TestRoundTrip2F:
    def test_recovers_vols_pinned_structure(self, curve):
        insts = [
            _swaption(2030, 5, _atm(curve, 2030, 5)),
            _swaption(2028, 3, _atm(curve, 2028, 3)),
            _swaption(2032, 4, _atm(curve, 2032, 4)),
        ]
        true = CheyetteModel2F(
            mean_reversion_1=jnp.array(0.5),
            mean_reversion_2=jnp.array(0.1),
            volatility_1=jnp.array(0.011),
            volatility_2=jnp.array(0.007),
            skew_1=jnp.array(0.1),
            skew_2=jnp.array(0.05),
            correlation=jnp.array(-0.5),
            initial_curve=curve,
        )
        market = cheyette_pde_prices(true, insts, "frozen", CFG_FROZEN_2F)
        guess = eqx.tree_at(
            lambda m: (m.volatility_1, m.volatility_2), true,
            (jnp.array(0.009), jnp.array(0.009)),
        )
        fit, _ = calibrate_cheyette_2f(
            insts, market, curve,
            initial_guess=guess,
            fixed_params={
                "mean_reversion_1": 0.5, "mean_reversion_2": 0.1,
                "skew_1": 0.1, "skew_2": 0.05, "correlation": -0.5,
            },
            config=CFG_FROZEN_2F, max_steps=200,
        )
        assert float(fit.volatility_1) == pytest.approx(0.011, rel=2e-2)
        assert float(fit.volatility_2) == pytest.approx(0.007, rel=3e-2)


# ── Guards ───────────────────────────────────────────────────────────


class TestGuards:
    def test_mismatched_counts(self, curve, strike_slice):
        with pytest.raises(ValueError, match="market prices"):
            calibrate_cheyette(
                strike_slice, jnp.ones(2), curve, config=CFG_FROZEN_1F
            )

    def test_unknown_fixed_param(self, curve, strike_slice):
        market = jnp.ones(len(strike_slice)) * 0.02
        with pytest.raises(ValueError, match="Unknown Cheyette parameter"):
            calibrate_cheyette(
                strike_slice, market, curve,
                fixed_params={"not_a_param": 1.0}, config=CFG_FROZEN_1F,
            )

    def test_unknown_engine(self, curve, strike_slice):
        market = jnp.ones(len(strike_slice)) * 0.02
        with pytest.raises(ValueError, match="Unknown engine"):
            calibrate_cheyette(
                strike_slice, market, curve, engine="bogus", config=CFG_FROZEN_1F
            )

    def test_curve_is_pinned_not_fitted(self, curve, strike_slice):
        true = CheyetteModel(
            mean_reversion=jnp.array(0.05),
            volatility=jnp.array(0.011),
            skew=jnp.array(0.1),
            initial_curve=curve,
        )
        market = cheyette_pde_prices(true, strike_slice, "frozen", CFG_FROZEN_1F)
        fit, _ = calibrate_cheyette(
            strike_slice, market, curve,
            fixed_params={"mean_reversion": 0.05},
            config=CFG_FROZEN_1F, max_steps=20,
        )
        assert jnp.array_equal(
            fit.initial_curve.discount_factors, curve.discount_factors
        )
        assert jnp.array_equal(
            fit.initial_curve.pillar_dates, curve.pillar_dates
        )
