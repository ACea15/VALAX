"""Hull-White calibration to a European swaption surface.

The decisive test is the synthetic round-trip: generate a surface *from* known
parameters, then recover them.  Any error in the pricer, the parameter
transforms, or the residual assembly shows up as a failure to return to the
generating values.

Also pins the honest limitation of a one-factor Gaussian model: it has two free
parameters and cannot reproduce an arbitrary market surface, so a flat
Black-76 vol surface must leave a visible residual.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.calibration.hull_white import (
    calibrate_hull_white,
    hw_swaption_prices,
    swaption_prices_from_vols,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.swaptions import _annuity

_DAY_COUNT = "act_365"
_NOTIONAL = 1_000_000.0
_TRUE_A = 0.08
_TRUE_SIGMA = 0.012

# A co-terminal-ish ATM grid: (expiry years, tenor years).
_GRID = [(1, 5), (2, 5), (3, 5), (5, 5), (1, 10), (2, 10), (5, 10), (7, 7), (10, 5)]


@pytest.fixture(scope="module")
def ref_date() -> int:
    return int(ymd_to_ordinal(2026, 1, 1))


@pytest.fixture(scope="module")
def curve(ref_date) -> DiscountCurve:
    years = [0.0] + [float(k) for k in range(1, 23)]
    pillars = jnp.array(
        [ref_date + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref_date).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.04 * times),
        reference_date=jnp.int32(ref_date),
        day_count=_DAY_COUNT,
    )


@pytest.fixture(scope="module")
def swaptions(ref_date, curve) -> list[Swaption]:
    """ATM payer swaptions across the expiry/tenor grid."""
    out = []
    for expiry_y, tenor_y in _GRID:
        expiry_ord = ref_date + expiry_y * 365
        fixed = jnp.array(
            [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
            dtype=jnp.int32,
        )
        ann = _annuity(jnp.int32(expiry_ord), fixed, curve, _DAY_COUNT)
        fwd = (curve(jnp.int32(expiry_ord)) - curve(fixed[-1])) / ann
        out.append(
            Swaption(
                expiry_date=jnp.int32(expiry_ord),
                fixed_dates=fixed,
                strike=fwd,
                notional=jnp.asarray(_NOTIONAL),
                is_payer=True,
                day_count=_DAY_COUNT,
            )
        )
    return out


@pytest.fixture(scope="module")
def true_model(curve) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.asarray(_TRUE_A),
        volatility=jnp.asarray(_TRUE_SIGMA),
        initial_curve=curve,
    )


@pytest.fixture(scope="module")
def synthetic_prices(true_model, swaptions):
    return hw_swaption_prices(true_model, swaptions)


# ── Round-trip recovery ───────────────────────────────────────────────

class TestSyntheticRoundTrip:
    @pytest.mark.parametrize("solver", ["gauss_newton", "bfgs"])
    def test_recovers_both_parameters(
        self, swaptions, synthetic_prices, curve, solver
    ):
        """Both solvers must return to the generating (a, sigma)."""
        start = HullWhiteModel(
            mean_reversion=jnp.asarray(0.04),
            volatility=jnp.asarray(0.008),
            initial_curve=curve,
        )
        fitted, _ = calibrate_hull_white(
            swaptions, synthetic_prices, curve,
            initial_guess=start, solver=solver,
        )
        assert float(fitted.mean_reversion) == pytest.approx(_TRUE_A, rel=1e-5)
        assert float(fitted.volatility) == pytest.approx(_TRUE_SIGMA, rel=1e-5)

    def test_residuals_vanish(self, swaptions, synthetic_prices, curve):
        fitted, _ = calibrate_hull_white(swaptions, synthetic_prices, curve)
        residual = hw_swaption_prices(fitted, swaptions) - synthetic_prices
        # Prices are O(1e4); 1e-6 absolute is ~1e-10 relative.
        assert float(jnp.max(jnp.abs(residual))) < 1e-6

    def test_recovers_sigma_with_mean_reversion_fixed(
        self, swaptions, synthetic_prices, curve
    ):
        """The recommended desk workflow: fix `a`, fit `sigma`."""
        fitted, _ = calibrate_hull_white(
            swaptions, synthetic_prices, curve,
            fixed_mean_reversion=jnp.asarray(_TRUE_A),
        )
        assert float(fitted.mean_reversion) == pytest.approx(_TRUE_A, rel=1e-12)
        assert float(fitted.volatility) == pytest.approx(_TRUE_SIGMA, rel=1e-5)

    @pytest.mark.parametrize(
        "start", [(0.02, 0.004), (0.20, 0.030), (0.08, 0.012), (0.5, 0.05)]
    )
    def test_insensitive_to_starting_point(
        self, swaptions, synthetic_prices, curve, start
    ):
        a0, s0 = start
        guess = HullWhiteModel(
            mean_reversion=jnp.asarray(a0),
            volatility=jnp.asarray(s0),
            initial_curve=curve,
        )
        fitted, _ = calibrate_hull_white(
            swaptions, synthetic_prices, curve, initial_guess=guess
        )
        assert float(fitted.mean_reversion) == pytest.approx(_TRUE_A, rel=1e-4)
        assert float(fitted.volatility) == pytest.approx(_TRUE_SIGMA, rel=1e-4)

    def test_fitted_parameters_stay_positive(
        self, swaptions, synthetic_prices, curve
    ):
        """The `positive()` transform must make negative output unreachable."""
        guess = HullWhiteModel(
            mean_reversion=jnp.asarray(0.001),
            volatility=jnp.asarray(0.0001),
            initial_curve=curve,
        )
        fitted, _ = calibrate_hull_white(
            swaptions, synthetic_prices, curve, initial_guess=guess
        )
        assert float(fitted.mean_reversion) > 0.0
        assert float(fitted.volatility) > 0.0


# ── Model limitations, asserted rather than assumed ───────────────────

class TestOneFactorLimits:
    def test_flat_black_surface_cannot_be_fitted_exactly(
        self, swaptions, curve
    ):
        """Two parameters cannot reproduce a flat lognormal vol surface.

        Hull-White generates a *normal* vol structure that decays with expiry;
        a flat Black-76 surface is a different shape, so a visible residual is
        the correct outcome. If this ever passed with ~0 residual, the pricer
        would not be responding to the surface at all.
        """
        market_vols = jnp.full((len(swaptions),), 0.25)
        market_prices = swaption_prices_from_vols(swaptions, curve, market_vols)

        fitted, _ = calibrate_hull_white(
            swaptions, market_prices, curve,
            fixed_mean_reversion=jnp.asarray(0.05),
        )
        rel = (hw_swaption_prices(fitted, swaptions) - market_prices) / market_prices
        rms = float(jnp.sqrt(jnp.mean(rel ** 2)))
        assert 0.01 < rms < 0.5, f"expected a visible but bounded misfit, got {rms}"

    def test_curve_is_not_a_calibration_degree_of_freedom(
        self, swaptions, synthetic_prices, curve
    ):
        """The fitted model keeps the supplied curve verbatim (exact-fit)."""
        fitted, _ = calibrate_hull_white(swaptions, synthetic_prices, curve)
        assert jnp.allclose(
            fitted.initial_curve.discount_factors, curve.discount_factors
        )


# ── API contract ──────────────────────────────────────────────────────

class TestAPI:
    def test_rejects_length_mismatch(self, swaptions, curve):
        with pytest.raises(ValueError, match="market prices"):
            calibrate_hull_white(swaptions, jnp.ones(3), curve)

    def test_rejects_unknown_solver(self, swaptions, synthetic_prices, curve):
        with pytest.raises(ValueError, match="Unknown solver"):
            calibrate_hull_white(
                swaptions, synthetic_prices, curve, solver="nelder_mead"
            )

    def test_vols_to_prices_roundtrip(self, swaptions, curve):
        """`swaption_prices_from_vols` must agree with the Black-76 pricer."""
        from valax.pricing.analytic.swaptions import swaption_price_black76

        vols = jnp.linspace(0.15, 0.35, len(swaptions))
        prices = swaption_prices_from_vols(swaptions, curve, vols)
        for i, sw in enumerate(swaptions):
            assert float(prices[i]) == pytest.approx(
                float(swaption_price_black76(sw, curve, vols[i])), rel=1e-12
            )

    def test_normal_quotes_use_bachelier(self, swaptions, curve):
        from valax.pricing.analytic.swaptions import swaption_price_bachelier

        vols = jnp.full((len(swaptions),), 0.01)
        prices = swaption_prices_from_vols(swaptions, curve, vols, normal=True)
        assert float(prices[0]) == pytest.approx(
            float(swaption_price_bachelier(swaptions[0], curve, vols[0])), rel=1e-12
        )

    def test_model_prices_are_jittable(self, true_model, swaptions):
        jitted = eqx.filter_jit(hw_swaption_prices)(true_model, swaptions)
        eager = hw_swaption_prices(true_model, swaptions)
        assert jnp.allclose(jitted, eager, rtol=1e-12)

    def test_model_prices_are_differentiable(self, true_model, swaptions):
        grads = eqx.filter_grad(
            lambda m: jnp.sum(hw_swaption_prices(m, swaptions))
        )(true_model)
        assert float(grads.volatility) > 0.0
        assert jnp.isfinite(grads.mean_reversion)
