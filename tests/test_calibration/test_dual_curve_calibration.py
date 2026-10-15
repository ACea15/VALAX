"""Dual-curve (OIS-discounted, tenor-projected) rates calibration.

Before this landed, every rates calibrator took one ``DiscountCurve`` used for
discounting *and* projection, while the P5.1 MC/PDE engines could already price
dual-curve.  A model fitted on a blended curve and then priced dual-curve
violates the model/market consistency invariant: the basis gets absorbed into
:math:`\\sigma`.  ``forward_curve`` closes that gap on both legs of the residual
— the model price *and* the vol-to-price conversion of the targets.

Gates:

1. **Single-curve non-regression.** Every existing call path is untouched:
   omitting ``forward_curve`` reproduces the previous fitted parameters
   exactly.  Multi-curve is an upgrade, not a replacement.
2. **Round-trip identifiability.** Targets generated from known parameters
   under two curves are recovered by the dual-curve fit.
3. **The basis actually matters.** Fitting dual-curve targets with the
   single-curve residual (the old behaviour) gives a *different*, biased
   :math:`\\sigma` — this is the bug the feature fixes, asserted directly.
4. **Consistency with pricing.** A model fitted dual-curve reprices its own
   targets through the dual-curve pricer.
5. **Honest refusal.** Cheyette, whose PDE engine has no discount/projection
   split, raises rather than silently ignoring a projection curve.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.calibration.g2pp import calibrate_g2pp, g2pp_swaption_prices
from valax.calibration.hull_white import (
    calibrate_hull_white,
    hw_swaption_prices,
    swaption_prices_from_vols,
)
from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.swaptions import swaption_price_black76


REF = ymd_to_ordinal(2025, 1, 1)


def _flat_curve(rate: float, n_years: int = 21) -> DiscountCurve:
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n_years)],
        dtype=jnp.int32,
    )
    times = (pillars - int(REF)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=REF,
        day_count="act_365",
    )


@pytest.fixture(scope="module")
def ois() -> DiscountCurve:
    return _flat_curve(0.03)


@pytest.fixture(scope="module")
def projection() -> DiscountCurve:
    return _flat_curve(0.035)


def _swaption(expiry_years: int, tail_years: int, strike: float = 0.032) -> Swaption:
    return Swaption(
        expiry_date=jnp.int32(int(ymd_to_ordinal(2025 + expiry_years, 1, 1))),
        fixed_dates=jnp.array(
            [
                int(ymd_to_ordinal(2025 + expiry_years + i, 1, 1))
                for i in range(1, tail_years + 1)
            ],
            dtype=jnp.int32,
        ),
        strike=jnp.array(strike),
        notional=jnp.array(1_000_000.0),
        is_payer=True,
        day_count="act_365",
    )


@pytest.fixture(scope="module")
def surface() -> tuple[Swaption, ...]:
    """A small ATM-ish expiry/tenor grid."""
    return tuple(
        _swaption(e, t) for e, t in ((1, 5), (2, 5), (3, 5), (2, 10), (5, 10))
    )


def _truth_hw(curve: DiscountCurve) -> HullWhiteModel:
    return HullWhiteModel(
        mean_reversion=jnp.array(0.08),
        volatility=jnp.array(0.012),
        initial_curve=curve,
    )


def _truth_g2pp(curve: DiscountCurve) -> G2PPModel:
    return G2PPModel(
        mean_reversion_x=jnp.array(0.50),
        mean_reversion_y=jnp.array(0.10),
        volatility_x=jnp.array(0.011),
        volatility_ratio=jnp.array(0.007) / jnp.array(0.011),
        correlation=jnp.array(-0.70),
        initial_curve=curve,
    )


# ── Gate 1: single-curve non-regression ───────────────────────────────

class TestSingleCurveNonRegression:
    def test_hull_white_default_matches_explicit_none(self, ois, surface):
        truth = _truth_hw(ois)
        targets = hw_swaption_prices(truth, surface)

        default = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08)
        ).value
        explicit = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08),
            forward_curve=None,
        ).value
        assert float(default.volatility[0]) == float(explicit.volatility[0])

    def test_hull_white_recovers_sigma_single_curve(self, ois, surface):
        """The pre-existing round-trip still holds."""
        truth = _truth_hw(ois)
        targets = hw_swaption_prices(truth, surface)
        fitted = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08)
        ).value
        assert float(fitted.volatility[0]) == pytest.approx(0.012, rel=1e-4)

    def test_projection_equal_to_discount_is_single_curve(self, ois, surface):
        truth = _truth_hw(ois)
        targets = hw_swaption_prices(truth, surface)
        single = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08)
        ).value
        same = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08),
            forward_curve=ois,
        ).value
        assert float(same.volatility[0]) == float(single.volatility[0])


# ── Gate 2: dual-curve round-trip ─────────────────────────────────────

class TestDualCurveRoundTrip:
    def test_hull_white_recovers_sigma(self, ois, projection, surface):
        """Targets built dual-curve are refitted dual-curve to the truth."""
        truth = _truth_hw(ois)
        targets = hw_swaption_prices(truth, surface, projection)

        fitted = calibrate_hull_white(
            surface, targets, ois,
            fixed_mean_reversion=jnp.array(0.08),
            forward_curve=projection,
        ).value
        assert float(fitted.volatility[0]) == pytest.approx(0.012, rel=1e-4)
        # The fitted model is curve-free: the discount curve it is exact-fitted
        # to is bound from the market at pricing time, not embedded.
        assert fitted.initial_curve is None

    def test_g2pp_recovers_vols_and_correlation(self, ois, projection, surface):
        truth = _truth_g2pp(ois)
        targets = g2pp_swaption_prices(truth, surface, projection)

        fitted = calibrate_g2pp(
            surface, targets, ois,
            fixed_params={
                "mean_reversion_x": jnp.array(0.50),
                "mean_reversion_y": jnp.array(0.10),
            },
            forward_curve=projection,
        ).value
        assert float(fitted.volatility_x[0]) == pytest.approx(0.011, rel=5e-3)
        assert float(fitted.volatility_y[0]) == pytest.approx(0.007, rel=5e-3)
        assert float(fitted.correlation) == pytest.approx(-0.70, rel=5e-2)


# ── Gate 3: the basis is not absorbable into sigma ─────────────────────

class TestBasisMatters:
    def test_ignoring_the_basis_biases_sigma(self, ois, projection, surface):
        """Fitting dual-curve targets single-curve gives the wrong sigma.

        This is the inconsistency the feature removes: the old code path had no
        way to express the projection curve, so the basis leaked into the fitted
        volatility.
        """
        truth = _truth_hw(ois)
        targets = hw_swaption_prices(truth, surface, projection)

        wrong = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08)
        ).value
        right = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08),
            forward_curve=projection,
        ).value
        assert float(right.volatility[0]) == pytest.approx(0.012, rel=1e-4)
        # The single-curve fit must miss, and by a margin far beyond solver noise.
        assert abs(float(wrong.volatility[0]) - 0.012) > 0.1 * 0.012

    def test_dual_curve_fit_reprices_its_targets(self, ois, projection, surface):
        truth = _truth_hw(ois)
        targets = hw_swaption_prices(truth, surface, projection)
        fitted = calibrate_hull_white(
            surface, targets, ois,
            fixed_mean_reversion=jnp.array(0.08), forward_curve=projection,
        ).value
        repriced = hw_swaption_prices(bind_curves(fitted, ois), surface, projection)
        assert jnp.allclose(repriced, targets, rtol=1e-4)


# ── Vol-to-price conversion ───────────────────────────────────────────

class TestVolsToPrices:
    def test_dual_curve_conversion_differs_and_matches_pricer(
        self, ois, projection, surface
    ):
        vols = jnp.full((len(surface),), 0.25)
        single = swaption_prices_from_vols(surface, ois, vols)
        dual = swaption_prices_from_vols(surface, ois, vols, forward_curve=projection)

        # The same quoted vol implies a different price once the forward swap
        # rate is projected off a different curve.
        assert not jnp.allclose(single, dual)
        expected = jnp.stack(
            [
                swaption_price_black76(sw, ois, vols[i], projection)
                for i, sw in enumerate(surface)
            ]
        )
        assert jnp.allclose(dual, expected, rtol=1e-12)

    def test_default_is_unchanged(self, ois, surface):
        vols = jnp.full((len(surface),), 0.25)
        assert jnp.array_equal(
            swaption_prices_from_vols(surface, ois, vols),
            swaption_prices_from_vols(surface, ois, vols, forward_curve=None),
        )

    def test_end_to_end_from_quoted_vols(self, ois, projection, surface):
        """Quotes -> dual-curve target prices -> dual-curve fit, consistently."""
        vols = jnp.full((len(surface),), 0.22)
        targets = swaption_prices_from_vols(
            surface, ois, vols, forward_curve=projection
        )
        fitted = calibrate_hull_white(
            surface, targets, ois,
            fixed_mean_reversion=jnp.array(0.08), forward_curve=projection,
        ).value
        model_prices = hw_swaption_prices(
            bind_curves(fitted, ois), surface, projection
        )
        # One factor cannot fit a whole surface exactly; require the fit to be
        # in the right ballpark rather than exact.
        assert jnp.all(jnp.abs(model_prices - targets) / targets < 0.35)
        assert float(fitted.volatility[0]) > 0.0


# ── Model-market binding ──────────────────────────────────────────────

class TestCurveIdBinding:
    def test_curve_id_recorded_on_fitted_model(self, ois, projection, surface):
        targets = hw_swaption_prices(_truth_hw(ois), surface, projection)
        fitted = calibrate_hull_white(
            surface, targets, ois,
            fixed_mean_reversion=jnp.array(0.08),
            forward_curve=projection,
            curve_id="USD.SOFR.OIS",
        ).value
        assert fitted.curve_id == "USD.SOFR.OIS"

    def test_g2pp_curve_id_recorded(self, ois, surface):
        targets = g2pp_swaption_prices(_truth_g2pp(ois), surface)
        fitted = calibrate_g2pp(
            surface, targets, ois,
            fixed_params={
                "mean_reversion_x": jnp.array(0.50),
                "mean_reversion_y": jnp.array(0.10),
            },
            curve_id="EUR.ESTR.OIS",
        ).value
        assert fitted.curve_id == "EUR.ESTR.OIS"

    def test_default_curve_id_is_unpinned(self, ois, surface):
        """No curve_id: the model binds to the instrument's discount curve."""
        targets = hw_swaption_prices(_truth_hw(ois), surface)
        fitted = calibrate_hull_white(
            surface, targets, ois, fixed_mean_reversion=jnp.array(0.08)
        ).value
        assert fitted.curve_id is None

    def test_fitted_model_binds_to_a_shocked_market(self, ois, surface):
        """A calibrated model with a recorded id binds from a MarketState."""
        from valax.core.market_state import MarketState
        from valax.curves.graph import CurveGraph

        targets = hw_swaption_prices(_truth_hw(ois), surface)
        fitted = calibrate_hull_white(
            surface, targets, ois,
            fixed_mean_reversion=jnp.array(0.08), curve_id="USD.SOFR.OIS",
        ).value
        shocked = MarketState(
            curves=CurveGraph(curves={"USD.SOFR.OIS": _flat_curve(0.04)})
        )
        rebound = bind_curves(fitted, shocked)
        assert jnp.allclose(
            rebound.initial_curve.discount_factors,
            _flat_curve(0.04).discount_factors,
        )
        assert float(rebound.volatility[0]) == float(fitted.volatility[0])


# ── Honest refusal where the engine cannot do it ──────────────────────

class TestCheyetteRefuses:
    def _setup(self):
        from valax.models.cheyette import CheyetteModel

        curve = _flat_curve(0.03)
        model = CheyetteModel(
            mean_reversion=jnp.array(0.05),
            volatility=jnp.array(0.01),
            skew=jnp.array(0.0),
            initial_curve=curve,
        )
        return curve, model

    def test_distinct_projection_curve_raises(self):
        from valax.calibration.cheyette import calibrate_cheyette

        curve, _ = self._setup()
        with pytest.raises(ValueError, match="does not yet support a projection"):
            calibrate_cheyette(
                [_swaption(2, 5)], jnp.array([1000.0]), curve,
                forward_curve=_flat_curve(0.035),
            )

    def test_2f_distinct_projection_curve_raises(self):
        from valax.calibration.cheyette import calibrate_cheyette_2f

        curve, _ = self._setup()
        with pytest.raises(ValueError, match="does not yet support a projection"):
            calibrate_cheyette_2f(
                [_swaption(2, 5)], jnp.array([1000.0]), curve,
                forward_curve=_flat_curve(0.035),
            )


# ── JAX behaviour of the dual-curve residual ──────────────────────────

class TestJax:
    def test_grad_flows_through_the_dual_curve_residual(
        self, ois, projection, surface
    ):
        """Exact autodiff Jacobian, including w.r.t. the projection curve."""
        targets = hw_swaption_prices(_truth_hw(ois), surface, projection)

        def sse(dfs):
            fc = eqx.tree_at(lambda c: c.discount_factors, projection, dfs)
            model = _truth_hw(ois)
            return jnp.sum((hw_swaption_prices(model, surface, fc) - targets) ** 2)

        grad = jax.grad(sse)(projection.discount_factors)
        assert bool(jnp.all(jnp.isfinite(grad)))
