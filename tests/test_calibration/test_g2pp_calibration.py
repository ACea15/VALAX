"""G2++ calibration to a European swaption surface.

    The decisive test is the synthetic round-trip.  G2++'s two mean reversions are
only weakly identified by an ATM surface (the classic ``a``/``b`` vs vol
degeneracy), so the load-bearing recovery test *pins the mean reversions* and
recovers the two volatilities and the correlation.  The objective also has
degenerate basins (``sigma -> 0`` with ``rho -> +/-1``), so recovery is started
from a sensible prior rather than an arbitrary point -- realistic desk
practice.  A separate test confirms that a full five-parameter fit at least
reprices the generating surface to tight tolerance (a valid, if non-unique,
optimum).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

from valax.calibration.g2pp import calibrate_g2pp, g2pp_swaption_prices
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.swaptions import _annuity

_DAY_COUNT = "act_365"
_NOTIONAL = 1_000_000.0

_TRUE = dict(
    mean_reversion_x=0.50,
    mean_reversion_y=0.10,
    volatility_x=0.012,
    volatility_y=0.007,
    correlation=-0.65,
)

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
def true_model(curve) -> G2PPModel:
    return G2PPModel(
        mean_reversion_x=jnp.asarray(_TRUE["mean_reversion_x"]),
        mean_reversion_y=jnp.asarray(_TRUE["mean_reversion_y"]),
        volatility_x=jnp.asarray(_TRUE["volatility_x"]),
        volatility_y=jnp.asarray(_TRUE["volatility_y"]),
        correlation=jnp.asarray(_TRUE["correlation"]),
        initial_curve=curve,
    )


@pytest.fixture(scope="module")
def market_prices(true_model, swaptions):
    return g2pp_swaption_prices(true_model, swaptions)


class TestRoundTrip:
    def test_recovers_vols_and_rho_with_pinned_mean_reversions(
        self, swaptions, market_prices, curve
    ):
        """Pin (a, b) at truth; recover (sigma, eta, rho)."""
        fitted, sol = calibrate_g2pp(
            swaptions,
            market_prices,
            curve,
            fixed_params={
                "mean_reversion_x": _TRUE["mean_reversion_x"],
                "mean_reversion_y": _TRUE["mean_reversion_y"],
            },
            initial_guess=G2PPModel(
                mean_reversion_x=jnp.asarray(0.5),
                mean_reversion_y=jnp.asarray(0.1),
                volatility_x=jnp.asarray(0.011),
                volatility_y=jnp.asarray(0.008),
                correlation=jnp.asarray(-0.5),
                initial_curve=curve,
            ),
        )
        assert float(fitted.volatility_x) == pytest.approx(_TRUE["volatility_x"], rel=2e-2)
        assert float(fitted.volatility_y) == pytest.approx(_TRUE["volatility_y"], rel=2e-2)
        assert float(fitted.correlation) == pytest.approx(_TRUE["correlation"], abs=2e-2)

    def test_full_fit_reprices_surface(self, swaptions, market_prices, curve):
        """A full 5-parameter fit reprices the generating surface tightly."""
        fitted, sol = calibrate_g2pp(swaptions, market_prices, curve)
        model_prices = g2pp_swaption_prices(fitted, swaptions)
        rel = jnp.abs(model_prices - market_prices) / jnp.abs(market_prices)
        assert float(jnp.max(rel)) < 1e-3


class TestGuards:
    def test_mismatched_counts_raises(self, swaptions, curve):
        with pytest.raises(ValueError):
            calibrate_g2pp(swaptions, jnp.ones(len(swaptions) + 1), curve)

    def test_unknown_fixed_param_raises(self, swaptions, market_prices, curve):
        with pytest.raises(ValueError):
            calibrate_g2pp(
                swaptions, market_prices, curve, fixed_params={"not_a_param": 1.0}
            )

    def test_curve_is_pinned_not_fitted(self, swaptions, market_prices, curve):
        fitted, _ = calibrate_g2pp(swaptions, market_prices, curve)
        # The curve is exact-fitted, not a calibration DOF: its discount
        # factors must come through untouched.
        assert jnp.array_equal(
            fitted.initial_curve.discount_factors, curve.discount_factors
        )
        assert jnp.array_equal(
            fitted.initial_curve.pillar_dates, curve.pillar_dates
        )
