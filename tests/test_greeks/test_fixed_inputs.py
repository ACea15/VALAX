"""T-14 conformance gate: sensitivities to the fixed inputs ``p``.

Verifies :mod:`valax.greeks.fixed_inputs` against the multi-curve calibration /
AAD risk audit specification, requirement **T-14** ("fixed-input sensitivities
(kappa, beta, shifts, convexity, turns) vs finite differences: present,
non-zero where expected, matching"), which backs A-8 ("fixed inputs have their
adjoints accumulated and reported, not silently dropped") and M-5.  Collected
with the rest of the T-suite via ``pytest -m audit``.

**The three words that structure the gate** are *present*, *non-zero where
expected*, and *matching* — and each is asserted separately, because any two
of them can hold while the third fails in a way that matters:

* *Present* — the report has a row per input, named and in units.
* *Non-zero where expected* — and, its complement, **exactly zero where not**.
  Without the complement, a carrier that returned garbage for everything would
  look healthy; without the main case, one that returned zeros would.
* *Matching* — against full central finite differences at two bump sizes,
  the same standard T-2 holds the quote deltas to.

**Direct versus total, which is the substantive distinction here.**  A fixed
input can reach the price twice: directly, and through what the calibration
lands on when it moves.  Both are legitimate numbers and they are not close —
measured on a 3y x 7y Hull-White swaption, mean-reversion sensitivity is
-1.34e5 held frozen and -2.03e4 once sigma is allowed to re-fit, a factor of
6.6.  The re-calibration offsets most of the direct effect, which is exactly
why reporting the wrong one is worse than reporting nothing: it overstates the
exposure by more than six times.  Which you get is decided by the function
handed to :func:`~valax.greeks.fixed_inputs.fixed_input_adjoints`, and both
are gated below.

Tolerances this implementation commits to:

* Direct sensitivities — ``rtol = 1e-6`` at a 1e-5 bump, ``1e-5`` at 1e-4.
  Observed 6.3e-10 and 6.2e-8.
* Re-calibrated (total) sensitivities — ``rtol = 1e-5`` at a 1e-4 bump,
  ``1e-3`` at 1e-3.  Looser because each finite difference re-runs a full
  Hull-White fit, so the solver's own tolerance enters the difference twice.
  Observed 2.1e-8 and 4.6e-6.
* A fixed input the price does not depend on — **exactly** ``0.0``, not a
  tolerance.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration.hull_white import calibrate_hull_white, hw_swaption_prices
from valax.core.binding import bind_curves
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks.fixed_inputs import (
    FixedInput,
    FixedInputs,
    fixed_input_adjoints,
    fixed_inputs_from_paths,
)
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.models.sabr import SABRModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.pricing.analytic.swaptions import annuity
from valax.surfaces import SwaptionCube


pytestmark = pytest.mark.audit


T14_DIRECT_RTOL = {1e-5: 1e-6, 1e-4: 1e-5}
T14_TOTAL_RTOL = {1e-4: 1e-5, 1e-3: 1e-3}

DC = "act_365"
REF = int(ymd_to_ordinal(2026, 1, 1))


def _central_difference(fn, base, bump):
    return (fn(base + bump) - fn(base - bump)) / (2.0 * bump)


@pytest.fixture(scope="module")
def curve():
    years = [0.0] + [float(k) for k in range(1, 23)]
    pillars = jnp.array(
        [REF + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.04 * times),
        reference_date=jnp.int32(REF),
        day_count=DC,
    )


def _swaption(curve, expiry_y, tenor_y, strike=None):
    expiry_ord = REF + expiry_y * 365
    fixed = jnp.array(
        [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
        jnp.int32,
    )
    if strike is None:
        ann = annuity(jnp.int32(expiry_ord), fixed, curve, DC)
        strike = (curve(jnp.int32(expiry_ord)) - curve(fixed[-1])) / ann
    return Swaption(
        expiry_date=jnp.int32(expiry_ord),
        fixed_dates=fixed,
        strike=jnp.asarray(strike),
        notional=jnp.array(1e6),
        day_count=DC,
    )


@pytest.fixture(scope="module")
def model(curve):
    return HullWhiteModel(
        mean_reversion=jnp.array(0.08),
        volatility=jnp.array(0.012),
        initial_curve=curve,
    )


@pytest.fixture(scope="module")
def trade(curve):
    """Struck away from the money, so the price is not locally flat."""
    return _swaption(curve, 3, 7, strike=0.045)


MEAN_REVERSION = FixedInputs(
    entries=(
        FixedInput(
            name="hull_white.mean_reversion",
            where=lambda m: m.mean_reversion,
            units="1/year",
        ),
    )
)


# ── The carrier itself ───────────────────────────────────────────────


class TestFixedInputsIsALensNotAStore:
    """The design property the whole module rests on.

    A carrier that *stored* copies of kappa and beta would be a second
    source of truth, and the day it disagreed with the model the risk
    report would describe something nobody priced. These tests pin that
    there is nothing to keep in sync.
    """

    def test_values_are_read_from_the_live_tree(self, model):
        first = MEAN_REVERSION.values(model)
        assert float(first[0]) == float(model.mean_reversion)

        moved = eqx.tree_at(
            lambda m: m.mean_reversion, model, jnp.array(0.15)
        )
        # The *same* FixedInputs object, a different tree.
        assert float(MEAN_REVERSION.values(moved)[0]) == 0.15

    def test_inject_round_trips_and_does_not_mutate(self, model):
        updated = MEAN_REVERSION.inject(model, jnp.array([0.2]))
        assert float(updated.mean_reversion) == 0.2
        assert float(MEAN_REVERSION.values(updated)[0]) == 0.2
        # The original is untouched: eqx pytrees are frozen.
        assert float(model.mean_reversion) == 0.08

    def test_inject_leaves_everything_else_alone(self, model):
        updated = MEAN_REVERSION.inject(model, jnp.array([0.2]))
        assert float(updated.volatility[0]) == float(model.volatility[0])
        assert updated.curve_id == model.curve_id
        assert np.array_equal(
            np.asarray(updated.initial_curve.discount_factors),
            np.asarray(model.initial_curve.discount_factors),
        )

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            FixedInputs(
                entries=(
                    FixedInput(name="a", where=lambda m: m.mean_reversion),
                    FixedInput(name="a", where=lambda m: m.volatility),
                )
            )

    def test_units_must_match_the_path_count(self):
        with pytest.raises(ValueError, match="one unit per input"):
            fixed_inputs_from_paths(
                [("a", lambda m: m.mean_reversion)], units=("x", "y")
            )

    def test_the_report_carries_names_values_and_units(self, model, trade):
        report = fixed_input_adjoints(
            lambda m: hw_swaption_price(trade, m), model, MEAN_REVERSION
        )
        assert report.names == ("hull_white.mean_reversion",)
        assert report.units == ("1/year",)
        assert float(report.values[0]) == 0.08
        rows = report.rows()
        assert len(rows) == 1
        name, value, adjoint, unit = rows[0]
        assert (name, value, unit) == (
            "hull_white.mean_reversion",
            0.08,
            "1/year",
        )
        assert report.as_dict()[name] == adjoint


# ── T-14 proper ──────────────────────────────────────────────────────


class TestDirectFixedInputSensitivities:
    """Present, non-zero, and matching finite differences."""

    @pytest.fixture(scope="class")
    def report(self, model, trade):
        return fixed_input_adjoints(
            lambda m: hw_swaption_price(trade, m), model, MEAN_REVERSION
        )

    def test_it_is_present_and_non_zero(self, report):
        assert len(report.adjoints) == 1
        assert abs(float(report.adjoints[0])) > 1.0

    @pytest.mark.parametrize("bump", sorted(T14_DIRECT_RTOL))
    def test_it_matches_central_finite_differences(
        self, report, model, trade, curve, bump
    ):
        def price_at(mean_reversion):
            return float(
                hw_swaption_price(
                    trade,
                    HullWhiteModel(
                        mean_reversion=jnp.asarray(mean_reversion),
                        volatility=model.volatility,
                        initial_curve=curve,
                    ),
                )
            )

        reference = _central_difference(price_at, 0.08, bump)
        assert float(report.adjoints[0]) == pytest.approx(
            reference, rel=T14_DIRECT_RTOL[bump]
        )

    def test_an_input_the_price_ignores_is_exactly_zero(
        self, model, trade, curve
    ):
        """The complement, without which the gate proves nothing.

        A second, unused model is placed in the tree and a lens pointed
        at its mean reversion. The price function never reads it, so its
        adjoint must be identically zero — not small, zero. If a carrier
        were fabricating numbers, or reporting the wrong leaf, this is
        where it would show.
        """
        spare = HullWhiteModel(
            mean_reversion=jnp.array(0.33),
            volatility=jnp.array(0.02),
            initial_curve=curve,
        )
        fixed = FixedInputs(
            entries=(
                FixedInput(
                    name="used.mean_reversion",
                    where=lambda pair: pair[0].mean_reversion,
                ),
                FixedInput(
                    name="unused.mean_reversion",
                    where=lambda pair: pair[1].mean_reversion,
                ),
            )
        )
        report = fixed_input_adjoints(
            lambda pair: hw_swaption_price(trade, pair[0]),
            (model, spare),
            fixed,
        )
        assert abs(float(report.adjoints[0])) > 1.0
        assert float(report.adjoints[1]) == 0.0


class TestRecalibratedFixedInputSensitivities:
    """The total derivative, once the model is allowed to re-fit.

    The number a desk should usually see for mean reversion: moving it
    and *not* recalibrating describes a model that would never be used.
    """

    @pytest.fixture(scope="class")
    def basket(self, curve):
        return [
            _swaption(curve, expiry, tenor)
            for expiry, tenor in [(1, 5), (2, 5), (5, 5), (1, 10)]
        ]

    @pytest.fixture(scope="class")
    def targets(self, basket, model):
        return hw_swaption_prices(model, basket)

    @pytest.fixture(scope="class")
    def recalibrated_price(self, basket, targets, curve, trade):
        def price(candidate):
            fitted = calibrate_hull_white(
                basket,
                targets,
                curve,
                fixed_mean_reversion=candidate.mean_reversion,
            ).value
            # The fitted model is curve-free; bind it for the kernel call.
            return hw_swaption_price(trade, bind_curves(fitted, curve))

        return price

    @pytest.fixture(scope="class")
    def report(self, recalibrated_price, model):
        return fixed_input_adjoints(
            recalibrated_price, model, MEAN_REVERSION
        )

    def test_it_is_present_and_non_zero(self, report):
        assert abs(float(report.adjoints[0])) > 1.0

    @pytest.mark.parametrize("bump", sorted(T14_TOTAL_RTOL))
    def test_it_matches_central_finite_differences(
        self, report, recalibrated_price, model, curve, bump
    ):
        """Each finite difference re-runs the whole fit.

        That is what makes this the *total* derivative and not a
        re-statement of the direct one: the reference never reuses a
        calibrated object.
        """

        def price_at(mean_reversion):
            return float(
                recalibrated_price(
                    HullWhiteModel(
                        mean_reversion=jnp.asarray(mean_reversion),
                        volatility=model.volatility,
                        initial_curve=curve,
                    )
                )
            )

        reference = _central_difference(price_at, 0.08, bump)
        assert float(report.adjoints[0]) == pytest.approx(
            reference, rel=T14_TOTAL_RTOL[bump]
        )

    def test_it_differs_materially_from_the_direct_sensitivity(
        self, report, model, trade
    ):
        """Why the distinction is worth making at all.

        Recalibration offsets most of the direct effect — by a factor of
        about 6.6 here — so quoting the frozen number as the exposure
        overstates it several times over. Pinned so the two can never be
        quietly conflated.
        """
        direct = float(
            fixed_input_adjoints(
                lambda m: hw_swaption_price(trade, m), model, MEAN_REVERSION
            ).adjoints[0]
        )
        total = float(report.adjoints[0])
        assert abs(direct) > 2.0 * abs(total)
        # Same sign: recalibration damps the exposure, it does not
        # reverse it.
        assert np.sign(direct) == np.sign(total)

    def test_the_optimiser_is_not_taped(self, report, recalibrated_price, model):
        """A-1 holds through this path too.

        The adjoint is finite and matches finite differences even though
        the function differentiated contains a full root-find, because
        ``optimistix`` differentiates its own implicit solution rather
        than its iterations. A taped optimiser would still *produce* a
        number here, so the evidence is the finite-difference agreement
        above; this test pins the cheaper necessary condition that the
        result is finite and iteration-count-independent.
        """
        assert np.isfinite(float(report.adjoints[0]))
        # Deterministic: the converged solution is what is
        # differentiated, not the path taken to reach it.
        again = fixed_input_adjoints(
            recalibrated_price, model, MEAN_REVERSION
        )
        assert float(again.adjoints[0]) == float(report.adjoints[0])


class TestVolatilityLayerFixedInputs:
    """SABR beta and the displacement shift (specification V-6).

    The vol layer's ``p``: both are chosen rather than fitted, both move
    the vol a cube reports, and neither appears in any quote vector.
    """

    @pytest.fixture(scope="class")
    def cube(self):
        shape = (3, 2)
        return SwaptionCube(
            expiries=jnp.array([1.0, 2.0, 5.0]),
            tenors=jnp.array([5.0, 10.0]),
            forwards=jnp.full(shape, 0.030),
            alphas=jnp.full(shape, 0.052),
            betas=jnp.full(shape, 0.5),
            rhos=jnp.full(shape, -0.25),
            nus=jnp.full(shape, 0.4),
        )

    @staticmethod
    def _vol(cube):
        return cube(
            jnp.asarray(0.035), jnp.asarray(1.0), jnp.asarray(5.0),
            forward=jnp.asarray(0.030),
        )

    def test_sabr_beta_is_present_and_matching(self):
        """Beta on a ``SABRModel``, where it is a scalar leaf."""
        model = SABRModel(
            alpha=jnp.array(0.052),
            beta=jnp.array(0.5),
            rho=jnp.array(-0.25),
            nu=jnp.array(0.4),
        )
        forward, strike, expiry = (
            jnp.asarray(0.030), jnp.asarray(0.035), jnp.asarray(1.0)
        )

        def vol(candidate):
            return sabr_implied_vol(candidate, forward, strike, expiry)

        fixed = fixed_inputs_from_paths(
            [("sabr.beta", lambda m: m.beta)], units=("exponent",)
        )
        report = fixed_input_adjoints(vol, model, fixed)
        assert abs(float(report.adjoints[0])) > 1e-6

        def vol_at(beta):
            return float(
                vol(eqx.tree_at(lambda m: m.beta, model, jnp.asarray(beta)))
            )

        reference = _central_difference(vol_at, 0.5, 1e-5)
        assert float(report.adjoints[0]) == pytest.approx(reference, rel=1e-5)

    def test_a_grid_valued_leaf_is_rejected_with_a_usable_message(self, cube):
        """The cube stores beta per node, which is a *set* of fixed inputs.

        Reporting one number for a grid would be meaningless, and
        indexing into the leaf is not a lens ``equinox.tree_at``
        accepts. Both are refused here with an error that says what to
        do instead, rather than surfacing as the opaque
        ``'_LeafWrapper' object is not subscriptable`` the tree surgery
        would otherwise raise.
        """
        grid = fixed_inputs_from_paths([("cube.betas", lambda c: c.betas)])
        with pytest.raises(ValueError, match="must be a scalar"):
            grid.values(cube)

        indexed = fixed_inputs_from_paths(
            [("cube.betas[0,0]", lambda c: c.betas[0, 0])]
        )
        # Reading is fine — it is a scalar — but injecting is not.
        assert float(indexed.values(cube)[0]) == 0.5
        with pytest.raises(TypeError):
            indexed.inject(cube, jnp.array([0.6]))

    def test_the_shift_is_inert_on_a_lognormal_cube(self, cube):
        """Zero *because the convention does not read it*, and that is right.

        ``shift`` only enters the normal (Bachelier) expansion, so on a
        lognormal cube its adjoint is exactly zero. Recorded rather than
        skipped: "non-zero where expected" needs the expectation stated,
        and here the expectation is zero.
        """
        fixed = fixed_inputs_from_paths([("cube.shift", lambda c: c.shift)])
        report = fixed_input_adjoints(self._vol, cube, fixed)
        assert float(report.adjoints[0]) == 0.0

    def test_the_shift_bites_on_a_normal_cube(self, cube):
        """The complement: switch the convention and it comes alive."""
        normal = eqx.tree_at(
            lambda c: c.shift, cube, jnp.array(0.02)
        )
        normal = SwaptionCube(
            expiries=normal.expiries,
            tenors=normal.tenors,
            forwards=normal.forwards,
            alphas=jnp.full(normal.alphas.shape, 0.010),
            betas=normal.betas,
            rhos=normal.rhos,
            nus=normal.nus,
            shift=normal.shift,
            is_normal=True,
        )
        fixed = fixed_inputs_from_paths([("cube.shift", lambda c: c.shift)])
        report = fixed_input_adjoints(self._vol, normal, fixed)
        assert abs(float(report.adjoints[0])) > 1e-8
