"""T-4, T-8 and T-9 conformance gates: limits, smoothness and repricing.

Verifies VALAX against the multi-curve calibration / AAD risk audit
specification, requirements:

* **T-4** — single-curve limit: set every projection curve equal to the
  discount curve (beta = 1, s = 0); results match an independent single-curve
  pricer.  Backs C-2 and M-2.
* **T-8** — smoothness scan: price and delta as a function of one quote over a
  fine grid of bumps show no steps, and delta is continuous.  Backs M-6, M-7,
  C-4 and R-6.
* **T-9** — the calibration basket repriced by the production PDE after
  calibration, with residuals reported.  Backs M-8.

Collected with the rest of the T-suite via ``pytest -m audit``.

**How T-8 avoids the tolerance trap.**  "No steps or jumps" is not directly
assertable: any smooth function has non-zero second differences, and any step
can be hidden under a loose enough tolerance.  Asserting the price is close to
a straight line over the scan would be worse still, because it has genuine
curvature there — the assertion would then be measuring the curvature and not
the smoothness.

The discriminator used instead is the **scaling under grid refinement**, which
separates the two cases cleanly rather than trading them off:

===================  ==========================  =====================
Quantity             Smooth                      With a step
===================  ==========================  =====================
``max|d2 price|``    ``O(h^2)`` — falls 4x       ``O(jump)`` — flat
``max|d delta|``     ``O(h)``   — falls 2x       ``O(jump)`` — flat
===================  ==========================  =====================

So halving the grid spacing *must* shrink the discrete second difference of
the price by about four and the first difference of delta by about two.  A
single discontinuity anywhere in the scan pins both ratios near one, whatever
the tolerance.  Observed: 3.99 and 2.00.

**What T-9 measures that a repricing check alone would not.**  The Hull-White
fit drives the *analytic* (Jamshidian) swaption price to its target, so
repricing the basket with the same formula is near-tautological — it returns
2e-15.  Pricing it with the **PDE** instead is a different discretisation of
the same model, so the residual is real, and refining the grid shows *what
kind* of error it is: the residual falls from 1.5e-4 to 3.7e-5 when the grid
doubles, a ratio of 4, so it is second-order discretisation error and not
calibration misfit.  Reporting a single residual would not distinguish those.

Tolerances this implementation commits to:

* **T-4** — the dual-curve code path with ``P_x == P_d`` must reproduce the
  single-curve path to ``rtol = 1e-15`` (one ulp).  Observed: exactly 0.0 for
  the float leg, par rate, swap price, annuity and caplet forward; 1.4e-17 on
  the swaption forward.  This is a telescoping identity, not an
  approximation, so near-exactness is the right bar.
  At the model layer, ``rtol = 1e-8`` on the refitted parameter.
* **T-8** — refinement ratios within 15% of their theoretical 4 and 2.
* **T-9** — PDE basket residual ``< 5e-4`` relative at a 200x200 grid, and
  the refinement ratio from 200x200 to 400x400 at least 2 (second order
  being 4).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration.hull_white import calibrate_hull_white, hw_swaption_prices
from valax.core.binding import bind_curves
from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves import (
    CurveSpec,
    DepositRate,
    IborSwapRate,
    OISSwapRate,
    bootstrap_curve_graph,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks.adjoint import curve_adjoint, curve_quote_deltas
from valax.instruments.rates import Caplet, InterestRateSwap, Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.caplets import caplet_forward
from valax.pricing.analytic.swaptions import (
    annuity,
    swap_price,
    swap_rate,
    swaption_forward_and_annuity,
)
from valax.pricing.pde.config import PDEConfig
from valax.pricing.pde.dispatch import pde_price_dispatch


pytestmark = pytest.mark.audit


T4_RTOL = 1e-15
T4_MODEL_RTOL = 1e-8
T8_RATIO_RTOL = 0.15
T9_RTOL = 5e-4

#: Valuation date as a plain ``int``, for building date arrays.
REF_ORDINAL = int(ymd_to_ordinal(2025, 1, 1))

#: Valuation date as a JAX scalar, which is what the quote pytrees and
#: ``bootstrap_curve_graph`` need: a bare Python ``int`` reaches
#: ``year_fraction`` as an ``int`` and has no ``.astype``.
REF = jnp.int32(REF_ORDINAL)

DC = "act_365"
OIS = "USD.SOFR.OIS"
P3M = "USD.TERM.3M"


def _d(year, month, day):
    return int(ymd_to_ordinal(year, month, day))


def _curve(
    zero_rate=0.040, slope=0.004, reference=REF_ORDINAL
) -> DiscountCurve:
    """A sloped curve, so nothing degenerates for want of curve shape."""
    years = [0.0] + [0.25 * k for k in range(1, 97)]
    pillars = jnp.array(
        [reference + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - reference).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-(zero_rate + slope * times) * times),
        reference_date=jnp.int32(reference),
        day_count=DC,
    )


# ── T-4: the single-curve limit ──────────────────────────────────────


class TestSingleCurveLimitAtThePricers:
    """beta = 1, s = 0 by passing the discount curve as the projection curve.

    The sharp form of T-4, and the one neither existing test covers.
    ``tests/test_curves/test_bootstrap_graph.py::TestSingleCurveDegeneracy``
    compares two *solvers*, and
    ``test_dual_curve_calibration.py::TestSingleCurveNonRegression``
    compares ``forward_curve=None`` against the default — which is the
    same branch, so it cannot detect a broken dual-curve path.

    Here ``forward_curve=curve`` is passed explicitly, so the
    **dual-curve code runs** and must land on the single-curve answer.
    That makes it a real test of
    ``swaptions._dual_curve_float_pv``: with one curve its summands
    telescope, ``sum_i (P(T_i-1)/P(T_i) - 1) P(T_i) = P(T_0) - P(T_n)``,
    and any error in the period alignment survives as a residual.
    """

    @pytest.fixture(scope="class")
    def curve(self):
        return _curve()

    @pytest.fixture(scope="class")
    def fixed_dates(self):
        return jnp.array(
            [REF_ORDINAL + 365 * (k + 1) for k in range(5)], jnp.int32
        )

    @pytest.fixture(scope="class")
    def swap(self, fixed_dates):
        return InterestRateSwap(
            start_date=jnp.int32(REF),
            fixed_dates=fixed_dates,
            fixed_rate=jnp.array(0.04),
            notional=jnp.array(1e6),
            day_count=DC,
        )

    def test_the_floating_leg_telescopes(self, curve, fixed_dates, swap):
        """The dual-curve float leg collapses to two curve reads."""
        dual = float(swap_rate(swap, curve, curve))
        single = float(swap_rate(swap, curve, None))
        assert dual == pytest.approx(single, rel=T4_RTOL)

    def test_the_par_rate_matches_an_independent_closed_form(
        self, curve, fixed_dates, swap
    ):
        """Against ``(P(T_0) - P(T_n)) / A`` built here, not by the pricer.

        The independent single-curve "pricer" T-4 asks for: on one curve
        the par rate has a closed form in two discount factors and the
        annuity, with the floating leg never enumerated at all.
        """
        reference = (
            float(curve(jnp.int32(REF))) - float(curve(fixed_dates[-1]))
        ) / float(annuity(jnp.int32(REF), fixed_dates, curve, DC))
        assert float(swap_rate(swap, curve, curve)) == pytest.approx(
            reference, rel=T4_RTOL
        )

    def test_the_swap_price_degenerates(self, curve, swap):
        assert float(swap_price(swap, curve, curve)) == pytest.approx(
            float(swap_price(swap, curve, None)), rel=T4_RTOL
        )

    def test_the_swaption_forward_and_annuity_degenerate(self, curve):
        swaption = Swaption(
            expiry_date=jnp.int32(REF_ORDINAL + 365),
            fixed_dates=jnp.array(
                [REF_ORDINAL + 365 * (k + 2) for k in range(5)], jnp.int32
            ),
            strike=jnp.array(0.04),
            notional=jnp.array(1e6),
            day_count=DC,
        )
        dual_f, dual_a = swaption_forward_and_annuity(swaption, curve, curve)
        single_f, single_a = swaption_forward_and_annuity(
            swaption, curve, None
        )
        assert float(dual_f) == pytest.approx(float(single_f), rel=T4_RTOL)
        # The annuity never consulted the projection curve, so this one
        # must be bit-identical rather than merely close.
        assert float(dual_a) == float(single_a)

    def test_the_caplet_forward_degenerates(self, curve):
        caplet = Caplet(
            fixing_date=jnp.int32(REF_ORDINAL + 365),
            start_date=jnp.int32(REF_ORDINAL + 365),
            end_date=jnp.int32(REF_ORDINAL + 456),
            strike=jnp.array(0.04),
            notional=jnp.array(1e6),
            day_count="act_360",
        )
        assert float(caplet_forward(caplet, curve, curve)) == float(
            caplet_forward(caplet, curve, None)
        )


class TestSingleCurveLimitAtTheBootstrap:
    """A two-curve graph whose basis is zero must give two equal curves.

    The Layer-1a statement of beta = 1: quote the IBOR swaps at exactly
    the OIS par rates of the same schedules, so the implied basis is
    zero, and the projection block must solve to the discount block.
    Nothing in the solver knows that, so it is a genuine joint-solve
    result rather than a short circuit.
    """

    def test_a_zero_basis_graph_has_two_identical_curves(self):
        pillars = [_d(2026, 1, 1), _d(2028, 1, 1), _d(2030, 1, 1)]
        specs = [
            CurveSpec(
                curve_id=cid,
                currency="USD",
                pillar_dates=jnp.array(pillars, jnp.int32),
                day_count=DC,
            )
            for cid in (OIS, P3M)
        ]
        schedules = [
            jnp.array([_d(y, 1, 1) for y in range(2026, end + 1)], jnp.int32)
            for end in (2026, 2028, 2030)
        ]

        # Discount block: a deposit plus two OIS swaps.
        ois_rates = (0.045, 0.047, 0.050)
        ois_quotes = [
            DepositRate(
                start_date=REF,
                end_date=pillars[0],
                rate=jnp.asarray(ois_rates[0]),
                day_count=DC,
                curves_touched=(OIS,),
            ),
            OISSwapRate(
                start_date=REF,
                fixed_dates=schedules[1],
                rate=jnp.asarray(ois_rates[1]),
                day_count=DC,
                curves_touched=(OIS,),
            ),
            OISSwapRate(
                start_date=REF,
                fixed_dates=schedules[2],
                rate=jnp.asarray(ois_rates[2]),
                day_count=DC,
                curves_touched=(OIS,),
            ),
        ]
        discount_result = bootstrap_curve_graph(REF, [specs[0]], ois_quotes)
        assert bool(discount_result.converged)
        discount_graph = discount_result.value
        discount = discount_graph[OIS]

        # Projection block: IBOR swaps quoted at the OIS par rate of the
        # same schedule, i.e. exactly zero basis.
        def par_rate(schedule):
            return (
                float(discount(jnp.int32(REF)))
                - float(discount(schedule[-1]))
            ) / float(annuity(jnp.int32(REF), schedule, discount, DC))

        ibor_quotes = [
            IborSwapRate(
                start_date=REF,
                fixed_dates=schedule,
                float_dates=schedule,
                fixing_dates=jnp.concatenate(
                    [jnp.array([REF_ORDINAL], jnp.int32), schedule[:-1]]
                ),
                rate=jnp.asarray(par_rate(schedule)),
                fixed_day_count=DC,
                float_day_count=DC,
                curves_touched=(OIS, P3M),
                index_id=P3M,
            )
            for schedule in schedules
        ]

        joint_result = bootstrap_curve_graph(
            REF, specs, ois_quotes + ibor_quotes
        )
        assert bool(joint_result.converged)
        graph = joint_result.value

        projection = graph[P3M]
        for pillar in pillars:
            assert float(projection(jnp.int32(pillar))) == pytest.approx(
                float(graph[OIS](jnp.int32(pillar))), rel=1e-10
            )

    def test_a_non_zero_basis_does_not_degenerate(self):
        """The complement, so the gate cannot pass by the curves being tied.

        Nothing in the solver forces the two blocks together; if the
        quotes imply a basis, the curves separate. Without this, a bug
        that aliased the two curves would satisfy the test above.
        """
        pillars = [_d(2028, 1, 1)]
        specs = [
            CurveSpec(
                curve_id=cid,
                currency="USD",
                pillar_dates=jnp.array(pillars, jnp.int32),
                day_count=DC,
            )
            for cid in (OIS, P3M)
        ]
        schedule = jnp.array([_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1)], jnp.int32)
        quotes = [
            OISSwapRate(
                start_date=REF,
                fixed_dates=schedule,
                rate=jnp.asarray(0.047),
                day_count=DC,
                curves_touched=(OIS,),
            ),
            IborSwapRate(
                start_date=REF,
                fixed_dates=schedule,
                float_dates=schedule,
                fixing_dates=jnp.array(
                    [REF_ORDINAL, _d(2026, 1, 1), _d(2027, 1, 1)], jnp.int32
                ),
                # 25 bp above the OIS quote: a real basis.
                rate=jnp.asarray(0.0495),
                fixed_day_count=DC,
                float_day_count=DC,
                curves_touched=(OIS, P3M),
                index_id=P3M,
            ),
        ]
        result = bootstrap_curve_graph(REF, specs, quotes)
        assert bool(result.converged)
        graph = result.value
        pillar = jnp.int32(pillars[0])
        assert abs(
            float(graph[P3M](pillar)) - float(graph[OIS](pillar))
        ) > 1e-4


class TestSingleCurveLimitAtTheModel:
    """The Layer-2 statement: a zero-basis dual-curve fit is the single one.

    ``calibrate_hull_white(..., forward_curve=curve)`` runs the
    dual-curve path — the frozen tenor-basis overlay of spec 6.2 — with
    a basis of exactly one. It must recover the same parameter the
    single-curve fit does, because every projected fixing's
    ``beta_i`` is then identically 1.
    """

    def test_the_dual_curve_fit_recovers_the_single_curve_parameter(self):
        curve = _curve(0.040, slope=0.0, reference=_d(2026, 1, 1))
        reference = int(curve.reference_date)
        swaptions = []
        for expiry_y, tenor_y in [(1, 5), (2, 5), (5, 5), (1, 10)]:
            expiry_ord = reference + expiry_y * 365
            fixed = jnp.array(
                [
                    expiry_ord + int(round(k * 365))
                    for k in range(1, tenor_y + 1)
                ],
                jnp.int32,
            )
            ann = annuity(jnp.int32(expiry_ord), fixed, curve, DC)
            forward = (curve(jnp.int32(expiry_ord)) - curve(fixed[-1])) / ann
            swaptions.append(
                Swaption(
                    expiry_date=jnp.int32(expiry_ord),
                    fixed_dates=fixed,
                    strike=forward,
                    notional=jnp.array(1e6),
                    day_count=DC,
                )
            )

        truth = HullWhiteModel(
            mean_reversion=jnp.array(0.08),
            volatility=jnp.array(0.012),
            initial_curve=curve,
        )
        targets = hw_swaption_prices(truth, swaptions)

        single = calibrate_hull_white(
            swaptions, targets, curve, fixed_mean_reversion=jnp.array(0.08)
        ).value
        dual = calibrate_hull_white(
            swaptions,
            targets,
            curve,
            fixed_mean_reversion=jnp.array(0.08),
            forward_curve=curve,
        ).value
        assert float(dual.volatility) == pytest.approx(
            float(single.volatility), rel=T4_MODEL_RTOL
        )


# ── T-8: the smoothness scan ─────────────────────────────────────────

_T8_PILLARS = [
    _d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1), _d(2030, 1, 1)
]
_T8_SCHEDULES = [
    [_d(2026, 1, 1), _d(2027, 1, 1)],
    [_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1)],
    [
        _d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1),
        _d(2029, 1, 1), _d(2030, 1, 1),
    ],
]
_T8_BASE = (0.045, 0.047, 0.048, 0.050)

#: Scan half-width: 2 bp, the larger of the two bump sizes T-2 validates at.
_T8_HALF_WIDTH = 2e-4

#: Grid sizes. The *ratio* is what matters — the finer grid must halve the
#: spacing — not the absolute count, which is kept small because every point
#: costs a full curve rebuild plus a Jacobian factorisation (~0.3 s).
_T8_COARSE, _T8_FINE = 17, 33

#: The quote scanned: the 5y OIS rate, which carries the most curvature of
#: the four (its annuity spans every pillar).
_T8_QUOTE_INDEX = 3


@pytest.fixture(scope="module")
def t8_spec():
    return CurveSpec(
        curve_id=OIS,
        currency="USD",
        pillar_dates=jnp.array(_T8_PILLARS, jnp.int32),
        day_count=DC,
    )


@pytest.fixture(scope="module")
def t8_trade():
    return InterestRateSwap(
        start_date=jnp.int32(REF),
        fixed_dates=jnp.array(_T8_SCHEDULES[2], jnp.int32),
        fixed_rate=jnp.array(0.046),
        notional=jnp.array(1e7),
        discount_id=OIS,
    )


def _t8_quotes(rates):
    quotes = [
        DepositRate(
            start_date=REF,
            end_date=_T8_PILLARS[0],
            rate=jnp.asarray(rates[0]),
            day_count=DC,
            curves_touched=(OIS,),
        )
    ]
    quotes += [
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(schedule, jnp.int32),
            rate=jnp.asarray(rate),
            day_count=DC,
            curves_touched=(OIS,),
        )
        for schedule, rate in zip(_T8_SCHEDULES, rates[1:])
    ]
    return quotes


def _t8_scan(spec, trade, n_points):
    """Price and AAD quote delta across ``n_points`` bumps of one quote.

    Everything is rebuilt from the quotes at every point — curve graph,
    adjoint handle, factorisation — which is the condition R-6 and M-6
    care about: nothing is carried over between bumps, so a step would
    not be smoothed away by reuse.
    """
    bumps = np.linspace(-_T8_HALF_WIDTH, _T8_HALF_WIDTH, n_points)
    prices, deltas = [], []
    for bump in bumps:
        rates = [
            base + (bump if i == _T8_QUOTE_INDEX else 0.0)
            for i, base in enumerate(_T8_BASE)
        ]
        quotes = _t8_quotes(rates)
        result = bootstrap_curve_graph(REF, [spec], quotes)
        assert bool(result.converged)
        graph = result.value
        market = MarketState(curves=graph)
        prices.append(float(price(trade, None, market).price))
        adjoint = curve_adjoint(graph, REF, [spec], quotes)
        deltas.append(
            float(
                curve_quote_deltas(trade, None, market, adjoint=adjoint)[
                    _T8_QUOTE_INDEX
                ]
            )
        )
    spacing = float(bumps[1] - bumps[0])
    return spacing, np.array(prices), np.array(deltas)


@pytest.fixture(scope="module")
def t8_scans(t8_spec, t8_trade):
    """Two scans over the same interval, the second at half the spacing."""
    return {
        "coarse": _t8_scan(t8_spec, t8_trade, _T8_COARSE),
        "fine": _t8_scan(t8_spec, t8_trade, _T8_FINE),
    }


class TestSmoothnessUnderGridRefinement:
    """T-8, phrased as a convergence rate rather than a tolerance."""

    def test_the_grid_really_was_refined(self, t8_scans):
        """Guards the premise every ratio below depends on."""
        coarse_h = t8_scans["coarse"][0]
        fine_h = t8_scans["fine"][0]
        assert fine_h == pytest.approx(coarse_h / 2.0, rel=1e-12)

    def test_price_second_differences_fall_as_h_squared(self, t8_scans):
        """Smooth: ``O(h^2)``, so a 4x drop.  A step: no drop at all."""
        coarse = np.max(np.abs(np.diff(t8_scans["coarse"][1], 2)))
        fine = np.max(np.abs(np.diff(t8_scans["fine"][1], 2)))
        assert coarse > 0.0 and fine > 0.0
        assert coarse / fine == pytest.approx(4.0, rel=T8_RATIO_RTOL)

    def test_delta_first_differences_fall_as_h(self, t8_scans):
        """Delta continuous: ``O(h)``, so a 2x drop."""
        coarse = np.max(np.abs(np.diff(t8_scans["coarse"][2])))
        fine = np.max(np.abs(np.diff(t8_scans["fine"][2])))
        assert coarse > 0.0 and fine > 0.0
        assert coarse / fine == pytest.approx(2.0, rel=T8_RATIO_RTOL)

    def test_delta_is_monotone_in_the_quote_with_no_reversals(self, t8_scans):
        """A step in the price shows up as a sign flip in delta's increments.

        Cheap and independent of the ratios above: across a scan this
        narrow the swap's delta moves one way throughout, so every
        consecutive difference must share a sign. A jump would put one
        large increment of the opposite sign among them.
        """
        increments = np.diff(t8_scans["fine"][2])
        assert np.all(increments > 0.0) or np.all(increments < 0.0)

    def test_the_aad_delta_tracks_the_slope_of_the_scanned_price(
        self, t8_scans
    ):
        """Ties the two scanned quantities together.

        The prices and the deltas are produced by different machinery —
        one by the pricer, one by the adjoint chain — so agreeing that
        the latter is the slope of the former is a real consistency
        statement, and it is free from the data already gathered.
        Central differences are ``O(h^2)``, hence the loose tolerance;
        the tight version of this comparison is T-2.
        """
        spacing, prices, deltas = t8_scans["fine"]
        central = (prices[2:] - prices[:-2]) / (2.0 * spacing)
        interior = deltas[1:-1]
        assert np.max(np.abs((central - interior) / interior)) < 1e-5

    def test_the_scan_actually_moves_the_price(self, t8_scans):
        """Anti-vacuity: a flat scan would pass every ratio test by zero.

        Asserted after the fact rather than assumed, because a trade
        insensitive to the scanned quote would make the whole gate
        meaningless while every assertion above held.
        """
        prices = t8_scans["fine"][1]
        assert abs(prices[-1] - prices[0]) > 1.0
        deltas = t8_scans["fine"][2]
        assert abs(float(np.mean(deltas))) > 1.0


# ── T-9: the basket repriced by the production PDE ───────────────────


@pytest.fixture(scope="module")
def t9_basket():
    """A calibrated Hull-White model and the basket it was fitted to."""
    reference = _d(2026, 1, 1)
    curve = _curve(0.040, slope=0.0, reference=reference)

    swaptions = []
    for expiry_y, tenor_y in [(1, 5), (2, 5), (3, 5), (5, 5), (1, 10), (5, 10)]:
        expiry_ord = reference + expiry_y * 365
        fixed = jnp.array(
            [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
            jnp.int32,
        )
        ann = annuity(jnp.int32(expiry_ord), fixed, curve, DC)
        forward = (curve(jnp.int32(expiry_ord)) - curve(fixed[-1])) / ann
        swaptions.append(
            Swaption(
                expiry_date=jnp.int32(expiry_ord),
                fixed_dates=fixed,
                strike=forward,
                notional=jnp.array(1e6),
                day_count=DC,
            )
        )

    truth = HullWhiteModel(
        mean_reversion=jnp.array(0.08),
        volatility=jnp.array(0.012),
        initial_curve=curve,
    )
    targets = hw_swaption_prices(truth, swaptions)
    # The calibrated model is curve-free; bind it for kernel-level repricing.
    fitted = bind_curves(
        calibrate_hull_white(
            swaptions, targets, curve, fixed_mean_reversion=jnp.array(0.08)
        ).value,
        curve,
    )
    return swaptions, targets, fitted


def _pde_basket_residuals(swaptions, targets, model, n_grid):
    """Relative residual of every basket member under the production PDE.

    One ``PDEConfig`` instance serves every member and is reused across
    calls, which is the frozen-grid condition of M-6: the grid is a
    static-field pytree, so it cannot vary with the model parameters or
    the curve.
    """
    config = PDEConfig(n_spot=n_grid, n_time=n_grid, rannacher_steps=2)
    priced = jnp.stack(
        [
            pde_price_dispatch(swaption, model, config).price
            for swaption in swaptions
        ]
    )
    return np.abs(np.asarray((priced - targets) / targets))


class TestBasketRepricedByTheProductionPDE:
    """T-9 / M-8: reprice the basket with the engine that prices the trade."""

    def test_the_analytic_refit_is_near_exact(self, t9_basket):
        """The baseline, and the reason it is not the gate.

        The fit drives the Jamshidian price to the target, so repricing
        with that same formula returns ~1e-15. Recorded explicitly so it
        is clear what the PDE comparison adds: a genuinely different
        discretisation of the same model.
        """
        swaptions, targets, fitted = t9_basket
        residuals = np.abs(
            np.asarray((hw_swaption_prices(fitted, swaptions) - targets) / targets)
        )
        assert residuals.max() < 1e-12

    def test_the_pde_reprices_the_basket_within_tolerance(self, t9_basket):
        swaptions, targets, fitted = t9_basket
        residuals = _pde_basket_residuals(swaptions, targets, fitted, 200)
        assert residuals.max() < T9_RTOL
        # Reported, per M-8: every member, not just the worst.
        assert residuals.shape == (len(swaptions),)
        assert np.all(np.isfinite(residuals))

    def test_the_residual_is_discretisation_error_not_misfit(self, t9_basket):
        """What a single reported residual could never tell you.

        Doubling the grid must shrink the residual. If the basket were
        genuinely mis-calibrated the residual would be insensitive to
        the grid, so this is the assertion that attributes the error to
        the PDE rather than to the fit. Second order predicts a factor
        of 4; the gate requires at least 2, which still excludes a
        grid-independent misfit.
        """
        swaptions, targets, fitted = t9_basket
        coarse = _pde_basket_residuals(swaptions, targets, fitted, 200).max()
        fine = _pde_basket_residuals(swaptions, targets, fitted, 400).max()
        assert fine < coarse
        assert coarse / fine > 2.0

    def test_the_grid_is_frozen_so_repricing_is_bit_identical(
        self, t9_basket
    ):
        """M-6: the same inputs must give the same number, exactly.

        A grid that moved with the parameters or the curve — rebuilt per
        call from the model's own volatility, say — would make prices
        step-shaped under a bump and the calibration Jacobian noisy.
        Bit-identical repricing is the observable consequence of the grid
        being static.
        """
        swaptions, targets, fitted = t9_basket
        first = _pde_basket_residuals(swaptions, targets, fitted, 200)
        second = _pde_basket_residuals(swaptions, targets, fitted, 200)
        assert np.array_equal(first, second)
