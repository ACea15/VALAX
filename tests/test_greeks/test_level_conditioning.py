"""T-13 conformance gate: condition numbers of dR/dtheta at every level.

Verifies :mod:`valax.greeks.adjoint` against the multi-curve calibration /
AAD risk audit specification, requirement **T-13** ("condition numbers of
dR/dtheta at every level: reported; alert above a stated threshold"), which
backs A-6 (adjoint numerical health) and C-5 (curve-build conditioning).
Collected with the rest of the T-suite via ``pytest -m audit``.

Why this matters at all: the implicit function theorem needs ``dR/dtheta``
invertible, and the *degree* of invertibility is what decides how much of the
double-precision mantissa survives the transposed solve.  A calibration can
converge beautifully -- tiny residuals, happy solver -- and still hand back
quote sensitivities whose leading digits are noise.  The residual norm does
not detect this; only the condition number does.

**Stated threshold.** :data:`~valax.greeks.adjoint.ILL_CONDITIONED_THRESHOLD`
= ``1e10``, which is roughly where a ``float64`` solve has shed ten of its
sixteen significant digits.  The constant is the single source of truth: this
file asserts the value once, and asserts that it is what
:func:`~valax.greeks.adjoint.condition_alert` defaults to, rather than
restating the number in every test.

**Stated tolerances.**

* Agreement between the three reports of the *same* Layer-1a Jacobian
  (``CalibrationResult.condition_number``, ``CurveAdjoint`` and the level-generic
  helper) -- ``rtol = 1e-10``.  They differentiate the same residual at the
  same point, so anything looser would let them silently diverge.
* Every level's condition number must be finite and ``>= 1.0 - 1e-10``; a
  condition number below one is not a number, it is a bug.
* The alert boundary itself is exact: ``<= threshold`` is silent,
  ``> threshold`` alerts.

**What is covered, level by level.**  Layer 1a is a real curve build; Layer 1b
a real SABR smile fit; Layer 2 a real Hull-White swaption fit.  The alert's
*ill-conditioned-but-finite* branch is driven by a synthetic Hilbert level
rather than a market one, and that is a deliberate, documented choice --
see :class:`TestAlertFiresAboveTheThreshold` for the measurements that
forced it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.calibration.hull_white import hw_swaption_prices
from valax.curves import (
    CurveSpec,
    DepositRate,
    OISSwapRate,
    bootstrap_curve_graph,
)
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks.adjoint import (
    ILL_CONDITIONED_THRESHOLD,
    CalibrationLevel,
    condition_alert,
    curve_adjoint,
    level_condition_number,
)
from valax.greeks.vol_adjoint import sabr_slice_level
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.pricing.analytic.swaptions import annuity


pytestmark = pytest.mark.audit


#: Agreement tolerance between reports of the same Jacobian.
T13_RTOL = 1e-10

REF = ymd_to_ordinal(2025, 1, 1)
OIS = "USD.SOFR.OIS"
DC = "act_365"


def _d(year, month, day):
    return int(ymd_to_ordinal(year, month, day))


# ── Layer 1a fixture: a realistic, overlapping OIS strip ─────────────

_PILLARS = [_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1), _d(2030, 1, 1)]
_ANNUITIES = [
    [_d(2026, 1, 1), _d(2027, 1, 1)],
    [_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1)],
    [
        _d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1),
        _d(2029, 1, 1), _d(2030, 1, 1),
    ],
]
_RATES = [0.045, 0.047, 0.048, 0.050]


def _ois_swap(fixed_dates, rate):
    return OISSwapRate(
        start_date=REF,
        fixed_dates=jnp.array(fixed_dates, jnp.int32),
        rate=jnp.asarray(rate),
        day_count=DC,
        curves_touched=(OIS,),
    )


def _curve_quotes(annuities=None, rates=None):
    """Deposit plus overlapping OIS swaps.

    The overlap matters: each swap's annuity spans every earlier pillar,
    so ``dR/dtheta`` is genuinely coupled rather than diagonal.  A strip
    of single-payment swaps would make the Jacobian a scaled identity
    and report a condition number of 1.0 no matter what the pillars did,
    which would make this whole gate vacuous.
    """
    annuities = _ANNUITIES if annuities is None else annuities
    rates = _RATES if rates is None else rates
    quotes = [
        DepositRate(
            start_date=REF,
            end_date=_PILLARS[0],
            rate=jnp.asarray(rates[0]),
            day_count=DC,
            curves_touched=(OIS,),
        )
    ]
    quotes += [
        _ois_swap(dates, rate)
        for dates, rate in zip(annuities, rates[1:])
    ]
    return quotes


@pytest.fixture(scope="module")
def curve_spec():
    return CurveSpec(
        curve_id=OIS,
        currency="USD",
        pillar_dates=jnp.array(_PILLARS, jnp.int32),
        day_count=DC,
    )


@pytest.fixture(scope="module")
def curve_level(curve_spec):
    """A converged Layer-1a build plus its adjoint handle."""
    quotes = _curve_quotes()
    result = bootstrap_curve_graph(REF, [curve_spec], quotes)
    assert bool(result.converged)
    graph = result.value
    adjoint = curve_adjoint(graph, REF, [curve_spec], quotes)
    return graph, result, adjoint


# ── Layer 1b fixture: a SABR smile ───────────────────────────────────


@pytest.fixture(scope="module")
def sabr_slice():
    """A fitted smile, with its own vols so the level sits at a solution."""
    model = SABRModel(
        alpha=jnp.array(0.012),
        beta=jnp.array(0.5),
        rho=jnp.array(-0.25),
        nu=jnp.array(0.45),
    )
    strikes = jnp.array([0.02, 0.03, 0.04, 0.05, 0.06])
    forward, expiry = jnp.array(0.04), jnp.array(5.0)
    vols = jax.vmap(
        lambda k: sabr_implied_vol(model, forward, k, expiry)
    )(strikes)
    return model, strikes, vols, forward, expiry


# ── Layer 2 fixture: a Hull-White swaption fit ───────────────────────


@pytest.fixture(scope="module")
def hull_white_level():
    """The (a, sigma) swaption fit, in both of its level forms.

    Returns builders for the two shapes a model fit legitimately takes:

    * ``stationarity(...)`` -- the square ``grad(SSE) = 0`` form, with a
      flat ``theta`` vector.  This is the form the implicit function
      theorem applies to directly, and the one
      :func:`~valax.greeks.vol_adjoint.sabr_slice_level` uses.
    * ``least_squares(...)`` -- the rectangular ``weighted price
      residuals`` form with a ``dict``-valued ``theta``, which is exactly
      the shape ``valax.calibration.hull_white._hw_price_residuals``
      hands to ``optimistix``.  Its Jacobian is the least-squares design
      matrix, non-square, and ``theta`` is a pytree.
    """
    ref = int(ymd_to_ordinal(2026, 1, 1))
    years = [0.0] + [float(k) for k in range(1, 23)]
    pillars = jnp.array(
        [ref + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - ref).astype(jnp.float64) / 365.0
    curve = DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-0.04 * times),
        reference_date=jnp.int32(ref),
        day_count=DC,
    )

    swaptions = []
    for expiry_y, tenor_y in [(1, 5), (2, 5), (3, 5), (5, 5), (1, 10), (5, 10)]:
        expiry_ord = ref + expiry_y * 365
        fixed = jnp.array(
            [expiry_ord + int(round(k * 365)) for k in range(1, tenor_y + 1)],
            dtype=jnp.int32,
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
    market = hw_swaption_prices(truth, swaptions)

    def price_residuals(mean_reversion, volatility, quotes):
        model = HullWhiteModel(
            mean_reversion=mean_reversion,
            volatility=volatility,
            initial_curve=curve,
        )
        return (hw_swaption_prices(model, swaptions) - quotes) / quotes

    def stationarity(free_mean_reversion: bool) -> CalibrationLevel:
        """Square ``grad(SSE) = 0``, flat theta -- the IFT-valid form."""

        def sse(theta, quotes):
            resid = price_residuals(
                theta[0] if free_mean_reversion else truth.mean_reversion,
                theta[-1],
                quotes,
            )
            return 0.5 * jnp.sum(resid * resid)

        theta = (
            jnp.array([truth.mean_reversion, truth.volatility])
            if free_mean_reversion
            else jnp.array([truth.volatility])
        )
        return CalibrationLevel(
            residual_fn=lambda th, quotes: jax.grad(sse)(th, quotes),
            theta=theta,
            inputs=market,
        )

    def least_squares() -> CalibrationLevel:
        """Rectangular price residuals, ``dict`` theta -- the solver's form."""
        return CalibrationLevel(
            residual_fn=lambda th, quotes: price_residuals(
                th["mean_reversion"], th["volatility"], quotes
            ),
            theta={
                "mean_reversion": truth.mean_reversion,
                "volatility": truth.volatility,
            },
            inputs=market,
        )

    return stationarity, least_squares, len(swaptions)


# ── Layer 1a ─────────────────────────────────────────────────────────


class TestLayer1aReportsItsConditionNumber:
    """The curve build: reported in two places, and they must agree."""

    def test_the_build_reports_a_finite_condition_number(self, curve_level):
        _, result, _ = curve_level
        kappa = float(result.condition_number())
        assert np.isfinite(kappa)
        assert kappa >= 1.0 - T13_RTOL

    def test_the_adjoint_handle_reports_the_same_number(self, curve_level):
        """Two code paths, one Jacobian.

        ``bootstrap_curve_graph`` reports the condition number it saw
        while solving; ``curve_adjoint`` re-evaluates the Jacobian at the
        recovered solution.  If these drift apart, the adjoint is not
        differentiating the system that was actually solved -- which is
        the single most consequential way this chain can be wrong, and
        one that no residual norm would reveal.
        """
        _, result, adjoint = curve_level
        assert float(adjoint.jacobian_condition_number) == pytest.approx(
            float(result.condition_number()), rel=T13_RTOL
        )

    def test_the_generic_helper_agrees_with_the_specialised_report(
        self, curve_level
    ):
        """Ties ``level_condition_number`` to the hand-rolled Layer-1a one.

        This is what licenses using the generic helper as *the* report at
        the levels that have no specialised one (1b and 2): it is
        calibrated here, against a level that does.
        """
        _, _, adjoint = curve_level
        assert float(
            level_condition_number(adjoint.level())
        ) == pytest.approx(
            float(adjoint.jacobian_condition_number), rel=T13_RTOL
        )

    def test_a_healthy_build_raises_no_alert(self, curve_level):
        _, result, adjoint = curve_level
        assert condition_alert(result.condition_number()) is None
        assert condition_alert(adjoint.jacobian_condition_number) is None


class TestPillarSpacingDoesNotDegradeConditioning:
    """A measured negative result, pinned so it is not re-derived.

    The intuitive way to build an ill-conditioned curve is to crowd two
    pillars together, and it **does not work here**: moving two pillars
    to one day apart leaves the condition number near 1.

    The reason is structural.  An OIS floating leg telescopes, so each
    swap residual is ``rate * annuity - (DF(start) - DF(maturity))``, and
    in log-DF space the ``DF(maturity)`` term (order 0.9) dominates the
    annuity term (order ``rate * tau * DF``, so order 0.05).  Each quote
    therefore very nearly pins its own maturity pillar and ``dR/dtheta``
    is near-diagonal, whatever the spacing.  That is a genuinely
    reassuring property of the parameterisation, not a gap in the test --
    and it is why the ill-conditioned *alert* has to be driven by a
    synthetic level below.
    """

    @pytest.mark.parametrize("gap_days", [1, 2])
    def test_near_coincident_pillars_stay_well_conditioned(self, gap_days):
        near = _d(2027, 1, 1) + gap_days
        pillars = [_d(2026, 1, 1), _d(2027, 1, 1), near, _d(2030, 1, 1)]
        spec = CurveSpec(
            curve_id=OIS,
            currency="USD",
            pillar_dates=jnp.array(pillars, jnp.int32),
            day_count=DC,
        )
        annuities = [
            [_d(2026, 1, 1), _d(2027, 1, 1)],
            [_d(2026, 1, 1), _d(2027, 1, 1), near],
            [_d(2026, 1, 1), _d(2027, 1, 1), near, _d(2030, 1, 1)],
        ]
        quotes = [
            DepositRate(
                start_date=REF,
                end_date=pillars[0],
                rate=jnp.asarray(0.045),
                day_count=DC,
                curves_touched=(OIS,),
            )
        ] + [
            _ois_swap(dates, rate)
            for dates, rate in zip(annuities, [0.047, 0.047, 0.050])
        ]

        result = bootstrap_curve_graph(REF, [spec], quotes)
        assert bool(result.converged)
        kappa = float(result.condition_number())
        assert np.isfinite(kappa)
        assert kappa < 10.0
        assert condition_alert(kappa) is None


class TestADuplicatedQuoteIsCaught:
    """The real operator error the alert exists to catch.

    Quoting the same instrument twice keeps the system square, so no
    shape check rejects it, but it makes two residual rows identical in
    ``theta`` and leaves one pillar unconstrained.  That is a singular
    ``dR/dtheta``: the implicit function theorem does not apply, and
    every sensitivity downstream is meaningless rather than imprecise.

    Note *which* signals fire.  The build itself does not converge and
    the condition number comes back non-finite, so the alert's singular
    branch is what catches it.  Both are asserted, because a future
    change that made this build converge to something would otherwise
    silently drop the gate.
    """

    def test_it_fails_to_converge_and_alerts(self, curve_spec):
        three_year = [_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1)]
        quotes = [
            DepositRate(
                start_date=REF,
                end_date=_PILLARS[0],
                rate=jnp.asarray(0.045),
                day_count=DC,
                curves_touched=(OIS,),
            ),
            _ois_swap([_d(2026, 1, 1), _d(2027, 1, 1)], 0.047),
            _ois_swap(three_year, 0.048),
            # The same 3y instrument again, at a marginally different
            # rate.  The 2030 pillar is now unconstrained.
            _ois_swap(three_year, 0.0481),
        ]

        result = bootstrap_curve_graph(REF, [curve_spec], quotes)
        assert not bool(result.converged)

        kappa = result.condition_number()
        assert not np.isfinite(float(kappa))

        alert = condition_alert(kappa, level_name="duplicated OIS strip")
        assert alert is not None
        assert "duplicated OIS strip" in alert
        assert "singular" in alert


# ── Layer 1b ─────────────────────────────────────────────────────────


class TestLayer1bReportsItsConditionNumber:
    """The volatility level, which has no specialised report of its own."""

    def test_a_fitted_smile_reports_a_finite_condition_number(
        self, sabr_slice
    ):
        model, strikes, vols, forward, expiry = sabr_slice
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        kappa = float(level_condition_number(level))
        assert np.isfinite(kappa)
        assert kappa >= 1.0 - T13_RTOL
        assert condition_alert(kappa, level_name="SABR slice") is None

    def test_freeing_beta_measurably_degrades_the_conditioning(
        self, sabr_slice
    ):
        """Puts a number on why beta is normally fixed.

        A SABR smile has three shape degrees of freedom and four
        parameters, so ``alpha`` and ``beta`` trade off against each
        other almost perfectly.  Freeing ``beta`` costs more than two
        orders of magnitude of conditioning here -- still inside the
        alert threshold, so this is a *reported* cost rather than a
        failure, which is exactly the distinction T-13 asks the
        diagnostic to support.
        """
        model, strikes, vols, forward, expiry = sabr_slice
        fixed = float(
            level_condition_number(
                sabr_slice_level(model, strikes, vols, forward, expiry)
            )
        )
        free = float(
            level_condition_number(
                sabr_slice_level(
                    model, strikes, vols, forward, expiry, free_beta=True
                )
            )
        )
        assert free > 100.0 * fixed
        # Both are still reportable rather than broken.
        assert np.isfinite(free)
        assert condition_alert(free) is None


# ── Layer 2 ──────────────────────────────────────────────────────────


class TestLayer2ReportsItsConditionNumber:
    """The model level, through the same level-generic helper."""

    def test_a_two_parameter_fit_reports_a_finite_condition_number(
        self, hull_white_level
    ):
        stationarity, _, _ = hull_white_level
        kappa = float(
            level_condition_number(stationarity(free_mean_reversion=True))
        )
        assert np.isfinite(kappa)
        assert kappa >= 1.0 - T13_RTOL
        assert condition_alert(kappa, level_name="Hull-White") is None

    def test_fixing_mean_reversion_improves_the_conditioning(
        self, hull_white_level
    ):
        """Quantifies the advice in ``valax.calibration.hull_white``.

        That module recommends holding ``a`` fixed because ``a`` and
        ``sigma`` are strongly correlated across a swaption surface.  The
        condition number is where that correlation becomes a number: the
        joint fit is orders of magnitude worse conditioned than the
        one-parameter fit.  Without a diagnostic at this level the advice
        stays folklore.
        """
        stationarity, _, _ = hull_white_level
        joint = float(
            level_condition_number(stationarity(free_mean_reversion=True))
        )
        sigma_only = float(
            level_condition_number(stationarity(free_mean_reversion=False))
        )
        assert joint > 100.0 * sigma_only

    def test_a_pytree_theta_on_a_rectangular_level_is_handled(
        self, hull_white_level
    ):
        """The shape the solver actually uses.

        ``optimistix`` drives Hull-White as a least-squares problem whose
        unknowns are a ``dict`` of raw parameters and whose residual is a
        flat vector of weighted price errors -- so ``dR/dtheta`` is
        ``n_quotes x n_params``, not square.  A condition number is still
        exactly the right diagnostic there (it is the conditioning of the
        design matrix, hence of the normal equations the fit solves), and
        it must not require the caller to pre-flatten the parameters.

        Non-square is also why this is reported separately from the
        square stationarity form above: the implicit function theorem
        needs the square one, so the two numbers answer different
        questions and neither substitutes for the other.
        """
        _, least_squares, n_quotes = hull_white_level
        level = least_squares()
        assert isinstance(level.theta, dict)

        kappa = level_condition_number(level)
        assert jnp.shape(kappa) == ()
        assert np.isfinite(float(kappa))
        assert float(kappa) >= 1.0 - T13_RTOL
        assert n_quotes > 2  # genuinely over-determined, not square by luck


# ── The alert itself ─────────────────────────────────────────────────


def _hilbert_level(n: int) -> CalibrationLevel:
    """``R(theta, q) = H theta - q`` for the n x n Hilbert matrix.

    The textbook ill-conditioned-but-nonsingular system, and a reference
    whose condition number is known independently of JAX (checked against
    NumPy below).  Linear in ``theta``, so ``dR/dtheta`` is exactly ``H``
    and the level sits at a solution for any ``theta`` with the matching
    ``q``.
    """
    matrix = jnp.asarray(
        np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    )
    return CalibrationLevel(
        residual_fn=lambda theta, q: matrix @ theta - q,
        theta=jnp.zeros(n),
        inputs=jnp.zeros(n),
    )


class TestAlertFiresAboveTheThreshold:
    """The reporting half of T-13.

    The ill-conditioned-but-finite case is driven by a synthetic Hilbert
    level, not a market one, and that is a deliberate choice rather than
    laziness: as :class:`TestPillarSpacingDoesNotDegradeConditioning`
    measures, a VALAX curve build resists crowding its pillars, and the
    worst *real* level reached anywhere in this file is the free-beta
    SABR fit at order 1e6 -- four orders inside the threshold.  Claiming
    a market fixture that straddled the threshold would mean inventing
    one, so the threshold logic is gated where it can be driven exactly,
    and the realistic failure mode (a duplicated quote) is gated
    separately in :class:`TestADuplicatedQuoteIsCaught`.
    """

    def test_the_hilbert_level_is_recognised_as_ill_conditioned(self):
        level = _hilbert_level(12)
        kappa = level_condition_number(level)

        # Independent confirmation that the fixture is what it claims:
        # NumPy's condition number for the same matrix.
        reference = np.linalg.cond(
            np.array([[1.0 / (i + j + 1) for j in range(12)] for i in range(12)])
        )
        assert float(kappa) == pytest.approx(reference, rel=1e-8)

        assert float(kappa) > ILL_CONDITIONED_THRESHOLD
        alert = condition_alert(kappa, level_name="Hilbert level")
        assert alert is not None
        assert "Hilbert level" in alert
        assert "ill-conditioned" in alert

    def test_a_singular_level_alerts_differently(self):
        """Singular is a different failure from merely ill-conditioned."""
        level = CalibrationLevel(
            residual_fn=lambda theta, q: jnp.zeros((3,)) @ theta - q,
            theta=jnp.zeros(3),
            inputs=jnp.zeros(3),
        )
        alert = condition_alert(
            level_condition_number(level), level_name="degenerate"
        )
        assert alert is not None
        assert "singular" in alert

    def test_the_boundary_is_exact(self):
        """At the threshold: silent.  One ulp above: alert."""
        assert condition_alert(ILL_CONDITIONED_THRESHOLD) is None
        assert (
            condition_alert(np.nextafter(ILL_CONDITIONED_THRESHOLD, np.inf))
            is not None
        )

    def test_the_threshold_is_the_documented_one(self):
        """Pins the stated threshold, in one place only."""
        assert ILL_CONDITIONED_THRESHOLD == 1e10

    def test_the_threshold_is_overridable_without_touching_the_default(self):
        """A desk with a tighter policy should not have to fork the module."""
        assert condition_alert(1e6) is None
        assert condition_alert(1e6, threshold=1e4) is not None
        assert condition_alert(1e6) is None

    def test_the_alert_names_the_level(self):
        """A multi-level report is useless if it cannot say which level."""
        alert = condition_alert(1e12, level_name="3M projection curve")
        assert alert is not None
        assert "3M projection curve" in alert
