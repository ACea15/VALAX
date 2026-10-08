"""Cross-validation: VALAX leg engine vs QuantLib's cashflow legs.

Stage 1 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

``tests/test_pricing/test_legs.py`` pins the leg engine against VALAX's
*own* earlier pricers, bit for bit. That guards drift but cannot catch a
convention both implementations share wrongly, because both sides of
that comparison are VALAX. This module supplies the independent
authority, and targets the two features the leg abstraction adds which
had no prior VALAX implementation at all:

* **amortisation** — against ``ql.FixedRateLeg`` built with a *vector*
  of notionals, which is QuantLib's native way to express a step-down.
* **overnight compounding** — against ``ql.OvernightLeg``, which
  compounds the daily index explicitly, day by day, rather than using
  the telescoping discount-factor ratio VALAX evaluates.

That second comparison is the more interesting one. VALAX prices a
compounded coupon as :math:`P_f(0,T^s)/P_f(0,T^e) - 1`, a single ratio
of two discount factors. QuantLib forms the daily product
:math:`\\prod_j (1 + f_j \\delta_j)`. These agree *analytically* —
each daily factor is itself a ratio of adjacent discount factors, so the
product telescopes — but they are completely different computations, and
agreement is therefore real evidence rather than a tautology.

Curve alignment
---------------
Both engines are handed the **same discount-factor vector at the same
dates** via ``ql.DiscountCurve`` and
:class:`~valax.curves.discount.DiscountCurve`, and every leg date is
placed *on* a curve pillar. Interpolation is then never consulted for
the fixed-leg comparisons, so an interpolation difference cannot
masquerade as a leg defect. ``ql.NullCalendar`` and ``ql.Unadjusted``
are used throughout for the same reason — business-day conformance is
gated separately in ``test_dates_ql.py``.

Measured agreement
------------------
Relative differences at the time of writing, on a 10mm five-year annual
leg over a sloped 3% + 40bp·t curve:

| Leg | VALAX | QuantLib | rel. diff |
|---|---|---|---|
| fixed, flat notional | 1,400,695.439396 | 1,400,695.439396 | 1.3e-15 |
| fixed, amortising | 899,343.842425 | 899,343.842425 | 1.6e-15 |
| overnight compounded | 2,211,992.169286 | 2,211,992.169286 | **0.0** |

The overnight row is exact: the daily product telescopes in floating
point as well as in algebra. The gates below are set an order or two
looser than these measurements so that ordinary last-bit movement does
not fail the suite, while still being far tighter than any real
convention error.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.legs import FixedLeg, FloatingLeg
from valax.pricing.analytic.legs import fixed_leg_pv, floating_leg_pv

# ── Paired market ────────────────────────────────────────────────────

_Y, _M, _D = 2026, 1, 1
REF = ymd_to_ordinal(_Y, _M, _D)
QL_REF = ql.Date(_D, _M, _Y)
NOTIONAL = 1.0e7
FIXED_RATE = 0.0315

#: Eleven annual pillars. Leg dates are drawn from this grid so both
#: curves are only ever queried where they carry an exact pillar.
_N_PILLARS = 11
_PILLAR_ORDINALS = [int(REF) + 365 * k for k in range(_N_PILLARS)]
_PILLAR_TIMES = [365.0 * k / 365.0 for k in range(_N_PILLARS)]


def _zero(t: float) -> float:
    """Sloped continuously-compounded zero curve, ``z(t) = 3% + 40bp t``."""
    return 0.030 + 0.0040 * t


_DFS = [
    float(jnp.exp(-_zero(t) * t).item()) if t > 0 else 1.0
    for t in _PILLAR_TIMES
]


@pytest.fixture(scope="module")
def paired_curves():
    """A VALAX curve and a QL curve carrying identical discount factors.

    Returns the VALAX :class:`CurveGraph`, the QL yield-term-structure
    handle, and the shared pillar ordinals.
    """
    ql.Settings.instance().evaluationDate = QL_REF

    valax_curve = DiscountCurve(
        pillar_dates=jnp.asarray(_PILLAR_ORDINALS, dtype=jnp.int32),
        discount_factors=jnp.asarray(_DFS),
        reference_date=jnp.asarray(int(REF), dtype=jnp.int32),
        day_count="act_365",
    )
    ql_curve = ql.DiscountCurve(
        [QL_REF + 365 * k for k in range(_N_PILLARS)],
        _DFS,
        ql.Actual365Fixed(),
        ql.NullCalendar(),
    )
    ql_curve.enableExtrapolation()
    return (
        CurveGraph(curves={"_default_": valax_curve}),
        ql.YieldTermStructureHandle(ql_curve),
        _PILLAR_ORDINALS,
    )


def test_the_two_curves_agree_at_every_pillar(paired_curves):
    """Self-consistency gate. Every leg comparison below rests on this,
    so a wire-up error must surface here rather than as a leg defect."""
    graph, handle, pillars = paired_curves
    curve = graph["_default_"]
    for k, ordinal in enumerate(pillars):
        valax_df = float(curve(jnp.asarray(ordinal, dtype=jnp.int32)))
        ql_df = handle.discount(QL_REF + 365 * k)
        assert valax_df == pytest.approx(ql_df, rel=1e-14)


# ── Fixed leg, flat and amortising ───────────────────────────────────

#: Five annual periods starting at the reference date.
_N_PERIODS = 5
_ACCRUAL_ENDS = [int(REF) + 365 * k for k in range(1, _N_PERIODS + 1)]
_ACCRUAL_STARTS = [int(REF)] + _ACCRUAL_ENDS[:-1]

#: A genuine step-down. Deliberately not monotone-to-zero and not all
#: equal, so neither a dropped factor nor a reversed array would pass.
_FACTORS = [1.0, 0.85, 0.65, 0.40, 0.20]


def _ql_schedule(n_periods: int = _N_PERIODS) -> ql.Schedule:
    """QL schedule built from the explicit date grid, unadjusted.

    Constructed from a date *vector* rather than from a rule, so the
    grid is the same integer dates VALAX is given and QuantLib's own
    roll/stub logic cannot introduce a difference.
    """
    dates = [QL_REF] + [QL_REF + 365 * k for k in range(1, n_periods + 1)]
    return ql.Schedule(
        ql.DateVector(dates), ql.NullCalendar(), ql.Unadjusted
    )


def _valax_fixed_leg(factors=None) -> FixedLeg:
    return FixedLeg(
        accrual_start_dates=jnp.asarray(_ACCRUAL_STARTS, dtype=jnp.int32),
        accrual_end_dates=jnp.asarray(_ACCRUAL_ENDS, dtype=jnp.int32),
        payment_dates=jnp.asarray(_ACCRUAL_ENDS, dtype=jnp.int32),
        fixed_rate=jnp.asarray(FIXED_RATE),
        notional=jnp.asarray(NOTIONAL),
        amortisation=None if factors is None else jnp.asarray(factors),
        day_count="act_360",
    )


def _ql_fixed_leg_npv(handle, notionals) -> float:
    leg = ql.FixedRateLeg(
        _ql_schedule(),
        ql.Actual360(),
        notionals,
        [FIXED_RATE],
    )
    return ql.CashFlows.npv(leg, handle, False, QL_REF)


class TestFixedLegMatchesQuantLib:
    def test_flat_notional(self, paired_curves):
        graph, handle, _ = paired_curves
        expected = _ql_fixed_leg_npv(handle, [NOTIONAL] * _N_PERIODS)
        got = float(fixed_leg_pv(_valax_fixed_leg(), graph))
        assert got == pytest.approx(expected, rel=1e-13)

    def test_amortising_notional(self, paired_curves):
        """The feature under test: VALAX's dimensionless factors against
        QuantLib's absolute per-period notionals."""
        graph, handle, _ = paired_curves
        notionals = [NOTIONAL * f for f in _FACTORS]
        expected = _ql_fixed_leg_npv(handle, notionals)
        got = float(fixed_leg_pv(_valax_fixed_leg(_FACTORS), graph))
        assert got == pytest.approx(expected, rel=1e-13)

    def test_the_two_notional_profiles_actually_differ(self, paired_curves):
        """Anti-vacuity, asserted on the **QuantLib** side so it cannot
        be satisfied by a VALAX no-op: the amortising and flat legs must
        be materially different numbers."""
        _, handle, _ = paired_curves
        flat = _ql_fixed_leg_npv(handle, [NOTIONAL] * _N_PERIODS)
        amortising = _ql_fixed_leg_npv(
            handle, [NOTIONAL * f for f in _FACTORS]
        )
        assert amortising < 0.75 * flat

    def test_amortisation_order_is_not_reversible(self, paired_curves):
        """Reversing the schedule must disagree with QuantLib, which
        pins the orientation of the factor array against an outside
        authority rather than against VALAX's own convention."""
        graph, handle, _ = paired_curves
        expected = _ql_fixed_leg_npv(
            handle, [NOTIONAL * f for f in _FACTORS]
        )
        reversed_pv = float(
            fixed_leg_pv(_valax_fixed_leg(list(reversed(_FACTORS))), graph)
        )
        assert reversed_pv != pytest.approx(expected, rel=1e-6)


# ── Compounded overnight leg ─────────────────────────────────────────


class TestCompoundedLegMatchesQuantLib:
    """VALAX's discount-factor ratio against QL's explicit daily product."""

    def _ql_overnight_leg_npv(self, handle, n_periods: int) -> float:
        index = ql.OvernightIndex(
            "TestON",
            0,
            ql.USDCurrency(),
            ql.NullCalendar(),
            ql.Actual360(),
            handle,
        )
        leg = ql.OvernightLeg([NOTIONAL], _ql_schedule(n_periods), index)
        return ql.CashFlows.npv(leg, handle, False, QL_REF)

    def test_compounded_leg_matches_the_daily_product(self, paired_curves):
        """The headline cross-check: a single ratio of discount factors
        against a product over every day of the period."""
        graph, handle, _ = paired_curves
        leg = FloatingLeg(
            accrual_start_dates=jnp.asarray(
                _ACCRUAL_STARTS, dtype=jnp.int32
            ),
            accrual_end_dates=jnp.asarray(_ACCRUAL_ENDS, dtype=jnp.int32),
            payment_dates=jnp.asarray(_ACCRUAL_ENDS, dtype=jnp.int32),
            spread=jnp.asarray(0.0),
            notional=jnp.asarray(NOTIONAL),
            compounding="compounded",
            day_count="act_360",
        )
        expected = self._ql_overnight_leg_npv(handle, _N_PERIODS)
        got = float(floating_leg_pv(leg, graph))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_the_overnight_leg_is_not_trivially_zero(self, paired_curves):
        """Anti-vacuity: a positive-rate curve must give a positive
        floating leg, so the comparison above is not 0 == 0."""
        _, handle, _ = paired_curves
        assert self._ql_overnight_leg_npv(handle, _N_PERIODS) > 0.1 * NOTIONAL
