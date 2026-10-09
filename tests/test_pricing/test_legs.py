"""Leg pytrees and the leg PV engine.

The leg engine does not introduce new arithmetic so much as generalise
three expressions that already shipped in narrower form. The risk is
therefore *drift*: a generalisation that quietly returns a slightly
different number than the curve bootstrap and the swap pricers already
produce would move prices across the library with nothing obviously
broken.

Every parity gate here is consequently a **bare ``==``** against the
pre-existing function, not an ``approx``. Summation order is pinned, not
merely the value. The classes follow the four-part shape of
``tests/test_curves/test_instrument_residuals.py::TestPaymentLagTelescoping``:

1. an ``@pytest.mark.audit`` test against a reference computed a
   *different way* (hand-rolled, or per-period instead of vectorised),
2. a bare ``==`` test on the preserved path,
3. a cross-branch test showing the new general path agrees with the
   preserved one where they must,
4. an **anti-vacuity** test showing the feature actually changes the
   answer — without which (1)-(3) could all pass on a no-op.

Fixtures are deliberately non-degenerate, per
``docs/architecture/numerical-pitfalls.md`` entry 3: the curve is sloped
so payment lags bite, amortisation factors genuinely vary, and the
seasoned leg's first period genuinely straddles the valuation date.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from equinox import EquinoxRuntimeError

from valax.curves.discount import DiscountCurve
from valax.curves.fixings import (
    FixingHistory,
    FixingSeries,
    empty_fixing_history,
)
from valax.curves.graph import CurveGraph
from valax.curves.instruments import (
    _arithmetic_average_forward,
    _compounded_float_leg_pv,
    _floating_leg_pv,
)
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.legs import (
    FixedLeg,
    FloatingLeg,
    MultiLegSwap,
    max_period_days,
)
from valax.pricing.analytic.legs import (
    fixed_leg_pv,
    floating_leg_pv,
    leg_pv,
    swap_fair_rate,
    swap_pv,
)
from valax.pricing.analytic.swaptions import annuity

REF = ymd_to_ordinal(2025, 1, 1)
NOTIONAL = jnp.asarray(1.0e7)
NO_FIXINGS = empty_fixing_history()

#: Pillars start a year before ``REF`` so a seasoned leg's accrual start
#: is still inside the curve's range.
_PILLAR_START = int(REF) - 400


def _sloped_curve(short: float, slope: float) -> DiscountCurve:
    """Curve with zero rate ``z(t) = short + slope * t`` (t in years).

    Sloped on purpose: on a flat curve the per-period lag discount factor
    is the same constant everywhere, which would let a leg that mishandled
    payment dates still look right.
    """
    pillars = jnp.asarray(
        [_PILLAR_START + 365 * k for k in range(14)], dtype=jnp.int32
    )
    times = (pillars - _PILLAR_START).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-(short + slope * times) * times),
        reference_date=jnp.asarray(_PILLAR_START, dtype=jnp.int32),
        day_count="act_365",
    )


def _annual_grid(start: int, n: int = 5):
    """``(accrual_start, accrual_end)`` for ``n`` contiguous annual periods."""
    ends = jnp.asarray(
        [start + 365 * k for k in range(1, n + 1)], dtype=jnp.int32
    )
    starts = jnp.concatenate(
        [jnp.asarray([start], dtype=jnp.int32), ends[:-1]]
    )
    return starts, ends


DISC = _sloped_curve(0.030, 0.0040)
FWD = _sloped_curve(0.035, 0.0050)
FWD_6M = _sloped_curve(0.037, 0.0055)
GRAPH = CurveGraph(
    curves={"_default_": DISC, "USD.SOFR": FWD, "USD.LIBOR.6M": FWD_6M}
)
STARTS, ENDS = _annual_grid(int(REF))


def _fixed(**kwargs) -> FixedLeg:
    base = dict(
        accrual_start_dates=STARTS,
        accrual_end_dates=ENDS,
        payment_dates=ENDS,
        fixed_rate=jnp.asarray(0.031),
        notional=NOTIONAL,
    )
    base.update(kwargs)
    return FixedLeg(**base)


def _floating(**kwargs) -> FloatingLeg:
    base = dict(
        accrual_start_dates=STARTS,
        accrual_end_dates=ENDS,
        payment_dates=ENDS,
        spread=jnp.asarray(0.0),
        notional=NOTIONAL,
    )
    base.update(kwargs)
    return FloatingLeg(**base)


# ── Fixed leg ────────────────────────────────────────────────────────


class TestFixedLegMatchesTheAnnuity:
    """A flat fixed leg is ``notional * rate * annuity``, exactly."""

    @pytest.mark.audit
    def test_matches_a_hand_rolled_per_period_sum(self):
        """Independent reference: a Python loop over periods, summing
        ``tau_i * DF_i`` one coupon at a time rather than vectorised."""
        leg = _fixed()
        taus = year_fraction(STARTS, ENDS, "act_360")
        expected = 0.0
        for i in range(leg.n_periods):
            expected += float(taus[i]) * float(DISC(ENDS[i]))
        expected *= float(NOTIONAL) * 0.031
        assert float(fixed_leg_pv(leg, GRAPH)) == pytest.approx(
            expected, rel=1e-13
        )

    def test_is_the_shipped_annuity_bit_for_bit(self):
        """Bare ``==`` against ``swaptions.annuity``: the fixed leg must
        not drift from the annuity every swap pricer already uses."""
        expected = (
            NOTIONAL * jnp.asarray(0.031) * annuity(REF, ENDS, DISC, "act_360")
        )
        assert fixed_leg_pv(_fixed(), GRAPH) == expected

    def test_pay_flag_flips_the_sign_exactly(self):
        """Negation is exact in IEEE-754, so this is an ``==`` too."""
        assert fixed_leg_pv(_fixed(pay=True), GRAPH) == -fixed_leg_pv(
            _fixed(), GRAPH
        )

    def test_a_payment_lag_moves_the_answer(self):
        """Anti-vacuity: discounting at a later date is worth less on a
        positive-rate curve, so ``payment_dates`` is genuinely read."""
        lagged = fixed_leg_pv(_fixed(payment_dates=ENDS + 5), GRAPH)
        assert float(lagged) < float(fixed_leg_pv(_fixed(), GRAPH))


# ── Amortisation ─────────────────────────────────────────────────────


class TestAmortisation:
    """Amortisation is a factor overlay that is inert at 1.0.

    The point of expressing amortisation as dimensionless factors rather
    than absolute per-period notionals is that the notional stays
    *outside* the summation. ``sum_i (N x_i)`` and ``N sum_i x_i`` differ
    in the last bit for most inputs, so absolute notionals could not have
    reproduced the flat pricer exactly.
    """

    def test_all_ones_is_the_flat_leg_bit_for_bit(self):
        """The acceptance criterion: a constant-notional amortising leg
        equals the flat pricer *exactly*, because multiplying by exactly
        ``1.0`` is exact."""
        ones = jnp.ones(len(ENDS))
        assert fixed_leg_pv(_fixed(amortisation=ones), GRAPH) == fixed_leg_pv(
            _fixed(), GRAPH
        )

    def test_all_ones_is_inert_on_a_floating_leg_too(self):
        ones = jnp.ones(len(ENDS))
        leg = _floating(compounding="compounded", forward_id="USD.SOFR")
        amortising = _floating(
            compounding="compounded",
            forward_id="USD.SOFR",
            amortisation=ones,
        )
        assert floating_leg_pv(amortising, GRAPH) == floating_leg_pv(
            leg, GRAPH
        )

    @pytest.mark.audit
    def test_matches_a_sum_of_single_period_flat_legs(self):
        """Independent reference: an amortising leg is the sum of
        one-period legs, each flat at its own outstanding notional.

        This recomputes the answer through a completely different route —
        many small legs instead of one factor array — so it would catch a
        factor applied to the wrong period (an off-by-one in the overlay).
        """
        factors = [1.0, 0.8, 0.6, 0.4, 0.2]
        leg = _fixed(amortisation=jnp.asarray(factors))
        expected = 0.0
        for i, factor in enumerate(factors):
            single = FixedLeg(
                accrual_start_dates=STARTS[i : i + 1],
                accrual_end_dates=ENDS[i : i + 1],
                payment_dates=ENDS[i : i + 1],
                fixed_rate=jnp.asarray(0.031),
                notional=NOTIONAL * factor,
            )
            expected += float(fixed_leg_pv(single, GRAPH))
        assert float(fixed_leg_pv(leg, GRAPH)) == pytest.approx(
            expected, rel=1e-13
        )

    def test_a_real_step_down_moves_the_answer(self):
        """Anti-vacuity: without this the tests above could pass on an
        overlay that was never applied."""
        amortising = _fixed(
            amortisation=jnp.asarray([1.0, 0.8, 0.6, 0.4, 0.2])
        )
        flat = fixed_leg_pv(_fixed(), GRAPH)
        assert float(fixed_leg_pv(amortising, GRAPH)) < float(flat)

    def test_factors_are_ordered_front_to_back(self):
        """``amortisation[i]`` must apply to period ``i``, not to the
        period that many steps from the end.

        Two mirror-image schedules: ``keeps_early`` pays its first three
        coupons, ``keeps_late`` its last three. Because later coupons are
        discounted harder, keeping the early ones is worth strictly more.
        A reversed or rolled overlay would get this backwards, so the
        inequality pins the array's orientation.
        """
        keeps_early = _fixed(
            amortisation=jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
        )
        keeps_late = _fixed(
            amortisation=jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
        )
        assert float(fixed_leg_pv(keeps_early, GRAPH)) > float(
            fixed_leg_pv(keeps_late, GRAPH)
        )


# ── Compounded floating leg ──────────────────────────────────────────


class TestCompoundedLegMatchesTheBootstrapLeg:
    """Parity with ``_compounded_float_leg_pv``, the curve-build leg."""

    def test_zero_lag_is_bit_for_bit(self):
        """The stated acceptance criterion."""
        leg = _floating(compounding="compounded")
        expected = NOTIONAL * _compounded_float_leg_pv(DISC, REF, ENDS, ENDS)
        assert floating_leg_pv(leg, GRAPH) == expected

    def test_lagged_payments_are_bit_for_bit(self):
        """Parity must survive a payment lag, which is the branch the
        bootstrap's own fast path does *not* take."""
        pay = ENDS + 3
        leg = _floating(compounding="compounded", payment_dates=pay)
        expected = NOTIONAL * _compounded_float_leg_pv(DISC, REF, ENDS, pay)
        assert floating_leg_pv(leg, GRAPH) == expected

    @pytest.mark.audit
    def test_telescopes_at_zero_lag(self):
        """Independent reference: with payment at accrual end and one
        curve, every term collapses and the sum telescopes to
        ``P(0,T_0) - P(0,T_n)`` — two curve reads, no summation."""
        leg = _floating(compounding="compounded")
        expected = float(NOTIONAL) * float(DISC(REF) - DISC(ENDS[-1]))
        assert float(floating_leg_pv(leg, GRAPH)) == pytest.approx(
            expected, abs=1e-6
        )

    def test_seasoning_branch_agrees_before_the_leg_starts(self):
        """Cross-branch: a valuation date at the leg's own start leaves
        nothing realised, so the general cut-over path must return the
        fast path's number *exactly*."""
        leg = _floating(compounding="compounded")
        assert floating_leg_pv(leg, GRAPH, NO_FIXINGS, REF) == floating_leg_pv(
            leg, GRAPH
        )

    def test_a_lag_breaks_the_telescope(self):
        """Anti-vacuity for the payment leg."""
        leg = _floating(compounding="compounded")
        lagged = _floating(compounding="compounded", payment_dates=ENDS + 10)
        assert float(floating_leg_pv(lagged, GRAPH)) < float(
            floating_leg_pv(leg, GRAPH)
        )

    def test_projection_curve_is_actually_used(self):
        """Anti-vacuity for ``forward_id``: a distinct projection curve
        must change the leg, or dual-curve would be decorative."""
        single = _floating(compounding="compounded")
        dual = _floating(compounding="compounded", forward_id="USD.SOFR")
        assert float(floating_leg_pv(dual, GRAPH)) != float(
            floating_leg_pv(single, GRAPH)
        )


# ── Simple floating leg ──────────────────────────────────────────────


def _simple_leg(**kwargs) -> FloatingLeg:
    base = dict(
        spread=jnp.asarray(0.0012),
        fixing_dates=STARTS,
        compounding="simple",
        index_id="USD.SOFR",
        forward_id="USD.SOFR",
    )
    base.update(kwargs)
    return _floating(**base)


class TestSimpleLegMatchesTheDualCurveLeg:
    """Parity with ``_floating_leg_pv``, the dual-curve IBOR-style leg."""

    def test_unseasoned_is_bit_for_bit(self):
        expected = NOTIONAL * _floating_leg_pv(
            GRAPH,
            NO_FIXINGS,
            REF,
            ENDS,
            STARTS,
            "act_360",
            "USD.SOFR",
            "_default_",
            "USD.SOFR",
            jnp.asarray(0.0012),
            ENDS,
        )
        assert floating_leg_pv(_simple_leg(), GRAPH) == expected

    @pytest.mark.audit
    def test_matches_a_hand_rolled_coupon_sum(self):
        """Independent reference: project each period's simple forward,
        add the spread, accrue and discount, one period at a time."""
        leg = _simple_leg()
        taus = year_fraction(STARTS, ENDS, "act_360")
        expected = 0.0
        for i in range(leg.n_periods):
            tau = float(taus[i])
            rate = (float(FWD(STARTS[i])) / float(FWD(ENDS[i])) - 1.0) / tau
            expected += (rate + 0.0012) * tau * float(DISC(ENDS[i]))
        expected *= float(NOTIONAL)
        assert float(floating_leg_pv(leg, GRAPH)) == pytest.approx(
            expected, rel=1e-12
        )

    def test_spread_is_added_to_the_rate(self):
        """Anti-vacuity plus a sign/placement check: raising the spread by
        one basis point adds one basis point of annuity, not something
        else."""
        base = floating_leg_pv(_simple_leg(), GRAPH)
        bumped = floating_leg_pv(
            _simple_leg(spread=jnp.asarray(0.0012 + 1e-4)), GRAPH
        )
        pv01 = float(NOTIONAL) * 1e-4 * float(
            annuity(REF, ENDS, DISC, "act_360")
        )
        assert float(bumped - base) == pytest.approx(pv01, rel=1e-12)

    def test_a_recorded_fixing_carries_no_projection_gradient(self):
        """Spec T-11: once a coupon's rate is observed it is market data,
        so it must have *exactly* zero sensitivity to the projection
        curve — not merely a small one."""
        history = FixingHistory(
            indices={
                "USD.SOFR": FixingSeries(
                    fixing_dates=STARTS[:1],
                    fixings=jnp.asarray([0.0291]),
                )
            }
        )
        one_period = _floating(
            accrual_start_dates=STARTS[:1],
            accrual_end_dates=ENDS[:1],
            payment_dates=ENDS[:1],
            spread=jnp.asarray(0.0),
            fixing_dates=STARTS[:1],
            compounding="simple",
            index_id="USD.SOFR",
            forward_id="USD.SOFR",
        )

        def pv(dfs):
            graph = CurveGraph(
                curves={
                    "_default_": DISC,
                    "USD.SOFR": eqx.tree_at(
                        lambda c: c.discount_factors, FWD, dfs
                    ),
                }
            )
            return floating_leg_pv(one_period, graph, history)

        grad = jax.grad(pv)(FWD.discount_factors)
        assert float(jnp.max(jnp.abs(grad))) == 0.0

    def test_an_unrecorded_fixing_does_carry_one(self):
        """Anti-vacuity for the test above: the zero gradient must come
        from the fixing being recorded, not from the leg ignoring its
        projection curve."""
        one_period = _floating(
            accrual_start_dates=STARTS[:1],
            accrual_end_dates=ENDS[:1],
            payment_dates=ENDS[:1],
            spread=jnp.asarray(0.0),
            fixing_dates=STARTS[:1],
            compounding="simple",
            index_id="USD.SOFR",
            forward_id="USD.SOFR",
        )

        def pv(dfs):
            graph = CurveGraph(
                curves={
                    "_default_": DISC,
                    "USD.SOFR": eqx.tree_at(
                        lambda c: c.discount_factors, FWD, dfs
                    ),
                }
            )
            return floating_leg_pv(one_period, graph, NO_FIXINGS)

        grad = jax.grad(pv)(FWD.discount_factors)
        assert float(jnp.max(jnp.abs(grad))) > 0.0


# ── Seasoning ────────────────────────────────────────────────────────

#: A leg that started 200 days before ``REF``, so its first period
#: genuinely straddles the valuation date.
_SEASONED_START = int(REF) - 200
_S_STARTS, _S_ENDS = _annual_grid(_SEASONED_START, 4)
_DAILY = jnp.arange(_SEASONED_START, int(REF), dtype=jnp.int32)
_SEASONED_HISTORY = FixingHistory(
    indices={
        "USD.SOFR": FixingSeries(
            fixing_dates=_DAILY, fixings=jnp.full(_DAILY.shape, 0.0325)
        )
    }
)


def _seasoned_leg(**kwargs) -> FloatingLeg:
    base = dict(
        accrual_start_dates=_S_STARTS,
        accrual_end_dates=_S_ENDS,
        payment_dates=_S_ENDS,
        spread=jnp.asarray(0.0),
        notional=NOTIONAL,
        compounding="compounded",
        index_id="USD.SOFR",
        forward_id="USD.SOFR",
    )
    base.update(kwargs)
    return FloatingLeg(**base)


class TestSeasoning:
    """Realised fixings blend into the straddling period."""

    @pytest.mark.audit
    def test_straddling_period_matches_the_compounded_factor_blend(self):
        """Independent reference built from
        ``FixingSeries.compounded_factor`` directly: the realised factor
        times the remaining projection, period by period."""
        leg = _seasoned_leg()
        realised = _SEASONED_HISTORY.compounded_factor(
            "USD.SOFR", jnp.asarray(_SEASONED_START), REF, "act_360"
        )
        expected = float(DISC(_S_ENDS[0])) * (
            float(realised) * float(FWD(REF)) / float(FWD(_S_ENDS[0])) - 1.0
        )
        for i in range(1, leg.n_periods):
            expected += float(DISC(_S_ENDS[i])) * (
                float(FWD(_S_STARTS[i])) / float(FWD(_S_ENDS[i])) - 1.0
            )
        expected *= float(NOTIONAL)
        got = floating_leg_pv(leg, GRAPH, _SEASONED_HISTORY, REF)
        assert float(got) == pytest.approx(expected, rel=1e-12)
        # The fixture must actually have accrued something.
        assert float(realised) > 1.0

    def test_a_settled_period_is_purely_realised(self):
        """A period wholly in the past has projection ratio exactly
        ``1.0``, so its coupon is the realised factor minus one — the
        cut-over formula degenerating correctly, checked bit for bit."""
        end = jnp.asarray([int(REF) - 100], dtype=jnp.int32)
        start = jnp.asarray([_SEASONED_START], dtype=jnp.int32)
        leg = FloatingLeg(
            accrual_start_dates=start,
            accrual_end_dates=end,
            payment_dates=end,
            spread=jnp.asarray(0.0),
            notional=jnp.asarray(1.0),
            compounding="compounded",
            index_id="USD.SOFR",
            forward_id="USD.SOFR",
        )
        realised = _SEASONED_HISTORY.compounded_factor(
            "USD.SOFR", start[0], end[0], "act_360"
        )
        expected = DISC(end) * (realised - 1.0)
        assert floating_leg_pv(leg, GRAPH, _SEASONED_HISTORY, REF) == expected

    def test_seasoning_moves_the_answer(self):
        """Anti-vacuity: the realised rate here (3.25%) differs from the
        projection, so ignoring the history must give a different PV."""
        leg = _seasoned_leg()
        seasoned = floating_leg_pv(leg, GRAPH, _SEASONED_HISTORY, REF)
        projected = floating_leg_pv(leg, GRAPH, None, REF)
        assert float(seasoned) != float(projected)

    def test_a_history_for_another_index_is_ignored(self):
        """Per-leg ``index_id`` must actually route the lookup."""
        leg = _seasoned_leg(index_id="USD.OTHER")
        assert floating_leg_pv(
            leg, GRAPH, _SEASONED_HISTORY, REF
        ) == floating_leg_pv(leg, GRAPH, NO_FIXINGS, REF)

    def test_no_history_projects_in_full_even_with_a_ref_date(self):
        """Without a history, every period must be *projected*, not
        zeroed — including periods already settled.

        The cut-over is deliberately conditional on a history existing.
        Were it applied unconditionally, a settled period would take its
        cut-over at its own accrual end, where the projection ratio is
        exactly ``1.0``, and so pay a silent zero index coupon rather
        than a projected one. Gated bit for bit against the no-valuation
        -date path, with a settled first period in the fixture.
        """
        settled_ends = jnp.asarray(
            [int(REF) - 100, int(REF) + 265], dtype=jnp.int32
        )
        settled_starts = jnp.asarray(
            [_SEASONED_START, int(REF) - 100], dtype=jnp.int32
        )
        leg = FloatingLeg(
            accrual_start_dates=settled_starts,
            accrual_end_dates=settled_ends,
            payment_dates=settled_ends,
            spread=jnp.asarray(0.0),
            notional=NOTIONAL,
            compounding="compounded",
            index_id="USD.SOFR",
            forward_id="USD.SOFR",
        )
        projected = floating_leg_pv(leg, GRAPH)
        assert floating_leg_pv(leg, GRAPH, None, REF) == projected
        assert floating_leg_pv(leg, GRAPH, NO_FIXINGS, REF) == projected
        # Anti-vacuity: supplying the history *does* change it, so the
        # equality above is a statement about absent data, not a leg
        # that ignores its fixings.
        assert float(
            floating_leg_pv(leg, GRAPH, _SEASONED_HISTORY, REF)
        ) != float(projected)

    def test_a_fixing_dated_on_the_valuation_date_is_excluded(self):
        """The cut-over window is half-open at the top: the overnight
        rate set *on* the valuation date is earned over the following
        day, which has not happened yet."""
        with_today = FixingHistory(
            indices={
                "USD.SOFR": FixingSeries(
                    fixing_dates=jnp.concatenate(
                        [_DAILY, jnp.asarray([int(REF)], dtype=jnp.int32)]
                    ),
                    fixings=jnp.concatenate(
                        [jnp.full(_DAILY.shape, 0.0325), jnp.asarray([0.25])]
                    ),
                )
            }
        )
        leg = _seasoned_leg()
        # A 25% rate dated exactly REF would be impossible to miss if it
        # leaked into the realised factor.
        assert floating_leg_pv(leg, GRAPH, with_today, REF) == floating_leg_pv(
            leg, GRAPH, _SEASONED_HISTORY, REF
        )


# ── Averaged leg ─────────────────────────────────────────────────────


class TestAveragedLeg:
    """Averaging discards interest-on-interest: a sum, not a product."""

    def test_averaged_is_worth_less_than_compounded(self):
        """``sum_j (rho_j - 1) < prod_j rho_j - 1`` for rho > 1, so the
        averaged leg must come in below the compounded one."""
        averaged = _floating(
            compounding="averaged", forward_id="USD.SOFR", index_id="USD.SOFR"
        )
        compounded = _floating(
            compounding="compounded",
            forward_id="USD.SOFR",
            index_id="USD.SOFR",
        )
        assert float(floating_leg_pv(averaged, GRAPH)) < float(
            floating_leg_pv(compounded, GRAPH)
        )

    @pytest.mark.audit
    def test_gap_is_the_second_order_compounding_term(self):
        """The averaged/compounded gap is interest-on-interest, of order
        ``r^2 * tau / 2`` per period — a magnitude check that would catch
        the two conventions being accidentally identical *or* wildly
        different."""
        averaged = _floating(
            compounding="averaged", forward_id="USD.SOFR", index_id="USD.SOFR"
        )
        compounded = _floating(
            compounding="compounded",
            forward_id="USD.SOFR",
            index_id="USD.SOFR",
        )
        gap = float(floating_leg_pv(compounded, GRAPH)) - float(
            floating_leg_pv(averaged, GRAPH)
        )
        taus = year_fraction(STARTS, ENDS, "act_360")
        rough = 0.0
        for i in range(len(ENDS)):
            tau = float(taus[i])
            rate = (float(FWD(STARTS[i])) / float(FWD(ENDS[i])) - 1.0) / tau
            rough += 0.5 * rate * rate * tau * float(DISC(ENDS[i]))
        rough *= float(NOTIONAL)
        assert gap == pytest.approx(rough, rel=0.15)
        assert gap > 0.0

    def test_an_explicit_span_matches_the_read_one_bit_for_bit(self):
        """The ``n_days`` parameter added to the shared scalar helper
        must be an *override*, not a different calculation — otherwise
        the bootstrap's averaged futures and the leg engine would
        silently diverge on the same convention."""
        span = int(ENDS[0] - STARTS[0])
        tau = year_fraction(STARTS[:1], ENDS[:1], "act_360")[0]
        read = _arithmetic_average_forward(FWD, STARTS[0], ENDS[0], tau)
        given = _arithmetic_average_forward(
            FWD, STARTS[0], ENDS[0], tau, n_days=span
        )
        assert read == given

    def test_seasoning_an_averaged_leg_is_refused(self):
        """The realised part of an arithmetic average is a sum of
        recorded rates, and ``FixingSeries`` only reduces them to a
        compounding product. Rather than quietly use the wrong one, the
        pricer refuses."""
        leg = _floating(
            compounding="averaged",
            forward_id="USD.SOFR",
            index_id="USD.SOFR",
        )
        with pytest.raises(ValueError, match="season an averaged leg"):
            floating_leg_pv(leg, GRAPH, _SEASONED_HISTORY, REF)


# ── Per-leg index descriptors ────────────────────────────────────────


class TestPerLegIndexDescriptors:
    """The point of the leg abstraction: three curves in one trade."""

    def _basis_swap(self) -> MultiLegSwap:
        """A single-currency tenor basis swap: OIS discounting, two
        different projection curves. Inexpressible with a singular
        ``forward_id`` on the trade."""
        return MultiLegSwap(
            legs=(
                _floating(
                    compounding="compounded",
                    index_id="USD.SOFR",
                    forward_id="USD.SOFR",
                ),
                _floating(
                    compounding="simple",
                    fixing_dates=STARTS,
                    spread=jnp.asarray(0.0015),
                    index_id="USD.LIBOR.6M",
                    forward_id="USD.LIBOR.6M",
                    pay=True,
                ),
            )
        )

    def test_three_curves_are_reported(self):
        swap = self._basis_swap()
        assert swap.curves_touched == (
            "_default_",
            "USD.SOFR",
            "USD.LIBOR.6M",
        )
        assert swap.indices_touched == ("USD.SOFR", "USD.LIBOR.6M")

    def test_each_leg_reads_its_own_projection_curve(self):
        """Bumping one projection curve must move exactly one leg, which
        is what proves the descriptors are not collapsing to one curve."""
        swap = self._basis_swap()
        bumped = CurveGraph(
            curves={
                "_default_": DISC,
                "USD.SOFR": FWD,
                "USD.LIBOR.6M": _sloped_curve(0.040, 0.0055),
            }
        )
        first_before = leg_pv(swap.legs[0], GRAPH)
        first_after = leg_pv(swap.legs[0], bumped)
        second_before = leg_pv(swap.legs[1], GRAPH)
        second_after = leg_pv(swap.legs[1], bumped)
        assert first_after == first_before
        assert float(second_after) != float(second_before)

    def test_swap_pv_is_the_sum_of_signed_legs(self):
        swap = self._basis_swap()
        expected = leg_pv(swap.legs[0], GRAPH) + leg_pv(swap.legs[1], GRAPH)
        assert swap_pv(swap, GRAPH) == expected

    def test_de_duplication_preserves_first_appearance_order(self):
        """Two legs sharing a discount curve must report it once, and the
        order must be deterministic rather than hash-dependent."""
        swap = MultiLegSwap(legs=(_fixed(), _fixed(pay=True)))
        assert swap.curves_touched == ("_default_",)
        assert swap.indices_touched == ()


# ── Par rate ─────────────────────────────────────────────────────────


class TestSwapFairRate:
    def _vanilla(self) -> MultiLegSwap:
        return MultiLegSwap(
            legs=(
                _fixed(pay=True),
                _floating(
                    compounding="compounded",
                    index_id="USD.SOFR",
                    forward_id="USD.SOFR",
                ),
            )
        )

    def test_par_rate_zeroes_the_swap(self):
        swap = self._vanilla()
        par = swap_fair_rate(swap, GRAPH)
        at_par = swap_pv(
            eqx.tree_at(lambda s: s.legs[0].fixed_rate, swap, par), GRAPH
        )
        assert float(at_par) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.audit
    def test_par_rate_is_float_pv_over_annuity(self):
        """Independent reference: the textbook ratio, built from the
        floating leg PV and the annuity separately."""
        swap = self._vanilla()
        float_pv = leg_pv(swap.legs[1], GRAPH)
        ann = NOTIONAL * annuity(REF, ENDS, DISC, "act_360")
        assert float(swap_fair_rate(swap, GRAPH)) == pytest.approx(
            float(float_pv / ann), rel=1e-12
        )

    def test_needs_exactly_one_fixed_leg(self):
        both_float = MultiLegSwap(
            legs=(
                _floating(compounding="compounded"),
                _floating(compounding="compounded", pay=True),
            )
        )
        with pytest.raises(ValueError, match="exactly one fixed leg"):
            swap_fair_rate(both_float, GRAPH)
        two_fixed = MultiLegSwap(legs=(_fixed(), _fixed(pay=True)))
        with pytest.raises(ValueError, match="exactly one fixed leg"):
            swap_fair_rate(two_fixed, GRAPH)


# ── JIT and derivatives ──────────────────────────────────────────────


class TestJitAndGradients:
    def test_fixed_leg_pv_jits(self):
        leg = _fixed(amortisation=jnp.asarray([1.0, 0.9, 0.8, 0.7, 0.6]))
        jitted = eqx.filter_jit(fixed_leg_pv)
        assert float(jitted(leg, GRAPH)) == pytest.approx(
            float(fixed_leg_pv(leg, GRAPH)), rel=1e-14
        )

    @pytest.mark.parametrize("compounding", ["simple", "compounded"])
    def test_floating_leg_pv_jits(self, compounding):
        kwargs = {"fixing_dates": STARTS} if compounding == "simple" else {}
        leg = _floating(
            compounding=compounding,
            index_id="USD.SOFR",
            forward_id="USD.SOFR",
            **kwargs,
        )
        jitted = eqx.filter_jit(floating_leg_pv)
        assert float(jitted(leg, GRAPH)) == pytest.approx(
            float(floating_leg_pv(leg, GRAPH)), rel=1e-14
        )

class TestAveragedLegUnderJit:
    """The averaged convention needs a *shape* at trace time.

    An arithmetic average is defined per rate day, so a period must be
    subdivided into its actual days — and that array's length depends on
    the dates' *values*. XLA requires shapes at trace time, so there are
    exactly three ways to price one, and this class pins all three plus
    the diagnostic for the case that genuinely cannot work.

    The resolution is **not** to make the dates static. That would
    recompile per distinct schedule and make legs unstackable, since
    differing statics change the treedef. Only the *bound* is promoted;
    see [JAX Patterns § 2.5](../architecture/jax-patterns.md).
    """

    def _leg(self, **kwargs):
        return _floating(
            compounding="averaged",
            index_id="USD.SOFR",
            forward_id="USD.SOFR",
            **kwargs,
        )

    def test_mode_1_eager(self):
        assert float(floating_leg_pv(self._leg(), GRAPH)) > 0.0

    def test_mode_2_closed_over_jits_without_a_bound(self):
        """With the leg closed over, its dates are compile-time
        constants, so the span can still be read — provided the read
        happens *before* indexing.

        ``np.asarray(dates)[i]`` is concrete; ``np.asarray(dates[i])``
        is not, because the ``jnp`` index is itself a traced operation.
        That ordering is the whole difference, and it is why scalar-dated
        ``MoneyMarketFuture`` has always worked closed-over while an
        array-dated leg did not.
        """
        leg = self._leg()
        eager = float(floating_leg_pv(leg, GRAPH))
        jitted = float(eqx.filter_jit(lambda g: floating_leg_pv(leg, g))(GRAPH))
        assert jitted == pytest.approx(eager, rel=1e-13)

    def test_mode_3_arg_passed_needs_a_bound(self):
        """Passed as an argument the dates are genuinely traced, so the
        span cannot be read at all — and the error says what to do."""
        with pytest.raises(ValueError, match="max_days"):
            eqx.filter_jit(floating_leg_pv)(self._leg(), GRAPH)

    def test_mode_3_arg_passed_jits_with_a_bound(self):
        leg = self._leg(max_days=max_period_days(STARTS, ENDS))
        eager = float(floating_leg_pv(self._leg(), GRAPH))
        jitted = float(eqx.filter_jit(floating_leg_pv)(leg, GRAPH))
        assert jitted == pytest.approx(eager, rel=1e-13)

    def test_the_bounded_path_is_bit_for_bit_the_eager_one(self):
        """Eagerly the two routes agree *exactly*, despite one padding
        every period to a common length.

        Clipping each period's grid at its own end date makes the surplus
        boundaries contribute ``P(T_e)/P(T_e) - 1``, which is exactly
        zero because a float divided by itself is exactly ``1.0``. So the
        padding is not merely negligible — it is inert.
        """
        bounded = self._leg(max_days=max_period_days(STARTS, ENDS))
        assert floating_leg_pv(bounded, GRAPH) == floating_leg_pv(
            self._leg(), GRAPH
        )

    def test_it_holds_on_ragged_periods(self):
        """One bound must serve periods of *different* lengths, which is
        the case padding exists for. A semiannual stub beside annual
        periods would be priced wrong by any scheme that assumed a
        uniform day count."""
        ends = jnp.asarray(
            [int(REF) + 365, int(REF) + 731, int(REF) + 912],
            dtype=jnp.int32,
        )
        starts = jnp.asarray(
            [int(REF), int(REF) + 365, int(REF) + 731], dtype=jnp.int32
        )
        spans = [365, 366, 181]
        assert [
            int(e - s) for s, e in zip(starts.tolist(), ends.tolist())
        ] == spans
        common = dict(
            accrual_start_dates=starts,
            accrual_end_dates=ends,
            payment_dates=ends,
        )
        eager = self._leg(**common)
        bounded = self._leg(**common, max_days=max_period_days(starts, ends))
        assert floating_leg_pv(bounded, GRAPH) == floating_leg_pv(
            eager, GRAPH
        )

    @pytest.mark.parametrize("jitted", [False, True])
    def test_an_undersized_bound_raises_rather_than_truncating(self, jitted):
        """The one way this design could be silently wrong.

        A bound shorter than the longest period does not error on its
        own — the grid just clips early and the average is taken over
        too few days. Measured at roughly *half* the correct rate for a
        200-day bound on annual periods, which is the kind of error that
        looks like a plausible number. ``eqx.error_if`` makes it loud,
        eagerly and under ``jit`` alike.
        """
        leg = self._leg(max_days=200)
        price = (
            eqx.filter_jit(floating_leg_pv) if jitted else floating_leg_pv
        )
        with pytest.raises(EquinoxRuntimeError):
            price(leg, GRAPH)

    def test_a_bound_is_rejected_on_a_convention_that_cannot_use_it(self):
        """Pairing guard: only the averaged convention subdivides a
        period, so only it has a shape to bound."""
        with pytest.raises(ValueError, match="only applies to"):
            _floating(compounding="compounded", max_days=366)

    def test_a_degenerate_bound_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1 day"):
            self._leg(max_days=0)

    def test_max_period_days_needs_concrete_dates(self):
        """The helper is deliberately eager — it is the step that turns a
        schedule into a static bound, so it must run before the trace."""

        @jax.jit
        def f(starts, ends):
            return max_period_days(starts, ends)

        with pytest.raises(ValueError, match="concrete dates"):
            f(STARTS, ENDS)

    def test_a_bounded_leg_can_be_stacked_and_vmapped(self):
        """The capability that static dates would have cost.

        Because only the bound is static and the dates stay dynamic, two
        legs share a treedef and so stack into one batched pytree. Had
        the dates been static, differing schedules would change the
        treedef and ``tree_map(stack, ...)`` would fail outright.
        """
        leg = self._leg(max_days=max_period_days(STARTS, ENDS))
        book = jax.tree_util.tree_map(lambda *x: jnp.stack(x), leg, leg)
        prices = eqx.filter_vmap(lambda l: floating_leg_pv(l, GRAPH))(book)
        assert prices.shape == (2,)
        assert float(prices[0]) == pytest.approx(
            float(floating_leg_pv(leg, GRAPH)), rel=1e-13
        )

    def test_the_bounded_path_differentiates_under_jit(self):
        leg = self._leg(max_days=max_period_days(STARTS, ENDS))

        def pv(dfs):
            graph = CurveGraph(
                curves={
                    "_default_": DISC,
                    "USD.SOFR": eqx.tree_at(
                        lambda c: c.discount_factors, FWD, dfs
                    ),
                }
            )
            return floating_leg_pv(leg, graph)

        grad = eqx.filter_jit(jax.grad(pv))(FWD.discount_factors)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.max(jnp.abs(grad))) > 0.0

    def test_swap_pv_jits_with_seasoning(self):
        swap = MultiLegSwap(legs=(_fixed(pay=True), _seasoned_leg()))

        @eqx.filter_jit
        def priced(trade, graph, history):
            return swap_pv(trade, graph, history, REF)

        assert float(
            priced(swap, GRAPH, _SEASONED_HISTORY)
        ) == pytest.approx(
            float(swap_pv(swap, GRAPH, _SEASONED_HISTORY, REF)), rel=1e-14
        )

    def test_rate_sensitivity_matches_finite_differences(self):
        """Autodiff through the leg engine against a central difference
        on the discount curve — a sanity check on the chain, since the
        library's Greeks are autodiff everywhere else."""
        leg = _fixed(amortisation=jnp.asarray([1.0, 0.9, 0.8, 0.7, 0.6]))

        def pv(dfs):
            graph = CurveGraph(
                curves={
                    "_default_": eqx.tree_at(
                        lambda c: c.discount_factors, DISC, dfs
                    )
                }
            )
            return fixed_leg_pv(leg, graph)

        base = DISC.discount_factors
        grad = jax.grad(pv)(base)
        h = 1e-7
        for i in (3, 6, 9):
            bump = jnp.zeros_like(base).at[i].set(h)
            fd = (float(pv(base + bump)) - float(pv(base - bump))) / (2 * h)
            assert float(grad[i]) == pytest.approx(fd, rel=1e-4)


# ── Construction guards ──────────────────────────────────────────────


class TestLegGuards:
    """Incoherent configurations raise at the user boundary."""

    def test_simple_leg_needs_fixing_dates(self):
        with pytest.raises(ValueError, match="needs fixing_dates"):
            _floating(compounding="simple")

    def test_compounded_leg_rejects_fixing_dates(self):
        """A field nothing would consult is a configuration error, not a
        harmless extra — accepting it would imply an observation shift
        that is not modelled."""
        with pytest.raises(ValueError, match="does not read"):
            _floating(compounding="compounded", fixing_dates=STARTS)

    def test_unknown_compounding_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown compounding"):
            _floating(compounding="in_arrears")

    def test_unknown_day_count_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown day count"):
            _fixed(day_count="act_366")

    def test_a_calendar_day_count_is_refused(self):
        """Bus/252 needs a calendar, and a leg carries none. Refused at
        construction rather than failing inside ``year_fraction`` at
        pricing time, and certainly rather than being given a default
        calendar the caller never chose."""
        with pytest.raises(ValueError, match="needs a business-day calendar"):
            _fixed(day_count="bus_252")

    def test_averaged_requires_an_actual_day_count(self):
        """The averaged forward is a day-count-weighted average, which
        collapses to a sum of daily ratios only when the period's year
        fraction equals the sum of its daily fractions. 30/360 breaks
        that, so it would compute a different quantity rather than an
        approximation of the same one."""
        with pytest.raises(ValueError, match="needs an actual/N day count"):
            _floating(compounding="averaged", day_count="30_360")
        # The same day count is fine on the conventions that do not
        # subdivide the period -- so the guard is specific, not blanket.
        assert _floating(
            compounding="compounded", day_count="30_360"
        ).day_count == "30_360"

    def test_ragged_schedule_is_rejected(self):
        with pytest.raises(ValueError, match="disagree"):
            _fixed(payment_dates=ENDS[:-1])

    def test_ragged_amortisation_is_rejected(self):
        with pytest.raises(ValueError, match="one entry per accrual period"):
            _fixed(amortisation=jnp.ones(3))

    def test_empty_leg_is_rejected(self):
        with pytest.raises(ValueError, match="no accrual periods"):
            _fixed(
                accrual_start_dates=STARTS[:0],
                accrual_end_dates=ENDS[:0],
                payment_dates=ENDS[:0],
            )

    def test_legs_must_be_a_tuple(self):
        with pytest.raises(TypeError, match="must be a tuple"):
            MultiLegSwap(legs=[_fixed()])

    def test_empty_swap_is_rejected(self):
        with pytest.raises(ValueError, match="no legs"):
            MultiLegSwap(legs=())

    def test_leg_pv_rejects_a_non_leg(self):
        with pytest.raises(TypeError, match="no implementation"):
            leg_pv("not a leg", GRAPH)

    def test_guards_survive_the_jit_boundary(self):
        """The guards are pre-trace, so a leg that was valid when built
        stays valid when traced — the checks must not fire spuriously."""
        leg = _floating(compounding="compounded", forward_id="USD.SOFR")
        assert float(eqx.filter_jit(floating_leg_pv)(leg, GRAPH)) != 0.0
