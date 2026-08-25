"""QuantLib cross-validation for VALAX interest rate pricers.

Workstream 1 of the rates session.  Tests three families of rate
pricers against QuantLib as an independent oracle:

1. **TestSwaptionQL** — European payer and receiver swaptions under
   Black-76 and Bachelier against ``ql.BlackSwaptionEngine`` /
   ``ql.BachelierSwaptionEngine``.  Tolerance: ``rel < 1e-4``.

2. **TestCapFloorQL** — Caps and floors under Black-76 and Bachelier
   against ``ql.BlackCapFloorEngine`` / ``ql.BachelierCapFloorEngine``.
   Tolerance: ``rel < 1e-4``.

3. **TestHWCallableTreeQL** — Callable and puttable bonds priced on the
   Hull-White trinomial tree against ``ql.TreeCallableFixedRateBondEngine``
   driven by a ``ql.HullWhite`` model calibrated to the same flat curve.
   Tolerance: ``rel < 5e-3`` (coarser: tree discretisation).

Conventions (matching ``_ql_adapters.py``):
- Evaluation date: ``DEFAULT_QL_DATE = ql.Date(1, 1, 2026)``.
- Day count: ``ql.Actual365Fixed()`` throughout.
- Calendar: ``ql.NullCalendar()`` — no business-day adjustments.
- Expiries and coupon schedules are integer-day-aligned.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import QuantLib as ql

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.instruments.rates import Cap, Caplet, Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.caplets import (
    cap_price_bachelier,
    cap_price_black76,
    caplet_price_bachelier,
    caplet_price_black76,
)
from valax.pricing.analytic.swaptions import (
    _annuity,
    swaption_price_bachelier,
    swaption_price_black76,
)
from valax.pricing.lattice.hull_white_tree import callable_bond_price, puttable_bond_price

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    ql_dates_from_year_offsets,
    ql_flat_curve,
    reset_evaluation_date,
    snap_expiry_to_days,
)

# ── Reference date ────────────────────────────────────────────────────

_REF_DATE_ORD = int(ymd_to_ordinal(2026, 1, 1))
_ACT365 = "act_365"


# ── Flat VALAX curve builder ──────────────────────────────────────────

def _flat_valax_curve(rate: float, pillar_years: list[float]) -> DiscountCurve:
    """Build a flat continuously-compounded VALAX discount curve.

    A ``t = 0`` pillar (``DF = 1``) is always prepended.  :class:`DiscountCurve`
    interpolates log-linearly with *flat* extrapolation, so without an anchor at
    the reference date every query before the first pillar silently returns that
    pillar's discount factor.  With the anchor in place, and because
    ``log DF = -r t`` is linear for a flat curve, interpolation is exact at every
    date -- no pillar needs to coincide with a schedule date.
    """
    ref = _REF_DATE_ORD
    years = [0.0] + [y for y in pillar_years if y > 0.0]
    pillars = jnp.array(
        [ref + int(round(y * 365)) for y in years],
        dtype=jnp.int32,
    )
    times = (pillars - ref).astype(jnp.float64) / 365.0
    dfs = jnp.exp(-jnp.asarray(rate) * times)
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=dfs,
        reference_date=jnp.int32(ref),
        day_count=_ACT365,
    )


# ─────────────────────────────────────────────────────────────────────
# Workstream 1a — Swaption QL validation
# ─────────────────────────────────────────────────────────────────────


class TestSwaptionQL:
    """European swaptions: VALAX Black-76 / Bachelier vs QuantLib."""

    # Parametrize over a small grid of (rate, vol, tenor_y, expiry_y, strike_offset)
    _CASES = [
        (0.04, 0.20, 5, 1, 0.00),   # ATM 5y into 1y
        (0.04, 0.25, 5, 1, 0.01),   # 1 % OTM
        (0.04, 0.15, 10, 2, -0.01), # 1 % ITM  10y tenor
        (0.05, 0.18, 3, 0.5, 0.00), # 3y tenor, 6m expiry
        (0.03, 0.22, 7, 2, 0.005),  # low-rate regime
    ]

    @pytest.fixture(params=_CASES, ids=[f"c{i}" for i in range(len(_CASES))])
    def setup(self, request):
        rate, lognorm_vol, tenor_y, expiry_y, strike_off = request.param
        reset_evaluation_date()

        # Day-align the expiry.
        exp_days, exp_eff = snap_expiry_to_days(expiry_y)

        ref = _REF_DATE_ORD
        # Fixed dates: annual from expiry to expiry + tenor.
        n_periods = int(tenor_y)
        expiry_ord = ref + exp_days
        fixed_ords = jnp.array(
            [expiry_ord + int(round(k * 365)) for k in range(1, n_periods + 1)],
            dtype=jnp.int32,
        )

        # Pillars at the expiry and at every fixed-leg date (plus the t=0
        # anchor added by the helper) so no schedule date is extrapolated.
        curve = _flat_valax_curve(
            rate, [exp_eff] + [exp_eff + k for k in range(1, n_periods + 1)]
        )

        # Forward swap rate for ATM reference.
        ann = float(_annuity(
            jnp.int32(expiry_ord), fixed_ords, curve, _ACT365
        ))
        df_s = float(curve(jnp.int32(expiry_ord)))
        df_e = float(curve(fixed_ords[-1]))
        fwd_swap_rate = (df_s - df_e) / ann
        strike = fwd_swap_rate + strike_off

        valax_sw = Swaption(
            expiry_date=jnp.int32(expiry_ord),
            fixed_dates=fixed_ords,
            strike=jnp.asarray(strike),
            notional=jnp.asarray(1_000_000.0),
            is_payer=True,
            day_count=_ACT365,
        )

        # ── QuantLib side ─────────────────────────────────────────────
        ql_disc = ql_flat_curve(rate)
        expiry_ql = DEFAULT_QL_DATE + ql.Period(exp_days, ql.Days)
        fixed_ql = [expiry_ql + ql.Period(int(round(k * 365)), ql.Days) for k in range(1, n_periods + 1)]
        schedule = ql.Schedule(
            expiry_ql,
            fixed_ql[-1],
            ql.Period(ql.Annual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Forward, False,
        )
        idx = ql.IborIndex(
            "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(),
            ql.NullCalendar(), ql.Unadjusted, False, ql.Actual365Fixed(),
            ql_disc,
        )
        ql_swap = ql.VanillaSwap(
            ql.VanillaSwap.Payer,
            1_000_000.0,
            schedule, strike, ql.Actual365Fixed(),
            schedule, idx, 0.0, ql.Actual365Fixed(),
        )
        exercise = ql.EuropeanExercise(expiry_ql)
        ql_swaption = ql.Swaption(ql_swap, exercise)

        # Receiver leg: same schedule/strike, opposite swap direction.
        ql_swap_recv = ql.VanillaSwap(
            ql.VanillaSwap.Receiver,
            1_000_000.0,
            schedule, strike, ql.Actual365Fixed(),
            schedule, idx, 0.0, ql.Actual365Fixed(),
        )
        ql_swaption_recv = ql.Swaption(ql_swap_recv, exercise)

        # Normal vol (roughly proportional to lognormal × fwd rate).
        norm_vol = lognorm_vol * max(fwd_swap_rate, 0.001)

        return {
            "valax_sw": valax_sw,
            "curve": curve,
            "ql_swaption": ql_swaption,
            "ql_swaption_recv": ql_swaption_recv,
            "ql_disc": ql_disc,
            "lognorm_vol": lognorm_vol,
            "norm_vol": norm_vol,
            "fwd_swap_rate": fwd_swap_rate,
        }

    def test_black76_payer(self, setup):
        v = float(swaption_price_black76(
            setup["valax_sw"], setup["curve"], jnp.asarray(setup["lognorm_vol"])
        ))
        engine = ql.BlackSwaptionEngine(setup["ql_disc"], ql.QuoteHandle(ql.SimpleQuote(setup["lognorm_vol"])))
        setup["ql_swaption"].setPricingEngine(engine)
        q = setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Black76 payer: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"

    def test_black76_receiver(self, setup):
        """VALAX receiver vs ``ql.BlackSwaptionEngine`` on a QL Receiver swap."""
        sw_r = Swaption(
            expiry_date=setup["valax_sw"].expiry_date,
            fixed_dates=setup["valax_sw"].fixed_dates,
            strike=setup["valax_sw"].strike,
            notional=setup["valax_sw"].notional,
            is_payer=False,
            day_count=_ACT365,
        )
        v = float(swaption_price_black76(
            sw_r, setup["curve"], jnp.asarray(setup["lognorm_vol"])
        ))
        engine = ql.BlackSwaptionEngine(
            setup["ql_disc"],
            ql.QuoteHandle(ql.SimpleQuote(setup["lognorm_vol"])),
        )
        setup["ql_swaption_recv"].setPricingEngine(engine)
        q = setup["ql_swaption_recv"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Black76 receiver: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"

    def test_payer_receiver_parity(self, setup):
        """payer - receiver == notional * annuity * (S - K)."""
        sw_p = setup["valax_sw"]
        sw_r = Swaption(
            expiry_date=sw_p.expiry_date,
            fixed_dates=sw_p.fixed_dates,
            strike=sw_p.strike,
            notional=sw_p.notional,
            is_payer=False,
            day_count=_ACT365,
        )
        vol = jnp.asarray(setup["lognorm_vol"])
        v_payer = float(swaption_price_black76(sw_p, setup["curve"], vol))
        v_recv = float(swaption_price_black76(sw_r, setup["curve"], vol))

        ann = float(_annuity(
            sw_p.expiry_date, sw_p.fixed_dates, setup["curve"], _ACT365,
        ))
        df_s = float(setup["curve"](sw_p.expiry_date))
        df_e = float(setup["curve"](sw_p.fixed_dates[-1]))
        fwd_s = (df_s - df_e) / ann
        parity = float(sw_p.notional) * ann * (fwd_s - float(sw_p.strike))
        assert abs((v_payer - v_recv) - parity) / max(abs(parity), 1.0) < 1e-10

    def test_bachelier_payer(self, setup):
        v = float(swaption_price_bachelier(
            setup["valax_sw"], setup["curve"], jnp.asarray(setup["norm_vol"])
        ))
        engine = ql.BachelierSwaptionEngine(
            setup["ql_disc"],
            ql.QuoteHandle(ql.SimpleQuote(setup["norm_vol"])),
        )
        setup["ql_swaption"].setPricingEngine(engine)
        q = setup["ql_swaption"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Bachelier payer: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"


# ─────────────────────────────────────────────────────────────────────
# Workstream 1b — Cap/Floor QL validation
# ─────────────────────────────────────────────────────────────────────


class TestCapFloorQL:
    """Caps and floors: VALAX Black-76 / Bachelier vs QuantLib."""

    _CASES = [
        (0.04, 0.25, 3, 0.04),   # ATM 3y annual cap
        (0.04, 0.20, 5, 0.05),   # OTM 5y cap
        (0.04, 0.20, 5, 0.03),   # ITM 5y cap
        (0.05, 0.18, 2, 0.05),   # 2y cap, ATM
        (0.03, 0.30, 4, 0.03),   # low-rate 4y cap
    ]

    @pytest.fixture(params=_CASES, ids=[f"c{i}" for i in range(len(_CASES))])
    def setup(self, request):
        rate, lognorm_vol, n_periods, strike = request.param
        reset_evaluation_date()

        ref = _REF_DATE_ORD

        # Annual caplet schedule: fixing at start of each period, end 1y later.
        period_years = list(range(1, n_periods + 1))
        fixing_ords = jnp.array(
            [ref + int(round((y - 1) * 365)) for y in period_years],
            dtype=jnp.int32,
        )
        start_ords = fixing_ords
        end_ords = jnp.array(
            [ref + int(round(y * 365)) for y in period_years],
            dtype=jnp.int32,
        )

        all_years = [0.0] + [float(y) for y in period_years]
        curve = _flat_valax_curve(rate, all_years[1:])

        valax_cap = Cap(
            fixing_dates=fixing_ords,
            start_dates=start_ords,
            end_dates=end_ords,
            strike=jnp.asarray(strike),
            notional=jnp.asarray(1_000_000.0),
            is_cap=True,
            day_count=_ACT365,
        )

        # ── QuantLib side ─────────────────────────────────────────────
        ql_disc = ql_flat_curve(rate)
        ql_dates_list = [DEFAULT_QL_DATE] + [
            DEFAULT_QL_DATE + ql.Period(int(round(y * 365)), ql.Days)
            for y in period_years
        ]
        schedule = ql.Schedule(
            ql_dates_list[0],
            ql_dates_list[-1],
            ql.Period(ql.Annual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Forward, False,
        )
        idx = ql.IborIndex(
            "Flat", ql.Period(ql.Annual), 0, ql.EURCurrency(),
            ql.NullCalendar(), ql.Unadjusted, False, ql.Actual365Fixed(),
            ql_disc,
        )
        # QuantLib >= 1.35: Cap/Floor take a coupon Leg, not (schedule, index).
        ql_leg = ql.IborLeg(
            [1_000_000.0], schedule, idx, ql.Actual365Fixed(), ql.Unadjusted,
        )
        ql_cap = ql.Cap(ql_leg, [strike])

        norm_vol = lognorm_vol * max(rate, 0.001)

        return {
            "valax_cap": valax_cap,
            "curve": curve,
            "ql_cap": ql_cap,
            "ql_leg": ql_leg,
            "ql_disc": ql_disc,
            "lognorm_vol": lognorm_vol,
            "norm_vol": norm_vol,
        }

    def test_black76_cap(self, setup):
        v = float(cap_price_black76(
            setup["valax_cap"], setup["curve"], jnp.asarray(setup["lognorm_vol"])
        ))
        engine = ql.BlackCapFloorEngine(
            setup["ql_disc"],
            ql.QuoteHandle(ql.SimpleQuote(setup["lognorm_vol"])),
        )
        setup["ql_cap"].setPricingEngine(engine)
        q = setup["ql_cap"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Black76 cap: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"

    def test_black76_floor(self, setup):
        from valax.instruments.rates import Cap as CapInst
        valax_floor = CapInst(
            fixing_dates=setup["valax_cap"].fixing_dates,
            start_dates=setup["valax_cap"].start_dates,
            end_dates=setup["valax_cap"].end_dates,
            strike=setup["valax_cap"].strike,
            notional=setup["valax_cap"].notional,
            is_cap=False,
            day_count=_ACT365,
        )
        v = float(cap_price_black76(
            valax_floor, setup["curve"], jnp.asarray(setup["lognorm_vol"])
        ))
        # Same coupon leg as the cap, so cap/floor differ only in optionality.
        ql_floor = ql.Floor(setup["ql_leg"], [float(setup["valax_cap"].strike)])
        engine = ql.BlackCapFloorEngine(
            setup["ql_disc"],
            ql.QuoteHandle(ql.SimpleQuote(setup["lognorm_vol"])),
        )
        ql_floor.setPricingEngine(engine)
        q = ql_floor.NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Black76 floor: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"

    def test_bachelier_cap(self, setup):
        v = float(cap_price_bachelier(
            setup["valax_cap"], setup["curve"], jnp.asarray(setup["norm_vol"])
        ))
        engine = ql.BachelierCapFloorEngine(
            setup["ql_disc"],
            ql.QuoteHandle(ql.SimpleQuote(setup["norm_vol"])),
        )
        setup["ql_cap"].setPricingEngine(engine)
        q = setup["ql_cap"].NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-4, f"Bachelier cap: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.2e}"

    def test_cap_floor_parity(self, setup):
        """cap - floor == swap (floating - fixed leg PV), by no-arb."""
        v_cap = float(cap_price_black76(
            setup["valax_cap"], setup["curve"], jnp.asarray(setup["lognorm_vol"])
        ))
        from valax.instruments.rates import Cap as CapInst
        valax_floor = CapInst(
            fixing_dates=setup["valax_cap"].fixing_dates,
            start_dates=setup["valax_cap"].start_dates,
            end_dates=setup["valax_cap"].end_dates,
            strike=setup["valax_cap"].strike,
            notional=setup["valax_cap"].notional,
            is_cap=False,
            day_count=_ACT365,
        )
        v_floor = float(cap_price_black76(
            valax_floor, setup["curve"], jnp.asarray(setup["lognorm_vol"])
        ))
        # cap - floor = sum_i notional * tau_i * DF_i * (F_i - K)
        from valax.dates.daycounts import year_fraction
        fixing_dates = setup["valax_cap"].fixing_dates
        start_dates = setup["valax_cap"].start_dates
        end_dates = setup["valax_cap"].end_dates
        curve = setup["curve"]
        notional = float(setup["valax_cap"].notional)
        K = float(setup["valax_cap"].strike)
        tau = year_fraction(start_dates, end_dates, _ACT365)
        df_s = curve(start_dates)
        df_e = curve(end_dates)
        F = (df_s / df_e - 1.0) / tau
        P = curve(end_dates)
        parity_rhs = float(jnp.sum(notional * tau * P * (F - K)))
        assert abs((v_cap - v_floor) - parity_rhs) / max(abs(parity_rhs), 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# Workstream 1c — HW callable/puttable tree vs QuantLib
# ─────────────────────────────────────────────────────────────────────


class TestHWCallableTreeQL:
    """Hull-White trinomial tree for callable/puttable bonds vs QL."""

    _CASES = [
        (0.05, 0.01, 0.10, 5, 2, 0.05),   # a=1%, sigma=10%, 5y 2-call
        (0.04, 0.02, 0.08, 7, 3, 0.04),   # a=2%, sigma=8%,  7y 3-call
        (0.06, 0.005, 0.12, 3, 1, 0.06),  # a=0.5%, sigma=12%, 3y 1-call
    ]

    @pytest.fixture(params=_CASES, ids=[f"c{i}" for i in range(len(_CASES))])
    def setup(self, request):
        rate, a, sigma, mat_y, n_call, coupon = request.param
        reset_evaluation_date()

        ref = _REF_DATE_ORD

        # Payment dates: annual coupons.
        payment_ords = jnp.array(
            [ref + int(round(k * 365)) for k in range(1, mat_y + 1)],
            dtype=jnp.int32,
        )
        # Call dates: spaced across the bond's life.
        call_step = max(1, mat_y // (n_call + 1))
        call_ords = jnp.array(
            [ref + int(round(k * call_step * 365)) for k in range(1, n_call + 1)],
            dtype=jnp.int32,
        )
        call_prices = jnp.ones(n_call)  # par calls

        valax_cb = CallableBond(
            payment_dates=payment_ords,
            settlement_date=jnp.int32(ref),
            coupon_rate=jnp.asarray(coupon),
            face_value=jnp.asarray(100.0),
            call_dates=call_ords,
            call_prices=call_prices,
            frequency=1,
            day_count=_ACT365,
        )
        valax_pb = PuttableBond(
            payment_dates=payment_ords,
            settlement_date=jnp.int32(ref),
            coupon_rate=jnp.asarray(coupon),
            face_value=jnp.asarray(100.0),
            put_dates=call_ords,
            put_prices=call_prices,
            frequency=1,
            day_count=_ACT365,
        )

        # VALAX HW model on flat curve.
        all_years = [float(k) for k in range(1, mat_y + 2)]
        curve = _flat_valax_curve(rate, all_years)
        hw_model = HullWhiteModel(
            mean_reversion=jnp.asarray(a),
            volatility=jnp.asarray(sigma),
            initial_curve=curve,
        )

        # ── QuantLib side ─────────────────────────────────────────────
        ql_disc = ql_flat_curve(rate)
        ql_hw = ql.HullWhite(ql_disc, float(a), float(sigma))

        maturity_ql = DEFAULT_QL_DATE + ql.Period(int(round(mat_y * 365)), ql.Days)
        schedule_ql = ql.Schedule(
            DEFAULT_QL_DATE,
            maturity_ql,
            ql.Period(ql.Annual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Forward, False,
        )
        ql_call_dates = [
            DEFAULT_QL_DATE + ql.Period(int(round(k * call_step * 365)), ql.Days)
            for k in range(1, n_call + 1)
        ]
        call_schedule = ql.CallabilitySchedule()
        for d in ql_call_dates:
            call_schedule.append(
                ql.Callability(
                    ql.BondPrice(100.0, ql.BondPrice.Clean),
                    ql.Callability.Call,
                    d,
                )
            )
        put_schedule = ql.CallabilitySchedule()
        for d in ql_call_dates:
            put_schedule.append(
                ql.Callability(
                    ql.BondPrice(100.0, ql.BondPrice.Clean),
                    ql.Callability.Put,
                    d,
                )
            )

        # QuantLib >= 1.35 requires paymentConvention / redemption / issueDate
        # positionally before the callability schedule.
        ql_callable = ql.CallableFixedRateBond(
            0, 100.0, schedule_ql, [coupon], ql.Actual365Fixed(),
            ql.Unadjusted, 100.0, DEFAULT_QL_DATE, call_schedule,
        )
        ql_puttable = ql.CallableFixedRateBond(
            0, 100.0, schedule_ql, [coupon], ql.Actual365Fixed(),
            ql.Unadjusted, 100.0, DEFAULT_QL_DATE, put_schedule,
        )

        return {
            "valax_cb": valax_cb,
            "valax_pb": valax_pb,
            "hw_model": hw_model,
            "ql_callable": ql_callable,
            "ql_puttable": ql_puttable,
            "ql_hw": ql_hw,
            "ql_disc": ql_disc,
        }

    def test_callable_bond_vs_ql(self, setup):
        v = float(callable_bond_price(setup["valax_cb"], setup["hw_model"], n_steps=200))
        engine = ql.TreeCallableFixedRateBondEngine(setup["ql_hw"], 200)
        setup["ql_callable"].setPricingEngine(engine)
        q = setup["ql_callable"].dirtyPrice()
        rel = abs(v - q) / max(abs(q), 1e-8)
        assert rel < 5e-3, (
            f"Callable bond: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.4f}"
        )

    def test_puttable_bond_vs_ql(self, setup):
        v = float(puttable_bond_price(setup["valax_pb"], setup["hw_model"], n_steps=200))
        engine = ql.TreeCallableFixedRateBondEngine(setup["ql_hw"], 200)
        setup["ql_puttable"].setPricingEngine(engine)
        q = setup["ql_puttable"].dirtyPrice()
        rel = abs(v - q) / max(abs(q), 1e-8)
        assert rel < 5e-3, (
            f"Puttable bond: VALAX={v:.4f}  QL={q:.4f}  rel={rel:.4f}"
        )

    def test_callable_lt_straight(self, setup):
        """Callable bond price must be <= straight (non-callable) bond price."""
        from valax.pricing.analytic.bonds import fixed_rate_bond_price
        from valax.instruments.bonds import FixedRateBond as FRB
        cb = setup["valax_cb"]
        straight = FRB(
            payment_dates=cb.payment_dates,
            settlement_date=cb.settlement_date,
            coupon_rate=cb.coupon_rate,
            face_value=cb.face_value,
            frequency=1,
            day_count=_ACT365,
        )
        p_call = float(callable_bond_price(cb, setup["hw_model"], n_steps=200))
        p_straight = float(fixed_rate_bond_price(straight, setup["hw_model"].initial_curve))
        assert p_call <= p_straight + 0.1, (
            f"Callable ({p_call:.4f}) should be <= straight ({p_straight:.4f})"
        )

    def test_puttable_gt_straight(self, setup):
        """Puttable bond price must be >= straight (non-puttable) bond price."""
        from valax.pricing.analytic.bonds import fixed_rate_bond_price
        from valax.instruments.bonds import FixedRateBond as FRB
        pb = setup["valax_pb"]
        straight = FRB(
            payment_dates=pb.payment_dates,
            settlement_date=pb.settlement_date,
            coupon_rate=pb.coupon_rate,
            face_value=pb.face_value,
            frequency=1,
            day_count=_ACT365,
        )
        p_put = float(puttable_bond_price(pb, setup["hw_model"], n_steps=200))
        p_straight = float(fixed_rate_bond_price(straight, setup["hw_model"].initial_curve))
        assert p_put >= p_straight - 0.1, (
            f"Puttable ({p_put:.4f}) should be >= straight ({p_straight:.4f})"
        )
