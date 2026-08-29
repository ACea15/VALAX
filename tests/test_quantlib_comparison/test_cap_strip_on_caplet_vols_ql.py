"""Stage 3 — chain validation: caplet vols → SABR surface → cap strip.

Stage 3 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

Chain tested:

  per-expiry caplet vol smiles → SABR fit per expiry
  (:func:`valax.surfaces.build_sabr_caplet_surface`) → cap priced as a strip
  of caplets, each reading its own vol from the surface
  (:func:`valax.pricing.analytic.cap_price`) → agreement with a QuantLib cap
  priced off an equivalent stripped optionlet vol structure.

**Design rule (mirrors ``test_exotics_on_sabr_surface_ql.py``).** The caplet
vol surface is built *once* on the VALAX side and its node values are *adopted*
into a QuantLib ``StrippedOptionletAdapter`` (via
:func:`ql_optionlet_vol_surface`).  Both engines then price the same cap off
this single shared surface, so the test isolates "cap-strip pricer reading a
caplet surface" from "calibrator producing a surface".  Because the cap's
strike and caplet expiries coincide with surface nodes — and both engines use
the same flat Act/365 curve — the agreement is to floating-point precision.

The two historical non-blockers are now resolved:

1. ``build_sabr_caplet_surface`` composes the SABR-per-expiry pipeline into a
   single VALAX convenience (used by ``shared_caplet_surface`` below).
2. The QuantLib ``OptionletStripper1`` fixture lives in ``_ql_adapters`` and is
   exercised by :class:`TestOptionletStripper1Fixture`.
"""

import jax.numpy as jnp
import pytest
import QuantLib as ql

from valax.dates.daycounts import ymd_to_ordinal
from valax.curves.discount import DiscountCurve
from valax.instruments.rates import Cap
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.pricing.analytic.rates_smile import cap_price
from valax.surfaces.optionlet_surface import build_sabr_caplet_surface

from tests.test_quantlib_comparison._ql_adapters import (
    DEFAULT_QL_DATE,
    reset_evaluation_date,
    ql_optionlet_vol_surface,
    ql_stripped_optionlet_surface,
)


# Integer-day-aligned schedule so VALAX (ordinal) and QL (Act/365 from the
# eval date) see bit-identical expiries — no leap-day drift.
_REF = int(ymd_to_ordinal(2026, 1, 1))
_RATE = 0.03
_EXPIRY_YEARS = [1, 2, 3, 4, 5]
_STRIKES = [0.02, 0.025, 0.030455, 0.035, 0.04]
_FWD = 0.030455   # annual forward of a flat 3% continuously-compounded curve


def _flat_curve() -> DiscountCurve:
    """Flat 3% Act/365 curve with a t=0 anchor and annual pillars to 6y."""
    years = [0.0] + [float(y) for y in range(1, 7)]
    pillars = jnp.array([_REF + int(round(y * 365)) for y in years], dtype=jnp.int32)
    times = (pillars - _REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-_RATE * times),
        reference_date=jnp.int32(_REF),
        day_count="act_365",
    )


@pytest.fixture(scope="module")
def shared_caplet_surface():
    """Build a VALAX caplet surface and adopt its node vols into a QL OVS."""
    reset_evaluation_date()
    curve = _flat_curve()
    expiries = jnp.array([float(y) for y in _EXPIRY_YEARS])
    strikes = jnp.array(_STRIKES)
    forwards = jnp.full((len(_EXPIRY_YEARS),), _FWD)

    # Self-consistent per-expiry smiles from a known SABR truth: calibration
    # recovers the params, so the surface is smooth and reproducible.
    truth = SABRModel(
        alpha=jnp.array(0.012), beta=jnp.array(0.5),
        rho=jnp.array(-0.25), nu=jnp.array(0.35),
    )
    strikes_per_expiry = [strikes for _ in _EXPIRY_YEARS]
    vols_per_expiry = [
        jnp.array([float(sabr_implied_vol(truth, jnp.array(_FWD), k, e)) for k in strikes])
        for e in expiries
    ]
    surface = build_sabr_caplet_surface(
        strikes_per_expiry, vols_per_expiry, forwards, expiries, fixed_beta=0.5,
    )

    # Sample the surface at its nodes and adopt into a QL optionlet structure.
    grid = [
        [float(surface(strikes[j], expiries[i])) for j in range(len(_STRIKES))]
        for i in range(len(_EXPIRY_YEARS))
    ]
    ql_fix = ql_optionlet_vol_surface(_EXPIRY_YEARS, _STRIKES, grid, _RATE)

    return {"curve": curve, "surface": surface, "ql": ql_fix}


def _valax_cap(strike: float, is_cap: bool) -> Cap:
    """Annual cap/floor: caplets fixing at 1..5y, paying at 2..6y."""
    fixing = jnp.array([_REF + int(round(y * 365)) for y in [1, 2, 3, 4, 5]], dtype=jnp.int32)
    end = jnp.array([_REF + int(round(y * 365)) for y in [2, 3, 4, 5, 6]], dtype=jnp.int32)
    return Cap(
        fixing_dates=fixing, start_dates=fixing, end_dates=end,
        strike=jnp.array(strike), notional=jnp.array(1_000_000.0),
        is_cap=is_cap, day_count="act_365",
    )


def _ql_cap(strike: float, is_cap: bool, ql_fix) -> ql.CapFloor:
    """QL cap/floor over the same annual 1..6y schedule reading the OVS."""
    today = DEFAULT_QL_DATE
    dates = [today + ql.Period(int(round(y * 365)), ql.Days) for y in [1, 2, 3, 4, 5, 6]]
    leg = ql.IborLeg([1_000_000.0], ql.Schedule(dates), ql_fix["index"],
                     ql.Actual365Fixed(), ql.Unadjusted)
    inst = ql.Cap(leg, [strike]) if is_cap else ql.Floor(leg, [strike])
    inst.setPricingEngine(ql.BlackCapFloorEngine(ql_fix["discount"], ql_fix["handle"]))
    return inst


class TestCapStripOnCapletVols:
    """VALAX cap-strip vs QuantLib, both reading one shared caplet surface."""

    @pytest.mark.parametrize("i", range(len(_EXPIRY_YEARS)))
    @pytest.mark.parametrize("j", range(len(_STRIKES)))
    def test_node_vol_equality(self, shared_caplet_surface, i, j):
        """Pre-flight: the QL adapter reproduces the VALAX surface at nodes.

        A failure here means the vol-grid adoption dropped state, so any
        downstream price disagreement would be ambiguous."""
        surface = shared_caplet_surface["surface"]
        v = float(surface(jnp.array(_STRIKES[j]), jnp.array(float(_EXPIRY_YEARS[i]))))
        q = shared_caplet_surface["ql"]["adapter"].volatility(
            float(_EXPIRY_YEARS[i]), _STRIKES[j], True
        )
        assert v == pytest.approx(q, abs=1e-12), (
            f"node ({_EXPIRY_YEARS[i]}y, K={_STRIKES[j]}): "
            f"VALAX={v:.12f}  QL={q:.12f}"
        )

    @pytest.mark.parametrize("is_cap", [True, False], ids=["cap", "floor"])
    @pytest.mark.parametrize("strike", [0.02, 0.030455, 0.04])
    def test_cap_strip_npv_matches(self, shared_caplet_surface, strike, is_cap):
        """Cap/floor priced as a caplet strip agrees to FP precision."""
        v = float(cap_price(
            _valax_cap(strike, is_cap),
            shared_caplet_surface["curve"],
            shared_caplet_surface["surface"],
        ))
        q = _ql_cap(strike, is_cap, shared_caplet_surface["ql"]).NPV()
        rel = abs(v - q) / max(abs(q), 1.0)
        assert rel < 1e-6, (
            f"strike={strike} is_cap={is_cap}: "
            f"VALAX={v:.6f}  QL={q:.6f}  rel={rel:.2e}"
        )


class TestOptionletStripper1Fixture:
    """Exercise the ``ql.OptionletStripper1`` fixture (former non-blocker #2).

    Strips caplet vols from flat cap (term) vols and checks the defining
    invariant: repricing the market caps with the stripped optionlet surface
    reproduces the flat-cap-vol prices."""

    _CAP_TENORS = [1, 2, 3, 4, 5]
    _CAP_STRIKES = [0.02, 0.03, 0.030455, 0.04]
    _TERM_VOL = 0.20

    @pytest.fixture
    def stripped(self):
        reset_evaluation_date()
        grid = [[self._TERM_VOL for _ in self._CAP_STRIKES] for _ in self._CAP_TENORS]
        return ql_stripped_optionlet_surface(
            self._CAP_TENORS, self._CAP_STRIKES, grid, _RATE, switch_strike=0.030455,
        )

    def test_flat_term_vols_strip_to_flat_caplet_vols(self, stripped):
        """A flat cap-vol surface strips to a flat caplet-vol surface."""
        for t in self._CAP_TENORS:
            v = stripped["adapter"].volatility(float(t), 0.03, True)
            assert v == pytest.approx(self._TERM_VOL, abs=5e-4)

    @pytest.mark.parametrize("n", _CAP_TENORS)
    def test_stripped_surface_reprices_input_caps(self, stripped, n):
        """Cap priced off the stripped optionlet vols == flat-term-vol price."""
        today = DEFAULT_QL_DATE
        dates = [today + ql.Period(int(round(y * 365)), ql.Days) for y in range(0, n + 1)]
        leg = ql.IborLeg([1_000_000.0], ql.Schedule(dates), stripped["index"],
                         ql.Actual365Fixed(), ql.Unadjusted)
        cap = ql.Cap(leg, [0.03])

        cap.setPricingEngine(ql.BlackCapFloorEngine(
            stripped["discount"], ql.QuoteHandle(ql.SimpleQuote(self._TERM_VOL)),
        ))
        flat_price = cap.NPV()
        cap.setPricingEngine(ql.BlackCapFloorEngine(
            stripped["discount"], stripped["handle"],
        ))
        stripped_price = cap.NPV()

        rel = abs(flat_price - stripped_price) / max(abs(flat_price), 1.0)
        assert rel < 1e-5, (
            f"n={n}y: flat={flat_price:.4f}  stripped={stripped_price:.4f}  rel={rel:.2e}"
        )
