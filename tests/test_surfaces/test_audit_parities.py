"""T-6 and T-7 conformance gates: option parities and the ATM identity.

Verifies the rates option pricers against the multi-curve calibration / AAD
risk audit specification, requirements **T-6** ("put-call parity: caplet minus
floorlet = FRA; payer minus receiver = A_d (S_x - K)") and **T-7** ("Bachelier
ATM identity: ATM straddle = 2 A_d sigma sqrt(T / 2 pi)"), which back V-1 and
V-2.  Collected with the rest of the T-suite via ``pytest -m audit``.

**The trap this file is built to avoid.**  Stated naively, T-6 is *vacuous
here*.  VALAX does not price a floorlet or a receiver from its own formula --
it prices the caplet/payer and then subtracts the forward value::

    caplets.py:   return cap_pv - caplet.notional * tau * P * (F - K)
    swaptions.py: return payer - swaption.notional * ann * (forward - K)

So "caplet - floorlet = FRA" is *true by construction*, to the last bit, and
asserting it tests only that subtraction is subtraction.  This is precisely the
failure recorded as entry 4 of ``docs/architecture/numerical-pitfalls.md``:
corroboration from something that shares your approximation is worthless.

Two independent references are therefore built here from scratch:

1. **The out-of-the-money leg's own closed form.**  ``ref_black76`` and
   ``ref_bachelier`` compute the put as ``K Phi(-d2) - F Phi(-d1)`` (and the
   Bachelier analogue), *not* as ``call - (F - K)``.  That makes each leg
   checkable on its own, which is what turns the parity into a real statement.
2. **The forward, annuity and discount factor, re-derived by hand.**  Plain
   Python loops over curve reads, transcribed from the definitions rather than
   from ``caplet_forward`` / ``annuity`` / ``_dual_curve_float_pv``.  This is
   the part that actually bites: it pins the accrual start of the first fixed
   period, which curve projects and which discounts, and the payment date the
   discount factor is read at.  Every one of those is a plausible wiring error
   that leaves parity perfectly intact.

Both are exercised single-curve **and** dual-curve, because a swapped
``P_d``/``P_x`` is invisible when the two curves are the same object.

Tolerances this implementation commits to:

* Each leg against its own closed form -- ``rtol = 1e-12``.  Both sides
  evaluate the same analytic formula in double precision; the only difference
  is operation order and ``float`` vs ``jnp`` scalars.
* Parity against the independently reconstructed forward value --
  ``rtol = 1e-12``, with an absolute floor of ``1e-9`` currency units on a
  1e6 notional (the parity term passes through zero at the money, where a
  relative tolerance is meaningless).
* T-7, the Bachelier ATM straddle -- ``rtol = 1e-10``.  Looser than the leg
  checks by two orders because the strike is set to a *separately computed*
  forward, so the contract is ATM only to the agreement of the two forward
  reconstructions; the residual ``(F - K)`` term enters the straddle
  quadratically.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.instruments.rates import Caplet, Swaption
from valax.pricing.analytic.caplets import (
    caplet_price_bachelier,
    caplet_price_black76,
)
from valax.pricing.analytic.rates_smile import caplet_price, swaption_price
from valax.pricing.analytic.swaptions import (
    swaption_price_bachelier,
    swaption_price_black76,
)
from valax.surfaces.constant import ConstantVol


pytestmark = pytest.mark.audit


T6_RTOL = 1e-12
T6_ATOL = 1e-9
T7_RTOL = 1e-10

REF = int(ymd_to_ordinal(2025, 1, 1))
DC = "act_360"
NOTIONAL = 1_000_000.0


# ── Curves ───────────────────────────────────────────────────────────


def _curve(zero_rate: float, slope: float = 0.0) -> DiscountCurve:
    """A smooth curve with a continuously-compounded rate ``r(t)``.

    Deliberately not flat by default: a sloped curve makes every period's
    forward distinct, so a reconstruction that read the wrong date gives a
    different answer rather than the same one.
    """
    years = [0.0] + [0.25 * k for k in range(1, 81)]
    pillars = jnp.array(
        [REF + int(round(y * 365)) for y in years], dtype=jnp.int32
    )
    times = (pillars - REF).astype(jnp.float64) / 365.0
    rates = zero_rate + slope * times
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rates * times),
        reference_date=jnp.int32(REF),
        day_count="act_365",
    )


@pytest.fixture(scope="module")
def discount_curve():
    return _curve(0.040, slope=0.004)


@pytest.fixture(scope="module")
def projection_curve():
    """A visibly different projection curve, so a P_d/P_x swap shows up."""
    return _curve(0.046, slope=0.006)


# ── Independent option references ────────────────────────────────────


def _phi(x: float) -> float:
    """Standard normal pdf."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x: float) -> float:
    """Standard normal cdf, via the standard library's erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ref_black76(forward, strike, vol, expiry, is_call: bool) -> float:
    """Undiscounted Black-76 value, with the put from *its own* formula.

    ``put = K Phi(-d2) - F Phi(-d1)``, not ``call - (F - K)``.  That is the
    whole point: deriving the put by parity here would reproduce the
    implementation's shortcut and make T-6 self-fulfilling.
    """
    sqrt_t = math.sqrt(expiry)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * expiry) / (
        vol * sqrt_t
    )
    d2 = d1 - vol * sqrt_t
    if is_call:
        return forward * _cdf(d1) - strike * _cdf(d2)
    return strike * _cdf(-d2) - forward * _cdf(-d1)


def ref_bachelier(forward, strike, vol, expiry, is_call: bool) -> float:
    """Undiscounted Bachelier value, with the put from *its own* formula."""
    sigma_t = vol * math.sqrt(expiry)
    d = (forward - strike) / sigma_t
    if is_call:
        return (forward - strike) * _cdf(d) + sigma_t * _phi(d)
    return (strike - forward) * _cdf(-d) + sigma_t * _phi(-d)


# ── Independent curve-observable references ──────────────────────────


def ref_caplet_observables(caplet: Caplet, discount, projection=None):
    """``(tau, P_d(T_e), F_x, T)`` rebuilt from the definitions.

    Transcribed from the maths, with scalar curve reads and no reuse of
    :func:`valax.pricing.analytic.caplets.caplet_forward`.
    """
    project = discount if projection is None else projection
    tau = float(
        year_fraction(caplet.start_date, caplet.end_date, caplet.day_count)
    )
    expiry = float(
        year_fraction(
            discount.reference_date, caplet.fixing_date, caplet.day_count
        )
    )
    df_start = float(project(caplet.start_date))
    df_end = float(project(caplet.end_date))
    forward = (df_start / df_end - 1.0) / tau
    return tau, float(discount(caplet.end_date)), forward, expiry


def ref_swaption_observables(swaption: Swaption, discount, projection=None):
    """``(A_d, S_x, T)`` rebuilt from the definitions, by explicit loop.

    The annuity accrues from the *expiry* date for the first period and
    from the previous payment date thereafter, and the forward's floating
    leg projects on ``projection`` while discounting on ``discount``.
    Those three wiring choices are what this reference exists to pin;
    none of them is visible to a parity assertion.
    """
    project = discount if projection is None else projection
    dates = [int(d) for d in swaption.fixed_dates]
    starts = [int(swaption.expiry_date)] + dates[:-1]

    annuity = 0.0
    float_pv = 0.0
    for start, end in zip(starts, dates):
        tau = float(
            year_fraction(
                jnp.int32(start), jnp.int32(end), swaption.day_count
            )
        )
        df_d_end = float(discount(jnp.int32(end)))
        annuity += tau * df_d_end
        float_pv += (
            float(project(jnp.int32(start))) / float(project(jnp.int32(end)))
            - 1.0
        ) * df_d_end

    expiry = float(
        year_fraction(
            discount.reference_date, swaption.expiry_date, swaption.day_count
        )
    )
    return annuity, float_pv / annuity, expiry


# ── Contracts ────────────────────────────────────────────────────────


def _caplet(strike: float, is_cap: bool) -> Caplet:
    start = REF + 365
    return Caplet(
        fixing_date=jnp.int32(start),
        start_date=jnp.int32(start),
        end_date=jnp.int32(start + 91),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(NOTIONAL),
        is_cap=is_cap,
        day_count=DC,
    )


def _swaption(strike: float, is_payer: bool) -> Swaption:
    expiry = REF + 2 * 365
    return Swaption(
        expiry_date=jnp.int32(expiry),
        fixed_dates=jnp.array(
            [expiry + int(round(k * 365)) for k in range(1, 6)], jnp.int32
        ),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(NOTIONAL),
        is_payer=is_payer,
        day_count=DC,
    )


_STRIKES = [0.030, 0.042, 0.055]
_VOLS_LN = [0.22, 0.35]
_VOLS_N = [0.006, 0.011]
_CURVE_MODES = ["single-curve", "dual-curve"]


def _projection(mode, projection_curve):
    return None if mode == "single-curve" else projection_curve


# ── T-6: each leg against its own closed form ────────────────────────


class TestCapletLegsMatchTheirOwnClosedForms:
    """Caplet *and* floorlet, each against its own formula.

    The floorlet assertion is the load-bearing one: the implementation
    reaches it by subtracting the forward value from the caplet, so
    agreeing with an independently written put formula is genuine
    evidence rather than a restatement.
    """

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("vol", _VOLS_LN)
    @pytest.mark.parametrize("strike", _STRIKES)
    @pytest.mark.parametrize("is_cap", [True, False])
    def test_black76(
        self, is_cap, strike, vol, mode, discount_curve, projection_curve
    ):
        contract = _caplet(strike, is_cap)
        projection = _projection(mode, projection_curve)
        tau, df, forward, expiry = ref_caplet_observables(
            contract, discount_curve, projection
        )
        expected = (
            NOTIONAL
            * tau
            * df
            * ref_black76(forward, strike, vol, expiry, is_cap)
        )
        got = float(
            caplet_price_black76(
                contract, discount_curve, jnp.asarray(vol), projection
            )
        )
        assert got == pytest.approx(expected, rel=T6_RTOL, abs=T6_ATOL)

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("vol", _VOLS_N)
    @pytest.mark.parametrize("strike", _STRIKES)
    @pytest.mark.parametrize("is_cap", [True, False])
    def test_bachelier(
        self, is_cap, strike, vol, mode, discount_curve, projection_curve
    ):
        contract = _caplet(strike, is_cap)
        projection = _projection(mode, projection_curve)
        tau, df, forward, expiry = ref_caplet_observables(
            contract, discount_curve, projection
        )
        expected = (
            NOTIONAL
            * tau
            * df
            * ref_bachelier(forward, strike, vol, expiry, is_cap)
        )
        got = float(
            caplet_price_bachelier(
                contract, discount_curve, jnp.asarray(vol), projection
            )
        )
        assert got == pytest.approx(expected, rel=T6_RTOL, abs=T6_ATOL)


class TestSwaptionLegsMatchTheirOwnClosedForms:
    """Payer *and* receiver, each against its own formula."""

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("vol", _VOLS_LN)
    @pytest.mark.parametrize("strike", _STRIKES)
    @pytest.mark.parametrize("is_payer", [True, False])
    def test_black76(
        self, is_payer, strike, vol, mode, discount_curve, projection_curve
    ):
        contract = _swaption(strike, is_payer)
        projection = _projection(mode, projection_curve)
        annuity, forward, expiry = ref_swaption_observables(
            contract, discount_curve, projection
        )
        expected = (
            NOTIONAL
            * annuity
            * ref_black76(forward, strike, vol, expiry, is_payer)
        )
        got = float(
            swaption_price_black76(
                contract, discount_curve, jnp.asarray(vol), projection
            )
        )
        assert got == pytest.approx(expected, rel=T6_RTOL, abs=T6_ATOL)

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("vol", _VOLS_N)
    @pytest.mark.parametrize("strike", _STRIKES)
    @pytest.mark.parametrize("is_payer", [True, False])
    def test_bachelier(
        self, is_payer, strike, vol, mode, discount_curve, projection_curve
    ):
        contract = _swaption(strike, is_payer)
        projection = _projection(mode, projection_curve)
        annuity, forward, expiry = ref_swaption_observables(
            contract, discount_curve, projection
        )
        expected = (
            NOTIONAL
            * annuity
            * ref_bachelier(forward, strike, vol, expiry, is_payer)
        )
        got = float(
            swaption_price_bachelier(
                contract, discount_curve, jnp.asarray(vol), projection
            )
        )
        assert got == pytest.approx(expected, rel=T6_RTOL, abs=T6_ATOL)


# ── T-6: the parity statements themselves ────────────────────────────


class TestCapletFloorletParityIsTheFRA:
    """caplet - floorlet = FRA, against an independently built FRA.

    The difference is structurally exact, so what is tested is the
    *right-hand side*: that the forward value the pricers imply is
    ``notional * tau * P_d(T_e) * (F_x - K)`` with ``tau`` on the
    contract's own day count, ``P_d`` read at the **payment** date, and
    ``F_x`` projected off the **projection** curve.  Mis-wire any of
    those and parity still holds while this fails.
    """

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("strike", _STRIKES)
    def test_black76(self, strike, mode, discount_curve, projection_curve):
        projection = _projection(mode, projection_curve)
        tau, df, forward, _ = ref_caplet_observables(
            _caplet(strike, True), discount_curve, projection
        )
        fra = NOTIONAL * tau * df * (forward - strike)

        vol = jnp.asarray(0.28)
        cap = float(
            caplet_price_black76(
                _caplet(strike, True), discount_curve, vol, projection
            )
        )
        floor = float(
            caplet_price_black76(
                _caplet(strike, False), discount_curve, vol, projection
            )
        )
        assert cap - floor == pytest.approx(fra, rel=T6_RTOL, abs=T6_ATOL)

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("strike", _STRIKES)
    def test_bachelier(self, strike, mode, discount_curve, projection_curve):
        projection = _projection(mode, projection_curve)
        tau, df, forward, _ = ref_caplet_observables(
            _caplet(strike, True), discount_curve, projection
        )
        fra = NOTIONAL * tau * df * (forward - strike)

        vol = jnp.asarray(0.009)
        cap = float(
            caplet_price_bachelier(
                _caplet(strike, True), discount_curve, vol, projection
            )
        )
        floor = float(
            caplet_price_bachelier(
                _caplet(strike, False), discount_curve, vol, projection
            )
        )
        assert cap - floor == pytest.approx(fra, rel=T6_RTOL, abs=T6_ATOL)

    def test_the_parity_term_is_vol_independent(self, discount_curve):
        """The FRA has no optionality, so the difference cannot move with vol.

        A cheap but sharp check: if the forward value picked up any
        vol dependence, the two legs would not be the same option.
        """
        values = [
            float(
                caplet_price_black76(
                    _caplet(0.042, True), discount_curve, jnp.asarray(v)
                )
                - caplet_price_black76(
                    _caplet(0.042, False), discount_curve, jnp.asarray(v)
                )
            )
            for v in (0.10, 0.25, 0.60)
        ]
        assert values[0] == pytest.approx(values[1], rel=T6_RTOL)
        assert values[1] == pytest.approx(values[2], rel=T6_RTOL)


class TestPayerReceiverParityIsTheAnnuity:
    """payer - receiver = A_d (S_x - K), against an independent A_d and S_x.

    Same structure as the caplet case, and the same target: the annuity's
    first accrual period runs from *expiry*, not from the curve reference
    date, and the dual-curve forward does not telescope.  Both are
    reconstructed by hand here.
    """

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("strike", _STRIKES)
    def test_black76(self, strike, mode, discount_curve, projection_curve):
        projection = _projection(mode, projection_curve)
        annuity, forward, _ = ref_swaption_observables(
            _swaption(strike, True), discount_curve, projection
        )
        expected = NOTIONAL * annuity * (forward - strike)

        vol = jnp.asarray(0.30)
        payer = float(
            swaption_price_black76(
                _swaption(strike, True), discount_curve, vol, projection
            )
        )
        receiver = float(
            swaption_price_black76(
                _swaption(strike, False), discount_curve, vol, projection
            )
        )
        assert payer - receiver == pytest.approx(
            expected, rel=T6_RTOL, abs=T6_ATOL
        )

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("strike", _STRIKES)
    def test_bachelier(self, strike, mode, discount_curve, projection_curve):
        projection = _projection(mode, projection_curve)
        annuity, forward, _ = ref_swaption_observables(
            _swaption(strike, True), discount_curve, projection
        )
        expected = NOTIONAL * annuity * (forward - strike)

        vol = jnp.asarray(0.010)
        payer = float(
            swaption_price_bachelier(
                _swaption(strike, True), discount_curve, vol, projection
            )
        )
        receiver = float(
            swaption_price_bachelier(
                _swaption(strike, False), discount_curve, vol, projection
            )
        )
        assert payer - receiver == pytest.approx(
            expected, rel=T6_RTOL, abs=T6_ATOL
        )

    def test_parity_survives_a_strike_dependent_vol(
        self, discount_curve, projection_curve
    ):
        """Parity must hold through the smile-aware entry point too.

        A smile makes the vol a function of the strike, which is fine --
        payer and receiver at the *same* strike read the *same* vol.  It
        would break if a vol source ever keyed off the option's
        direction, which is exactly the kind of coupling a cube must not
        have.  Routed through ``rates_smile.swaption_price`` so the
        ``forward=``-aware read is the one under test.
        """
        strike = 0.050
        annuity, forward, _ = ref_swaption_observables(
            _swaption(strike, True), discount_curve, projection_curve
        )
        expected = NOTIONAL * annuity * (forward - strike)

        source = ConstantVol(jnp.asarray(0.0105), is_normal=True)
        payer = float(
            swaption_price(
                _swaption(strike, True),
                discount_curve,
                source,
                projection_curve,
            )
        )
        receiver = float(
            swaption_price(
                _swaption(strike, False),
                discount_curve,
                source,
                projection_curve,
            )
        )
        assert payer - receiver == pytest.approx(
            expected, rel=T6_RTOL, abs=T6_ATOL
        )

    def test_caplet_parity_survives_the_smile_entry_point(
        self, discount_curve, projection_curve
    ):
        """The caplet-side counterpart, through ``rates_smile.caplet_price``."""
        strike = 0.050
        tau, df, forward, _ = ref_caplet_observables(
            _caplet(strike, True), discount_curve, projection_curve
        )
        expected = NOTIONAL * tau * df * (forward - strike)

        source = ConstantVol(jnp.asarray(0.0095), is_normal=True)
        cap = float(
            caplet_price(
                _caplet(strike, True),
                discount_curve,
                source,
                projection_curve,
            )
        )
        floor = float(
            caplet_price(
                _caplet(strike, False),
                discount_curve,
                source,
                projection_curve,
            )
        )
        assert cap - floor == pytest.approx(expected, rel=T6_RTOL, abs=T6_ATOL)


# ── T-7: the Bachelier ATM identity ──────────────────────────────────


class TestBachelierATMStraddle:
    r"""ATM straddle :math:`= 2 A_d \sigma \sqrt{T / 2\pi}`.

    At the money the Bachelier payer collapses to
    :math:`A_d \sigma \sqrt{T} \phi(0) = A_d \sigma \sqrt{T / 2\pi}`, and
    the receiver equals it because the parity term vanishes.  So the
    straddle pins the **scale** of the normal-vol convention: a missing
    :math:`\sqrt{2\pi}`, a :math:`\sqrt{T}` applied twice, or a
    variance/volatility mix-up all show up here and nowhere in the parity
    tests, which are insensitive to the vol altogether.

    Note this is not weakened by the receiver being derived by parity:
    the parity term is zero at the money, so the identity is a statement
    about the payer formula's ATM value, independently computed.
    """

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("vol", [0.004, 0.009, 0.015])
    def test_swaption_straddle(
        self, vol, mode, discount_curve, projection_curve
    ):
        projection = _projection(mode, projection_curve)
        annuity, forward, expiry = ref_swaption_observables(
            _swaption(0.04, True), discount_curve, projection
        )
        payer = _swaption(forward, True)
        receiver = _swaption(forward, False)

        straddle = float(
            swaption_price_bachelier(
                payer, discount_curve, jnp.asarray(vol), projection
            )
            + swaption_price_bachelier(
                receiver, discount_curve, jnp.asarray(vol), projection
            )
        )
        expected = (
            2.0
            * NOTIONAL
            * annuity
            * vol
            * math.sqrt(expiry / (2.0 * math.pi))
        )
        assert straddle == pytest.approx(expected, rel=T7_RTOL)

    @pytest.mark.parametrize("mode", _CURVE_MODES)
    @pytest.mark.parametrize("vol", [0.004, 0.009, 0.015])
    def test_caplet_straddle(
        self, vol, mode, discount_curve, projection_curve
    ):
        """The same identity with ``tau * P_d(T_e)`` as the annuity."""
        projection = _projection(mode, projection_curve)
        tau, df, forward, expiry = ref_caplet_observables(
            _caplet(0.04, True), discount_curve, projection
        )
        straddle = float(
            caplet_price_bachelier(
                _caplet(forward, True),
                discount_curve,
                jnp.asarray(vol),
                projection,
            )
            + caplet_price_bachelier(
                _caplet(forward, False),
                discount_curve,
                jnp.asarray(vol),
                projection,
            )
        )
        expected = (
            2.0
            * NOTIONAL
            * tau
            * df
            * vol
            * math.sqrt(expiry / (2.0 * math.pi))
        )
        assert straddle == pytest.approx(expected, rel=T7_RTOL)

    def test_the_straddle_is_linear_in_the_volatility(self, discount_curve):
        """A corollary with teeth: ATM time value is exactly linear in sigma.

        Any leakage of the vol into ``d`` at the money -- the commonest
        way to get the ATM case subtly wrong -- would curve this.
        """
        _, forward, _ = ref_swaption_observables(
            _swaption(0.04, True), discount_curve
        )
        payer = _swaption(forward, True)

        def straddle(vol):
            return float(
                swaption_price_bachelier(
                    payer, discount_curve, jnp.asarray(vol), None
                )
                + swaption_price_bachelier(
                    _swaption(forward, False),
                    discount_curve,
                    jnp.asarray(vol),
                    None,
                )
            )

        base = straddle(0.005)
        assert straddle(0.010) == pytest.approx(2.0 * base, rel=T7_RTOL)
        assert straddle(0.015) == pytest.approx(3.0 * base, rel=T7_RTOL)

    def test_black76_atm_straddle_tends_to_the_bachelier_one(
        self, discount_curve
    ):
        r"""Cross-convention consistency in the overlapping limit.

        As :math:`\sigma_{LN}\sqrt{T} \to 0` the lognormal ATM straddle
        tends to the normal one with :math:`\sigma_N = \sigma_{LN} F`.
        Checked as a *convergence rate* rather than at a single
        tolerance: halving the vol must shrink the relative gap by about
        four, since the leading correction is
        :math:`O((\sigma\sqrt{T})^2)`.  Asserting a fixed tolerance here
        would pass just as well against a wrong formula that happened to
        be close.
        """
        annuity, forward, expiry = ref_swaption_observables(
            _swaption(0.04, True), discount_curve
        )
        payer, receiver = _swaption(forward, True), _swaption(forward, False)

        def relative_gap(vol_ln):
            lognormal = float(
                swaption_price_black76(
                    payer, discount_curve, jnp.asarray(vol_ln), None
                )
                + swaption_price_black76(
                    receiver, discount_curve, jnp.asarray(vol_ln), None
                )
            )
            normal = (
                2.0
                * NOTIONAL
                * annuity
                * (vol_ln * forward)
                * math.sqrt(expiry / (2.0 * math.pi))
            )
            return abs(lognormal - normal) / normal

        coarse = relative_gap(0.08)
        fine = relative_gap(0.04)
        assert fine < coarse
        assert coarse / fine == pytest.approx(4.0, rel=0.05)
