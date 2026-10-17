"""One small sample trade per ``price()`` route, for the route contract test.

Every registered library ``(instrument, model)`` pair has an entry in
:data:`SAMPLES`: a builder returning the instrument, the model, and the
market entries its price must be sensitive to (anti-vacuity). All samples
price against one shared :data:`MARKET`, which also carries **decoy**
entries no sample names (a curve, a spot, a dividend, a surface, an FX spot,
an inflation curve) so the contract can check that a route's gradient is
exactly zero outside what the trade declares.

Adding a route means adding its sample here — the contract test fails for a
registered pair without one (see ``docs/guide/tutorial-unified-pricing.md``
§10).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from valax.core.market_state import MarketState
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.curves.inflation import from_zc_rates
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.instruments.bonds import (
    CallableBond,
    FixedRateBond,
    FloatingRateBond,
    PuttableBond,
)
from valax.instruments.fx import FXForward, FXVanillaOption
from valax.instruments.inflation import (
    InflationCapFloor,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)
from valax.instruments.legs import FloatingLeg, MultiLegSwap
from valax.instruments.options import (
    AmericanOption,
    AsianOption,
    DigitalOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    SpreadOption,
    VarianceSwap,
    WorstOfBasketOption,
)
from valax.instruments.rates import (
    BermudanSwaption,
    CMSSpreadSwap,
    Cap,
    Caplet,
    CrossCurrencySwap,
    InterestRateSwap,
    OISSwap,
    Swaption,
)
from valax.models import (
    BlackScholesModel,
    CheyetteModel,
    CheyetteModel2F,
    G2PPModel,
    HestonModel,
    HullWhiteModel,
    ImpliedVolModel,
    LMMModel,
    LocalVolModel,
    MultiAssetGBMModel,
    SLVModel,
)
from valax.models.lmm import ExponentialCorrelation, PiecewiseConstantVol
from valax.pricing.lattice import TreeConfig
from valax.pricing.lattice.binomial import BinomialConfig
from valax.pricing.mc import MCConfig
from valax.pricing.pde.config import PDEConfig, PDEConfig2D
from valax.surfaces import (
    ConstantVol,
    LeverageGrid,
    OptionletVolSurface,
    SVIVolSurface,
    SwaptionCube,
)

jax.config.update("jax_enable_x64", True)

REF = int(ymd_to_ordinal(2026, 1, 1))
DC = "act_365"
F3M, F6M, EUR = "USD.SOFR.3M", "USD.SOFR.6M", "EUR.ESTR.OIS"
CUBE, OPTIONLET, CPI_VOL = "USD.SOFR.CUBE", "USD.SOFR.OPTIONLET", "USD.CPI.VOL"
FX_VOL = "EURUSD.VOL"

#: Decoys: entries no sample names. A route's market gradient must be exactly
#: zero on them.
DECOYS = {
    "curves": "JPY.TONA.OIS",
    "spots": "NKY",
    "dividends": "NKY",
    "surfaces": "NKY.VOL",
    "fx_spots": "GBPUSD",
    "inflation_curves": "EUR.HICP",
}


def _curve(short: float, slope: float) -> DiscountCurve:
    pillars = jnp.array([REF + 365 * k for k in range(21)], dtype=jnp.int32)
    t = (pillars - REF) / 365.0
    return DiscountCurve(pillar_dates=pillars,
                         discount_factors=jnp.exp(-(short + slope * t) * t),
                         reference_date=jnp.int32(REF), day_count=DC)


def _cpi(rate: float, base: float):
    pillars = jnp.array([REF + 365 * k for k in range(1, 11)], dtype=jnp.int32)
    return from_zc_rates(jnp.int32(REF), pillars, jnp.full(10, rate), jnp.asarray(base))


def _svi(vol: float = 0.2) -> SVIVolSurface:
    expiries = jnp.array([0.05, 0.5, 1.0, 2.0])
    return SVIVolSurface(
        expiries=expiries, forwards=jnp.full_like(expiries, 100.0),
        a_vec=vol ** 2 * expiries, b_vec=jnp.zeros_like(expiries),
        rho_vec=jnp.zeros_like(expiries), m_vec=jnp.zeros_like(expiries),
        sigma_vec=jnp.full_like(expiries, 0.1),
    )


def _cube() -> SwaptionCube:
    shape = (4, 3)
    return SwaptionCube(
        expiries=jnp.array([0.5, 1.0, 2.0, 5.0]), tenors=jnp.array([1.0, 5.0, 10.0]),
        forwards=jnp.full(shape, 0.03), alphas=jnp.full(shape, 0.035),
        betas=jnp.full(shape, 0.5), rhos=jnp.full(shape, -0.25),
        nus=jnp.full(shape, 0.4),
    )


def _optionlet() -> OptionletVolSurface:
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    n = expiries.shape[0]
    return OptionletVolSurface(
        expiries=expiries, forwards=jnp.full(n, 0.03),
        alphas=jnp.linspace(0.03, 0.04, n), betas=jnp.full(n, 0.5),
        rhos=jnp.full(n, -0.25), nus=jnp.full(n, 0.4),
    )


MARKET = MarketState(
    curves=CurveGraph(curves={
        "_default_": _curve(0.030, 0.0015),
        F3M: _curve(0.033, 0.0015),
        F6M: _curve(0.035, 0.0015),
        EUR: _curve(0.020, 0.0010),
        DECOYS["curves"]: _curve(0.005, 0.0005),
    }),
    spots={"_default_": jnp.asarray(100.0), "SPX": jnp.asarray(100.0),
           "SX5E": jnp.asarray(50.0), DECOYS["spots"]: jnp.asarray(80.0)},
    dividends={"_default_": jnp.asarray(0.01), DECOYS["dividends"]: jnp.asarray(0.02)},
    surfaces={"_default_": ConstantVol(jnp.asarray(0.2)), CUBE: _cube(),
              OPTIONLET: _optionlet(), CPI_VOL: ConstantVol(jnp.asarray(0.3)),
              FX_VOL: ConstantVol(jnp.asarray(0.1)),
              DECOYS["surfaces"]: ConstantVol(jnp.asarray(0.25))},
    fx_spots={"EURUSD": jnp.asarray(1.1), DECOYS["fx_spots"]: jnp.asarray(1.3)},
    inflation_curves={"USD.CPI": _cpi(0.025, 300.0),
                      DECOYS["inflation_curves"]: _cpi(0.02, 120.0)},
)


@dataclasses.dataclass(frozen=True)
class RouteSample:
    """A sample trade for one pair.

    Attributes:
        instrument: The instrument.
        model: The model, or ``None``.
        nonzero: ``(slot, id)`` market entries the price must move with.
    """

    instrument: Any
    model: Any
    nonzero: tuple[tuple[str, str], ...]


# ── Instruments ──────────────────────────────────────────────────────


def _years(*ys) -> jnp.ndarray:
    return jnp.array([REF + 365 * y for y in ys], dtype=jnp.int32)


def european():
    return EuropeanOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0))


def american():
    return AmericanOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0), is_call=False)


def barrier():
    return EquityBarrierOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0),
                               barrier=jnp.asarray(130.0), is_up=True,
                               is_knock_in=False, smoothing=0.5)


def asian():
    return AsianOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0))


def lookback():
    return LookbackOption(expiry=jnp.asarray(1.0), is_fixed_strike=True,
                          strike=jnp.asarray(100.0))


def variance_swap():
    return VarianceSwap(expiry=jnp.asarray(1.0), strike_var=jnp.asarray(0.04),
                        notional_var=jnp.asarray(1e4))


def digital():
    return DigitalOption(strike=jnp.asarray(100.0), expiry=jnp.asarray(1.0),
                         payout=jnp.asarray(1.0))


def spread():
    return SpreadOption(expiry=jnp.asarray(1.0), strike=jnp.asarray(45.0),
                        notional=jnp.asarray(1.0), underlying_ids=("SPX", "SX5E"))


def worst_of():
    return WorstOfBasketOption(expiry=jnp.asarray(1.0), strike=jnp.asarray(1.0),
                               notional=jnp.asarray(1.0), n_assets=2,
                               underlying_ids=("SPX", "SX5E"))


def swaption():
    return Swaption(expiry_date=_years(2)[0], fixed_dates=_years(3, 4, 5, 6, 7),
                    strike=jnp.asarray(0.03), notional=jnp.asarray(1.0), day_count=DC)


def bermudan():
    return BermudanSwaption(exercise_dates=_years(2, 3, 4), fixed_dates=_years(3, 4, 5, 6),
                            strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
                            day_count=DC)


def caplet():
    return Caplet(fixing_date=_years(1)[0], start_date=_years(1)[0],
                  end_date=jnp.int32(REF + 365 + 182), strike=jnp.asarray(0.03),
                  notional=jnp.asarray(1.0), day_count=DC)


def cap():
    starts = jnp.array([REF + 182 * k for k in range(2, 8)], dtype=jnp.int32)
    return Cap(fixing_dates=starts, start_dates=starts, end_dates=starts + 182,
               strike=jnp.asarray(0.03), notional=jnp.asarray(1.0), day_count=DC)


_BOND = dict(
    payment_dates=generate_schedule(2026, 7, 1, 2031, 1, 1, frequency=2).payment_dates,
    settlement_date=jnp.int32(REF), coupon_rate=jnp.asarray(0.04),
    face_value=jnp.asarray(100.0), frequency=2, day_count=DC,
)
_CALLS = jnp.array([int(ymd_to_ordinal(y, 1, 1)) for y in (2028, 2029, 2030)], dtype=jnp.int32)


def fixed_bond():
    return FixedRateBond(**_BOND)


def callable_bond():
    return CallableBond(call_dates=_CALLS, call_prices=jnp.full(3, 1.0), **_BOND)


def puttable_bond():
    return PuttableBond(put_dates=_CALLS, put_prices=jnp.full(3, 1.0), **_BOND)


def frn():
    pay = generate_schedule(2026, 7, 1, 2031, 1, 1, frequency=2).payment_dates
    fix = jnp.concatenate([jnp.array([REF], dtype=jnp.int32), pay[:-1]])
    return FloatingRateBond(payment_dates=pay, fixing_dates=fix,
                            settlement_date=jnp.int32(REF), spread=jnp.asarray(0.002),
                            face_value=jnp.asarray(100.0), day_count=DC)


def irs():
    return InterestRateSwap(start_date=jnp.int32(REF), fixed_dates=_years(1, 2, 3, 4, 5),
                            fixed_rate=jnp.asarray(0.031), notional=jnp.asarray(1e6),
                            day_count=DC)


def ois():
    return OISSwap(start_date=jnp.int32(REF), fixed_dates=_years(1, 2, 3, 4, 5),
                   float_dates=_years(1, 2, 3, 4, 5), fixed_rate=jnp.asarray(0.031),
                   notional=jnp.asarray(1e6), day_count=DC)


def cms_spread():
    return CMSSpreadSwap(start_date=jnp.int32(REF), payment_dates=_years(1, 2, 3, 4, 5),
                         fixed_rate=jnp.asarray(0.002), notional=jnp.asarray(1e6),
                         cms_tenor_long=10, cms_tenor_short=2, day_count=DC)


def multi_leg():
    starts = jnp.array([REF + 2 + 182 * k for k in range(8)], dtype=jnp.int32)

    def leg(forward_id, **kw):
        return FloatingLeg(accrual_start_dates=starts, accrual_end_dates=starts + 182,
                           payment_dates=starts + 182, notional=jnp.asarray(1e6),
                           compounding="simple", fixing_dates=starts,
                           index_id=forward_id, forward_id=forward_id, day_count=DC, **kw)

    return MultiLegSwap(legs=(leg(F3M, spread=jnp.asarray(0.0)),
                              leg(F6M, spread=jnp.asarray(-0.001), pay=True)))


def xccy():
    return CrossCurrencySwap(
        start_date=jnp.int32(REF + 2),
        payment_dates=jnp.array([REF + 2 + 91 * k for k in range(1, 13)], dtype=jnp.int32),
        maturity_date=jnp.int32(REF + 2 + 91 * 12),
        domestic_notional=jnp.asarray(1.1e6), foreign_notional=jnp.asarray(1e6),
        basis_spread=jnp.asarray(-0.0015), exchange_notional=False,
        currency_pair="EUR/USD", foreign_discount_id=EUR)


def zcis():
    return ZeroCouponInflationSwap(
        effective_date=jnp.int32(REF), maturity_date=_years(5)[0],
        fixed_rate=jnp.asarray(0.022), notional=jnp.asarray(1e6),
        base_cpi=jnp.asarray(300.0), inflation_id="USD.CPI")


def yyis():
    return YearOnYearInflationSwap(
        effective_date=_years(1)[0], payment_dates=_years(2, 3, 4, 5),
        fixed_rate=jnp.asarray(0.022), notional=jnp.asarray(1e6),
        base_cpi=jnp.asarray(300.0), inflation_id="USD.CPI")


def inflation_cap():
    return InflationCapFloor(
        effective_date=_years(1)[0], payment_dates=_years(2, 3, 4, 5),
        strike=jnp.asarray(0.024), notional=jnp.asarray(1e6),
        base_cpi=jnp.asarray(300.0), inflation_id="USD.CPI")


# ── Models ───────────────────────────────────────────────────────────


def bs():
    return BlackScholesModel()   # vol, rate, dividend from the market


def heston():
    return HestonModel(v0=jnp.asarray(0.04), kappa=jnp.asarray(2.0),
                       theta=jnp.asarray(0.04), xi=jnp.asarray(0.3),
                       rho=jnp.asarray(-0.7))


def local_vol():
    return LocalVolModel(surface=_svi())


def slv():
    h = HestonModel(v0=jnp.asarray(0.04), kappa=jnp.asarray(2.0),
                    theta=jnp.asarray(0.04), xi=jnp.asarray(0.3),
                    rho=jnp.asarray(-0.7), rate=jnp.asarray(0.03),
                    dividend=jnp.asarray(0.01))
    lev = LeverageGrid.flat(log_moneyness_grid=jnp.linspace(-0.3, 0.3, 5),
                            time_grid=jnp.linspace(0.1, 1.0, 4), value=1.0)
    return SLVModel.from_heston_and_leverage(h, _svi(), lev)


def multi_asset():
    return MultiAssetGBMModel(vols=jnp.array([0.2, 0.3]), rate=jnp.asarray(0.03),
                              dividends=jnp.zeros(2),
                              correlation=jnp.array([[1.0, 0.5], [0.5, 1.0]]))


def hw():
    return HullWhiteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01))


def g2pp():
    return G2PPModel(mean_reversion_x=jnp.asarray(0.5), mean_reversion_y=jnp.asarray(0.1),
                     volatility_x=jnp.asarray(0.01), volatility_ratio=jnp.asarray(0.8),
                     correlation=jnp.asarray(-0.7))


def cheyette():
    return CheyetteModel(mean_reversion=jnp.asarray(0.05), volatility=jnp.asarray(0.01),
                         skew=jnp.asarray(0.1))


def cheyette2f():
    return CheyetteModel2F(
        mean_reversion_1=jnp.asarray(0.5), mean_reversion_2=jnp.asarray(0.1),
        volatility_1=jnp.asarray(0.012), volatility_ratio=jnp.asarray(0.75),
        skew_1=jnp.asarray(0.0), skew_2=jnp.asarray(0.0), correlation=jnp.asarray(-0.5))


# ── The registry ─────────────────────────────────────────────────────

_C = ("curves", "_default_")
_S = ("spots", "_default_")
_NONE = type(None)


def _equity(instrument, model, extra=()):
    return lambda: RouteSample(instrument(), model(), (_S,) + tuple(extra))


def _rates(instrument, model, extra=()):
    return lambda: RouteSample(instrument(), model(), (_C,) + tuple(extra))


SAMPLES: dict[tuple[type, type], Callable[[], RouteSample]] = {
    # Equity.
    (EuropeanOption, BlackScholesModel): _equity(european, bs, [_C, ("surfaces", "_default_")]),
    (EuropeanOption, HestonModel): _equity(european, heston),
    (EuropeanOption, LocalVolModel): _equity(european, local_vol),
    (EuropeanOption, SLVModel): _equity(european, slv),
    (AmericanOption, BlackScholesModel): _equity(american, bs),
    (DigitalOption, BlackScholesModel): _equity(digital, bs),
    (SpreadOption, MultiAssetGBMModel): lambda: RouteSample(
        spread(), multi_asset(), (("spots", "SPX"), ("spots", "SX5E"))),
    (WorstOfBasketOption, MultiAssetGBMModel): lambda: RouteSample(
        worst_of(), multi_asset(), ()),   # return-space payoff: scale-free in spots
}
for _inst in (barrier, asian, lookback, variance_swap):
    _cls = type(_inst())
    for _model in (bs, heston, local_vol, slv):
        SAMPLES[(_cls, type(_model()))] = _equity(_inst, _model)
# Variance swaps are scale-free in spot too: no spot delta to require. A
# variance swap has no strike, so Black-Scholes cannot sample the surface for
# it: its model carries an explicit vol (rate and dividend from the market).
for _model in (lambda: BlackScholesModel(vol=jnp.asarray(0.2)), heston, local_vol, slv):
    SAMPLES[(VarianceSwap, type(_model()))] = (
        lambda m=_model: RouteSample(variance_swap(), m(), ()))

# Rates, model-free.
SAMPLES.update({
    (FixedRateBond, _NONE): lambda: RouteSample(fixed_bond(), None, (_C,)),
    (FloatingRateBond, _NONE): lambda: RouteSample(frn(), None, (_C,)),
    (InterestRateSwap, _NONE): lambda: RouteSample(irs(), None, (_C,)),
    (OISSwap, _NONE): lambda: RouteSample(ois(), None, (_C,)),
    (MultiLegSwap, _NONE): lambda: RouteSample(
        multi_leg(), None, (_C, ("curves", F3M), ("curves", F6M))),
    (CrossCurrencySwap, _NONE): lambda: RouteSample(
        xccy(), None, (_C, ("curves", EUR), ("fx_spots", "EURUSD"))),
    (ZeroCouponInflationSwap, _NONE): lambda: RouteSample(
        zcis(), None, (_C, ("inflation_curves", "USD.CPI"))),
    (YearOnYearInflationSwap, _NONE): lambda: RouteSample(
        yyis(), None, (_C, ("inflation_curves", "USD.CPI"))),
    # Market implied vol.
    (Caplet, ImpliedVolModel): lambda: RouteSample(
        caplet(), ImpliedVolModel(OPTIONLET), (_C, ("surfaces", OPTIONLET))),
    (Cap, ImpliedVolModel): lambda: RouteSample(
        cap(), ImpliedVolModel(OPTIONLET), (_C, ("surfaces", OPTIONLET))),
    (Swaption, ImpliedVolModel): lambda: RouteSample(
        swaption(), ImpliedVolModel(CUBE), (_C, ("surfaces", CUBE))),
    (InflationCapFloor, ImpliedVolModel): lambda: RouteSample(
        inflation_cap(), ImpliedVolModel(CPI_VOL),
        (_C, ("inflation_curves", "USD.CPI"), ("surfaces", CPI_VOL))),
    # FX: domestic + foreign curves and the keyed FX spot.
    (FXForward, _NONE): lambda: RouteSample(
        FXForward(strike=jnp.asarray(1.08), expiry=jnp.asarray(1.0),
                  notional_foreign=jnp.asarray(1e6), currency_pair="EUR/USD",
                  foreign_discount_id=EUR),
        None, (_C, ("curves", EUR), ("fx_spots", "EURUSD"))),
    (FXVanillaOption, ImpliedVolModel): lambda: RouteSample(
        FXVanillaOption(strike=jnp.asarray(1.12), expiry=jnp.asarray(1.0),
                        notional_foreign=jnp.asarray(1e6), currency_pair="EUR/USD",
                        foreign_discount_id=EUR),
        ImpliedVolModel(FX_VOL),
        (_C, ("curves", EUR), ("fx_spots", "EURUSD"), ("surfaces", FX_VOL))),
})

# Rates under the short-rate models (curve-free; the market binds the curve).
_SHORT_RATE = {HullWhiteModel: hw, G2PPModel: g2pp, CheyetteModel: cheyette,
               CheyetteModel2F: cheyette2f}
_RATES_INSTRUMENTS = {
    Swaption: swaption, BermudanSwaption: bermudan, Caplet: caplet, Cap: cap,
    FixedRateBond: fixed_bond, FloatingRateBond: frn, CallableBond: callable_bond,
    PuttableBond: puttable_bond, CMSSpreadSwap: cms_spread,
}
for _icls, _ibuild in _RATES_INSTRUMENTS.items():
    for _mcls, _mbuild in _SHORT_RATE.items():
        SAMPLES[(_icls, _mcls)] = _rates(_ibuild, _mbuild)

# Rates under the LMM (curve-free). Every period sits on the model's annual
# tenor grid T_k = REF + k years, k = 1..5, and the Bermudan's swap ends at
# its last date — the LMM recipes map the trade onto the grid from the dates.
_LMM_GRID = _years(1, 2, 3, 4, 5)


def lmm():
    return LMMModel(tenor_dates=_LMM_GRID,
                    vol_structure=PiecewiseConstantVol(vols=jnp.full((4, 4), 0.2)),
                    corr_structure=ExponentialCorrelation(beta=jnp.asarray(0.1)))


def lmm_caplet():
    return Caplet(fixing_date=_LMM_GRID[1], start_date=_LMM_GRID[1],
                  end_date=_LMM_GRID[2], strike=jnp.asarray(0.03),
                  notional=jnp.asarray(1.0), day_count=DC)


def lmm_cap():
    return Cap(fixing_dates=_LMM_GRID[:-1], start_dates=_LMM_GRID[:-1],
               end_dates=_LMM_GRID[1:], strike=jnp.asarray(0.03),
               notional=jnp.asarray(1.0), day_count=DC)


def lmm_swaption():
    return Swaption(expiry_date=_LMM_GRID[1], fixed_dates=_LMM_GRID[2:],
                    strike=jnp.asarray(0.03), notional=jnp.asarray(1.0), day_count=DC)


def lmm_bermudan():
    return BermudanSwaption(exercise_dates=_LMM_GRID[1:4], fixed_dates=_LMM_GRID[2:],
                            strike=jnp.asarray(0.03), notional=jnp.asarray(1.0),
                            day_count=DC)


for _icls, _ibuild in {Caplet: lmm_caplet, Cap: lmm_cap, Swaption: lmm_swaption,
                       BermudanSwaption: lmm_bermudan}.items():
    SAMPLES[(_icls, LMMModel)] = _rates(_ibuild, lmm)

#: Registered routes ``price()`` cannot run yet, named so they are never
#: sampled silently. Empty: the LMM routes, the last ones, landed with D18.
KNOWN_UNPRICEABLE: set[tuple[type, type]] = set()


def cheap(config: Any) -> Any:
    """A small version of a method's default config, for a fast contract run."""
    sizes = {
        MCConfig: dict(n_paths=64, n_steps=8),
        PDEConfig: dict(n_spot=40, n_time=16),
        PDEConfig2D: dict(n_x=16, n_y=12, n_time=8),
        BinomialConfig: dict(n_steps=24),
        TreeConfig: dict(n_steps=24),
    }
    for cls, fields in sizes.items():
        if isinstance(config, cls):
            return dataclasses.replace(config, **fields)
    return config
