"""Example 20 — FX, inflation, cross-currency and multi-leg trades on one market.

Every trade here prices through the one ``price(instrument, model, market)``
contract against a single multi-currency ``MarketState``, and every risk number
comes from ``valax.greeks.market_risk`` — one backward pass keyed by the market
entries the trade *declares* through its static selectors. Where it helps, the
example drops to the kernel layer (``valax.pricing.analytic.{fx,inflation,legs,
rates_exotics}``) to show ``price()`` is the same function underneath.

  1. Market: USD SOFR OIS (domestic discounting), USD 3M and 6M projection
     curves, EUR ESTR OIS (foreign), the EURUSD spot, a USD CPI curve, flat
     vols for EURUSD and USD CPI — plus decoy entries no trade names.
  2. FX: an ``FXForward`` against covered interest parity by hand; an
     ``FXVanillaOption`` under ``ImpliedVolModel("EURUSD.VOL")``, put–call
     parity against the forward, and ``price()`` = ``garman_kohlhagen_price``.
  3. Cross-currency and multi-leg: a ``CrossCurrencySwap`` with and without
     notional exchange (the exchanged swap collapses to its spread annuity;
     the unexchanged one carries FX and EUR-curve risk; the par basis from the
     kernel reprices to zero), and a 3M-vs-6M ``MultiLegSwap`` basis swap
     (= Σ of its legs, = a hand computation, flips sign with pay/receive,
     par spread from affinity).
  4. Inflation: ZCIS at the breakeven prices to zero; a YoY swap's par rate
     from two prices; an ``InflationCapFloor`` cap − floor = the YoY swap at
     the strike.
  5. ``market_risk``: declared entries per trade, non-zero where expected,
     the full market gradient exactly zero on decoys and everything not
     declared; FX delta and a CPI-pillar sensitivity against central FD.
  6. JIT / batching: ``eqx.filter_jit`` of one trade and ``eqx.filter_vmap``
     over a strike ladder of FX options, equal to the eager loop.

Run::

    python examples/20_unified_fx_inflation_xccy.py
"""

# %% Imports and setup
from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import valax
from valax.core import MarketState, available_methods, declared_entries, price
from valax.curves.discount import DiscountCurve, zero_rate
from valax.curves.graph import CurveGraph
from valax.curves.inflation import forward_cpi, from_zc_rates, zc_inflation_rate
from valax.dates.daycounts import year_fraction, ymd_to_ordinal
from valax.greeks import market_risk
from valax.instruments.fx import FXForward, FXVanillaOption
from valax.instruments.inflation import (
    InflationCapFloor,
    YearOnYearInflationSwap,
    ZeroCouponInflationSwap,
)
from valax.instruments.legs import FloatingLeg, MultiLegSwap
from valax.instruments.rates import CrossCurrencySwap
from valax.models import ImpliedVolModel
from valax.pricing.analytic.fx import fx_forward_rate, garman_kohlhagen_price
from valax.pricing.analytic.inflation import zcis_breakeven_rate
from valax.pricing.analytic.legs import leg_pv
from valax.pricing.analytic.rates_exotics import cross_currency_basis_spread
from valax.pricing.analytic.swaptions import annuity
from valax.surfaces import ConstantVol

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
SWAP_YEARS = 3 if FAST else 5          # basis swap and xccy maturity
N_LADDER = 9 if FAST else 41           # FX option strikes in the vmapped ladder
LAST_YOY_YEAR = 2029 if FAST else 2031  # YoY swap / cap schedule end


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


def rel(a, b) -> float:
    """Relative gap |a - b| / max(1, |b|)."""
    return abs(float(a) - float(b)) / max(1.0, abs(float(b)))


REF = jnp.int32(int(ymd_to_ordinal(2026, 1, 1)))
USD, USD3M, USD6M, EUR = "USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M", "EUR.ESTR.OIS"
CPI, FX, FX_VOL, CPI_VOL = "USD.CPI", "EURUSD", "EURUSD.VOL", "USD.CPI.VOL"
DECOYS = {"curves": "JPY.TONA.OIS", "fx_spots": "GBPUSD", "inflation_curves": "EUR.HICP",
          "surfaces": "GBPUSD.VOL", "spots": "SPX", "dividends": "SPX"}

# %% 1. Market
header("1. One multi-currency MarketState (with decoys)")

TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0)


def zero_curve(zeros: tuple[float, ...]) -> DiscountCurve:
    """An act/365 curve with pillars at REF + round(365 t) and the given zero rates."""
    days = jnp.round(jnp.asarray((0.0,) + TENORS) * 365.0).astype(jnp.int32)
    z = jnp.asarray((zeros[0],) + zeros)
    return DiscountCurve(pillar_dates=REF + days, discount_factors=jnp.exp(-z * days / 365.0),
                         reference_date=REF, day_count="act_365")


usd_z = (0.0430, 0.0425, 0.0410, 0.0385, 0.0372, 0.0365, 0.0368, 0.0375, 0.0385)
eur_z = (0.0200, 0.0198, 0.0195, 0.0205, 0.0215, 0.0230, 0.0242, 0.0255, 0.0268)
curves = {
    USD: zero_curve(usd_z),
    USD3M: zero_curve(tuple(z + 0.0010 for z in usd_z)),   # +10 bp tenor basis
    USD6M: zero_curve(tuple(z + 0.0018 for z in usd_z)),   # +18 bp tenor basis
    EUR: zero_curve(eur_z),
    DECOYS["curves"]: zero_curve((0.0050,) * len(TENORS)),
}

# CPI pillars on calendar anniversaries, *including the reference date* — see the
# note in section 4 on why the reference pillar matters.
CPI_YEARS = (0, 1, 2, 3, 5, 7, 10)
cpi_pillars = jnp.stack([ymd_to_ordinal(2026 + k, 1, 1) for k in CPI_YEARS])
CPI_ZC = jnp.array([0.0285, 0.0285, 0.0270, 0.0260, 0.0250, 0.0245, 0.0240])
BASE_CPI = jnp.asarray(320.0)

market = MarketState(
    curves=CurveGraph(curves=curves),
    surfaces={FX_VOL: ConstantVol(jnp.asarray(0.085)), CPI_VOL: ConstantVol(jnp.asarray(0.35)),
              DECOYS["surfaces"]: ConstantVol(jnp.asarray(0.09))},
    spots={DECOYS["spots"]: jnp.asarray(5000.0)},
    dividends={DECOYS["dividends"]: jnp.asarray(0.013)},
    fx_spots={FX: jnp.asarray(1.0850), DECOYS["fx_spots"]: jnp.asarray(1.2700)},
    inflation_curves={
        CPI: from_zc_rates(REF, cpi_pillars, CPI_ZC, BASE_CPI),
        DECOYS["inflation_curves"]: from_zc_rates(REF, cpi_pillars, jnp.full(7, 0.02),
                                                  jnp.asarray(125.0)),
    },
)
S = market.fx_spots[FX]

print(f"  {'curve':<16}{'1Y zero':>10}{'5Y zero':>10}{'10Y zero':>10}")
for cid, c in market.curves.items():
    zs = [float(zero_rate(c, REF + 365 * y)) for y in (1, 5, 10)]
    tag = "   (decoy)" if cid == DECOYS["curves"] else ""
    print(f"  {cid:<16}" + "".join(f"{z:>10.3%}" for z in zs) + tag)
print(f"  fx_spots: " + ", ".join(f"{k}={float(v):.4f}" for k, v in market.fx_spots.items()))
for iid, ic in market.inflation_curves.items():
    zc5 = float(zc_inflation_rate(ic, ymd_to_ordinal(2031, 1, 1)))
    print(f"  inflation {iid:<9} base CPI {float(ic.base_cpi):.1f}   5Y breakeven {zc5:.3%}")
print("  surfaces: " + ", ".join(f"{k}={float(v.vol):.1%}" for k, v in market.surfaces.items()))
print(f"  decoys (named by no trade): {sorted(set(DECOYS.values()))}")

# %% 2. FX
header("2. FX: forward vs covered interest parity, option parity, kernel layer")

N_EUR = jnp.asarray(10e6)
T_FWD = jnp.asarray(1.0)
fwd = FXForward(strike=jnp.asarray(1.1000), expiry=T_FWD, notional_foreign=N_EUR,
                currency_pair="EUR/USD", discount_id=USD, foreign_discount_id=EUR)
pv_fwd = price(fwd, None, market).price

# Covered interest parity by hand: F = S·P_EUR(T)/P_USD(T); PV = N·(S·P_EUR − K·P_USD).
d_T = REF + T_FWD * 365.0                      # act/365 ordinal of the expiry
df_usd, df_eur = market.curves[USD](d_T), market.curves[EUR](d_T)
F_cip = S * df_eur / df_usd
pv_cip = N_EUR * (S * df_eur - fwd.strike * df_usd)
print(f"  1Y EURUSD forward: CIP by hand {float(F_cip):.6f}   "
      f"fx_forward_rate kernel {float(fx_forward_rate(S, zero_rate(market.curves[USD], d_T), zero_rate(market.curves[EUR], d_T), T_FWD)):.6f}")
print(f"  buy 10mm EUR at 1.1000: price() {float(pv_fwd):,.4f} USD   by hand {float(pv_cip):,.4f}")
assert rel(pv_fwd, pv_cip) <= 1e-12
at_fair = eqx.tree_at(lambda f: f.strike, fwd, F_cip)
assert abs(float(price(at_fair, None, market).price)) <= 1e-12 * float(N_EUR)
print("  ✓ the FX forward is covered interest parity off the two keyed curves; zero at F")

fx_model = ImpliedVolModel(FX_VOL)
T_OPT = jnp.asarray(0.75)                      # deliberately off a curve pillar
opt_kw = dict(strike=jnp.asarray(1.1200), expiry=T_OPT, notional_foreign=N_EUR,
              currency_pair="EUR/USD", discount_id=USD, foreign_discount_id=EUR)
call = FXVanillaOption(is_call=True, **opt_kw)
put = FXVanillaOption(is_call=False, **opt_kw)   # is_call is static: a new pytree
pv_call = price(call, fx_model, market).price
pv_put = price(put, fx_model, market).price
fwd_k = FXForward(strike=call.strike, expiry=T_OPT, notional_foreign=N_EUR,
                  currency_pair="EUR/USD", discount_id=USD, foreign_discount_id=EUR)
pv_fwd_k = price(fwd_k, None, market).price
print(f"  9M 1.1200 call {float(pv_call):,.2f}   put {float(pv_put):,.2f}   "
      f"methods {available_methods(call, fx_model)}")
print(f"  call − put {float(pv_call - pv_put):,.6f}   forward at 1.1200 {float(pv_fwd_k):,.6f}")
assert abs(float(pv_call - pv_put - pv_fwd_k)) <= 1e-10 * float(N_EUR * S)
print("  ✓ put–call parity: C − P = the same-strike FX forward, priced by a different route")

# The kernel layer: Garman–Kohlhagen on the zero rates to the option's own expiry.
d_opt = REF + T_OPT * 365.0
r_d = -jnp.log(market.curves[USD](d_opt)) / T_OPT
r_f = -jnp.log(market.curves[EUR](d_opt)) / T_OPT
vol = market.surfaces[FX_VOL](call.strike, T_OPT)
pv_gk = garman_kohlhagen_price(call, S, vol, r_d, r_f)
print(f"  garman_kohlhagen_price(call, S, σ={float(vol):.1%}, r_d={float(r_d):.4%}, "
      f"r_f={float(r_f):.4%}) = {float(pv_gk):,.6f}")
assert rel(pv_call, pv_gk) <= 1e-12
print("  ✓ price() is the GK kernel on the surface vol and each curve's zero rate to expiry")

# %% 3. Cross-currency and multi-leg
header("3. Cross-currency swap and a 3M-vs-6M multi-leg basis swap")

start = REF + 2
n_q = 4 * SWAP_YEARS
pay_q = start + 91 * jnp.arange(1, n_q + 1, dtype=jnp.int32)
N_USD = N_EUR * S
xccy_kw = dict(start_date=start, payment_dates=pay_q, maturity_date=pay_q[-1],
               domestic_notional=N_USD, foreign_notional=N_EUR,
               basis_spread=jnp.asarray(-0.0015), currency_pair="EUR/USD",
               discount_id=USD, foreign_discount_id=EUR)
xccy_ex = CrossCurrencySwap(exchange_notional=True, **xccy_kw)
xccy_no = CrossCurrencySwap(exchange_notional=False, **xccy_kw)
pv_ex = price(xccy_ex, None, market).price
pv_no = price(xccy_no, None, market).price

# With notional exchange each currency's float leg telescopes against its
# exchanges: PV = N_USD · s · A_USD (the spread annuity on the domestic curve).
A_usd = annuity(start, pay_q, market.curves[USD], xccy_ex.day_count)
print(f"  {SWAP_YEARS}Y EUR/USD xccy, recv USD+(−15 bp) / pay EUR, N = 10mm EUR")
print(f"    with notional exchange    PV {float(pv_ex):>14,.2f}   N·s·A_USD {float(N_USD * xccy_ex.basis_spread * A_usd):>14,.2f}")
assert rel(pv_ex, N_USD * xccy_ex.basis_spread * A_usd) <= 1e-9
print(f"    without notional exchange PV {float(pv_no):>14,.2f}")

# Without exchange, the foreign float leg is −S·N_EUR·(P_EUR(start) − P_EUR(end)):
# ∂PV/∂S is exactly that bracket, and the EUR curve matters.
dS_hand = -N_EUR * (market.curves[EUR](start) - market.curves[EUR](pay_q[-1]))
risk_no = market_risk(xccy_no, None, market)
print(f"    ∂PV/∂EURUSD: market_risk {float(risk_no.fx_spots[FX]):,.2f}   by hand {float(dS_hand):,.2f}")
assert rel(risk_no.fx_spots[FX], dS_hand) <= 1e-12
assert float(jnp.max(jnp.abs(risk_no.curves[EUR]))) > 0.0
print("  ✓ exchanged xccy = spread annuity; unexchanged xccy carries EURUSD and EUR-curve risk")

# Unexchanged, the USD float leg (higher rates) outweighs the EUR one by far, so the
# par spread is large and negative; it is a number to reprice at, not a market basis.
s_par = cross_currency_basis_spread(xccy_no, market.curves[USD], market.curves[EUR], S)
at_par = eqx.tree_at(lambda x: x.basis_spread, xccy_no, s_par)
pv_par = price(at_par, None, market).price
print(f"    par basis (kernel cross_currency_basis_spread) {float(s_par) * 1e4:+.2f} bp → "
      f"price() at par {float(pv_par):.2e}")
assert abs(float(pv_par)) <= 1e-14 * float(N_USD)
print("  ✓ the kernel's par basis reprices to zero through price()")

# 3M-vs-6M tenor basis swap as legs: receive 3M + spread quarterly, pay 6M semiannually,
# both discounted on SOFR OIS.
N_BASIS = jnp.asarray(50e6)


def float_leg(period_days: int, n: int, forward_id: str, spread: float, pay: bool) -> FloatingLeg:
    starts = start + period_days * jnp.arange(n, dtype=jnp.int32)
    return FloatingLeg(accrual_start_dates=starts, accrual_end_dates=starts + period_days,
                       payment_dates=starts + period_days, spread=jnp.asarray(spread),
                       notional=N_BASIS, fixing_dates=starts, pay=pay, compounding="simple",
                       day_count="act_360", index_id=forward_id, discount_id=USD,
                       forward_id=forward_id)


leg3 = float_leg(91, n_q, USD3M, 0.0005, pay=False)
leg6 = float_leg(182, 2 * SWAP_YEARS, USD6M, 0.0, pay=True)
basis = MultiLegSwap(legs=(leg3, leg6))
pv_basis = price(basis, None, market).price
legs_kernel = [leg_pv(leg, market.curves) for leg in basis.legs]
print(f"\n  {SWAP_YEARS}Y basis swap, N = 50mm: recv 3M+5 bp {float(legs_kernel[0]):>14,.2f}"
      f"   pay 6M {float(legs_kernel[1]):>14,.2f}")
print(f"    price() {float(pv_basis):,.4f}   Σ leg_pv {float(sum(legs_kernel)):,.4f}")
assert rel(pv_basis, sum(legs_kernel)) <= 1e-13


def hand_leg(leg: FloatingLeg) -> jax.Array:
    """N Σ (P_f(s)/P_f(e) − 1 + s·τ) · P_OIS(pay), with the sign of the leg."""
    fc, dc = market.curves[leg.forward_id], market.curves[USD]
    tau = year_fraction(leg.accrual_start_dates, leg.accrual_end_dates, leg.day_count)
    coupon = fc(leg.accrual_start_dates) / fc(leg.accrual_end_dates) - 1.0 + leg.spread * tau
    pv = leg.notional * jnp.sum(coupon * dc(leg.payment_dates))
    return -pv if leg.pay else pv


assert rel(pv_basis, hand_leg(leg3) + hand_leg(leg6)) <= 1e-10
flipped = MultiLegSwap(legs=(float_leg(91, n_q, USD3M, 0.0005, pay=True),
                             float_leg(182, 2 * SWAP_YEARS, USD6M, 0.0, pay=False)))
assert float(price(flipped, None, market).price) == -float(pv_basis)
print("  ✓ price() = Σ legs (kernel) = hand projection; flipping pay/receive negates it exactly")

# PV is affine in the 3M spread, so the par spread needs no solver.
def basis_pv(spread: jax.Array) -> jax.Array:
    return price(eqx.tree_at(lambda b: b.legs[0].spread, basis, spread), None, market).price


s0 = jnp.asarray(0.0)
basis_par = s0 - basis_pv(s0) / jax.grad(basis_pv)(s0)
print(f"    par 3M spread {float(basis_par) * 1e4:+.3f} bp → PV {float(basis_pv(basis_par)):.2e}")
assert abs(float(basis_pv(basis_par))) <= 1e-14 * float(N_BASIS)
print("  ✓ the par basis from one gradient (PV is affine in the spread) prices to zero")

# %% 4. Inflation
header("4. Inflation: ZCIS breakeven, YoY par rate, cap − floor = YoY swap")

N_INF = jnp.asarray(25e6)
mat5 = ymd_to_ordinal(2031, 1, 1)
zcis = ZeroCouponInflationSwap(effective_date=REF, maturity_date=mat5,
                               fixed_rate=jnp.asarray(0.0200), notional=N_INF,
                               base_cpi=BASE_CPI, discount_id=USD, inflation_id=CPI)
cpi_curve = market.inflation_curves[CPI]
pv_zcis = price(zcis, None, market).price
T5 = year_fraction(REF, mat5, zcis.day_count)
ratio = forward_cpi(cpi_curve, mat5) / BASE_CPI
pv_zcis_hand = N_INF * (ratio - (1.0 + zcis.fixed_rate) ** T5) * market.curves[USD](mat5)
print(f"  5Y ZCIS recv CPI / pay 2.00%: price() {float(pv_zcis):,.4f}   by hand {float(pv_zcis_hand):,.4f}")
assert rel(pv_zcis, pv_zcis_hand) <= 1e-12

# A ZCIS is *not* affine in its fixed rate ((1+K)^T), so take the breakeven from
# the kernel and check it against the curve it was built from.
k_be = zcis_breakeven_rate(zcis, cpi_curve)
pv_be = price(eqx.tree_at(lambda z: z.fixed_rate, zcis, k_be), None, market).price
print(f"  breakeven (zcis_breakeven_rate) {float(k_be):.6%}   input 5Y ZC rate {float(CPI_ZC[4]):.6%}"
      f"   PV at breakeven {float(pv_be):.2e}")
assert abs(float(k_be - CPI_ZC[4])) <= 1e-13
assert abs(float(pv_be)) <= 1e-12 * float(N_INF)
print("  ✓ a ZCIS struck at the breakeven has zero PV; the breakeven is the curve's own ZC rate")

yoy_pay = jnp.stack([ymd_to_ordinal(y, 1, 1) for y in range(2028, LAST_YOY_YEAR + 1)])
yoy_eff = ymd_to_ordinal(2027, 1, 1)
yoy = YearOnYearInflationSwap(effective_date=yoy_eff, payment_dates=yoy_pay,
                              fixed_rate=jnp.asarray(0.0240), notional=N_INF,
                              base_cpi=BASE_CPI, discount_id=USD, inflation_id=CPI)


def yoy_pv(k: jax.Array) -> jax.Array:
    return price(eqx.tree_at(lambda y: y.fixed_rate, yoy, k), None, market).price


pv_yoy = yoy_pv(yoy.fixed_rate)
# Affine in K: two prices give the slope (−N·Σ τ·DF) and the par rate.
k_a, k_b = jnp.asarray(0.0), jnp.asarray(0.01)
slope = (yoy_pv(k_b) - yoy_pv(k_a)) / (k_b - k_a)
k_par = k_a - yoy_pv(k_a) / slope
print(f"  {len(yoy_pay)}-period YoY swap (1Y fwd start) at 2.40%: PV {float(pv_yoy):,.2f}"
      f"   par rate {float(k_par):.6%} → PV {float(yoy_pv(k_par)):.2e}")
assert abs(float(yoy_pv(k_par))) <= 1e-9 * float(N_INF)

cpi_model = ImpliedVolModel(CPI_VOL)
K_CAP = jnp.asarray(0.0250)
cap = InflationCapFloor(effective_date=yoy_eff, payment_dates=yoy_pay, strike=K_CAP,
                        notional=N_INF, base_cpi=BASE_CPI, is_cap=True,
                        discount_id=USD, inflation_id=CPI)
floor = InflationCapFloor(effective_date=yoy_eff, payment_dates=yoy_pay, strike=K_CAP,
                          notional=N_INF, base_cpi=BASE_CPI, is_cap=False,
                          discount_id=USD, inflation_id=CPI)
pv_cap = price(cap, cpi_model, market).price
pv_floor = price(floor, cpi_model, market).price
# Caplets pay N·(YoY − K) with no accrual fraction while the YoY fixed leg pays
# N·K·τ_i, so the parity is exact when every τ_i = 1 (calendar years, act/act).
taus = year_fraction(jnp.concatenate([yoy_eff[None], yoy_pay[:-1]]), yoy_pay, yoy.day_count)
assert bool(jnp.all(jnp.abs(taus - 1.0) <= 1e-15))
pv_yoy_k = yoy_pv(K_CAP)
print(f"  YoY cap {float(pv_cap):,.2f}   floor {float(pv_floor):,.2f}   at K = 2.50%, σ = 35%")
print(f"  cap − floor {float(pv_cap - pv_floor):,.6f}   YoY swap at 2.50% {float(pv_yoy_k):,.6f}")
assert float(pv_cap) > 0.0 and float(pv_floor) > 0.0
assert abs(float(pv_cap - pv_floor - pv_yoy_k)) <= 1e-10 * float(N_INF)
print("  ✓ cap − floor at K = the receive-inflation YoY swap at fixed rate K")

# Why the CPI curve carries a pillar at the reference date (CPI = base): forward_cpi
# extrapolates flat *backwards* from the first pillar, so without it CPI(today) would
# read CPI(1Y) and a spot-starting YoY period would project a zero first coupon.
no_ref = from_zc_rates(REF, cpi_pillars[1:], CPI_ZC[1:], BASE_CPI)
cpi_today_no_ref = forward_cpi(no_ref, REF)
print(f"  CPI(today): with a reference pillar {float(forward_cpi(cpi_curve, REF)):.4f}   "
      f"without {float(cpi_today_no_ref):.4f} (= CPI(1Y) {float(no_ref.forward_cpis[0]):.4f})")
assert abs(float(forward_cpi(cpi_curve, REF)) - float(BASE_CPI)) <= 1e-12 * float(BASE_CPI)
assert abs(float(cpi_today_no_ref) - float(no_ref.forward_cpis[0])) <= 1e-12 * float(BASE_CPI)
print("  ✓ the reference pillar pins CPI(today) = base; without it the curve extrapolates flat")

# %% 5. market_risk
header("5. market_risk: declared entries, decoys, and FD sanity checks")

# (instrument, model, entries that must carry a non-zero sensitivity)
book = {
    "FX forward": (fwd, None, {"curves": (USD, EUR), "fx_spots": (FX,)}),
    "FX call": (call, fx_model, {"curves": (USD, EUR), "fx_spots": (FX,), "surfaces": (FX_VOL,)}),
    "xccy (exch)": (xccy_ex, None, {"curves": (USD,)}),
    "xccy (no exch)": (xccy_no, None, {"curves": (USD, EUR), "fx_spots": (FX,)}),
    "3M/6M basis": (basis, None, {"curves": (USD, USD3M, USD6M)}),
    "ZCIS": (zcis, None, {"curves": (USD,), "inflation_curves": (CPI,)}),
    "YoY swap": (yoy, None, {"curves": (USD,), "inflation_curves": (CPI,)}),
    "YoY cap": (cap, cpi_model, {"curves": (USD,), "inflation_curves": (CPI,),
                                 "surfaces": (CPI_VOL,)}),
    "YoY floor": (floor, cpi_model, {"curves": (USD,), "inflation_curves": (CPI,),
                                     "surfaces": (CPI_VOL,)}),
}
SLOTS = ("curves", "spots", "dividends", "fx_spots", "inflation_curves", "surfaces")


def slot_tables(g: MarketState) -> dict[str, dict]:
    return {"curves": g.curves.curves, "spots": g.spots, "dividends": g.dividends,
            "fx_spots": g.fx_spots, "inflation_curves": g.inflation_curves,
            "surfaces": g.surfaces}


def max_abs(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return max((float(jnp.max(jnp.abs(x))) for x in leaves), default=0.0)


for name, (inst, model, must_move) in book.items():
    r = market_risk(inst, model, market)
    full = eqx.filter_grad(lambda m: price(inst, model, m).price)(market)
    tables = slot_tables(full)
    shown = {s: getattr(r.declared, s) for s in SLOTS if getattr(r.declared, s)}
    print(f"  {name:<15} PV {float(r.price):>15,.2f}   declares {shown}")
    assert r.declared == declared_entries(inst, model, market)
    # Non-zero where the trade economically depends on the market.
    report = {"curves": r.curves, "fx_spots": r.fx_spots,
              "inflation_curves": r.inflation_curves, "surfaces": r.surfaces}
    for slot, ids in must_move.items():
        for i in ids:
            assert max_abs(report[slot][i]) > 0.0, (name, slot, i)
    # Exactly zero on every entry not declared (the decoys included), and on
    # fixings / credit curves, which no trade here names.
    worst = 0.0
    for slot in SLOTS:
        for key, sub in tables[slot].items():
            if key not in getattr(r.declared, slot):
                worst = max(worst, max_abs(sub))
    worst = max(worst, max_abs(full.fixings), max_abs(full.credit_curves))
    assert worst == 0.0, (name, worst)
    # The report is the full gradient read at the declared keys.
    for c in r.declared.curves:
        assert bool(jnp.all(r.curves[c] == tables["curves"][c].discount_factors))

print("  ✓ every trade moves only with what it declares; decoys and the rest are exactly 0")

# The exchanged xccy still *declares* EUR and EURUSD — its pricer reads them — but
# they cancel to round-off: declared is "may depend on", not "does depend on".
r_ex = market_risk(xccy_ex, None, market)
print(f"  exchanged xccy: |∂PV/∂EURUSD| = {abs(float(r_ex.fx_spots[FX])):.1e}, "
      f"max|∂PV/∂DF_EUR| = {max_abs(r_ex.curves[EUR]):.1e}  (round-off of a cancellation)")
assert abs(float(r_ex.fx_spots[FX])) <= 1e-6 * float(N_EUR)

# FD sanity checks on the market itself (FD only as a check, per AGENTS.md).
r_call = market_risk(call, fx_model, market)
h_fx = 1e-5 * float(S)
up = eqx.tree_at(lambda m: m.fx_spots[FX], market, S + h_fx)
dn = eqx.tree_at(lambda m: m.fx_spots[FX], market, S - h_fx)
fd_fx = (price(call, fx_model, up).price - price(call, fx_model, dn).price) / (2 * h_fx)
gap_fx = abs(float(r_call.fx_spots[FX] - fd_fx)) / abs(float(fd_fx))
print(f"\n  FX call ∂PV/∂EURUSD: market_risk {float(r_call.fx_spots[FX]):,.4f}   "
      f"central FD {float(fd_fx):,.4f}   rel gap {gap_fx:.1e}")
assert gap_fx <= 1e-7

r_cap = market_risk(cap, cpi_model, market)
k_pil = CPI_YEARS.index(3)                       # the 3Y CPI pillar
cpis = cpi_curve.forward_cpis
h_cpi = 1e-6 * float(cpis[k_pil])


def bumped(dh: float) -> MarketState:
    return eqx.tree_at(lambda m: m.inflation_curves[CPI].forward_cpis, market,
                       cpis.at[k_pil].add(dh))


fd_cpi = (price(cap, cpi_model, bumped(h_cpi)).price
          - price(cap, cpi_model, bumped(-h_cpi)).price) / (2 * h_cpi)
ad_cpi = r_cap.inflation_curves[CPI][k_pil]
gap_cpi = abs(float(ad_cpi - fd_cpi)) / abs(float(fd_cpi))
print(f"  YoY cap ∂PV/∂CPI(3Y pillar): market_risk {float(ad_cpi):,.4f}   "
      f"central FD {float(fd_cpi):,.4f}   rel gap {gap_cpi:.1e}")
# A 1e-4 relative CPI bump leaves a ~6e-6 truncation error (the YoY rate moves
# ~0.4% of itself); the 1e-6 bump used here is past that and above round-off.
assert gap_cpi <= 1e-6
print("  ✓ FX delta and CPI-pillar sensitivity agree with central finite differences")

# %% 6. JIT / batching
header("6. JIT and batching")


@eqx.filter_jit
def pv(inst, model, m: MarketState) -> jax.Array:
    return price(inst, model, m).price


t0 = time.perf_counter()
jit_cap = pv(cap, cpi_model, market).block_until_ready()
t1 = time.perf_counter()
jit_cap2 = pv(cap, cpi_model, market).block_until_ready()
t2 = time.perf_counter()
print(f"  filter_jit YoY cap {float(jit_cap):,.6f} (first call {t1 - t0:.2f}s, second {t2 - t1:.4f}s)")
assert rel(jit_cap, pv_cap) <= 1e-12 and float(jit_cap2) == float(jit_cap)

strikes = jnp.linspace(0.95, 1.25, N_LADDER) * S


@eqx.filter_jit
def ladder(ks: jax.Array) -> jax.Array:
    return eqx.filter_vmap(
        lambda k: price(eqx.tree_at(lambda o: o.strike, call, k), fx_model, market).price
    )(ks)


batched = ladder(strikes)
eager = jnp.stack([price(eqx.tree_at(lambda o: o.strike, call, k), fx_model, market).price
                   for k in strikes])
gap = float(jnp.max(jnp.abs(batched - eager)))
print(f"  {N_LADDER}-strike EURUSD call ladder {float(strikes[0]):.4f}…{float(strikes[-1]):.4f}: "
      f"max |vmap − eager| = {gap:.1e}")
for k, v in list(zip(strikes.tolist(), batched.tolist()))[:: max(1, N_LADDER // 4)]:
    print(f"    K = {k:.4f}   PV {v:>14,.2f}")
assert gap <= 1e-12 * float(N_EUR)
assert bool(jnp.all(jnp.diff(batched) < 0.0))   # call value falls with strike
print("  ✓ one compiled, vmapped program equals the eager loop; calls decrease in strike")
