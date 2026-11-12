"""Example 22 — Multi-asset and path-dependent equity exotics on one keyed market.

One ``MarketState`` holds several equities by id; every trade names the
assets it is written on, every price goes through ``price()``, the same
instruments are priced under several models, and risk comes from
``valax.greeks.market_risk`` — no per-instrument risk code. One section drops
to the kernel layer to price a payoff ``price()`` does not route.

  1. Market: SPX, SX5E and a decoy NKY keyed by id (spots, dividends, a vol
     surface each — SVI for SPX, flat for the others), one OIS curve.
  2. Multi-asset: a spread option and a worst-of call under
     ``MultiAssetGBMModel`` (spots by ``underlying_ids``). The zero-strike
     spread is checked against Margrabe (exact), the struck one against Kirk
     (an approximation); the worst-of falls as correlation falls (common
     random numbers); the correlation sensitivity by ``model_gradient``
     against a common-random-number bump.
  3. Path-dependent ladder: European (sanity row), up-and-out barrier,
     Asian, lookback and variance swap under Black-Scholes, Heston, local vol
     and SLV (leverage calibrated on the SPX smile). Anchors: BS European vs
     the closed form, BS barrier MC (Broadie-Glasserman-Kou shifted) vs PDE,
     in + out = vanilla on the same paths, the BS variance-swap fair strike
     σ², Heston European MC vs the analytic price, the LV / SLV vanillas vs
     the smile. The model spread is printed, never claimed to vanish.
  4. Risk: ``market_risk`` for the spread and the barrier — declared entries
     only, an exact zero on every undeclared (decoy) asset, spot deltas vs a
     central bump with common random numbers, and Euler homogeneity of the
     zero-strike spread as an exact check.
  5. Kernel layer: a dual digital (pays if both assets finish above their
     strikes) priced by hand on ``generate_correlated_gbm_paths`` — the same
     paths ``price()`` used for the spread — against its exact bivariate-normal
     value, and its deltas by ``jax.grad`` through the path generator.
  6. JIT: ``eqx.filter_jit`` of a ``price()`` call and ``eqx.filter_vmap``
     over a strike ladder of barriers (BS PDE), equal to the eager loop.

Run::

    python examples/22_unified_multi_asset_exotics.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float

import valax  # noqa: F401  (enables float64)
from valax.calibration import calibrate_slv_leverage
from valax.core import MarketState, auto_method, available_methods, effective_config, price
from valax.core.binding import project_rate
from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks import market_gradient, market_risk, model_gradient
from valax.instruments.options import (
    AsianOption,
    EquityBarrierOption,
    EuropeanOption,
    LookbackOption,
    SpreadOption,
    VarianceSwap,
    WorstOfBasketOption,
)
from valax.models import (
    BlackScholesModel,
    HestonModel,
    LocalVolModel,
    MultiAssetGBMModel,
    SLVModel,
)
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.pricing.analytic.spread import kirk_price, margrabe_price
from valax.pricing.analytic.variance_swap import variance_swap_price
from valax.pricing.mc import MCConfig, generate_correlated_gbm_paths
from valax.pricing.mc.payoffs import spread_option_mc_payoff
from valax.pricing.pde.config import PDEConfig
from valax.surfaces import ConstantVol, SVIVolSurface

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
# Multi-asset GBM log-Euler is exact, and spread / worst-of are terminal
# payoffs: one time step suffices, so the budget goes into paths.
MC_MA = MCConfig(n_paths=20_000 if FAST else 200_000, n_steps=1)
# Path-dependent ladder: 50 monitoring dates.
MC_PATH = MCConfig(n_paths=4_000 if FAST else 40_000, n_steps=50)
# Variance swaps: the recipe annualises with 252 observations a year, so the
# step count must be 252 · T (see section 3).
MC_VS = MCConfig(n_paths=2_000 if FAST else 8_000, n_steps=252)
# SLV leverage: one Guyon–Henry-Labordère pass (self-consistent by itself).
SLV_PARTICLES = 10_000 if FAST else 30_000
SLV_ROWS = 12 if FAST else 24
PDE = PDEConfig(n_spot=200, n_time=200)

KEY_MA, KEY_PATH, KEY_VS, KEY_SLV = jax.random.split(jax.random.PRNGKey(22), 4)
REF = int(ymd_to_ordinal(2026, 1, 1))
T = jnp.asarray(1.0)
OIS = "USD.OIS"
SPX, SX5E, DECOY = "SPX", "SX5E", "NKY"


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


def bump(mkt: MarketState, asset: str, h: float) -> MarketState:
    """The market with one asset's spot moved by ``h`` (a new pytree)."""
    return eqx.tree_at(lambda m: m.spots[asset], mkt, mkt.spots[asset] + h)


# %% 1. Market
header("1. One keyed market: three equities, one OIS curve")

pillars = jnp.array([REF + 365 * k for k in range(11)], dtype=jnp.int32)
t_pil = (pillars - REF) / 365.0
curve = DiscountCurve(
    pillar_dates=pillars, discount_factors=jnp.exp(-(0.03 + 0.002 * t_pil) * t_pil),
    reference_date=jnp.int32(REF), day_count="act_365",
)
SPOT = {SPX: 100.0, SX5E: 95.0, DECOY: 80.0}
DIV = {SPX: 0.015, SX5E: 0.025, DECOY: 0.02}
RATE_1Y = float(project_rate(curve, T))
# A skewed SVI smile for SPX, its slice forwards on the curve's carry.
svi_t = jnp.array([0.25, 0.5, 1.0, 2.0])
spx_svi = SVIVolSurface(
    expiries=svi_t, forwards=SPOT[SPX] * jnp.exp((RATE_1Y - DIV[SPX]) * svi_t),
    a_vec=0.03 * svi_t, b_vec=0.08 * svi_t, rho_vec=jnp.full(4, -0.6),
    m_vec=jnp.full(4, 0.05), sigma_vec=jnp.full(4, 0.2),
)
market = MarketState(
    curves=CurveGraph(curves={OIS: curve}),
    spots={a: jnp.asarray(s) for a, s in SPOT.items()},
    dividends={a: jnp.asarray(q) for a, q in DIV.items()},
    surfaces={f"{SPX}.VOL": spx_svi, f"{SX5E}.VOL": ConstantVol(jnp.asarray(0.25)),
              f"{DECOY}.VOL": ConstantVol(jnp.asarray(0.22))},
)
FWD_SPX = SPOT[SPX] * np.exp((RATE_1Y - DIV[SPX]) * float(T))
VOL = {SPX: float(spx_svi(jnp.asarray(FWD_SPX), T)), SX5E: 0.25}

print(f"  curve {OIS}: 1Y zero rate {RATE_1Y:.4%} (the only curve: '_default_' resolves to it)")
print(f"  {'asset':<6}{'spot':>8}{'div':>8}  surface")
for a in SPOT:
    s = market.surfaces[f"{a}.VOL"]
    desc = (f"SVI, 1Y vols 80/100/120% K: "
            + "/".join(f"{float(s(jnp.asarray(k), T)):.1%}" for k in (80.0, 100.0, 120.0))
            if isinstance(s, SVIVolSurface) else f"flat {float(s.vol):.0%}")
    print(f"  {a:<6}{SPOT[a]:>8.2f}{DIV[a]:>8.2%}  {a}.VOL: {desc}")
print(f"  {DECOY} is a decoy: no trade below names it, so every gradient on it must be 0.")

# %% 2. Multi-asset: spread and worst-of under correlated GBM
header("2. Spread and worst-of under MultiAssetGBMModel, through price()")

RHO = 0.5


def ma_model(rho) -> MultiAssetGBMModel:
    """Two-asset GBM on (SPX, SX5E). ``rate=None`` is bound from the curve.

    The model owns its vols and dividends (set here from the market's 1Y ATM
    vols and dividend entries); only the rate is late-bound by ``price()``.
    """
    corr = jnp.array([[1.0, 0.0], [0.0, 1.0]]) + rho * jnp.array([[0.0, 1.0], [1.0, 0.0]])
    return MultiAssetGBMModel(
        vols=jnp.array([VOL[SPX], VOL[SX5E]]), rate=None,
        dividends=jnp.array([DIV[SPX], DIV[SX5E]]), correlation=corr,
    )


ma = ma_model(jnp.asarray(RHO))
exchange = SpreadOption(expiry=T, strike=jnp.asarray(0.0), notional=jnp.asarray(1.0),
                        underlying_ids=(SPX, SX5E))
spread = SpreadOption(expiry=T, strike=jnp.asarray(5.0), notional=jnp.asarray(1.0),
                      underlying_ids=(SPX, SX5E))
wof_call = WorstOfBasketOption(expiry=T, strike=jnp.asarray(0.95), notional=jnp.asarray(100.0),
                               n_assets=2, is_call=True, underlying_ids=(SPX, SX5E))
wof_put = WorstOfBasketOption(expiry=T, strike=jnp.asarray(1.0), notional=jnp.asarray(100.0),
                              n_assets=2, is_call=False, underlying_ids=(SPX, SX5E))

print(f"  {'trade':<16}{'methods':<9}{'auto':<6}{'config':<30}")
for name, trade in (("spread", spread), ("worst-of", wof_call)):
    print(f"  {name:<16}{','.join(available_methods(trade, ma)):<9}{auto_method(trade, ma):<6}"
          f"{str(effective_config(trade, ma)):<30}")
print(f"  (the default MC config; this example passes its own: {MC_MA})")


def mc(trade, model, mkt=market, cfg=MC_MA, key=KEY_MA):
    return price(trade, model, mkt, method="mc", config=cfg, key=key)


s1, s2 = jnp.asarray(SPOT[SPX]), jnp.asarray(SPOT[SX5E])
v1, v2 = jnp.asarray(VOL[SPX]), jnp.asarray(VOL[SX5E])
q1, q2 = jnp.asarray(DIV[SPX]), jnp.asarray(DIV[SX5E])
r_ex = mc(exchange, ma)
margrabe = float(margrabe_price(exchange, s1, s2, v1, v2, jnp.asarray(RHO), q1, q2))
z_m = (float(r_ex.price) - margrabe) / float(r_ex.stderr)
r_sp = mc(spread, ma)
kirk = float(kirk_price(spread, s1, s2, v1, v2, jnp.asarray(RHO), jnp.asarray(RATE_1Y), q1, q2))
z_k = (float(r_sp.price) - kirk) / float(r_sp.stderr)
print(f"\n  ρ = {RHO}, {MC_MA.n_paths} paths")
print(f"  exchange (K=0): MC {float(r_ex.price):.4f} ± {float(r_ex.stderr):.4f}   "
      f"Margrabe (exact) {margrabe:.4f}   z = {z_m:+.2f}")
print(f"  spread  (K=5):  MC {float(r_sp.price):.4f} ± {float(r_sp.stderr):.4f}   "
      f"Kirk (approx.)   {kirk:.4f}   z = {z_k:+.2f}")
assert abs(z_m) < 3.0, z_m
print("  ✓ the zero-strike spread matches Margrabe's exact price within 3 σ_MC")
assert abs(z_k) < 4.0, z_k
print("  ✓ Kirk's approximation is within MC error of the struck spread at this budget")

print(f"\n  {'ρ':>5}{'worst-of call 95%':>22}{'worst-of put 100%':>22}   (same random numbers)")
wof = {}
for rho in (0.8, 0.5, 0.2):
    m = ma_model(jnp.asarray(rho))
    c, p = mc(wof_call, m), mc(wof_put, m)
    wof[rho] = (float(c.price), float(p.price))
    print(f"  {rho:>5.1f}{float(c.price):>14.4f} ± {float(c.stderr):.4f}"
          f"{float(p.price):>14.4f} ± {float(p.stderr):.4f}")
calls = [wof[r][0] for r in (0.8, 0.5, 0.2)]
puts = [wof[r][1] for r in (0.8, 0.5, 0.2)]
assert calls[0] > calls[1] > calls[2] and puts[0] < puts[1] < puts[2]
print("  ✓ lower correlation: the worst-of call falls, the worst-of put rises")

g = model_gradient(wof_call, ma, market, method="mc", config=MC_MA, key=KEY_MA)
d_rho_ad = float(g.correlation[0, 1] + g.correlation[1, 0])   # symmetric entry
h_rho = 1e-3
d_rho_fd = (float(mc(wof_call, ma_model(jnp.asarray(RHO + h_rho))).price)
            - float(mc(wof_call, ma_model(jnp.asarray(RHO - h_rho))).price)) / (2 * h_rho)
print(f"\n  ∂(worst-of call)/∂ρ: model_gradient {d_rho_ad:+.6f}   central bump (CRN) {d_rho_fd:+.6f}")
print(f"  per-asset vegas ∂/∂σ: {[round(float(v), 4) for v in g.vols]}")
assert d_rho_ad > 0.0
assert abs(d_rho_ad - d_rho_fd) <= 1e-3 * abs(d_rho_fd), (d_rho_ad, d_rho_fd)
print("  ✓ the correlation sensitivity by AD through Cholesky + paths matches the bump")

# %% 3. Path-dependent ladder across models
header("3. Path-dependent ladder: the same trades under BS, Heston, LV, SLV")

K = jnp.asarray(100.0)
BARRIER = 130.0
european = EuropeanOption(strike=K, expiry=T, underlying_id=SPX)
uo_call = EquityBarrierOption(strike=K, expiry=T, barrier=jnp.asarray(BARRIER), is_up=True,
                              is_knock_in=False, underlying_id=SPX)
asian = AsianOption(strike=K, expiry=T, underlying_id=SPX)
lookback = LookbackOption(expiry=T, is_fixed_strike=True, strike=K, underlying_id=SPX)
var_swap = VarianceSwap(expiry=T, strike_var=jnp.asarray(0.04), notional_var=jnp.asarray(1e4),
                        underlying_id=SPX)

# BS has no surface selector: with vol=None it reads only the market's
# '_default_' surface, which this keyed market does not have. Its vol is the
# SPX 1Y ATM-forward vol, given explicitly; rate and dividend are late-bound
# (curve at the trade's expiry, SPX's dividend).
bs = BlackScholesModel(vol=jnp.asarray(VOL[SPX]))
heston = HestonModel(v0=jnp.asarray(0.055), kappa=jnp.asarray(1.5), theta=jnp.asarray(0.055),
                     xi=jnp.asarray(0.5), rho=jnp.asarray(-0.7))
local_vol = LocalVolModel(surface=spx_svi)
# SLV: one leverage pass on the SPX smile over the Heston backbone. The
# particle method needs a concrete flat carry; the priced model leaves
# rate / dividend to the market like the others.
heston_carry = eqx.tree_at(lambda m: (m.rate, m.dividend), heston,
                           (jnp.asarray(RATE_1Y), jnp.asarray(DIV[SPX])),
                           is_leaf=lambda x: x is None)
lev_fit = calibrate_slv_leverage(
    heston_carry, spx_svi, market.spot(SPX), jnp.linspace(-0.5, 0.5, 11),
    jnp.linspace(1.0 / SLV_ROWS, 1.0, SLV_ROWS), SLV_PARTICLES, KEY_SLV,
    method="kernel", n_iterations=1,
)
slv = SLVModel.from_heston_and_leverage(heston, spx_svi, lev_fit.value)
models = {"BS": bs, "Heston": heston, "LV": local_vol, "SLV": slv}
print(f"  SLV leverage: one pass, {SLV_PARTICLES} particles, {SLV_ROWS} time rows × 11 k; "
      f"clip fraction {float(lev_fit.diagnostics.extras['clip_fraction']):.1%}")

trades = {"European": european, "UO barrier": uo_call, "Asian": asian,
          "lookback": lookback, "var swap": var_swap}
print(f"\n  auto method        " + "".join(f"{n:<10}" for n in models)
      + "available (BS / Heston)")
for name, trade in trades.items():
    print(f"  {name:<19}" + "".join(f"{auto_method(trade, m):<10}" for m in models.values())
          + f"{','.join(available_methods(trade, bs))} / "
          f"{','.join(available_methods(trade, heston))}")
print("  (this section prices everything by MC to compare models on one engine)")


def ladder_price(trade, model):
    cfg = MC_VS if isinstance(trade, VarianceSwap) else MC_PATH
    key = KEY_VS if isinstance(trade, VarianceSwap) else KEY_PATH
    return price(trade, model, market, method="mc", config=cfg, key=key)


table = {(t, m): ladder_price(trade, model)
         for t, trade in trades.items() for m, model in models.items()}
print(f"\n  MC prices ± σ_MC ({MC_PATH.n_paths} paths × {MC_PATH.n_steps} steps; "
      f"var swap {MC_VS.n_paths} × {MC_VS.n_steps}); var swap as fair vol")
print(f"  {'trade':<12}" + "".join(f"{m:>18}" for m in models) + f"{'max−min':>10}")
df_1y = float(jnp.exp(-RATE_1Y * T))


def fair_vol(res) -> float:
    """Fair variance-swap vol from its value at strike 0.04 (notional 1e4)."""
    return float(np.sqrt(0.04 + float(res.price) / (1e4 * df_1y)))


for t in trades:
    vals = [table[(t, m)] for m in models]
    if t == "var swap":
        cells = [f"{fair_vol(v):>12.2%} ± {float(v.stderr) / (2e4 * df_1y * fair_vol(v)):.2%}"
                 for v in vals]
        spread_ = max(fair_vol(v) for v in vals) - min(fair_vol(v) for v in vals)
        print(f"  {t:<12}" + "".join(f"{c:>18}" for c in cells) + f"{spread_:>10.2%}")
    else:
        ps = [float(v.price) for v in vals]
        print(f"  {t:<12}" + "".join(f"{p:>10.4f} ± {float(v.stderr):.3f}" for p, v in zip(ps, vals))
              + f"{max(ps) - min(ps):>10.4f}")
print("  The European row is pinned by the smile (LV / SLV) or close to it; the exotics are not:")
print("  the vanillas fix the marginals, not the path dynamics.")

# --- Anchors ---
print()
q_spx = DIV[SPX]
bs_eu = float(black_scholes_price(european, market.spot(SPX), jnp.asarray(VOL[SPX]),
                                  jnp.asarray(RATE_1Y), jnp.asarray(q_spx)))
r = table[("European", "BS")]
z = (float(r.price) - bs_eu) / float(r.stderr)
print(f"  BS European: MC {float(r.price):.4f} vs closed form {bs_eu:.4f}  (z = {z:+.2f})")
assert abs(z) < 3.0
print("  ✓ BS Monte Carlo European within 3 σ_MC of black_scholes_price")

r = table[("European", "Heston")]
h_an = float(price(european, heston, market, method="analytic").price)
z = (float(r.price) - h_an) / float(r.stderr)
print(f"  Heston European: MC {float(r.price):.4f} vs analytic {h_an:.4f}  (z = {z:+.2f})")
assert abs(z) < 3.0
print("  ✓ Heston Monte Carlo (QE) European within 3 σ_MC of the analytic price")

smile_eu = float(black_scholes_price(european, market.spot(SPX), spx_svi(K, T),
                                     jnp.asarray(RATE_1Y), jnp.asarray(q_spx)))
for m in ("LV", "SLV"):
    r = table[("European", m)]
    z = (float(r.price) - smile_eu) / float(r.stderr)
    print(f"  {m} European: MC {float(r.price):.4f} vs smile (BS at σ(K)) {smile_eu:.4f}  (z = {z:+.2f})")
    assert abs(z) < 4.0, (m, z)
print("  ✓ LV and SLV reprice the SPX smile at the strike within MC noise (+ leverage noise)")

# Barrier: PDE monitors continuously, MC at 50 dates. Broadie-Glasserman-Kou:
# a discrete barrier H prices like a continuous one at H·exp(+β σ √Δt),
# β = 0.5826, so the continuous barrier H is mimicked by a discrete one at
# H·exp(−β σ √Δt).
pde_uo = float(price(uo_call, bs, market, method="pde", config=PDE).price)
mc_uo = table[("UO barrier", "BS")]
dt = float(T) / MC_PATH.n_steps
bgk = eqx.tree_at(lambda b: b.barrier, uo_call,
                  jnp.asarray(BARRIER * np.exp(-0.5826 * VOL[SPX] * np.sqrt(dt))))
mc_bgk = ladder_price(bgk, bs)
z_raw = (float(mc_uo.price) - pde_uo) / float(mc_uo.stderr)
z_bgk = (float(mc_bgk.price) - pde_uo) / float(mc_bgk.stderr)
print(f"\n  BS up-and-out: PDE (continuous monitoring) {pde_uo:.4f}")
print(f"    MC, {MC_PATH.n_steps} monitoring dates     {float(mc_uo.price):.4f}  (z = {z_raw:+.1f})")
print(f"    MC, BGK-shifted barrier       {float(mc_bgk.price):.4f}  (z = {z_bgk:+.2f})")
assert z_raw > 3.0
assert abs(z_bgk) < 3.0, z_bgk
print("  ✓ discrete monitoring overprices the knock-out; the BGK shift closes the gap to the PDE")

# In + out = vanilla, path by path, on the same key — whatever the smoothing.
print(f"\n  {'smoothing':>10}{'up-and-in':>12}{'up-and-out':>12}{'in + out':>12}{'vanilla MC':>12}")
eu_mc = float(table[("European", "Heston")].price)
for smoothing in (0.0, 0.5, 2.0):
    legs = [float(ladder_price(EquityBarrierOption(
        strike=K, expiry=T, barrier=jnp.asarray(BARRIER), is_up=True, is_knock_in=ki,
        smoothing=smoothing, underlying_id=SPX), heston).price) for ki in (True, False)]
    print(f"  {smoothing:>10.1f}{legs[0]:>12.4f}{legs[1]:>12.4f}{sum(legs):>12.4f}{eu_mc:>12.4f}")
    assert abs(sum(legs) - eu_mc) <= 1e-10 * eu_mc
print("  ✓ in + out = vanilla to round-off under Heston, for every smoothing width: the")
print("    sigmoid weight h goes to the knock-in and 1 − h to the knock-out on each path.")
print("    Smoothing moves value between the legs (a bias vs the hard barrier), never the sum.")

# Variance swap: the BS fair strike is σ² (Euler log-returns on 252 steps).
r = table[("var swap", "BS")]
vs_an = float(variance_swap_price(var_swap, jnp.asarray(VOL[SPX]), jnp.asarray(RATE_1Y)))
z = (float(r.price) - vs_an) / float(r.stderr)
print(f"\n  BS variance swap: MC value {float(r.price):.3f} ± {float(r.stderr):.3f} vs "
      f"N·(σ² − K)·DF = {vs_an:.3f}  (z = {z:+.2f}); fair vol {fair_vol(r):.2%} vs σ {VOL[SPX]:.2%}")
assert abs(z) < 3.0
print("  ✓ the BS fair variance strike is σ² within MC error")
# Pitfall: realised variance is annualised with 252 observations a year and
# price() cannot pass another annual_factor, so n_steps must be 252 · T.
half = price(var_swap, bs, market, method="mc", config=MCConfig(n_paths=MC_VS.n_paths, n_steps=126), key=KEY_VS)
fv_half = fair_vol(half) ** 2 / VOL[SPX] ** 2
print(f"  pitfall: with 126 steps over 1Y the same swap's fair variance is {fv_half:.2f} σ²"
      " (252 / 126 = 2):")
print("  the recipe annualises as if every step were one trading day.")
assert abs(fv_half - 2.0) < 0.1
lv_fv, atm = fair_vol(table[("var swap", "LV")]), VOL[SPX]
print(f"  LV fair vol {lv_fv:.2%} > ATM vol {atm:.2%}: the skew's OTM puts are in the log contract")
assert lv_fv > atm

# %% 4. Risk from market_risk
header("4. market_risk: declared entries, decoys at exactly zero, deltas vs bumps")


def show(risk) -> None:
    d = risk.declared
    print(f"    declared curves {d.curves}, spots {d.spots}, dividends {d.dividends}, "
          f"surfaces {d.surfaces}")
    print("    spot deltas " + ", ".join(f"{a} {float(v):+.6f}" for a, v in risk.spots.items())
          + "; dividend " + ", ".join(f"{a} {float(v):+.4f}" for a, v in risk.dividends.items())
          + f"; Σ∂/∂DF {sum(float(jnp.sum(v)) for v in risk.curves.values()):+.4f}")


def decoys_zero(grad: MarketState, used: tuple[str, ...]) -> None:
    """Every spot, dividend and surface leaf outside ``used`` is exactly 0."""
    for a in SPOT:
        if a in used:
            continue
        assert float(grad.spots[a]) == 0.0 and float(grad.dividends[a]) == 0.0, a
    for sid, surf in grad.surfaces.items():
        assert all(float(jnp.max(jnp.abs(leaf))) == 0.0
                   for leaf in jax.tree_util.tree_leaves(surf)), sid


def fd_delta(trade, model, asset, h, **kw) -> float:
    up = price(trade, model, bump(market, asset, h), **kw).price
    dn = price(trade, model, bump(market, asset, -h), **kw).price
    return float(up - dn) / (2 * h)


mc_kw = dict(method="mc", config=MC_MA, key=KEY_MA)
print("  Zero-strike spread (MultiAssetGBMModel, MC):")
risk_ex = market_risk(exchange, ma, market, **mc_kw)
show(risk_ex)
assert risk_ex.declared.spots == (SPX, SX5E) and risk_ex.declared.curves == (OIS,)
decoys_zero(market_gradient(exchange, ma, market, **mc_kw), (SPX, SX5E))
print(f"  ✓ {DECOY} (spot, dividend, surface) and every surface: gradient exactly 0")
# The model owns its dividends: the declared market dividends are read by
# resolve() but never reach the paths, so their gradient is a structural 0.
assert all(float(v) == 0.0 for v in risk_ex.dividends.values())
print("  ✓ declared SPX / SX5E dividends carry 0: MultiAssetGBMModel uses its own `dividends`")
# A small bump: with common random numbers the sample price is piecewise
# linear in spot (one kink per path), so the central difference converges to
# the pathwise derivative as the bump shrinks.
for a in (SPX, SX5E):
    fd = fd_delta(exchange, ma, a, 1e-5 * SPOT[a], **mc_kw)
    print(f"    Δ_{a}: market_risk {float(risk_ex.spots[a]):+.6f}   central bump (CRN) {fd:+.6f}")
    assert abs(float(risk_ex.spots[a]) - fd) <= 1e-4 * abs(fd) + 1e-6
euler = SPOT[SPX] * float(risk_ex.spots[SPX]) + SPOT[SX5E] * float(risk_ex.spots[SX5E])
print(f"    S₁Δ₁ + S₂Δ₂ = {euler:.10f}   price = {float(risk_ex.price):.10f}")
assert abs(euler - float(risk_ex.price)) <= 1e-10 * float(risk_ex.price)
print("  ✓ deltas match the bumps; Euler's theorem holds exactly (degree-1 homogeneous payoff)")
# The exchange option does not depend on the rate (the drift cancels the
# discount path by path), the struck spread does: rate=None was bound from
# the curve inside the differentiated path, so its rho lands on USD.OIS.
risk_sp = market_risk(spread, ma, market, **mc_kw)
rho_ex = sum(float(jnp.sum(v)) for v in risk_ex.curves.values())
rho_sp = float(jnp.sum(risk_sp.curves[OIS]))
print(f"    Σ ∂PV/∂DF on {OIS}: exchange {rho_ex:+.2e}, struck spread {rho_sp:+.4f}")
assert abs(rho_ex) < 1e-8 and abs(rho_sp) > 1.0
print("  ✓ curve risk of the struck spread through the late-bound model rate; none for K = 0")

print("\n  Up-and-out call (BS, PDE — deterministic):")
pde_kw = dict(method="pde", config=PDE)
risk_bar = market_risk(uo_call, bs, market, **pde_kw)
show(risk_bar)
assert risk_bar.declared.spots == (SPX,) and risk_bar.declared.surfaces == ()
decoys_zero(market_gradient(uo_call, bs, market, **pde_kw), (SPX,))
# The PDE recipe centres its grid on a stop-gradient spot: AD differentiates
# the read-off on a fixed mesh, while a bump re-centres the mesh and so adds
# O(dx²)/h grid noise — a tiny bump (0.01) is off by ~4% here. A bump of half
# a grid cell (~0.5 index point) and a finer grid both settle on the AD value.
fd = fd_delta(uo_call, bs, SPX, 0.5, **pde_kw)
fine = PDEConfig(n_spot=400, n_time=400)
ad_fine = float(market_risk(uo_call, bs, market, method="pde", config=fine).spots[SPX])
ad = float(risk_bar.spots[SPX])
print(f"    Δ_SPX: market_risk {ad:+.6f} (400×400 grid {ad_fine:+.6f})   central bump ±0.5 {fd:+.6f}")
assert abs(ad - ad_fine) <= 2e-4 * abs(ad)
assert abs(ad - fd) <= 2e-3 * abs(fd)
print(f"  ✓ SX5E, {DECOY} and every surface exactly 0; the PDE delta is grid-converged and")
print("    matches the bump")
print("    (the BS model carries its vol, so no surface is declared: vega is a model")
print("    greek here — model_gradient(...).vol)")

print("\n  Up-and-out call (Heston, MC, sigmoid-smoothed barrier for a pathwise delta):")
uo_smooth = EquityBarrierOption(strike=K, expiry=T, barrier=jnp.asarray(BARRIER), is_up=True,
                                is_knock_in=False, smoothing=1.0, underlying_id=SPX)
hmc_kw = dict(method="mc", config=MC_PATH, key=KEY_PATH)
risk_h = market_risk(uo_smooth, heston, market, **hmc_kw)
show(risk_h)
decoys_zero(market_gradient(uo_smooth, heston, market, **hmc_kw), (SPX,))
fd = fd_delta(uo_smooth, heston, SPX, 1e-2, **hmc_kw)
print(f"    Δ_SPX: market_risk {float(risk_h.spots[SPX]):+.6f}   central bump (CRN) {fd:+.6f}")
assert abs(float(risk_h.spots[SPX]) - fd) <= 1e-3 * abs(fd)
print("  ✓ pathwise delta through the QE paths and the smoothed payoff matches the bump")

# %% 5. Kernel layer: a payoff price() does not route
header("5. Kernel layer: a dual digital on the public path generator")

# The model price() actually ran: rate bound at the trade's horizon.
ma_bound = eqx.tree_at(lambda m: m.rate, ma, project_rate(market.curve(OIS), T),
                       is_leaf=lambda x: x is None)
spots0 = jnp.stack([market.spot(SPX), market.spot(SX5E)])
df = jnp.exp(-ma_bound.rate * T)
paths = generate_correlated_gbm_paths(ma_bound, spots0, T, MC_MA.n_steps, MC_MA.n_paths, KEY_MA)
kernel_ex = float(df * jnp.mean(spread_option_mc_payoff(paths, exchange)))
print(f"  exchange option on the kernel paths {kernel_ex:.12f}   price() {float(r_ex.price):.12f}")
assert abs(kernel_ex - float(r_ex.price)) <= 1e-12 * kernel_ex
print("  ✓ same generator, key and config: price() is this computation, bit for bit")

K_DD = jnp.array([100.0, 95.0])   # pays 1 if SPX_T ≥ 100 and SX5E_T ≥ 95
EPS = 0.5                         # sigmoid width (index points) for the smoothed version


def dual_digital(spots: Float[Array, " 2"], smooth: bool) -> Float[Array, ""]:
    """Discounted dual digital on paths generated from ``spots``."""
    st = generate_correlated_gbm_paths(ma_bound, spots, T, MC_MA.n_steps, MC_MA.n_paths,
                                       KEY_MA)[:, -1, :]
    hit = jax.nn.sigmoid((st - K_DD) / EPS) if smooth else (st >= K_DD).astype(st.dtype)
    return df * jnp.mean(jnp.prod(hit, axis=1))


def dual_digital_exact(spots: Float[Array, " 2"]) -> Float[Array, ""]:
    """DF · P(Z₁ ≥ a₁, Z₂ ≥ a₂) for the correlated log-normals (1-D quadrature)."""
    s = ma_bound.vols * jnp.sqrt(T)
    m = jnp.log(spots / K_DD) + (ma_bound.rate - ma_bound.dividends - 0.5 * ma_bound.vols**2) * T
    a1, a2 = -m / s
    x = jnp.linspace(a1, a1 + 12.0, 40_001)
    f = jax.scipy.stats.norm.pdf(x) * jax.scipy.stats.norm.cdf((RHO * x - a2) / np.sqrt(1 - RHO**2))
    return df * jnp.trapezoid(f, x)


hard = float(dual_digital(spots0, False))
smooth = float(dual_digital(spots0, True))
exact = float(dual_digital_exact(spots0))
p_hat = hard / float(df)
se = float(df) * np.sqrt(p_hat * (1 - p_hat) / MC_MA.n_paths)
print(f"\n  dual digital: MC {hard:.5f} ± {se:.5f}   smoothed (ε={EPS}) {smooth:.5f}   "
      f"exact (bivariate normal) {exact:.5f}")
assert abs(hard - exact) < 3 * se
print("  ✓ the hand-built kernel price is within 3 σ_MC of the exact value")

delta_ad = jax.grad(lambda s: dual_digital(s, True))(spots0)
delta_ex = jax.grad(dual_digital_exact)(spots0)
# Its standard error: GBM terminals are linear in spot (S_T = S_0 · G), so the
# per-path pathwise derivative is explicit; its mean is what jax.grad returned.
st = paths[:, -1, :]
sig = jax.nn.sigmoid((st - K_DD) / EPS)
per_path = df * (sig * (1 - sig) / EPS) * (st / spots0) * sig[:, ::-1]
delta_se = np.asarray(jnp.std(per_path, axis=0)) / np.sqrt(MC_MA.n_paths)
assert np.allclose(np.asarray(jnp.mean(per_path, axis=0)), np.asarray(delta_ad), rtol=1e-10)
for i, a in enumerate((SPX, SX5E)):
    h = 1e-5 * float(spots0[i])
    e = jnp.zeros(2).at[i].set(h)
    fd = float(dual_digital(spots0 + e, True) - dual_digital(spots0 - e, True)) / (2 * h)
    z = (float(delta_ad[i]) - float(delta_ex[i])) / delta_se[i]
    print(f"    Δ_{a}: jax.grad through the generator {float(delta_ad[i]):+.6f} ± {delta_se[i]:.6f}"
          f"   bump (CRN) {fd:+.6f}   exact {float(delta_ex[i]):+.6f}  (z = {z:+.2f})")
    assert abs(float(delta_ad[i]) - fd) <= 1e-5 * abs(fd)
    assert abs(z) < 3.0
print("  ✓ AD through generate_correlated_gbm_paths equals the bump on the smoothed payoff")
print("    and the exact delta within 3 σ_MC (the symmetric sigmoid's bias is O(ε²); a hard")
print("    digital has no pathwise delta at all)")

# %% 6. JIT and vmap
header("6. filter_jit of price() and filter_vmap over a strike ladder")


@eqx.filter_jit
def barrier_pv(trade: EquityBarrierOption, mkt: MarketState) -> Float[Array, ""]:
    return price(trade, bs, mkt, method="pde", config=PDE).price


jit_pv = float(barrier_pv(uo_call, market))
print(f"  up-and-out under filter_jit {jit_pv:.12f}   eager {pde_uo:.12f}")
assert abs(jit_pv - pde_uo) <= 1e-12 * pde_uo

strikes = jnp.linspace(90.0, 115.0, 6)
ladder = eqx.filter_jit(eqx.filter_vmap(
    lambda k: barrier_pv(eqx.tree_at(lambda b: b.strike, uo_call, k), market)))(strikes)
eager = [float(price(eqx.tree_at(lambda b: b.strike, uo_call, k), bs, market,
                     method="pde", config=PDE).price) for k in strikes]
print("  strike  " + "".join(f"{float(k):>9.1f}" for k in strikes))
print("  vmapped " + "".join(f"{float(v):>9.4f}" for v in ladder))
assert np.allclose(np.asarray(ladder), eager, rtol=1e-10, atol=1e-12)
assert np.all(np.diff(np.asarray(ladder)) < 0)
print("  ✓ one compiled, vmapped PDE ladder equals the eager loop; value falls with strike")
