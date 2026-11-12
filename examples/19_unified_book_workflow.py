"""Example 19 — A rates book end to end: quotes → curves → risk → PCA scenarios.

The ``MarketState`` / ``price()`` successor of the retired examples 08
(six-stage workflow) and 09 (PCA curve shocks → bond P&L). It also shows the
two public API layers working together: ``price()`` routes every trade, and
the low-level kernels (``valax.pricing.analytic.bonds``, ``valax.risk.factors``,
``valax.risk.shocks``) supply what a router does not return — yields,
durations, a fitted factor model.

  1. Quotes → curves: synthetic OIS + 3M quotes, jointly bootstrapped.
  2. A mixed book — swaps, fixed-rate bonds, an FRN — through ``price()``; the
     policy (``available_methods`` / ``auto_method`` / ``effective_config``)
     says how each trade is priced.
  3. ``market_risk``: one backward pass per trade, keyed by the curves the
     trade declares. The per-trade reports sum to the gradient of the whole
     book; zero-rate DV01s are bucketed and confirmed by a ±1 bp ``shock``.
  4. Dropping to the kernels: ``price()`` of a bond *is*
     ``fixed_rate_bond_price`` on the resolved curve; YTM, modified duration
     and convexity from the kernel layer explain a 25 bp curve move.
  5. A PCA factor model fitted to a synthetic curve history
     (``zero_rate_returns_from_snapshots`` → ``fit_rates_pca``).
  6. PC and basis moves as ``KeyedScenario``s → ``shock`` → revaluation,
     vmapped and compiled once. A basis-only move leaves the bonds untouched;
     P&L explain against the ``market_risk`` report; PC-score sensitivities by
     one ``jax.grad``; VaR / ES.

Run::

    python examples/19_unified_book_workflow.py
"""

# %% Imports and setup
from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import calibrate_market
from valax.core import auto_method, available_methods, effective_config, price
from valax.dates.daycounts import year_fraction
from valax.greeks import bucketed_curve_deltas, market_risk
from valax.instruments.bonds import FixedRateBond, FloatingRateBond
from valax.market import (
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    rates_snapshot,
    sample_swap_book,
)
from valax.market.keyed_scenario import KeyedScenario, stack_keyed_scenarios
from valax.pricing.analytic.bonds import (
    convexity,
    fixed_rate_bond_price,
    modified_duration,
    yield_to_maturity,
)
from valax.risk.bucketing import tenor_bucket_map
from valax.risk.factors import fit_rates_pca, zero_rate_returns_from_snapshots
from valax.risk.shocks import bump_curve_zero_rates
from valax.risk.state_shocks import shock

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
N_DAYS = 120 if FAST else 500          # length of the synthetic curve history
N_SCENARIOS = 64 if FAST else 1000     # PCA scenarios for P&L explain / VaR
HORIZON = 10                           # scenario horizon in days


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
cfg = default_config()
spec = RatesWorldSpec(
    pillar_years=(1, 2, 3, 5, 10) if FAST else (1, 2, 3, 5, 7, 10, 15, 20),
)
OIS, PROJ = spec.ois_id, spec.projection_id("3M")
CURVES = (OIS, PROJ)

# %% 1. Quotes → curves
header("1. Quotes → curves (OIS + 3M, joint bootstrap)")

snap, world = rates_snapshot(registry, cfg, spec, bp_noise=0.5)
calibrated = calibrate_market(snap.chain, snap.quotes, strict=True)
market = calibrated.market
ref = world.reference_date
res = calibrated.results[0]
print(f"  {len(snap.parts['curves'].keys)} quotes  converged={bool(res.diagnostics.converged)}"
      f"  max|R|={float(res.diagnostics.max_abs_residual):.1e}")

curve_ois = market.curves[OIS]
t_pillar = year_fraction(curve_ois.reference_date, curve_ois.pillar_dates, curve_ois.day_count)
for cid in CURVES:
    assert market.curves[cid].pillar_dates.shape == curve_ois.pillar_dates.shape
N_PILLARS = int(curve_ois.pillar_dates.shape[0])
DF = {cid: market.curves[cid].discount_factors for cid in CURVES}
print(f"  pillars (years): {[round(float(t), 2) for t in t_pillar]}")

# %% 2. A mixed book through price()
header("2. A mixed book through price()")


def semiannual(years: int) -> jax.Array:
    return ref + 182 * jnp.arange(1, 2 * years + 1, dtype=jnp.int32)


def accrual_starts(pay: jax.Array) -> jax.Array:
    return jnp.concatenate([ref[None], pay[:-1]])


def fixed_bond(years: int, coupon: float) -> FixedRateBond:
    pay = semiannual(years)
    return FixedRateBond(
        payment_dates=pay, settlement_date=ref, coupon_rate=jnp.asarray(coupon),
        face_value=jnp.asarray(1e6), accrual_start_dates=accrual_starts(pay),
        frequency=2, day_count=world.day_count, discount_id=OIS,
    )


frn_dates = semiannual(5)
swaps = sample_swap_book(registry, world, n_swaps=3 if FAST else 6, off_market_bp=25.0)
book = {f"swap {i}": s for i, s in enumerate(swaps)}
book |= {
    "bond 2Y": fixed_bond(2, 0.035),
    "bond 5Y": fixed_bond(5, 0.040),
    "bond 10Y": fixed_bond(10, 0.045),
    "FRN 5Y": FloatingRateBond(
        payment_dates=frn_dates, fixing_dates=accrual_starts(frn_dates),
        settlement_date=ref, spread=jnp.asarray(0.0015), face_value=jnp.asarray(1e6),
        accrual_start_dates=accrual_starts(frn_dates), day_count=world.day_count,
        discount_id=OIS, forward_id=PROJ,
    ),
}
BONDS = ("bond 2Y", "bond 5Y", "bond 10Y")

pvs: dict[str, float] = {}
print(f"  {'trade':<10}{'type':<18}{'methods':<12}{'auto':<10}{'config':<8}{'PV':>16}")
for name, trade in book.items():
    result = price(trade, None, market)
    assert result.method == auto_method(trade, None)
    pvs[name] = float(result.price)
    print(f"  {name:<10}{type(trade).__name__:<18}{','.join(available_methods(trade, None)):<12}"
          f"{auto_method(trade, None):<10}{str(effective_config(trade, None)):<8}{pvs[name]:>16,.2f}")
pv0 = sum(pvs.values())
print(f"  book PV = {pv0:,.2f}")
print("  Curve-discounted trades take model=None; the policy picks 'analytic' and no")
print("  config. Swaps and the FRN discount on OIS and project on 3M; bonds name OIS only.")


def book_value(m) -> jax.Array:
    return sum(price(trade, None, m).price for trade in book.values())


# %% 3. market_risk per trade
header("3. market_risk: one backward pass per trade, keyed by declared curves")

risk = {name: market_risk(trade, None, market) for name, trade in book.items()}
for name, r in risk.items():
    assert abs(float(r.price) - pvs[name]) <= 1e-9 * max(1.0, abs(pvs[name]))
    print(f"  {name:<10} declares curves {r.declared.curves}")
for name in BONDS:
    assert set(risk[name].declared.curves) == {OIS}

# ∂PV/∂DF(pillar) of the whole book, per curve, from the per-trade reports.
book_dDF = {
    cid: sum(r.curves[cid] for r in risk.values() if cid in r.curves) for cid in CURVES
}
full = eqx.filter_grad(book_value)(market)
for cid in CURVES:
    gap = float(jnp.max(jnp.abs(full.curves.curves[cid].discount_factors - book_dDF[cid])))
    assert gap <= 1e-6 * max(1.0, float(jnp.max(jnp.abs(book_dDF[cid])))), gap
print("  ✓ Σ per-trade reports = gradient of the whole book (nothing outside the declared curves)")

# Zero-rate DV01 per pillar: ∂PV/∂r_i = ∂PV/∂DF_i · (−t_i DF_i), per +1 bp.
dv01 = {cid: book_dDF[cid] * (-t_pillar * DF[cid]) * 1e-4 for cid in CURVES}
edges = jnp.asarray([b for b in (1.0, 2.0, 5.0, 10.0, 20.0) if b <= float(t_pillar[-1]) + 1e-9])
bmap = tenor_bucket_map(t_pillar, edges)
print(f"\n  Bucketed zero-rate DV01 (PV change per +1 bp):")
print(f"  {'curve':<16}" + "".join(f"{float(e):>11.0f}Y" for e in edges))
for cid in CURVES:
    bucketed = bucketed_curve_deltas(dv01[cid], bmap)
    total = float(jnp.sum(dv01[cid]))
    assert abs(float(jnp.sum(bucketed)) - total) <= 1e-8 * max(1.0, abs(total))
    print(f"  {cid:<16}" + "".join(f"{float(b):12,.0f}" for b in bucketed))

# The same sensitivity by revaluation: ±1 bp parallel shocks through shock().
print()
for cid in CURVES:
    up = shock(market, KeyedScenario(curve_shocks={cid: jnp.full(N_PILLARS, 1e-4)}))
    dn = shock(market, KeyedScenario(curve_shocks={cid: jnp.full(N_PILLARS, -1e-4)}))
    fd = (float(book_value(up)) - float(book_value(dn))) / 2
    ad = float(jnp.sum(dv01[cid]))
    print(f"  parallel +1 bp on {cid:<14}  market_risk {ad:14,.2f}   shock FD {fd:14,.2f}")
    assert abs(ad - fd) <= 1e-5 * max(1.0, abs(fd)) + 1e-2
print("  ✓ the bucketed report sums to the parallel DV01 that full revaluation measures")

# %% 4. The kernel layer
header("4. The kernel layer: price() routes to these functions")

bond = book["bond 5Y"]
kernel_pv = fixed_rate_bond_price(bond, market.curves[OIS])
assert abs(float(kernel_pv) - pvs["bond 5Y"]) <= 1e-9 * abs(pvs["bond 5Y"])
print(f"  price(bond, None, market) = {pvs['bond 5Y']:,.4f}")
print(f"  fixed_rate_bond_price(bond, curve) = {float(kernel_pv):,.4f}   (same function underneath)")

ytm = yield_to_maturity(bond, kernel_pv)
dur = modified_duration(bond, ytm)
cvx = convexity(bond, ytm)
print(f"  YTM {float(ytm):.4%}   modified duration {float(dur):.3f}   convexity {float(cvx):.2f}")

dr = 25e-4
moved = shock(market, KeyedScenario(curve_shocks={OIS: jnp.full(N_PILLARS, dr)}))
actual = float(price(bond, None, moved).price) - float(kernel_pv)
# A parallel continuously-compounded zero-rate move dr is, per semiannual period,
# a yield move dy; the residual gap is act/365 vs 1/2-year period times (~0.3%).
dy = 2.0 * ((1.0 + ytm / 2.0) * jnp.exp(dr / 2.0) - 1.0) - ytm
approx = float(kernel_pv * (-dur * dy + 0.5 * cvx * dy**2))
print(f"  +25 bp curve move: repriced ΔP {actual:,.2f}   duration + convexity {approx:,.2f}")
assert abs(approx - actual) <= 1e-2 * abs(actual)
print("  ✓ yield-space analytics from the kernels explain the curve move price() measures")

# %% 5. A factor model from a curve history
header("5. A factor model from a synthetic curve history (kernel layer)")

body_t = t_pillar[1:]                         # the reference pillar never moves
x = body_t / 2.0                              # Nelson-Siegel loadings, λ = 2Y
ns_slope = (1.0 - jnp.exp(-x)) / x
loadings = jnp.stack([jnp.ones_like(body_t), ns_slope, ns_slope - jnp.exp(-x)], axis=1)
daily_sd = jnp.array([6e-4, 4e-4, 3e-4])      # 6 / 4 / 3 bp per day
k_scores, k_noise = jax.random.split(registry.key("example19.curve_history"))
daily = (jax.random.normal(k_scores, (N_DAYS, 3)) * daily_sd) @ loadings.T
daily = daily + 3e-5 * jax.random.normal(k_noise, (N_DAYS, body_t.shape[0]))
path = jnp.cumsum(daily, axis=0)
history = [bump_curve_zero_rates(curve_ois, jnp.concatenate([jnp.zeros(1), z])) for z in path]

returns = zero_rate_returns_from_snapshots(history, curve_ois.pillar_dates[1:])
pca = fit_rates_pca(returns, body_t, n_components=3)
print(f"  {N_DAYS} days, {body_t.shape[0]} pillars; 3 PCs explain "
      f"{float(pca.fraction_explained):.2%} of the variance")
print(f"  one-sigma daily scores (bp): {[round(float(jnp.sqrt(v)) * 1e4, 2) for v in pca.eigenvalues]}")
print(f"  {'pillar':>8}{'PC1':>9}{'PC2':>9}{'PC3':>9}{'R²':>9}")
for t, row, r2 in zip(body_t.tolist(), pca.jacobian.tolist(), pca.r_squared_per_pillar(returns).tolist()):
    print(f"  {t:>7.2f}Y{row[0]:>+9.3f}{row[1]:>+9.3f}{row[2]:>+9.3f}{r2:>9.2%}")
assert float(pca.fraction_explained) > 0.95
assert bool(jnp.all(pca.jacobian[:, 0] > 0.0))   # PC1 is a parallel-up move

# %% 6. Keyed scenarios
header("6. Keyed scenarios: PCA and basis moves, revalued in one compiled program")

J = pca.jacobian
J_full = jnp.concatenate([jnp.zeros((1, 3)), J], axis=0)   # add the reference pillar
sd = jnp.sqrt(pca.eigenvalues * HORIZON)                  # one-sigma HORIZON-day scores


def keyed(scores: jax.Array, basis_bp) -> KeyedScenario:
    """A PC move on both curves, plus a 3M-vs-OIS basis move on the projection curve."""
    dz = J_full @ scores
    return KeyedScenario(curve_shocks={OIS: dz, PROJ: dz + basis_bp * 1e-4})


@eqx.filter_jit
def revalue(scenarios: KeyedScenario) -> jax.Array:
    return jax.vmap(lambda sc: book_value(shock(market, sc)) - pv0)(scenarios)


e = jnp.eye(3)
named = {
    f"+1σ PC1 (level), {HORIZON}d": keyed(e[0] * sd, 0.0),
    f"-1σ PC1 (level), {HORIZON}d": keyed(-e[0] * sd, 0.0),
    f"+1σ PC2 (slope), {HORIZON}d": keyed(e[1] * sd, 0.0),
    f"+1σ PC3 (curvature), {HORIZON}d": keyed(e[2] * sd, 0.0),
    "+5 bp 3M basis": keyed(jnp.zeros(3), 5.0),
}
named_pnl = revalue(stack_keyed_scenarios(list(named.values())))
for label, pnl in zip(named, named_pnl):
    print(f"  {label:<28} ΔPV {float(pnl):>+16,.2f}")

basis_market = shock(market, named["+5 bp 3M basis"])
for name in BONDS:
    assert abs(float(price(book[name], None, basis_market).price) - pvs[name]) <= 1e-9 * pvs[name]
print("  ✓ a basis-only move leaves the bonds untouched: they declare OIS only")

# Random PC scenarios, as one batched KeyedScenario (leaves carry the scenario axis).
scores = jax.random.normal(registry.key("example19.scenarios"), (N_SCENARIOS, 3)) * sd


def batch(size: float) -> KeyedScenario:
    dz = (size * scores) @ J_full.T
    return KeyedScenario(curve_shocks={OIS: dz, PROJ: dz})


t0 = time.perf_counter()
pnl = revalue(batch(1.0))
pnl.block_until_ready()
t1 = time.perf_counter()
pnl = revalue(batch(1.0))
pnl.block_until_ready()
t2 = time.perf_counter()
print(f"\n  {N_SCENARIOS} PCA scenarios: first call (compile + run) {t1 - t0:.1f}s, "
      f"second call {t2 - t1:.3f}s")


def explained(sc: KeyedScenario) -> jax.Array:
    """First-order P&L from the market_risk report: Σ ∂PV/∂DF · ΔDF."""
    return sum(
        (DF[cid] * jnp.expm1(-sc.curve_shocks[cid] * t_pillar)) @ book_dDF[cid]
        for cid in CURVES
    )


def unexplained_ratio(size: float) -> float:
    full_pnl = revalue(batch(size))
    return float(jnp.sqrt(jnp.mean((full_pnl - explained(batch(size))) ** 2)) / jnp.std(full_pnl))


corr = float(np.corrcoef(np.asarray(pnl), np.asarray(explained(batch(1.0))))[0, 1])
r_full, r_half = unexplained_ratio(1.0), unexplained_ratio(0.5)
print(f"  corr(full revaluation, market_risk explain) = {corr:.6f}")
print(f"  RMS unexplained / σ(P&L): full-size {r_full:.2e}, half-size {r_half:.2e}")
assert corr > 0.99
assert r_half < 0.75 * r_full
print("  ✓ the unexplained part is second order: it halves as the shocks halve")

# PC-score sensitivities: one jax.grad through shock() and price(), checked
# against the chain rule on the market_risk report.
g_pc = jax.grad(lambda s: book_value(shock(market, keyed(s, 0.0))))(jnp.zeros(3))
g_chain = sum((book_dDF[cid] * (-t_pillar * DF[cid])) @ J_full for cid in CURVES)
print(f"  ∂PV/∂(PC score), per 1 bp of score: {[round(float(g) * 1e-4, 2) for g in g_pc]}")
assert jnp.allclose(g_pc, g_chain, rtol=1e-8, atol=1e-6)
print("  ✓ jax.grad through the scenario = market_risk report × PCA loadings")

losses = -np.asarray(pnl)
var99 = np.quantile(losses, 0.99)
es99 = losses[losses >= var99].mean()
print(f"\n  {HORIZON}-day 99% VaR {var99:,.2f}   99% ES {es99:,.2f}   (book PV {pv0:,.2f})")
