"""Example 13 — Quote scenarios: recalibrate and reprice a book in one program.

Scenario risk in quote space: shock the *quotes*, recalibrate the whole chain
under every scenario, reprice the book, and compare with the first-order
explanation from the quote risk computed once at the base.

  1. Synthetic equity snapshot (OIS strip, spot, dividend, SABR smiles) and a
     small option book priced off the smile.
  2. Book quote risk at the base: one adjoint sweep with the summed cotangent.
  3. Correlated quote scenarios — rate level/slope, spot, vol level/skew with
     spot-vol correlation — stacked into one ``QuoteSet``.
  4. Full revaluation: ``jax.vmap`` of (calibrate_market → price the book)
     over every scenario, compiled once. Every scenario reports whether every
     level converged.
  5. P&L explain: full revaluation vs Σ q̄·Δq. The unexplained remainder is
     the second-order (gamma / vanna / volga) part; it shrinks with the shock
     size, which is checked by repeating with half-size shocks.
  6. VaR / ES from the full-revaluation P&L vector.

Run::

    python examples/13_unified_quote_scenarios.py
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
from valax.core.pricing import price
from valax.greeks import chain_adjoint, market_gradient
from valax.greeks.quote_risk import tree_add
from valax.market import (
    EquityWorldSpec,
    QuoteSet,
    SeedRegistry,
    default_config,
    equity_snapshot,
    sample_equity_trades,
    sample_quote_scenarios,
)
from valax.models.black_scholes import BlackScholesModel

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
N_SCENARIOS = 64 if FAST else 1000


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
cfg = default_config()
spec = EquityWorldSpec(expiries=(0.5, 1.0) if FAST else (0.25, 0.5, 1.0, 2.0))
model = BlackScholesModel()  # vol read off the calibrated smile at each strike

# %% 1. Snapshot and book
header("1. Snapshot and book")

snap, world = equity_snapshot(registry, cfg, spec, bp_noise=0.5, vol_bp_noise=10.0)
base = calibrate_market(snap.chain, snap.quotes, snap.fixed, strict=True)
book = sample_equity_trades(
    registry, world, n_per_kind=3 if FAST else 6, kinds=("european",),
    expiry_range=(0.4, 1.0),
)
positions = jnp.asarray([1.0, -2.0, 1.5, -1.0, 2.0, 1.0] * 2)[: len(book)]


def book_value(market):
    return sum(n * price(t, model, market).price for n, t in zip(positions, book))


pv0 = float(book_value(base.market))
print(f"  {len(book)} options, book PV {pv0:.4f}")

# %% 2. Book quote risk at the base
header("2. Book quote risk at the base (one sweep)")

book_bar = None
for n, t in zip(positions, book):
    g = market_gradient(t, model, base.market)
    book_bar = tree_add(book_bar, jax.tree.map(lambda x: n * x, g))
risk = chain_adjoint(base, book_bar)
groups = {
    "curve (per bp)": [k for k in snap.parts["curves"].keys],
    "spot": [snap.parts["smile"].spot_key],
    "smile (per vol pt)": [k for row in snap.parts["smile"].vol_keys for k in row],
}
scale = {"curve (per bp)": 1e-4, "spot": 1.0, "smile (per vol pt)": 1e-2}
for name, keys in groups.items():
    print(f"  Σ ∂PV/∂q {name:<20}{sum(float(risk.quotes[k]) for k in keys) * scale[name]:12.5f}")

# %% 3. Quote scenarios
header(f"3. {N_SCENARIOS} correlated quote scenarios")

def scenarios(size: float, stream: str) -> QuoteSet:
    return sample_quote_scenarios(
        registry, snap.quotes, N_SCENARIOS,
        rate_tenors=snap.parts["curves"].tenors,
        vol_coords=snap.parts["smile"].coords,
        spot_keys=(snap.parts["smile"].spot_key,),
        rate_level_bp=10.0 * size, rate_slope_bp=5.0 * size,
        vol_level_bp=150.0 * size, vol_skew_bp=50.0 * size,
        spot_rel=0.03 * size, stream_name=stream,
    )

shocked = scenarios(1.0, "example13.scenarios")
spot_key = snap.parts["smile"].spot_key
print(f"  spot range [{float(jnp.min(shocked[spot_key])):.2f}, "
      f"{float(jnp.max(shocked[spot_key])):.2f}]")

# %% 4. Full revaluation, vmapped and compiled once
header("4. Full revaluation: recalibrate + reprice every scenario")


@eqx.filter_jit
def revalue(quotes: QuoteSet):
    def one(q):
        cal = calibrate_market(snap.chain, q, snap.fixed)
        return book_value(cal.market) - pv0, cal.converged
    return jax.vmap(one)(quotes)


t0 = time.perf_counter()
pnl, converged = revalue(shocked)
pnl.block_until_ready()
t1 = time.perf_counter()
pnl_again, _ = revalue(shocked)
pnl_again.block_until_ready()
t2 = time.perf_counter()
print(f"  first call (compile + run) {t1 - t0:.1f}s, second call {t2 - t1:.3f}s")
print(f"  converged in {int(jnp.sum(converged))}/{N_SCENARIOS} scenarios")
assert bool(jnp.all(converged))

# %% 5. P&L explain
header("5. P&L explain: full revaluation vs first-order quote risk")


def explained(quotes: QuoteSet) -> jax.Array:
    return sum(
        risk.quotes[k] * (quotes[k] - snap.quotes[k]) for k in risk.quotes.keys()
    )


def unexplained_ratio(size: float, stream: str) -> float:
    q = scenarios(size, stream)
    full, _ = revalue(q)
    first = explained(q)
    return float(jnp.sqrt(jnp.mean((full - first) ** 2)) / jnp.std(full))


first = explained(shocked)
corr = float(np.corrcoef(np.asarray(pnl), np.asarray(first))[0, 1])
r_full = unexplained_ratio(1.0, "example13.scenarios")
r_half = unexplained_ratio(0.5, "example13.scenarios")
print(f"  corr(full, explained) = {corr:.4f}")
print(f"  RMS unexplained / σ(P&L): full-size shocks {r_full:.3f}, half-size {r_half:.3f}")
assert corr > 0.95
assert r_half < 0.75 * r_full  # second-order remainder shrinks with the shock
print("  ✓ the unexplained part is second order: it halves as the shocks halve")

# %% 6. VaR / ES from the full-revaluation P&L
header("6. Risk measures from full revaluation")

losses = -np.asarray(pnl)
var99 = np.quantile(losses, 0.99)
es99 = losses[losses >= var99].mean()
print(f"  99% VaR {var99:.4f}   99% ES {es99:.4f}   (book PV {pv0:.4f})")
print("\n  Next steps (roadmap): the same loop over a heterogeneous book grouped by")
print("  structure, with per-group vmaps, once the portfolio layer lands.")
