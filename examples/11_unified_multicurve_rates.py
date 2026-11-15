"""Example 11 — Multi-curve rates: synthetic quotes → curves → swap book risk.

  1. Sample a true OIS curve and a 3M projection curve (OIS + tenor basis),
     and quote the standard strip on them: deposits, OIS swaps, 3M IBOR swaps.
     Quotes carry Gaussian noise of half a basis point.
  2. Bootstrap both curves jointly as one calibration level; report
     convergence, conditioning and how far the noise moved the curves.
  3. Price a book of swaps that discount on OIS and project on 3M.
  4. Quote risk of the whole book in ONE adjoint sweep (sum the trades'
     market cotangents first), checked against the per-trade sum, the
     autodiff path, and full-recalibration finite differences.
  5. The OIS quotes carry risk even for exposure that sits on the projection
     curve: the 3M curve is solved *through* the OIS curve, so moving an OIS
     quote moves the 3M nodes. Pillar deltas cannot show this; quote risk does.
  6. The single-curve limit: zero basis gives identical curves from the same
     multi-curve plumbing.

Run::

    python examples/11_unified_multicurve_rates.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import calibrate_market, condition_alert
from valax.core.pricing import price
from valax.greeks import (
    chain_adjoint,
    market_gradient,
    quote_risk_adjoint,
    quote_risk_autodiff,
    reconcile_quote_risk,
)
from valax.greeks.quote_risk import tree_add
from valax.market import (
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    rates_snapshot,
    sample_swap_book,
)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
cfg = default_config()
spec = RatesWorldSpec(pillar_years=(1, 2, 5, 10) if FAST else (1, 2, 3, 5, 7, 10, 15, 20))

# %% 1. Synthetic quotes from a true two-curve market
header("1. Synthetic OIS + 3M quotes (0.5 bp noise)")

snap, world = rates_snapshot(registry, cfg, spec, bp_noise=0.5)
curves = snap.parts["curves"]
print(f"  {'key':<16}{'clean par':>12}{'quoted':>12}")
for k in curves.keys:
    print(f"  {k:<16}{float(curves.clean[k]):12.5%}{float(snap.quotes[k]):12.5%}")

# %% 2. Joint bootstrap as one calibration level
header("2. Joint bootstrap")

calibrated = calibrate_market(snap.chain, snap.quotes, strict=True)
res = calibrated.results[0]
kappa = res.condition_number()
print(f"  converged={bool(res.diagnostics.converged)}  max|R|={float(res.diagnostics.max_abs_residual):.1e}"
      f"  steps={int(res.diagnostics.n_steps)}  cond(dR/dθ)={float(kappa):.2e}")
assert condition_alert(kappa, level_name="USD curves") is None
for cid in world.curve_ids:
    fitted = calibrated.market.curves[cid].discount_factors
    true = world.truth[cid].discount_factors
    print(f"  {cid:<14} max |DF_fit - DF_true| = {float(jnp.max(jnp.abs(fitted - true))):.2e}"
          "  (from the quote noise only)")

# %% 3. A swap book priced off both curves
header("3. Swap book (discount OIS, project 3M)")

book = sample_swap_book(registry, world, n_swaps=4 if FAST else 8, off_market_bp=25.0)
pvs = [float(price(s, None, calibrated.market).price) for s in book]
for s, pv in zip(book, pvs):
    years = int(round((int(s.fixed_dates[-1]) - int(world.reference_date)) / 365))
    side = "pay " if s.pay_fixed else "recv"
    print(f"  {side} {years:>2}Y  K={float(s.fixed_rate):.4%}  N={float(s.notional):>12,.0f}"
          f"  PV={pv:>14,.2f}")
book_pv = sum(pvs)
print(f"  book PV = {book_pv:,.2f}")

# %% 4. Book quote risk in one sweep
header("4. Book quote risk: one sweep for the whole book")

book_bar = None
for s in book:
    book_bar = tree_add(book_bar, market_gradient(s, None, calibrated.market))
book_risk = chain_adjoint(calibrated, book_bar)

per_trade = [quote_risk_adjoint(s, None, calibrated) for s in book]
for k in curves.keys:
    total = sum(float(r.quotes[k]) for r in per_trade)
    assert abs(float(book_risk.quotes[k]) - total) <= 1e-8 * max(1.0, abs(total))

# Independent path on the largest trade.
big = int(np.argmax([float(s.notional) for s in book]))
rec = reconcile_quote_risk(
    per_trade[big],
    quote_risk_autodiff(book[big], None, snap.chain, snap.quotes),
    rtol=1e-6, atol=1e-6,
)
assert rec.ok, rec

# Full-recalibration finite differences on two quotes (1 bp bumps).
def book_value(quotes):
    m = calibrate_market(snap.chain, quotes).market
    return sum(float(price(s, None, m).price) for s in book)


print(f"  {'quote':<16}{'∂PV/∂q per bp':>16}{'FD per bp':>16}")
for k in (curves.keys[1], curves.keys[-1]):
    fd = (book_value(snap.quotes.bump(k, 1e-4)) - book_value(snap.quotes.bump(k, -1e-4))) / 2
    ad = float(book_risk.quotes[k]) * 1e-4
    print(f"  {k:<16}{ad:16,.2f}{fd:16,.2f}")
    assert abs(ad - fd) <= 1e-4 * max(1.0, abs(fd)) + 1e-2

print(f"\n  Full quote-sensitivity vector (PV change per +1 bp in each quote):")
for k in curves.keys:
    print(f"    {k:<16}{float(book_risk.quotes[k]) * 1e-4:14,.2f}")
print("  ✓ one sweep = Σ per-trade risk = autodiff = full-revaluation FD")

# %% 5. Indirect discount delta
header("5. Why OIS quotes carry projection risk")

proj = spec.projection_id("3M")
pillar = market_gradient(book[big], None, calibrated.market).curves.curves
ois_pillar = float(jnp.sum(pillar[spec.ois_id].discount_factors))
proj_pillar = float(jnp.sum(pillar[proj].discount_factors))
print(f"  trade {big}: Σ∂PV/∂DF on OIS {ois_pillar:,.0f}, on 3M {proj_pillar:,.0f}")
ois_q = sum(float(per_trade[big].quotes[k]) for k in curves.keys if ".OIS." in k)
irs_q = sum(float(per_trade[big].quotes[k]) for k in curves.keys if ".IRS3M." in k)
print(f"  quote risk: Σ over OIS quotes {ois_q * 1e-4:,.2f}/bp, over 3M swap quotes {irs_q * 1e-4:,.2f}/bp")
print("  The 3M swap quotes absorb the projection exposure; the OIS quotes carry the")
print("  discounting exposure *net* of how the 3M curve moves when OIS moves.")

# %% 6. Single-curve limit
header("6. Single-curve limit (zero basis)")

snap0, world0 = rates_snapshot(registry, cfg, RatesWorldSpec(
    pillar_years=spec.pillar_years, zero_basis=True,
))
cal0 = calibrate_market(snap0.chain, snap0.quotes, strict=True)
gap = float(jnp.max(jnp.abs(
    cal0.market.curves[spec.ois_id].discount_factors
    - cal0.market.curves[proj].discount_factors
)))
print(f"  max |DF_OIS - DF_3M| after calibration = {gap:.1e}")
assert gap < 1e-10
print("  ✓ the multi-curve node reproduces the single-curve market when the basis is zero")
