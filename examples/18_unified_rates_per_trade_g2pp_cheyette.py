"""Example 18 — Rates: per-trade G2++ and Cheyette on the calibrated cube → Bermudan risk.

The two-factor and skewed siblings of example 14's per-trade Hull-White:

  1. Sample a true rates market (OIS + 3M projection curves, a SABR swaption
     cube in normal vols) and quote it with Gaussian noise: 0.5 bp on rates,
     0.5 normal-vol bp on the cube; calibrate curves → cube.
  2. Calibrate G2++ **per trade** by a **co-terminal bootstrap**: each
     Bermudan gets a piecewise-constant first-factor vol σ(t), one bucket per
     exercise date, solved so that every co-terminal European (priced off the
     cube by the Brigo-Mercurio closed form) is repriced exactly. The mean
     reversions (a, b), the correlation ρ and the factor-vol ratio η/σ are
     held fixed as fixed inputs, so their sensitivities are reported. Same-
     shaped baskets are fitted in one ``eqx.filter_vmap``. For comparison, the
     least-squares alternative — two constant vols (σ, η) — needs an extra
     short-tenor option to be well posed and still misses the co-terminals.
  3. Price the Bermudans through ``price`` (G2++ 2-D ADI PDE) and measure the
     grid's own swaption error against the closed form.
  4. Quote risk of one Bermudan by the adjoint sweep (λ_m through cube and
     curves) and independently by autodiff through every calibration and
     the PDE; the basket target shifts give the vega profile, one entry per
     exercise date; one target shift and one cube quote are checked by full
     recalibration.
  5. **Exact vs engine-consistent calibration**: bootstrap the same basket on
     the Bermudan's own PDE grid instead. The closed-form fit has portable
     parameters (provenance ``exact``); the PDE fit reprices its basket
     exactly *as the grid prices it* (``discretised``, config recorded), so
     the grid's swaption bias does not add to its basket miss. Full mode adds
     a coarser grid and a converged reference for the Bermudan.
  6. Cheyette (frozen-y PDE): σ(t) bootstrapped on the co-terminals with the
     mean reversion κ and the skew β held fixed. The Cheyette PDE has no
     discount/projection split, so it runs on a **single-curve** market (OIS
     discounts and projects) and ``price`` refuses a trade that names a
     distinct projection curve. The fit records its engine-bound provenance;
     ``check_engine`` refuses another grid. Adjoint vs. autodiff and an FD
     check (full mode; the test suite runs them too).
  7. Model risk: Hull-White, G2++ and Cheyette, each bootstrapped per trade to
     the same Bermudan's co-terminals on the same single-curve market — every
     model reprices them exactly, so the spread is exercise-value model risk.

Run::

    python examples/18_unified_rates_per_trade_g2pp_cheyette.py
"""

# %% Imports and setup
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import valax
from valax.calibration import (
    CalibrationChain,
    calibrate_market,
    calibrate_model,
    coterminal_swaptions,
    positive,
    swaption_basket_node,
)
from valax.calibration.cheyette import cheyette_provenance
from valax.calibration.result import CalibrationError, check_engine
from valax.core.binding import bind_curves
from valax.core.pricing import price
from valax.greeks import quote_risk_adjoint, quote_risk_autodiff, reconcile_quote_risk
from valax.instruments.rates import BermudanSwaption
from valax.market import (
    QuoteSet,
    RatesVolSpec,
    RatesWorldSpec,
    SeedRegistry,
    default_config,
    rates_vol_snapshot,
)
from valax.models.cheyette import CheyetteModel
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price
from valax.pricing.analytic.rates_smile import swaption_price
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.pricing.pde.config import PDEConfig, PDEConfig2D
from valax.surfaces.constant import ConstantVol

jax.config.update("jax_enable_x64", True)

FAST = os.environ.get("VALAX_EXAMPLES_FAST") == "1"
VOL_NOISE_BP = 0.5
# Bermudan grids: G2++ (two-factor ADI) and Cheyette (frozen-y, one factor).
# The G2++ basket is priced by the closed form; Cheyette's (no closed form)
# on its own grid, which makes its parameters engine-bound.
PDE_G2 = (PDEConfig2D(n_x=41, n_y=41, n_time=40, x_range=5.0) if FAST
          else PDEConfig2D(n_x=81, n_y=81, n_time=100, x_range=5.0))
PDE_CH = (PDEConfig(n_spot=101, n_time=60, spot_range=5.0) if FAST
          else PDEConfig(n_spot=201, n_time=150, spot_range=5.0))
PDE_HW = PDEConfig(n_spot=100, n_time=60) if FAST else PDEConfig(n_spot=200, n_time=150)
# Fixed inputs: G2++ mean reversions, correlation and factor-vol ratio η/σ;
# Cheyette κ and skew β; Hull-White mean reversion.
A_KEY, B_KEY, RHO_KEY, RATIO_KEY = "USD.G2.A", "USD.G2.B", "USD.G2.RHO", "USD.G2.RATIO"
KAPPA_KEY, BETA_KEY, HW_A_KEY = "USD.CHEY.KAPPA", "USD.CHEY.BETA", "USD.HW.MEAN_REVERSION"
G2_FIXED = {A_KEY: 0.8, B_KEY: 0.03, RHO_KEY: -0.7, RATIO_KEY: 1.6}
N_EX = 3  # exercise dates (2Y, 3Y, 4Y) = co-terminals = σ(t) buckets


def header(s: str) -> None:
    print(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


registry = SeedRegistry(master_seed=20260214, library_version=valax.__version__)
vol_spec = RatesVolSpec(
    expiry_years=(1, 2, 5) if FAST else (1, 2, 3, 5),
    tenor_years=(2, 5) if FAST else (2, 3, 5, 7),
    cap_years=(1,),
)


def market_chain(rates_spec: RatesWorldSpec, extra_fixed: dict):
    """Noisy snapshot → calibrated curves → cube (the market levels)."""
    snap, world = rates_vol_snapshot(
        registry, default_config(), rates_spec, vol_spec,
        bp_noise=0.5, vol_bp_noise=VOL_NOISE_BP,
    )
    chain = CalibrationChain(nodes=snap.chain.nodes[:2])  # curves → cube
    quotes = snap.quotes.select(chain.quote_keys)
    fixed = snap.fixed.merge(QuoteSet.from_dict(extra_fixed))
    calibrated = calibrate_market(chain, quotes, fixed, strict=True)
    return snap, world, chain, quotes, fixed, calibrated


def bermudan(world, strike: float, *, multi_curve: bool = True) -> BermudanSwaption:
    """2Y-into-5Y Bermudan payer, annual exercise at 2Y, 3Y, 4Y."""
    ref = int(world.rates.reference_date)
    return BermudanSwaption(
        exercise_dates=jnp.array([ref + 365 * k for k in (2, 3, 4)], dtype=jnp.int32),
        fixed_dates=jnp.array([ref + 365 * k for k in range(3, 8)], dtype=jnp.int32),
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1.0),
        day_count=world.rates.day_count,
        discount_id=world.discount_id,
        forward_id=world.forward_id if multi_curve else None,
    )


def lsq_basket(trade) -> tuple:
    """Co-terminal swaptions + the 1Y-tenor swaption at the first exercise.

    For the two-constant-vol least-squares fit only: the co-terminal set pins
    the overall vol level; the short-tenor option (one coupon of the first
    exercise's underlying) pins the split between the fast (a) and slow (b)
    factor, which the co-terminals alone cannot.
    """
    co = coterminal_swaptions(trade)
    short = eqx.tree_at(lambda s: s.fixed_dates, co[0], co[0].fixed_dates[:1])
    return co + (short,)


def target_keys(name: str, n: int) -> tuple[str, ...]:
    return tuple(f"{name}.TARGET.{i}" for i in range(n))


G2_TEMPLATE = G2PPModel(
    mean_reversion_x=jnp.asarray(0.8), mean_reversion_y=jnp.asarray(0.03),
    volatility_x=jnp.asarray(0.006), volatility_ratio=jnp.asarray(1.6),
    correlation=jnp.asarray(-0.7),
)


def g2_node(basket, cube_id, name="g2", method="analytic", config=None):
    """Per-trade G2++ bootstrap: σ(t) per co-terminal, a/b/ρ/ratio held.

    The basket engine is ``(method, config)``.
    """
    keys = target_keys(name, len(basket))
    node = swaption_basket_node(
        name,
        template=G2_TEMPLATE,
        basket=basket,
        cube_id=cube_id,
        transforms={"volatility_x": positive()},
        fixed_fields={"mean_reversion_x": A_KEY, "mean_reversion_y": B_KEY,
                      "correlation": RHO_KEY, "volatility_ratio": RATIO_KEY},
        target_keys=keys,
        mode="coterminal_bootstrap",
        method=method,
        config=config,
    )
    return node, keys


def g2_lsq_node(basket, cube_id, name="g2lsq"):
    """The least-squares alternative: constant σ and ratio, a/b/ρ held."""
    keys = target_keys(name, len(basket))
    node = swaption_basket_node(
        name,
        template=G2_TEMPLATE,
        basket=basket,
        cube_id=cube_id,
        transforms={"volatility_x": positive(), "volatility_ratio": positive()},
        fixed_fields={"mean_reversion_x": A_KEY, "mean_reversion_y": B_KEY,
                      "correlation": RHO_KEY},
        target_keys=keys,
    )
    return node, keys


def tree_index(tree, i):
    return jax.tree.map(lambda x: x[i], tree)


def buckets(v) -> str:
    return "  ".join(f"{float(s):.5f}" for s in np.ravel(v))


# %% 1. Multi-curve market
header("1. Noisy quotes → curves → swaption cube (multi-curve: OIS + 3M)")

rates_spec = RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10))
g2_fixed = {
    **G2_FIXED,
    **{k: 0.0 for k in target_keys("g2", N_EX)},
    **{k: 0.0 for k in target_keys("g2pde", N_EX)},
    **{k: 0.0 for k in target_keys("g2lsq", N_EX + 1)},
    **{k: 0.0 for k in target_keys("g2lsqco", N_EX)},
}
snap, world, chain, quotes, fixed, calibrated = market_chain(rates_spec, g2_fixed)
vols = snap.parts["vols"]
cube = calibrated.market.surfaces[vol_spec.cube_id]
cube_fit = calibrated.result(vols.cube_node.name)
rmse_bp = float(jnp.sqrt(jnp.mean(cube_fit.diagnostics.rmse ** 2))) * 1e4
print(f"  {len(snap.parts['curves'].keys)} curve quotes, cube cells {cube_fit.labels}")
print(f"  cube fit RMSE {rmse_bp:.2f} normal-vol bp (quote noise {VOL_NOISE_BP} bp)")
assert rmse_bp < 1.5 * VOL_NOISE_BP

# %% 2. Per-trade G2++, batched
header("2. G2++ σ(t) bootstrapped per trade (a, b, ρ, η/σ fixed), closed-form basket")

ref = int(world.rates.reference_date)
atm = float(jnp.mean(cube.forwards))
trades = [bermudan(world, atm), bermudan(world, atm + 0.004)]
baskets = [coterminal_swaptions(t) for t in trades]
assert all(len(b) == N_EX for b in baskets)


def g2_fit(basket):
    node, _ = g2_node(basket, vol_spec.cube_id)
    return node.calibrate(QuoteSet(), calibrated.market, fixed)


stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *baskets)
batched = eqx.filter_vmap(g2_fit)(stacked)
assert bool(jnp.all(batched.diagnostics.converged))
assert batched.value.initial_curve is None  # curve-free: price binds the curve
assert batched.value.volatility_x.shape == (len(trades), N_EX)
# The basket is a traced argument (breakpoints included), so one compiled fit
# serves every trade; Newton to 1e-12 per bucket, so batched and one-by-one
# fits agree to round-off.
g2_fit_one = eqx.filter_jit(g2_fit)
print("  σ(t) buckets [0, 2Y) [2Y, 3Y) [3Y, ∞) — breakpoints at the exercise dates")
for i, (t, b) in enumerate(zip(trades, baskets)):
    one = g2_fit_one(b)
    np.testing.assert_allclose(batched.value.volatility_x[i], one.value.volatility_x,
                               rtol=1e-10)
    miss = float(jnp.max(jnp.abs(batched.diagnostics.residuals[i])))
    print(f"  trade {i}: K={float(t.strike):.3%}  σ(t) = {buckets(batched.value.volatility_x[i])}"
          f"  (η = {G2_FIXED[RATIO_KEY]}·σ)  max basket miss {miss:.0e}")
    assert miss < 1e-10
print(f"  filter_vmap fits == trade-by-trade fits (rtol 1e-10); provenance "
      f"{one.provenance.method} / {one.provenance.engine_kind}")
assert one.provenance.engine_kind == "exact"

# The least-squares alternative for trade 0: two constant vols (σ, η = rσ).
# On the co-terminals alone the two factors cannot be told apart (the level
# is near-singular); the extra 2Yx1Y option makes it well posed, but two
# parameters still cannot reprice the co-terminals.
lsq_node, _ = g2_lsq_node(lsq_basket(trades[0]), vol_spec.cube_id)
lsq = lsq_node.calibrate(QuoteSet(), calibrated.market, fixed)
co_lsq_node, _ = g2_lsq_node(baskets[0], vol_spec.cube_id, name="g2lsqco")
co_cond = float(co_lsq_node.calibrate(QuoteSet(), calibrated.market, fixed).condition_number())
lsq_cond = float(lsq.condition_number())
boot_cond = float(g2_fit_one(baskets[0]).condition_number())
print(f"  least squares, constant σ={float(lsq.value.volatility_x[0]):.5f} "
      f"η={float(lsq.value.volatility_y[0]):.5f} (ratio "
      f"{float(lsq.value.volatility_ratio[0]):.2f}): basket RMSE "
      f"{float(lsq.diagnostics.rmse):.1e} (relative)")
print(f"  cond(∂R/∂θ): bootstrap {boot_cond:.1e}; least squares {lsq_cond:.1e} with the "
      f"2Yx1Y option, {co_cond:.1e} on the co-terminals alone")
assert lsq_cond < 1e5 < co_cond

# %% 3. Price the book
header("3. Bermudans through price() (G2++ 2-D PDE); the grid's own swaption error")

models = [tree_index(batched.value, i) for i in range(len(trades))]
disc = calibrated.market.curve(world.discount_id)
fwd = calibrated.market.curve(world.forward_id)
for i, (t, m, b) in enumerate(zip(trades, models, baskets)):
    pv = float(price(t, m, calibrated.market, method="pde", config=PDE_G2).price)
    euro = max(float(price(sw, m, calibrated.market, method="pde", config=PDE_G2).price)
               for sw in b)
    print(f"  trade {i}: Bermudan {pv * 1e4:8.2f} bp  ≥ max co-terminal European {euro * 1e4:8.2f} bp")
    assert pv >= euro - 1e-6
worst = 0.0
for sw in baskets[0]:
    mkt = float(swaption_price(sw, disc, cube, fwd))
    pde = float(price(sw, models[0], calibrated.market, method="pde", config=PDE_G2).price)
    exact = float(g2pp_swaption_price(sw, bind_curves(models[0], disc), fwd))
    assert abs(exact / mkt - 1.0) < 1e-10  # the bootstrap reprices the cube exactly
    worst = max(worst, abs(pde / exact - 1.0))
    tenor = (int(sw.fixed_dates[-1]) - int(sw.expiry_date)) / 365
    print(f"    {(int(sw.expiry_date) - ref) / 365:.0f}Yx{tenor:.0f}Y: cube {mkt * 1e4:7.2f} bp"
          f"  Brigo-Mercurio {exact * 1e4:7.2f} bp  G2++ PDE {pde * 1e4:7.2f} bp")
print(f"  max |PDE / closed form - 1| = {worst:.1e}: the Bermudan's grid misprices the")
print("  basket the model was fitted to by this much (see section 5)")
assert worst < (1e-2 if FAST else 2e-3)

# %% 4. Quote risk of trade 0
header("4. Quote risk of trade 0: adjoint (λ_m) vs autodiff, vega profile, FD")

trade, basket = trades[0], baskets[0]
node, keys = g2_node(basket, vol_spec.cube_id)
model_result = calibrate_model(node, quotes, calibrated.market, fixed, strict=True)
adj = quote_risk_adjoint(
    trade, None, calibrated, model_result=model_result, method="pde", config=PDE_G2
)
ad = quote_risk_autodiff(
    trade, None, chain, quotes, fixed, model_node=node, method="pde", config=PDE_G2
)
rec = reconcile_quote_risk(adj, ad, rtol=1e-6, atol=1e-10)
print(f"  adjoint vs autodiff: max |Δ| = {rec.max_abs_diff:.1e}  ok={rec.ok}")
assert rec.ok, rec
print("  level stationarity: "
      + ", ".join(f"{n}={float(v):.0e}" for n, v in adj.level_residuals.items()))

q = adj.quotes
curve_keys = snap.parts["curves"].keys
print(f"  rho: Σ ∂V/∂(curve quotes) per bp {sum(float(q[k]) for k in curve_keys) * 1e4 * 1e-4:+.4f} bp")
print(f"  fixed inputs: ∂V/∂a {float(adj.fixed[A_KEY]):+.2e}  ∂V/∂b {float(adj.fixed[B_KEY]):+.2e}"
      f"  ∂V/∂ρ {float(adj.fixed[RHO_KEY]):+.2e}  ∂V/∂(η/σ) {float(adj.fixed[RATIO_KEY]):+.2e}")
assert float(adj.fixed[RATIO_KEY]) != 0.0  # the held factor split is a live risk input

print("  λ_m vega profile across the basket (∂V per 1 normal-vol bp of each co-terminal):")
profile = []
for k, sw in zip(keys, basket):
    def basket_price(v, _sw=sw):
        return swaption_price(_sw, disc, ConstantVol(vol=v, is_normal=True), fwd)

    s_k, _ = swaption_forward_and_annuity(sw, disc, fwd)
    expiry = (int(sw.expiry_date) - ref) / 365.0
    tenor = (int(sw.fixed_dates[-1]) - int(sw.expiry_date)) / 365.0
    vega_k = jax.grad(basket_price)(cube(sw.strike, expiry, tenor, forward=s_k))
    profile.append(float(adj.fixed[k]) * float(vega_k) * 1e-4)
    print(f"    {expiry:.0f}Yx{tenor:.0f}Y:  ∂V/∂T = {float(adj.fixed[k]):+.4f}"
          f"   → {profile[-1] * 1e4:+.4f} bp per vol bp")
assert len(profile) == len(trade.exercise_dates)  # one λ_m entry per exercise date
assert model_result.labels == keys
cube_total = sum(float(q[k]) for row in vols.swaption_keys for ks in row for k in ks) * 1e-4
print(f"  total vega: cube quotes {cube_total * 1e4:+.4f} bp  vs  Σ basket profile "
      f"{sum(profile) * 1e4:+.4f} bp per vol bp")
np.testing.assert_allclose(cube_total, sum(profile), rtol=2e-2)
assert cube_total > 0.0  # long a payer Bermudan: long vol


@eqx.filter_jit
def pv_fn(qq, pp):
    """Full recalibration: curves → cube → per-trade G2++ → Bermudan PDE."""
    market = calibrate_market(chain, qq, pp).market
    return price(trade, node.calibrate(qq, market, pp).value, market,
                 method="pde", config=PDE_G2).price


vol_key = vols.swaption_keys[vol_spec.expiry_years.index(2)][vol_spec.tenor_years.index(5)][
    vol_spec.strike_offsets_bp.index(0.0)]  # 2Yx5Y ATM: the cell the 2Yx5Y basket option reads
# The PDE smooths its payoff and exercise projection over each node's cell
# (PDEConfig2D.smooth_payoff, the default), so the discrete price is C^1 in its
# inputs and an ordinary step works: no grid kink to straddle, and at h = 1e-5
# the recalibration's stopping noise is negligible against the slope.
# Adjoint and autodiff are the exact derivative of the discrete map.
for kind, key, h in (("p", keys[1], 1e-5), ("q", vol_key, 1e-5), ("p", RATIO_KEY, 1e-5)):
    if kind == "p":
        up = pv_fn(quotes, fixed.with_values({key: fixed[key] + h}))
        dn = pv_fn(quotes, fixed.with_values({key: fixed[key] - h}))
        got = float(adj.fixed[key])
    else:
        up = pv_fn(quotes.with_values({key: quotes[key] + h}), fixed)
        dn = pv_fn(quotes.with_values({key: quotes[key] - h}), fixed)
        got = float(adj.quotes[key])
    fd = float((up - dn) / (2 * h))
    print(f"  full-recalibration FD ∂V/∂{key}: {fd:+.6f}  vs adjoint {got:+.6f}"
          f"  (rel {got / fd - 1:+.1e})")
    assert abs(fd) > 1e-4, key  # a live sensitivity, not 0 == 0
    np.testing.assert_allclose(got, fd, rtol=2e-4)

# %% 5. Exact vs engine-consistent calibration
header("5. Exact (closed-form) vs engine-consistent (PDE-grid) bootstrap of trade 0")

# The same co-terminals, bootstrapped on the Bermudan's own grid: every one
# is repriced exactly *as this grid prices it*. Recorded as discretised, with
# the grid; its σ(t) differs from the exact fit's by the grid's swaption error.
pde_node, pde_keys = g2_node(basket, vol_spec.cube_id, name="g2pde",
                             method="pde", config=PDE_G2)
pde_fit = pde_node.calibrate(quotes, calibrated.market, fixed)
assert bool(pde_fit.converged)
exact_fit = model_result
print(f"  {'basket engine':<22}{'σ(t) buckets':>28}   provenance")
for label, res in (("closed form", exact_fit), ("PDE (Bermudan grid)", pde_fit)):
    p = res.provenance
    print(f"  {label:<22}{buckets(res.value.volatility_x):>28}   {p.method} / {p.engine_kind}")
assert exact_fit.provenance.engine_kind == "exact"
assert pde_fit.provenance.engine_kind == "discretised"
assert pde_fit.provenance.config == PDE_G2
rel = np.max(np.abs(np.asarray(pde_fit.value.volatility_x)
                    / np.asarray(exact_fit.value.volatility_x) - 1.0))
assert rel < 5e-2, rel


def basket_miss_on(model, cfg):
    """Max relative basket miss vs the cube, priced on grid ``cfg``."""
    return max(
        abs(float(price(sw, model, calibrated.market, method="pde", config=cfg).price)
            / float(swaption_price(sw, disc, cube, fwd)) - 1.0)
        for sw in basket
    )


def bermudan_on(model, cfg):
    return float(price(trade, model, calibrated.market, method="pde", config=cfg).price)


miss_exact, miss_pde = (basket_miss_on(r.value, PDE_G2) for r in (exact_fit, pde_fit))
v_exact, v_pde = (bermudan_on(r.value, PDE_G2) for r in (exact_fit, pde_fit))
print(f"  max basket miss as the Bermudan's grid prices it: exact fit {miss_exact:.1e} "
      f"(the grid's swaption error), PDE fit {miss_pde:.0e} (exact on its engine)")
assert miss_pde < 1e-10 < miss_exact
print(f"  Bermudan on that grid: exact fit {v_exact * 1e4:.3f} bp, PDE fit {v_pde * 1e4:.3f} bp "
      f"(Δ {(v_pde - v_exact) * 1e4:+.3f} bp)")
if not FAST:
    # Converged reference: the exact fit's Bermudan on a much finer grid.
    fine = PDEConfig2D(n_x=161, n_y=161, n_time=200, x_range=5.0)
    v_ref = bermudan_on(exact_fit.value, fine)
    coarse = PDEConfig2D(n_x=41, n_y=41, n_time=40, x_range=5.0)
    coarse_fit = g2_node(basket, vol_spec.cube_id, name="g2pde",
                         method="pde", config=coarse)[0].calibrate(
        quotes, calibrated.market, fixed)
    print(f"  converged reference (exact fit, {fine.n_x}x{fine.n_y}x{fine.n_time}): "
          f"{v_ref * 1e4:.3f} bp; Bermudan error by grid:")
    for cfg, on_engine in ((coarse, coarse_fit), (PDE_G2, pde_fit)):
        e_exact = bermudan_on(exact_fit.value, cfg) - v_ref
        e_pde = bermudan_on(on_engine.value, cfg) - v_ref
        print(f"    {cfg.n_x}x{cfg.n_y}x{cfg.n_time}: exact fit {e_exact * 1e4:+.3f} bp,"
              f" engine-consistent fit {e_pde * 1e4:+.3f} bp")
        # The grid's Bermudan error largely cancels against its basket's.
        assert abs(e_pde) < abs(e_exact), (cfg, e_pde, e_exact)
print("  The closed form gives portable parameters and smooth, cheap risk; the")
print("  engine-consistent fit trades that for a Bermudan priced against a basket")
print("  repriced exactly on its own grid. The provenance records which one each is.")

# %% 6. Cheyette on a single-curve market
header("6. Cheyette σ(t) bootstrapped per trade, κ and β fixed (single-curve, frozen-y PDE)")

# Cheyette's PDE recipes take no projection curve: on the multi-curve market
# price() refuses rather than silently dropping the 3M basis.
probe = CheyetteModel(mean_reversion=jnp.asarray(0.03), volatility=jnp.asarray(0.008),
                      skew=jnp.asarray(0.0))
try:
    price(trade, probe, calibrated.market, method="pde", config=PDE_CH)
    raise AssertionError("Cheyette priced a multi-curve trade")
except ValueError as exc:
    print(f"  multi-curve trade refused: {str(exc).splitlines()[0][:76]}...")

sc_fixed = {
    KAPPA_KEY: 0.03, BETA_KEY: -0.25, HW_A_KEY: 0.03, **G2_FIXED,
    **{k: 0.0 for k in target_keys("ch", N_EX)},
    **{k: 0.0 for k in target_keys("g2", N_EX)},
    **{k: 0.0 for k in target_keys("hw", N_EX)},
}
sc_snap, sc_world, sc_chain, sc_quotes, sc_fixed_q, sc_cal = market_chain(
    RatesWorldSpec(pillar_years=(1, 2, 3, 5, 7, 10), projection_tenors=()), sc_fixed,
)
assert sc_world.forward_id == sc_world.discount_id  # OIS discounts and projects
sc_cube = sc_cal.market.surfaces[vol_spec.cube_id]
sc_trade = bermudan(sc_world, float(jnp.mean(sc_cube.forwards)), multi_curve=False)
sc_basket = coterminal_swaptions(sc_trade)
ch_keys = target_keys("ch", N_EX)
ch_node = swaption_basket_node(
    "ch",
    template=CheyetteModel(mean_reversion=jnp.asarray(0.03), volatility=jnp.asarray(0.008),
                           skew=jnp.asarray(-0.25)),
    basket=sc_basket,
    cube_id=vol_spec.cube_id,
    transforms={"volatility": positive()},
    fixed_fields={"mean_reversion": KAPPA_KEY, "skew": BETA_KEY},
    target_keys=ch_keys,
    mode="coterminal_bootstrap",
    method="pde",
    config=PDE_CH,
)
ch_res = calibrate_model(ch_node, sc_quotes, sc_cal.market, sc_fixed_q, strict=True)
ch = ch_res.value
assert ch.initial_curve is None
ch_miss = float(jnp.max(jnp.abs(ch_res.diagnostics.residuals)))
print(f"  σ(t) = {buckets(ch.volatility)}  β={float(ch.skew):+.2f} (held)  max basket miss "
      f"{ch_miss:.0e} on its grid")
assert ch_miss < 1e-10
prov = ch_res.provenance
assert prov == cheyette_provenance("frozen", PDE_CH)
print(f"  provenance recorded by the node: {prov.method} / {prov.engine_kind}"
      " (frozen-y parameters are engine-bound)")
assert check_engine(ch_res, "pde", PDE_CH) is ch_res
finer = PDEConfig(n_spot=2 * PDE_CH.n_spot, n_time=2 * PDE_CH.n_time, spot_range=5.0)
try:
    check_engine(ch_res, "pde", finer)
    raise AssertionError("an engine-bound fit was accepted on another grid")
except CalibrationError as exc:
    print(f"  check_engine on a finer grid refused: {str(exc)[:68]}...")
ch_pv = float(price(sc_trade, ch, sc_cal.market, method="pde", config=PDE_CH).price)
ch_euro = max(float(price(sw, ch, sc_cal.market, method="pde", config=PDE_CH).price)
              for sw in sc_basket)
print(f"  Bermudan {ch_pv * 1e4:.2f} bp ≥ max co-terminal European {ch_euro * 1e4:.2f} bp")
assert ch_pv >= ch_euro - 1e-6

if FAST:
    print("  (adjoint vs autodiff and the FD check of the Cheyette level run in full")
    print("   mode and in tests/test_calibration/test_coterminal_bootstrap.py)")
else:
    ch_adj = quote_risk_adjoint(sc_trade, None, sc_cal, model_result=ch_res,
                                method="pde", config=PDE_CH)
    ch_ad = quote_risk_autodiff(sc_trade, None, sc_chain, sc_quotes, sc_fixed_q,
                                model_node=ch_node, method="pde", config=PDE_CH)
    ch_rec = reconcile_quote_risk(ch_adj, ch_ad, rtol=1e-6, atol=1e-10)
    print(f"  adjoint vs autodiff: max |Δ| = {ch_rec.max_abs_diff:.1e}  ok={ch_rec.ok}"
          f"  (stationarity {float(ch_adj.level_residuals['ch']):.0e})")
    assert ch_rec.ok, ch_rec
    ch_pv_fn = eqx.filter_jit(lambda pp: price(
        sc_trade, ch_node.calibrate(sc_quotes, sc_cal.market, pp).value,
        sc_cal.market, method="pde", config=PDE_CH,
    ).price)
    k1, h = ch_keys[1], 1e-6
    fd = float((ch_pv_fn(sc_fixed_q.with_values({k1: h}))
                - ch_pv_fn(sc_fixed_q.with_values({k1: -h}))) / (2 * h))
    print(f"  FD ∂V/∂T_1 {fd:+.6f} vs adjoint {float(ch_adj.fixed[k1]):+.6f};"
          f"  ∂V/∂κ {float(ch_adj.fixed[KAPPA_KEY]):+.2e}  ∂V/∂β {float(ch_adj.fixed[BETA_KEY]):+.2e}")
    np.testing.assert_allclose(float(ch_adj.fixed[k1]), fd, rtol=5e-4)

# %% 7. Model risk on one trade
header("7. Model risk: HW vs G2++ vs Cheyette, each bootstrapped on the same co-terminals")

hw_node = swaption_basket_node(
    "hw",
    template=HullWhiteModel(mean_reversion=jnp.asarray(0.03), volatility=jnp.asarray(0.008)),
    basket=sc_basket, cube_id=vol_spec.cube_id,
    transforms={"volatility": positive()}, fixed_fields={"mean_reversion": HW_A_KEY},
    target_keys=target_keys("hw", N_EX), mode="coterminal_bootstrap",
)
sc_g2_node, _ = g2_node(sc_basket, vol_spec.cube_id)
fits = {
    "Hull-White": (calibrate_model(hw_node, sc_quotes, sc_cal.market, sc_fixed_q, strict=True),
                   PDE_HW, "Jamshidian"),
    "G2++": (calibrate_model(sc_g2_node, sc_quotes, sc_cal.market, sc_fixed_q, strict=True),
             PDE_G2, "Brigo-Mercurio"),
    "Cheyette": (ch_res, PDE_CH, "frozen-y PDE"),
}
sc_disc = sc_cal.market.curve(sc_world.discount_id)
pvs, misses = {}, {}
for name, (res, cfg, engine) in fits.items():
    m = res.value
    assert float(jnp.max(jnp.abs(res.diagnostics.residuals))) < 1e-10  # exact on its engine
    pvs[name] = float(price(sc_trade, m, sc_cal.market, method="pde", config=cfg).price)
    misses[name] = max(
        abs(float(price(sw, m, sc_cal.market, method="pde", config=cfg).price)
            / float(swaption_price(sw, sc_disc, sc_cube, None)) - 1.0)
        for sw in sc_basket
    )
    print(f"  {name:10s} Bermudan {pvs[name] * 1e4:8.2f} bp   basket exact on {engine:15s}"
          f" max miss on the Bermudan's grid {misses[name]:.0e}")
spread = (max(pvs.values()) - min(pvs.values())) / pvs["Hull-White"]
print(f"  model spread {spread:.2%} of the HW price. Every model reprices the co-terminal")
print("  Europeans exactly on its calibration engine (what is left above is each Bermudan")
print("  grid's own swaption error), so the spread is the exercise-value model risk the")
print("  Europeans do not pin down.")
assert all(v < 1e-2 for v in misses.values()), misses
assert spread < 0.10
print("\n  All checks passed.")
