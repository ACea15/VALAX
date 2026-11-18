# Pricing

VALAX ships with runnable example scripts in the `examples/` directory. Each file uses `# %%` cell markers for interactive execution in VS Code, PyCharm, or any IDE that supports the Python cell format.

## Two layers, both supported

VALAX has two public API layers, and the examples cover both on purpose:

- **The kernel layer**: the pricing functions themselves (`black_scholes_price`, `fixed_rate_bond_price`, `mc_price`, `pde_price`, `binomial_price`, the path generators, the `valax.risk` building blocks). They take explicit arguments, with no market container and no dispatch, so you can compose them freely: a custom payoff on generated paths, a `jax.grad` through your own pipeline, or a quantity `price()` does not return (a yield, a duration, a factor model). Examples **01–06** stay on this layer, as do [`comparisons/`](https://github.com/ACea15/VALAX/tree/main/examples/comparisons) and the [pitfalls](#pitfalls-things-that-do-not-work-and-why).
- **The unified layer**: `price(instrument, model, market)` on a keyed `MarketState`, with risk from `valax.greeks.market_risk` and quote risk through the calibration chain. It routes to the kernels above and does not replace them. Examples **10 onward** use it, and several (10, 14, 18, 19) drop down to the kernels where that is the clearer tool.

The contract behind the unified layer is the [Unified Pricing tutorial](guide/tutorial-unified-pricing.md). The kernel layer is not legacy. An example is retired only when an API it is built on is scheduled for removal (see [Retired examples](#retired-examples)).

## Running Examples

```bash
# Run any example directly
python examples/01_equity_options.py

# Or open in your IDE and run cells interactively (Shift+Enter in VS Code)
```

## Example Index

| File | Topics | What You'll Learn |
|------|--------|-------------------|
| [`01_equity_options.py`](#equity-options) | Black-Scholes, Greeks, implied vol, portfolio vmap | Defining instruments, pricing, all Greeks via autodiff, portfolio-level vectorization, JIT compilation |
| [`02_sabr_smile.py`](#sabr-smile) | SABR model, vol smile, calibration | Generating smiles, parameter sensitivity (rho/nu/beta), model Greeks, fitting to market data with LM/BFGS |
| [`03_fixed_income.py`](#fixed-income) | Curves, bonds, duration, KRD | Building discount curves, bond pricing, YTM, duration/convexity/key-rate durations via autodiff, DV01 |
| [`04_rates_derivatives.py`](#rates-derivatives) | Caps, floors, swaps, swaptions | Caplet pricing (Black-76 + Bachelier), cap strips, swap NPV and par rates, swaption pricing, rate Greeks |
| [`05_monte_carlo.py`](#monte-carlo) | GBM, Heston, SABR paths, exotics | Path generation, convergence analysis, Asian and barrier options, Heston smile extraction |
| [`06_pde_and_lattice.py`](#pde-and-lattice) | Crank-Nicolson, binomial trees | PDE grid convergence, CRR trees, American vs European puts, early exercise premium, method comparison |
| [`10_unified_market_tiers.py`](#unified-calibration-pricing-and-risk) | Unified quotes → calibrate → price → quote risk | One option on three markets — flat Black-Scholes, bootstrapped curve + SABR smile, + calibrated Heston — through the same workflow |
| [`11_unified_multicurve_rates.py`](#unified-calibration-pricing-and-risk) | Multi-curve bootstrap, swap book, quote risk | Synthetic OIS + 3M quotes → joint bootstrap → swap book → book quote risk in one adjoint sweep (vs autodiff and FD), single-curve limit |
| [`12_unified_equity_smile_and_model.py`](#unified-calibration-pricing-and-risk) | Equity smiles, Heston, MC quote risk | Noisy equity quotes → curve, SABR smiles, Heston → European / barrier / Asian prices → Monte Carlo quote risk with fixed-input sensitivities |
| [`13_unified_quote_scenarios.py`](#unified-calibration-pricing-and-risk) | Quote scenarios, VaR / ES | Correlated quote scenarios → `vmap` recalibration + repricing compiled once → P&L explain vs first-order quote risk → VaR / ES |
| [`14_unified_rates_per_trade_model.py`](#per-trade-model-calibration-for-bermudans) | Swaption cube, per-trade Hull–White, Bermudan risk | Curves → swaption cube → per-trade Hull–White σ(t) bootstrapped on co-terminal baskets (`filter_vmap`) → Bermudan via `price()` PDE → quote risk; adjoint vs autodiff; λ_m vega profile vs full recalibration |
| [`15_unified_rates_caps_optionlet_chain.py`](#caps-and-floors-off-the-optionlet-chain) | Cap strip, optionlet SABR, cap/floor risk | Curves → cap→caplet strip → optionlet SABR as chain nodes → caps / floors through `price()` → quote risk to cap and curve quotes; adjoint vs autodiff vs FD |
| [`16_unified_calibration_diagnostics.py`](#diagnosing-a-failed-calibration) | Failure naming, conditioning, engine provenance | Bad quotes (NaN, off-market, stale) → `check()` / `failed_labels()` name the cell or bucket; healthy nodes unaffected; ill-conditioning alert; `check_engine` refusing a frozen-y Cheyette fit on another engine; fix and recalibrate |
| [`17_unified_slv_fixed_point.py`](#slv-as-a-fixed-point-level) | SLV leverage, fixed-point diagnostics, MC exotic risk | Smile → Heston → SLV leverage as a fixed-point level (max\|ΔL\|, `converged`, clip fraction vs `n_iterations`) → barrier under LV / Heston / SLV → vega by AAD through the unrolled passes vs FD |
| [`18_unified_rates_per_trade_g2pp_cheyette.py`](#per-trade-g2-and-cheyette) | Per-trade G2++ and Cheyette, model risk | Per-trade G2++ σ(t) (multi-curve, closed-form basket, 2-D PDE Bermudan) and Cheyette σ(t) (single-curve, frozen-y PDE), both bootstrapped → quote risk reconciled; exact vs engine-consistent G2++ calibration; HW vs G2++ vs Cheyette Bermudan on the same basket |
| [`19_unified_book_workflow.py`](#a-rates-book-end-to-end) | Rates book, `market_risk`, kernels, PCA + keyed scenarios | OIS + 3M quotes → swaps / bonds / FRN through `price()` → per-trade `market_risk` (bucketed DV01s, ±1 bp `shock` check) → bond YTM / duration / convexity from the kernels → `fit_rates_pca` → PC and basis `KeyedScenario`s under `vmap`, P&L explain, PC-score sensitivities, VaR / ES |
| [`20_unified_fx_inflation_xccy.py`](#fx-inflation-and-cross-currency-on-one-market) | FX, cross-currency, multi-leg, inflation, declared-entry risk | One multi-currency `MarketState` with decoys → FX forward / option, cross-currency and 3M-vs-6M basis swaps, ZCIS / YoY / cap / floor through `price()` → parity and kernel identities → `market_risk` exactly zero outside declared entries, FD-confirmed |
| [`21_unified_rates_engine_ladder.py`](#one-swaption-every-engine) | Hull–White, G2++, Cheyette 1F/2F, LMM across engines | One European and one Bermudan swaption under every registered (model, method) → within-model engine agreement against closed forms, Bermudan ≥ European, the `auto` policy and its errors → curve deltas agree across engines, LMM path-wise risk |
| [`22_unified_multi_asset_exotics.py`](#multi-asset-and-path-dependent-exotics) | Spread / worst-of, barrier / Asian / lookback / variance swap across models | Keyed three-asset market → `MultiAssetGBMModel` vs Margrabe / Kirk, correlation risk → exotic ladder under BS / Heston / LV / SLV with exact anchors → `market_risk` spot deltas → a kernel-built dual digital with `jax.grad` deltas |

---

## Equity Options

**`examples/01_equity_options.py`** — The starting point. Covers the core workflow: define an instrument, price it, compute Greeks, and scale to portfolios.

```python
from valax.instruments.options import EuropeanOption
from valax.pricing.analytic.black_scholes import black_scholes_price
from valax.greeks.autodiff import greeks

option = EuropeanOption(strike=jnp.array(100.0), expiry=jnp.array(1.0), is_call=True)
price = black_scholes_price(option, spot, vol, rate, dividend)
g = greeks(black_scholes_price, option, spot, vol, rate, dividend)
# g["delta"], g["gamma"], g["vega"], g["vanna"], g["volga"], g["theta"], ...
```

Highlights:

- **All Greeks from one call** — price, delta, gamma, vega, theta, rho, vanna, volga
- **Higher-order Greeks** — compose `jax.grad` for speed (d³P/dS³) or any custom sensitivity
- **Portfolio vmap** — `batch_price` and `batch_greeks` price thousands of options in one vectorized call
- **JIT compilation** — ~3 µs per Black-Scholes price after compilation

---

## SABR Smile

**`examples/02_sabr_smile.py`** — Volatility smile modeling and calibration with SABR.

```python
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_price
from valax.calibration.sabr import calibrate_sabr

model = SABRModel(alpha=jnp.array(0.25), beta=jnp.array(0.5),
                  rho=jnp.array(-0.3), nu=jnp.array(0.4))

# Generate a smile
smile = jax.vmap(lambda K: sabr_implied_vol(model, forward, K, expiry))(strikes)

# Calibrate to market — a CalibrationResult; the fitted SABRModel is .value
result = calibrate_sabr(strikes, market_vols, forward, expiry, fixed_beta=jnp.array(0.5))
result.check()                      # raises CalibrationError if not converged
fitted = result.value
```

Highlights:

- **Parameter sensitivity** — see how rho controls skew, nu controls wings, beta controls the backbone
- **Model risk** — differentiate price w.r.t. SABR parameters (d(price)/d(alpha), d(price)/d(rho))
- **Calibration** — Levenberg-Marquardt (default), BFGS via `settings=SolverSettings(method="bfgs", ...)`, and weighted fitting, all gradient-based via optimistix

---

## Fixed Income

**`examples/03_fixed_income.py`** — Discount curves, bond pricing, and the autodiff advantage for risk measures.

```python
from valax.curves.discount import DiscountCurve
from valax.pricing.analytic.bonds import fixed_rate_bond_price, key_rate_durations

price = fixed_rate_bond_price(bond, curve)
krd = key_rate_durations(bond, curve)  # all pillar sensitivities in one backward pass
```

Highlights:

- **Curve construction** — build from zero rates, query at arbitrary dates, extract forwards
- **Bond analytics** — YTM solver (Newton-Raphson with autodiff Jacobian), modified duration, convexity
- **Key-rate durations** — one `jax.grad` call gives sensitivity to every curve pillar simultaneously
- **DV01** — parallel and key-rate basis point values

---

## Rates Derivatives

**`examples/04_rates_derivatives.py`** — Caps, floors, swaps, and swaptions on a realistic curve.

```python
from valax.pricing.analytic.caplets import caplet_price_black76, cap_price_black76
from valax.pricing.analytic.swaptions import swap_rate, swaption_price_black76

caplet_pv = caplet_price_black76(caplet, curve, vol)
par = swap_rate(swap, curve)
swaption_pv = swaption_price_black76(swaption, curve, swaption_vol)
```

Highlights:

- **Caplets** — Black-76 and Bachelier pricing, cap/floor parity
- **Swaps** — NPV, par swap rate, DV01 via autodiff through the curve
- **Swaptions** — Black-76 and Bachelier, vega computation

---

## Monte Carlo

**`examples/05_monte_carlo.py`** — Path generation, vanilla and exotic pricing, and convergence.

```python
from valax.pricing.mc.paths import generate_gbm_paths, generate_heston_paths
from valax.pricing.mc.sabr_paths import generate_sabr_paths
from valax.pricing.mc.engine import mc_price_with_stderr

paths = generate_gbm_paths(model, spot, T, n_steps, n_paths, key)
mc_p, mc_se = mc_price_with_stderr(option, spot, model, config, key)
```

Highlights:

- **Three path generators** — GBM, Heston (correlated 2D SDE), SABR, all via diffrax
- **Convergence** — MC error shrinks as $1/\sqrt{N}$, validated against analytical solutions
- **Exotic payoffs** — Asian (arithmetic average), barrier (up-and-out knock-out)
- **Heston smile** — extract the implied vol smile from MC prices across strikes

---

## PDE and Lattice

**`examples/06_pde_and_lattice.py`** — Numerical methods and American option pricing.

```python
from valax.pricing.pde.solvers import pde_price, PDEConfig
from valax.pricing.lattice.binomial import binomial_price, BinomialConfig

pde_p = pde_price(option, spot, vol, rate, dividend, PDEConfig(n_spot=300, n_time=300))
american_p = binomial_price(put, spot, vol, rate, dividend, BinomialConfig(n_steps=500, american=True))
```

Highlights:

- **Crank-Nicolson PDE** — second-order accurate, grid convergence demonstrated
- **CRR binomial tree** — European and American exercise
- **Early exercise premium** — American put premium increases with moneyness
- **Method comparison** — all three methods (analytic, PDE, lattice) converge to the same price and Greeks

---

## Unified Calibration, Pricing and Risk

**`examples/10_…` to `examples/22_…`** — the unified workflow on synthetic quote universes: noisy quotes → a `CalibrationChain` → `price(instrument, model, market)` → quote risk by one adjoint sweep, reconciled against autodiff through every calibration. The walkthrough is [From Quotes to Risk](guide/calibration-to-risk.md); the quote generators are described in [Synthetic Market Data](guide/synthetic_market.md#synthetic-quote-universes).

- **10** — one option on three market tiers (flat, curve + smile, + Heston), same workflow.
- **11** — OIS + 3M joint bootstrap, swap book quote risk in one sweep.
- **12** — equity curve, SABR smiles and Heston; exotic MC prices with quote and fixed-input risk.
- **13** — correlated quote scenarios, `vmap` recalibration + repricing, P&L explain, VaR / ES.

### Per-trade model calibration for Bermudans

**`examples/14_unified_rates_per_trade_model.py`** — Rates end to end with a model calibrated **per trade**: curves → swaption SABR cube (on the curve-implied forwards) → a Hull–White σ(t) bootstrapped on each Bermudan's own co-terminal basket → Bermudan priced through `price()` on the PDE → quote risk.

```python
snap, world = rates_vol_snapshot(registry, default_config(), rates_spec, vol_spec,
                                 bp_noise=0.5, vol_bp_noise=0.5)
calibrated = calibrate_market(chain, quotes, fixed, strict=True)    # curves → cube

node = swaption_basket_node("hw", template=HullWhiteModel(...),
                            basket=coterminal_swaptions(bermudan),
                            cube_id=vol_spec.cube_id, transforms={"volatility": positive()},
                            fixed_fields={"mean_reversion": A_KEY}, target_keys=keys,
                            mode="coterminal_bootstrap")   # one σ bucket per exercise
batched = eqx.filter_vmap(fit)(stacked_baskets)   # one fit per trade, curve-free models
pv = price(bermudan, model, calibrated.market, method="pde", config=PDE).price

adj = quote_risk_adjoint(bermudan, None, calibrated, model_result=model_result,
                         method="pde", config=PDE)
```

Highlights:

- **Batched per-trade bootstraps** — same-shaped baskets calibrate in one `eqx.filter_vmap`; each trade gets its own σ(t) (one bucket per exercise date, breakpoints as traced data), and the fitted `HullWhiteModel`s are curve-free (`price` binds the market curve), so they stack.
- **Model reprices its basket exactly** — each co-terminal European priced by Jamshidian through `price()` (the auto-default analytic route for `Swaption` × `HullWhiteModel`) matches the cube to ≤ 1e-13; a constant σ fitted by least squares to the same basket is printed alongside and misses by up to 2.8%.
- **Adjoint vs autodiff** — the model level's multiplier λ_m is pushed through the cube and curve levels and reconciled against autodiff through every calibration and the PDE.
- **λ_m as a vega profile** — one entry per exercise date: the basket target sensitivities, converted to per-vol-bp vega, sum to the total cube vega, and one entry is checked against full recalibration by finite differences.

### Caps and floors off the optionlet chain

**`examples/15_unified_rates_caps_optionlet_chain.py`** — the cap side of the rates vol chain: curves → `caplet_strip_node` (cap → caplet bootstrap per strike column, on the curve-projected forwards) → `optionlet_sabr_node` (one SABR smile per optionlet expiry, no quotes of its own) → caps and floors → quote risk.

```python
chain = CalibrationChain(nodes=(curve_node, vols.strip_node, vols.optionlet_node))
calibrated = calibrate_market(chain, quotes, fixed, strict=True)
SMILE = ImpliedVolModel(vol_spec.optionlet_id)   # the optionlet surface's id
pv = price(cap, SMILE, calibrated.market).price
adj = quote_risk_adjoint(cap, SMILE, calibrated)
ad = quote_risk_autodiff(cap, SMILE, chain, quotes, fixed)
```

Highlights:

- **Truth recovery** — noise-free quotes recover the strip and the curves to solver precision; with 0.5 bp noise the strip error and SABR RMSE are bounded by the noise.
- **Book through `price()`** — on- and off-grid strikes and an unquoted maturity; cap − floor equals the forward strip to round-off. The caps are priced under `ImpliedVolModel(surface_id)`, the built-in route that reads the named optionlet surface (`cap_price` underneath).
- **Where the vega lands** — on the cap quotes the caplets read, and structurally zero on longer maturities; Σ quote vegas ≈ the parallel-surface vega. The curve risk includes the strip's own dependence on the forwards (compared against rho with the surface frozen).
- **Adjoint vs autodiff vs FD** — reconciled with an absolute tolerance scaled to the largest sensitivity (the optionlet level is a non-zero-residual least-squares fit, where the two paths agree to ~1e-8 of scale), then checked against full-recalibration finite differences.

### Diagnosing a failed calibration

**`examples/16_unified_calibration_diagnostics.py`** — what the `CalibrationResult` contract buys when the quotes are bad. Each scene is self-checked:

1. **Clean baseline** — every level converged, with its label count, max residual and condition number.
2. **A NaN swaption vol** — `strict=True` raises naming the cell (`'2Yx10Y'`); `failed_labels()` returns exactly that cell, and the other cells of the batched level are bit-identical to the baseline.
3. **An off-market cap quote** — an infeasible strip bucket is named by strike and cap maturity (`'K=-0.86% cap 2Y'`, matching the quote key `USD.CAP.2Y.-86`), installs NaN so the later bucket of its column fails too, and `root_cause_labels()` names the first; a stale but strippable quote passes the strip and is caught downstream by the optionlet SABR.
4. **Unconvergeable settings and bad curve quotes** — `max_steps=1` on the curves or a smile is reported, never silently accepted; the curve failure names the long-end cluster of quotes (4 of 14), not all of them, and a NaN or infeasible (−50%) curve quote is named alone.
5. **Conditioning** — a fit on nearly-redundant strikes converges but trips `condition_alert` / `check(max_condition_number=...)` against `ILL_CONDITIONED_THRESHOLD`.
6. **Engine provenance** — `check_engine` refuses a frozen-y Cheyette fit on a different PDE grid or on MC, accepts the matching engine, and never refuses exact (Jamshidian) or discretised fits.
7. **Fix and recalibrate** — clean quotes restore the baseline exactly.

The contract itself is in [Calibration § Solvers](guide/calibration.md#solvers) and [§ 8 Conditioning](guide/calibration.md#8-is-the-fit-trustworthy-conditioning-at-every-level).

### SLV as a fixed-point level

**`examples/17_unified_slv_fixed_point.py`** — the documented [SLV fixed-point strategy](guide/slv.md#how-the-fixed-point-is-reported) made runnable: smile → two-expiry Heston backbone (κ fixed) → `calibrate_slv(heston=…)` swept over `n_iterations` → an up-and-out call under local vol, Heston and SLV → quote risk.

- **Backbone conditioning** — `calibrate_heston` with all five parameters free does not converge on the 1Y smile alone and does on both smiles (per-quote `expiry`).
- **Fixed-point table** — max|ΔL| (the last increment, the level's residual — Monte Carlo noise, since every pass is a self-consistent Guyon–Henry-Labordère sweep), `converged`, clip fraction and the vanilla repricing error per `n_iterations`; `n_iterations=1` is never converged and its `check()` raises, the first converged pass passes `check()` together with its Heston child.
- **Model picture** — all three models reprice the ATM vanilla within Monte Carlo error; the barrier prices differ, SLV between local vol and Heston. Local vol runs on the raw SABR surface (the default `LocalVolGuard` keeps deep-wing paths finite).
- **Risk by plain AAD, not IFT** — `jax.grad` through the calibrations, the unrolled leverage passes and the SLV Monte Carlo, checked against a central difference that reruns the same passes on the same random numbers with the Heston backbone held; through the backbone the QE variance scheme makes the sample only piecewise smooth, and the example shows the bump FD failing to settle there.

The backbone uses the full calibrated vol-of-vol: no mixing fraction and no hand-clamped wings are needed.

### Per-trade G2++ and Cheyette

**`examples/18_unified_rates_per_trade_g2pp_cheyette.py`** — Example 14 with the two richer short-rate models, plus the model risk between all three.

- **G2++ (multi-curve)** — σ(t) bootstrapped per trade, one bucket per exercise date, with a, b, ρ and the factor ratio η/σ as fixed inputs; the basket priced by `price`'s default for the pair (the Brigo–Mercurio closed form), the Bermudan by the 2-D ADI PDE. Every co-terminal is repriced exactly and the level's condition number is ~7. For comparison, the least-squares fit of two constant vols needs the 1Y-tenor swaption at the first exercise date to separate the two factors (condition number 5.8e3 with it, 5.6e5 on the co-terminals alone) and still misses the basket by 1.5%. ∂V/∂(η/σ), the price of holding the factor split fixed, is reported and checked by finite differences. Same-shaped baskets fit in one `eqx.filter_vmap`.
- **Exact vs engine-consistent calibration** — the same basket bootstrapped on the Bermudan's own PDE grid instead. The closed-form fit has portable parameters (provenance `exact`); the PDE fit reprices the basket exactly *as that grid prices it* (`discretised`, grid recorded), so the grid's swaption bias (5.7e-4) does not reach the basket. The example prints both σ(t), the basket miss on the grid and the Bermudan under each. Full mode adds a converged reference (161×161×200): on each grid the engine-consistent fit's Bermudan error is several times smaller (41×41×40: −0.624 vs −0.096 bp; 81×81×100: −0.117 vs +0.005 bp) because it cancels against the basket's — the reason to calibrate on the pricing engine, at the cost of grid-bound parameters.
- **Cheyette (single-curve)** — σ(t) bootstrapped through the frozen-y PDE with κ and the skew β held as fixed inputs (identifying β would need a strike slice and a least-squares fit, which is ill-conditioned per trade; see the [register](applications/calibration-register.md#cheyette-1f-2f)). `price` refuses a trade with a distinct projection curve (Cheyette carries no tenor-basis overlay), so this section runs on a single-curve market; the node records the engine-bound provenance and `check_engine` refuses another grid.
- **Risk** — the adjoint (λ_m) is reconciled with autodiff (rtol 1e-6) and checked against full-recalibration finite differences; the G2++ vega profile across the basket sums to the total cube vega.
- **Model risk** — Hull–White, G2++ and Cheyette each bootstrapped on the same trade's co-terminals on the same market. Every model reprices them exactly on its calibration engine, so the Bermudan spread across models (1.7%) is the exercise-value model risk the Europeans do not pin down, plus each Bermudan grid's own swaption error (printed).

The validation tests behind this are in `tests/test_calibration/test_coterminal_bootstrap.py` (bootstrap) and `tests/test_calibration/test_per_trade_basket.py` (least squares).

### A rates book end to end

**`examples/19_unified_book_workflow.py`** takes one rates book from quotes to scenario risk using both layers: `price()` routes every trade, and the kernels supply what a router does not return. It succeeds the retired examples 08 and 09.

```python
calibrated = calibrate_market(snap.chain, snap.quotes, strict=True)       # OIS + 3M
market = calibrated.market
pv = price(swap, None, market).price                                      # policy: analytic
r = market_risk(bond, None, market)            # r.curves keyed by r.declared.curves
ytm = yield_to_maturity(bond, fixed_rate_bond_price(bond, market.curves[OIS]))
pca = fit_rates_pca(zero_rate_returns_from_snapshots(history, pillars), times)
pnl = jax.vmap(lambda sc: book_value(shock(market, sc)) - pv0)(scenarios)  # KeyedScenario batch
```

Highlights:

- **Policy in the open**: each trade prints its `available_methods`, `auto_method` and `effective_config`. Curve-discounted trades take `model=None` and price analytically with no config.
- **`market_risk` per trade**: one backward pass per trade, keyed only by the curves that trade declares (the bonds name OIS alone). The per-trade reports sum to the gradient of the whole book. Zero-rate DV01s are bucketed with `tenor_bucket_map`, and a ±1 bp `shock` confirms them by revaluation.
- **The kernel layer**: the routed bond price equals `fixed_rate_bond_price` on the resolved curve. YTM, modified duration and convexity from `valax.pricing.analytic.bonds` explain a 25 bp curve move to within the compounding-convention gap (0.2% on the example's 5Y bond).
- **Factor model**: `fit_rates_pca` on a synthetic curve history (`zero_rate_returns_from_snapshots`) recovers the level, slope and curvature moves.
- **Keyed scenarios**: PC and basis moves are built as `KeyedScenario`s, applied with `shock` and revalued under `vmap` in one compiled program. A basis-only move leaves the bonds unchanged. The first-order explain comes from the `market_risk` report, and its second-order remainder halves with the shock size. PC-score sensitivities come from one `jax.grad` and match the chain rule on the report. The section ends with VaR and ES.

### FX, inflation and cross-currency on one market

**`examples/20_unified_fx_inflation_xccy.py`** prices FX, cross-currency, multi-leg and inflation trades against one multi-currency `MarketState`: USD OIS / 3M / 6M, EUR ESTR, the EURUSD spot, a USD CPI curve, EURUSD and CPI vols, and decoy entries that no trade names. The payoff maths lives in the [FX](guide/fx.md), [Inflation](guide/inflation.md) and [Fixed Income](guide/fixed-income.md) guides. This example checks the routes against market identities:

- **FX**: the forward matches covered interest parity by hand, and call − put equals the same-strike forward. `price()` equals `garman_kohlhagen_price` under `ImpliedVolModel("EURUSD.VOL")`.
- **Cross-currency and multi-leg**: with notional exchange the cross-currency swap collapses to N·s·A_USD; without it, its FX delta matches the hand formula. The kernel's par basis (`cross_currency_basis_spread`) reprices to zero. The 3M-vs-6M `MultiLegSwap` equals Σ `leg_pv`, flips sign exactly with pay/receive, and its par spread (one `jax.grad`, since PV is linear in the spread) prices to zero.
- **Inflation**: `zcis_breakeven_rate` recovers the curve's 5Y zero-coupon rate and the swap prices to zero there. The YoY par rate comes from two prices. Cap − floor equals the YoY swap at the strike; the example first asserts that every accrual fraction is 1, because caplets carry no accrual fraction.
- **Declared-entry risk**: for all nine trades the full market gradient is exactly zero on the decoys and on every undeclared entry, and `market_risk` equals it on the declared ones. The FX delta and a CPI-pillar sensitivity are confirmed by central differences (relative gaps 3e-10 and 5e-8).
- **JIT**: `eqx.filter_jit` of a YoY cap, and a strike ladder of FX calls under `eqx.filter_vmap` that matches the eager loop.

The example also records three inflation pitfalls, listed on the [roadmap](roadmap.md#35-inflation-derivatives). CPI before the first pillar extrapolates flat backwards, so the curve needs a pillar at the reference date. Caplet vol time is measured to the period start. `index_lag` is not read by any kernel.

### One swaption, every engine

**`examples/21_unified_rates_engine_ladder.py`** prices one 2Y-into-5Y at-the-money European and the matching Bermudan (exercise 2Y…6Y) through `price()` under Hull–White, G2++, Cheyette 1F/2F and the LMM, by every engine each pair registers. Only `model=`, `method=` and `config=` change. The instrument and the single-curve `MarketState` stay fixed, and the curve-free models are bound to the market curve by `price()`. Engine mechanics are in the [Lattice](guide/lattice.md), [PDE](guide/pde.md) and [Monte Carlo](guide/monte-carlo.md) guides.

- **Within-model agreement, with measured tolerances**: the Hull–White PDE is within 5e-6 of Jamshidian (full grid). The tree's error is first order (0.17/n, measured), and Richardson extrapolation brings it to 1.4e-6. The G2++ PDE is within 3e-4 of the Brigo–Mercurio closed form, and G2++ and Cheyette MC are within 2σ of their references. Engines are never compared across models.
- **Bermudan ≥ European** under every model and engine (+28–31%), and the Hull–White tree agrees with the PDE. The LMM Bermudan is a Longstaff–Schwartz lower bound that reports no standard error, so its uncertainty is the spread over four seeds.
- **Policy**: every pair's `available_methods`, `auto_method` and `effective_config`. `method="auto"` equals the explicit method, and bad requests (no such pricer, an unknown method name, MC without a key) raise a `ValueError` naming the fix.
- **Risk through the same call**: `market_risk` gives the same curve pillar deltas under analytic, PDE and tree (relative gap 2e-5 for the PDE and 9e-5 for the tree, full mode). Its parallel DV01 matches a ±1 bp `shock` to 1.5e-6. The LMM reports path-wise curve deltas with a fixed key. The PDE route also puts a sensitivity of about −price on the t = 0 pillar, whose discount factor is fixed at 1; the example asserts and explains this, and excludes that pillar from the comparison.

### Multi-asset and path-dependent exotics

**`examples/22_unified_multi_asset_exotics.py`** puts three equities (SPX on an SVI smile, SX5E, and a decoy NKY) on one keyed `MarketState` and prices exotics through `price()`. The payoffs are documented in [Equity Exotics](guide/equity-exotics.md).

- **Multi-asset**: a spread option and a worst-of under `MultiAssetGBMModel`. The zero-strike spread is checked against Margrabe (exact; z = −0.9) and the struck spread against Kirk (z = −0.8). Worst-of calls fall and puts rise as correlation falls (common random numbers). The correlation sensitivity from `model_gradient` matches a bump.
- **Model ladder**: European, up-and-out barrier, Asian, lookback and variance swap under Black–Scholes, Heston, local vol and SLV, each with exact anchors:
  - Heston MC against its closed form; the BS European and variance swap against σ.
  - The BS barrier by MC against the PDE, after the Broadie–Glasserman–Kou continuity shift (z = −0.4, against z = +15 unshifted).
  - In + out equals the vanilla to 1e-10 at every smoothing.

  The differences between models are printed, not asserted away: for example, the barrier spans 3.08 to 5.84 across the four models.
- **Risk**: `market_risk` for the spread and the barrier lists declared entries for the right assets only, with exactly zero gradient on the decoy. Spot deltas match bumps, and the zero-strike spread satisfies Euler's identity S₁Δ₁ + S₂Δ₂ = price exactly.
- **Kernel layer**: the spread is rebuilt on `generate_correlated_gbm_paths` with `spread_option_mc_payoff`, bit-identical to `price()`. A dual digital, which no route prices, is then built on the same paths, checked against its exact bivariate-normal value, and differentiated by `jax.grad` through the generator.
- **JIT**: `eqx.filter_jit` of one `price()` call, and a barrier strike ladder on the PDE under `eqx.filter_vmap`.

Two limitations of the routes are demonstrated and asserted, and both are on the [roadmap](roadmap.md#36-equity-exotics). `BlackScholesModel` reads only the `_default_` surface, so a per-asset BS vol is passed explicitly. The variance-swap MC route annualises at 252 per year, so `n_steps` must equal 252·T; at 126 steps over one year it returns 2σ².

### Retired examples

These scripts were built on APIs that the [roadmap](roadmap.md) schedules for removal (`MarketData`, `MarketScenario`, the legacy `valax.risk.var`), so they were retired in favour of `MarketState`-based successors. The kernel-layer examples 01–06 are kept on purpose.

| Retired | Built on | Use instead |
|---|---|---|
| `07_synthetic_market.py` | `MarketData`, `batch_price` | [10](#unified-calibration-pricing-and-risk) (one option on three synthetic market tiers). The generators are documented in [Synthetic Market Data](guide/synthetic_market.md). |
| `08_end_to_end_workflow.py` | `MarketData`, SABR on a noisy smile | 10–13 and [19](#a-rates-book-end-to-end). The calibration-residual loop is still tested in `tests/test_market/test_calibration_residual.py`. |
| `09_pca_rates_pnl.py` | `MarketData`, `MarketScenario` | [19](#a-rates-book-end-to-end) (PCA on `MarketState` via `KeyedScenario`) |
| `comparisons/08_risk_greeks_var.py` | `MarketData`, legacy `valax.risk.var` | [13](#unified-calibration-pricing-and-risk) (quote scenarios, VaR / ES) and [19](#a-rates-book-end-to-end) |

---

## Pitfalls — things that do *not* work, and why

**[`examples/pitfalls/`](https://github.com/ACea15/VALAX/tree/main/examples/pitfalls)** — a lab bench of **known failure modes**: numerical traps, solver fragilities and modelling gotchas found while building VALAX, each reduced to the smallest runnable reproduction and handed to you with the knobs still attached.

These are deliberately *red*. They are not examples of correct usage — they demonstrate something broken or subtle so you can poke at it: flip a solver, tighten a tolerance, re-break the fix and watch it fail again. Each script carries a **SYMPTOM / ROOT CAUSE / KNOBS** header and a **WAY OUT** section naming the fix without applying it, and is wrapped so it runs end-to-end even where the underlying call diverges or raises.

| Script | Pitfall |
|---|---|
| [`01_sabr_vol_convention_mismatch.py`](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/01_sabr_vol_convention_mismatch.py) | Fitting a **normal**-quoted SABR smile with the **lognormal** Hagan formula. The solver reports a small residual and returns nonsense parameters — so a small calibration residual is necessary but *not sufficient*. |
| [`02_where_guard_kills_the_derivative.py`](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/02_where_guard_kills_the_derivative.py) | A `jnp.where` guard on a removable singularity with the **right value and no derivative**. Two "independent" adjoints agreed bit-for-bit and were both wrong, because they shared the broken primitive; the finite difference nobody believed was right all along. |
| [`03_unrolled_series_compile_time.py`](https://github.com/ACea15/VALAX/blob/main/examples/pitfalls/03_unrolled_series_compile_time.py) | A correct, fast formula whose **traced graph** is enormous: a series written as an expanded double sum is unrolled by JAX, so values and run time are fine but compile time scales with the program *as written* — a quote-risk compile through the G2++ PDE took 427 s instead of 4 s. |

The prose counterpart is [Numerical Pitfalls](architecture/numerical-pitfalls.md), which keeps one write-up per bug — symptom, the signals that pointed the wrong way, root cause, fix, and the generalisable lesson — together with the debugging methodology that localised it.
