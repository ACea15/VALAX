# Pricing

VALAX ships with runnable example scripts in the `examples/` directory. Each file uses `# %%` cell markers for interactive execution in VS Code, PyCharm, or any IDE that supports the Python cell format.

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
| [`07_synthetic_market.py`](#synthetic-market) | Synthetic generators, basket pricing, market tapes | Reproducible market state via `SeedRegistry`, multi-asset snapshot + correlation, batched analytic pricing, 5-step market evolution |
| [`08_end_to_end_workflow.py`](#end-to-end-workflow) | Stages 1 → 6, calibration on synthetic, arbitrage injection | Ground-truth SABR → noisy smile → calibration to within the noise floor; portfolio pricing + autodiff Greeks; deliberate calendar-arb injection demonstrating the xfail backlog |
| [`10_unified_market_tiers.py`](#unified-calibration-pricing-and-risk) | Unified quotes → calibrate → price → quote risk | One option on three markets — flat Black-Scholes, bootstrapped curve + SABR smile, + calibrated Heston — through the same workflow |
| [`11_unified_multicurve_rates.py`](#unified-calibration-pricing-and-risk) | Multi-curve bootstrap, swap book, quote risk | Synthetic OIS + 3M quotes → joint bootstrap → swap book → book quote risk in one adjoint sweep (vs autodiff and FD), single-curve limit |
| [`12_unified_equity_smile_and_model.py`](#unified-calibration-pricing-and-risk) | Equity smiles, Heston, MC quote risk | Noisy equity quotes → curve, SABR smiles, Heston → European / barrier / Asian prices → Monte Carlo quote risk with fixed-input sensitivities |
| [`13_unified_quote_scenarios.py`](#unified-calibration-pricing-and-risk) | Quote scenarios, VaR / ES | Correlated quote scenarios → `vmap` recalibration + repricing compiled once → P&L explain vs first-order quote risk → VaR / ES |
| [`14_unified_rates_per_trade_model.py`](#per-trade-model-calibration-for-bermudans) | Swaption cube, per-trade Hull–White, Bermudan risk | Curves → swaption cube → per-trade Hull–White σ(t) bootstrapped on co-terminal baskets (`filter_vmap`) → Bermudan via `price()` PDE → quote risk; adjoint vs autodiff; λ_m vega profile vs full recalibration |
| [`15_unified_rates_caps_optionlet_chain.py`](#caps-and-floors-off-the-optionlet-chain) | Cap strip, optionlet SABR, cap/floor risk | Curves → cap→caplet strip → optionlet SABR as chain nodes → caps / floors through `price()` → quote risk to cap and curve quotes; adjoint vs autodiff vs FD |
| [`16_unified_calibration_diagnostics.py`](#diagnosing-a-failed-calibration) | Failure naming, conditioning, engine provenance | Bad quotes (NaN, off-market, stale) → `check()` / `failed_labels()` name the cell or bucket; healthy nodes unaffected; ill-conditioning alert; `check_engine` refusing a frozen-y Cheyette fit on another engine; fix and recalibrate |
| [`17_unified_slv_fixed_point.py`](#slv-as-a-fixed-point-level) | SLV leverage, fixed-point diagnostics, MC exotic risk | Smile → Heston → SLV leverage as a fixed-point level (max\|ΔL\|, `converged`, clip fraction vs `n_iterations`) → barrier under LV / Heston / SLV → vega by AD through the unrolled passes vs FD |
| [`18_unified_rates_per_trade_g2pp_cheyette.py`](#per-trade-g2-and-cheyette) | Per-trade G2++ and Cheyette, model risk | Per-trade G2++ σ(t) (multi-curve, closed-form basket, 2-D PDE Bermudan) and Cheyette σ(t) (single-curve, frozen-y PDE), both bootstrapped → quote risk reconciled; exact vs engine-consistent G2++ calibration; HW vs G2++ vs Cheyette Bermudan on the same basket |

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

## Synthetic Market

**`examples/07_synthetic_market.py`** — Generate a complete market snapshot from scratch and price a small basket.  No external data required.

```python
from valax.market import (
    SeedRegistry, SyntheticMarketConfig,
    sample_market_with_correlation, sample_option_portfolio,
    evolve_market, OptionPortfolioSpec,
)

registry = SeedRegistry(master_seed=20260101, library_version=valax.__version__)
cfg = SyntheticMarketConfig(n_assets=4)
md, corr = sample_market_with_correlation(registry, cfg)        # snapshot + corr
port = sample_option_portfolio(registry, md,
                                OptionPortfolioSpec(n_per_asset=5))
prices = batch_price(black_scholes_price, *port["calls"], ...)
tape = evolve_market(registry, md, dates, corr)                 # 5-step path
```

Highlights:

- **Reproducible by design** — every random draw routed through `SeedRegistry.key(name)`; bumping `version=` is the only non-renaming way to change a stream's bytes.
- **Canonical types** — generators emit `MarketData`, `DiscountCurve`, `EuropeanOption`, and a correlation matrix that the rest of the library accepts unchanged.
- **Tape generation** — `evolve_market` produces a stacked `MarketData` whose `spots` carries a leading `n_dates` axis.

---

## End-to-End Workflow

**`examples/08_end_to_end_workflow.py`** — The full six-stage workflow from synthetic ground truth to calibration, pricing, and arbitrage stress testing.  This is the script that demonstrates the **non-tautological calibration check**: the SABR fit residual lands at the injected observation noise floor, proving the calibrator is bottlenecked by data quality, not optimiser convergence.

```python
sabr_truth   = sample_sabr_params(registry, cfg)                  # Stage 1
smile_noisy  = synthesize_sabr_smile(registry, sabr_truth, F, T,
                                      strikes, vol_bp_noise=15)   # Stage 2
sabr_fit     = calibrate_sabr(strikes, smile_noisy, F, T,
                              fixed_beta=sabr_truth.beta).value   # Stage 3
port         = sample_option_portfolio(registry, md, spec)        # Stage 4
prices       = batch_price(black_scholes_price, *port["calls"]...) # Stage 5
bad_w, diag  = inject_calendar_arb(w_clean, i=1, j=3)             # Stage 6
```

Highlights:

- **Non-tautological calibration check** — the fit RMSE is compared against the *empirical* noise RMSE on the same draw; a passing test means the optimiser is at the noise floor, not that "calibrating SABR on SABR-generated data recovers SABR" (which would only test the optimiser).
- **Closed loop in one script** — every stage uses outputs of the previous, so any inconsistency surfaces immediately.
- **Arbitrage backlog reporting** — Stage 6 deliberately breaks a vector of total variances and prints the diagnosis the library *would* raise once detection lands.  The corresponding xfail tests turn green the day a detector is added.

---

## Unified Calibration, Pricing and Risk

**`examples/10_…` to `examples/18_…`** — the unified workflow on synthetic quote universes: noisy quotes → a `CalibrationChain` → `price(instrument, model, market)` → quote risk by one adjoint sweep, reconciled against autodiff through every calibration. The walkthrough is [From Quotes to Risk](guide/calibration-to-risk.md); the quote generators are described in [Synthetic Market Data](guide/synthetic_market.md#synthetic-quote-universes).

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
- **Risk by AD, not IFT** — `jax.grad` through the calibrations, the unrolled leverage passes and the SLV Monte Carlo, checked against a central difference that reruns the same passes on the same random numbers with the Heston backbone held; through the backbone the QE variance scheme makes the sample only piecewise smooth, and the example shows the bump FD failing to settle there.

The backbone uses the full calibrated vol-of-vol: no mixing fraction and no hand-clamped wings are needed.

### Per-trade G2++ and Cheyette

**`examples/18_unified_rates_per_trade_g2pp_cheyette.py`** — Example 14 with the two richer short-rate models, plus the model risk between all three.

- **G2++ (multi-curve)** — σ(t) bootstrapped per trade, one bucket per exercise date, with a, b, ρ and the factor ratio η/σ as fixed inputs; the basket priced by `price`'s default for the pair (the Brigo–Mercurio closed form), the Bermudan by the 2-D ADI PDE. Every co-terminal is repriced exactly and the level's condition number is ~7. For comparison, the least-squares fit of two constant vols needs the 1Y-tenor swaption at the first exercise date to separate the two factors (condition number 5.8e3 with it, 5.6e5 on the co-terminals alone) and still misses the basket by 1.5%. ∂V/∂(η/σ), the price of holding the factor split fixed, is reported and checked by finite differences. Same-shaped baskets fit in one `eqx.filter_vmap`.
- **Exact vs engine-consistent calibration** — the same basket bootstrapped on the Bermudan's own PDE grid instead. The closed-form fit has portable parameters (provenance `exact`); the PDE fit reprices the basket exactly *as that grid prices it* (`discretised`, grid recorded), so the grid's swaption bias (5.7e-4) does not reach the basket. The example prints both σ(t), the basket miss on the grid and the Bermudan under each. Full mode adds a converged reference (161×161×200): on each grid the engine-consistent fit's Bermudan error is several times smaller (41×41×40: −0.624 vs −0.096 bp; 81×81×100: −0.117 vs +0.005 bp) because it cancels against the basket's — the reason to calibrate on the pricing engine, at the cost of grid-bound parameters.
- **Cheyette (single-curve)** — σ(t) bootstrapped through the frozen-y PDE with κ and the skew β held as fixed inputs (identifying β would need a strike slice and a least-squares fit, which is ill-conditioned per trade; see the [register](applications/calibration-register.md#cheyette-1f-2f)). `price` refuses a trade with a distinct projection curve (Cheyette carries no tenor-basis overlay), so this section runs on a single-curve market; the node records the engine-bound provenance and `check_engine` refuses another grid.
- **Risk** — the adjoint (λ_m) is reconciled with autodiff (rtol 1e-6) and checked against full-recalibration finite differences; the G2++ vega profile across the basket sums to the total cube vega.
- **Model risk** — Hull–White, G2++ and Cheyette each bootstrapped on the same trade's co-terminals on the same market. Every model reprices them exactly on its calibration engine, so the Bermudan spread across models (1.7%) is the exercise-value model risk the Europeans do not pin down, plus each Bermudan grid's own swaption error (printed).

The validation tests behind this are in `tests/test_calibration/test_coterminal_bootstrap.py` (bootstrap) and `tests/test_calibration/test_per_trade_basket.py` (least squares).

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
