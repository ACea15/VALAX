# Examples

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

# Calibrate to market
fitted, sol = calibrate_sabr(strikes, market_vols, forward, expiry, fixed_beta=jnp.array(0.5))
```

Highlights:

- **Parameter sensitivity** — see how rho controls skew, nu controls wings, beta controls the backbone
- **Model risk** — differentiate price w.r.t. SABR parameters (d(price)/d(alpha), d(price)/d(rho))
- **Calibration** — Levenberg-Marquardt, BFGS, and weighted fitting, all gradient-based via optimistix

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
sabr_fit, _  = calibrate_sabr(strikes, smile_noisy, F, T,
                              fixed_beta=sabr_truth.beta)         # Stage 3
port         = sample_option_portfolio(registry, md, spec)        # Stage 4
prices       = batch_price(black_scholes_price, *port["calls"]...) # Stage 5
bad_w, diag  = inject_calendar_arb(w_clean, i=1, j=3)             # Stage 6
```

Highlights:

- **Non-tautological calibration check** — the fit RMSE is compared against the *empirical* noise RMSE on the same draw; a passing test means the optimiser is at the noise floor, not that "calibrating SABR on SABR-generated data recovers SABR" (which would only test the optimiser).
- **Closed loop in one script** — every stage uses outputs of the previous, so any inconsistency surfaces immediately.
- **Arbitrage backlog reporting** — Stage 6 deliberately breaks a vector of total variances and prints the diagnosis the library *would* raise once detection lands.  The corresponding xfail tests turn green the day a detector is added.
