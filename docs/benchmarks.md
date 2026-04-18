# Benchmarks and QuantLib Validation

VALAX is validated against [QuantLib](https://www.quantlib.org/) — the
industry-standard open-source quantitative finance library — on every core
pricing function. This page documents **two different things**:

1. **Numerical validation** (available today): cross-checks that VALAX
   and QuantLib agree on prices and Greeks to a specified tolerance. This is a
   **compliance matrix**, not a speed benchmark.
2. **Performance benchmarks** (planned): GPU/TPU throughput comparisons versus
   QuantLib on CPU. Tracked on the [roadmap](roadmap.md) as Priority P5.2.

## 1. Validation Scope

The test suite at `tests/test_quantlib_comparison/` contains 7 files that cross-
check the main pricing surfaces. Each test file corresponds to one product or
method area.

| File | Area | What is compared |
|------|------|------------------|
| [`test_european_options_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_european_options_ql.py) | Analytical equity options | Call/put prices, Greeks (Δ, Γ, Vega), implied vol round-trip |
| [`test_fixed_income_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_fixed_income_ql.py) | Curves, bonds | Discount factors at pillars, bond prices, YTM, modified duration, KRD sum-to-duration |
| [`test_heston_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_heston_ql.py) | Heston MC | Call prices within 3 SE of QL engine; smile skew direction; risk-neutral drift; variance mean-reversion |
| [`test_monte_carlo_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_monte_carlo_ql.py) | GBM MC | Call price within 3 SE of BS; convergence with path count; Asian/barrier bounds; risk-neutral drift; path initial condition |
| [`test_pde_lattice_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_pde_lattice_ql.py) | PDE and lattice | Crank-Nicolson vs BS and vs QL FDM; CRR vs BS and vs QL CRR; American put premium vs QL |
| [`test_risk_greeks_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_risk_greeks_ql.py) | Risk engine | Greeks (Δ, Γ, Vega, ρ, Θ) vs QL; full-reval repricing under spot/vol/rate bumps; 2nd-order attribution accuracy; parametric VaR |
| [`test_sabr_ql.py`](https://github.com/acea/valax/blob/main/tests/test_quantlib_comparison/test_sabr_ql.py) | Hagan-SABR | Implied vol across strikes and parameter sets; ATM vol |

## 2. Compliance Matrix

Each row below is a single pytest assertion. "Tolerance" is the absolute or
relative bound that must hold for the test to pass; failures cause the
comparison to fail.

### 2.1 European Options (`test_european_options_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| Call price (ATM) | VALAX BSM vs QL `AnalyticEuropeanEngine` | `abs < 1e-10` |
| Put price (ATM) | VALAX BSM vs QL, via put-call parity | `abs < 1e-10` |
| Call prices across 7 strikes (80–120) | VALAX BSM vs QL | `abs < 1e-10` |
| Delta | `jax.grad` vs `option.delta()` | `abs < 1e-10` |
| Gamma | nested `jax.grad` vs `option.gamma()` | `abs < 1e-10` |
| Vega | `jax.grad` vs `option.vega()` | `abs < 1e-8` |
| Implied vol (round-trip) | `implied_vol(price(σ)) = σ` | `abs < 1e-12` |
| Implied vol vs QL | VALAX Newton-Raphson vs QL `impliedVolatility` | `abs < 1e-8` |

### 2.2 Fixed Income (`test_fixed_income_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| Discount factor at each pillar | `DiscountCurve(t)` vs QL `discount(t)` | `abs < 1e-12` |
| Bond price | `fixed_rate_bond_price` vs QL `FixedRateBond.cleanPrice()` | `rel < 5e-4` |
| Yield to maturity | VALAX Newton solver vs QL `Bond.bondYield()` | `abs < 5e-3` |
| Modified duration | `jax.grad`-based vs QL `BondFunctions.duration` | `abs < 0.2` |
| KRDs sum to modified duration | Internal consistency check | `abs < 0.2` |

### 2.3 Heston (`test_heston_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| MC call prices across 3 strikes | VALAX MC vs QL `AnalyticHestonEngine` | `< 3 SE` |
| Smile skew direction | `price(K_low) / price(K_high)` ratio with negative ρ | `> 3.0` |
| Risk-neutral drift | MC terminal mean vs $S_0 e^{(r-q)T}$ | `rel < 2%` |
| Variance mean-reversion | MC time-averaged variance vs θ | `abs < 0.01` |

### 2.4 Monte Carlo — GBM (`test_monte_carlo_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| MC call vs Black-Scholes | VALAX MC vs analytical BS | `< 3 SE` |
| MC standard error | SE < 1.0 for 10k paths | Sanity |
| Asian < European | `price_asian < price_european` | Monotonic |
| Barrier < European | `price_up_and_out < price_european` | Monotonic |
| Risk-neutral drift | MC terminal mean vs $S_0 e^{(r-q)T}$ | `rel < 2%` |
| Paths start at spot | `paths[:, 0] == S_0` | `allclose` |

### 2.5 PDE and Lattice (`test_pde_lattice_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| Crank-Nicolson call converges to BS | VALAX PDE vs analytical BS | `rel < 1e-3` |
| VALAX PDE vs QL FDM | VALAX Crank-Nicolson vs `FDEuropeanEngine` | `rel < 5e-3` |
| PDE delta vs BS | `jax.grad` through PDE vs BSM closed-form delta | `abs < 0.01` |
| CRR European converges to BS | VALAX binomial vs analytical BS | `rel < 5e-3` |
| CRR vs QL CRR | VALAX binomial vs `BinomialVanillaEngine("crr")` | `rel < 1e-3` |
| American put premium (CRR) | VALAX vs QL | `abs < 0.01` |
| American ≥ European | `amer >= euro - 1e-10` | Monotonic |

### 2.6 SABR (`test_sabr_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| Implied vol across parameter sets and strikes | Hagan formula vs QL `sabrVolatility` | `abs < 1e-10` |
| ATM vol | Hagan at $K = F$ vs QL | `abs < 1e-12` |

### 2.7 Risk and Greeks (`test_risk_greeks_ql.py`)

| Quantity | Comparison | Tolerance |
|----------|------------|-----------|
| Delta, Gamma, Vega, Rho | `greeks()` vs QL Greeks | `abs < 1e-4` |
| Theta | `greeks()` per-day vs QL per-day theta | `abs < 1e-3` |
| Full reval under spot bump | VALAX repricing vs QL repricing | `abs < 1e-4` |
| Full reval under vol bump | VALAX repricing vs QL repricing | `abs < 1e-4` |
| Full reval under rate bump | VALAX repricing vs QL repricing | `abs < 1e-4` |
| Full reval under combined bump | VALAX repricing vs QL repricing | `abs < 1e-4` |
| P&L attribution: delta_spot | `attr["delta_spot"]` vs `QL.delta × ΔS` | `abs < 1e-3` |
| P&L attribution: gamma_spot | `attr["gamma_spot"]` vs $\tfrac{1}{2}\,\mathrm{QL.gamma} \times \Delta S^2$ | `abs < 1e-3` |
| P&L attribution: delta_vol | `attr["delta_vol"]` vs `QL.vega × Δσ` | `abs < 1e-3` |
| P&L attribution: delta_rate | `attr["delta_rate"]` vs `QL.rho × Δr` | `abs < 5e-3` |
| Attribution actual P&L | `attr["actual"]` vs QL NPV difference | `abs < 1e-4` |
| 2nd-order attribution tighter than 1st | Internal consistency (unexplained shrinks) | Monotonic |
| Parametric VaR | VALAX delta-normal vs manual QL-derived VaR | `abs < 1e-4` |
| Parametric VaR positivity / scaling | `VaR(99%) > VaR(95%)`, `VaR > 0` | Monotonic |

## 3. Methodology Notes

- **Deterministic seeds.** All Monte Carlo tests use fixed `jax.random.PRNGKey`
  values so results are reproducible. Path counts are documented in each test.
- **Tolerance choice.** Analytic-vs-analytic comparisons use machine precision
  (`1e-10` to `1e-12`). MC vs analytical uses the 3-sigma band. Autodiff
  Greeks vs QL's own Greeks use `1e-10` when both are closed-form (Black-Scholes
  vs BSM `AnalyticEuropeanEngine`) and looser when a numerical method is
  involved (e.g., QL's vega uses an `AnalyticEuropeanEngine` numerical bump
  internally, giving `< 1e-8`).
- **QuantLib version.** The comparison suite is pinned against whatever version
  of `QuantLib` is installed locally. There is no strict upper bound; if you
  see a QL version mismatch, please file an issue.
- **Optional dependency.** `QuantLib` is intentionally **not** in the `[dev]`
  extras — it is a heavy dependency that slows down CI. Install it separately
  to run these tests:

  ```bash
  pip install QuantLib
  pytest tests/test_quantlib_comparison
  ```

## 4. What is *not* yet benchmarked

The VALAX ↔ QuantLib comparison is comprehensive for every area where
QuantLib has a canonical implementation. Several newer VALAX modules do not yet
have QL comparison coverage, either because QL's implementation differs
materially or because the feature is VALAX-specific:

- **Hull-White trinomial tree** — QL's tree uses a different truncation
  boundary; direct price comparison requires careful step-matching. Tracked
  for a future comparison test.
- **Inflation pricers** (ZCIS, YYIS, inflation cap/floor) — QL's inflation
  module uses a different curve representation (YY or ZC rate curve vs.
  VALAX's forward CPI curve). Cross-validation planned.
- **Spread options** (Margrabe / Kirk) — QL has no native Margrabe pricer;
  validation is against the closed-form formula directly.
- **Rates exotics** (XCCY, TRS, CMS swap/cap/floor, range accrual) — QL's
  versions use different conventions (e.g. compounded vs simple). Planned
  as part of the rates-exotics test expansion.
- **Multi-curve bootstrapping** — QL has `PiecewiseYieldCurve`, which can be
  pointed to match VALAX's log-DF log-linear approach. The simultaneous Newton
  solver gives identical DFs at pillars; interpolation differences show up
  between pillars. Planned as part of fixed-income expansion.
- **Risk ladders and waterfall** — VALAX-specific; validated internally
  against a sum-of-rungs identity rather than QL.

## 5. Performance Benchmarks (Planned — P5.2)

Numerical agreement with QL is the baseline. The **competitive differentiator**
VALAX promises — end-to-end JAX for portable GPU/TPU acceleration — is not yet
benchmarked with published numbers.

The roadmap item [P5.2 "GPU/TPU Benchmark Suite"](roadmap.md) calls for:

- Batch European option pricing: 10k–100k instruments via `jax.vmap`, CPU
  vs. GPU vs. QL CPU.
- Full-revaluation Monte Carlo VaR: 1k–10k paths × 100–1000 portfolio
  positions, CPU vs. GPU.
- XVA exposure simulation (depends on Priority P3.1 XVA suite).

!!! info "Why not today?"
    Performance numbers in this space are notoriously sensitive to:

    - JAX tracing vs compilation time (a "cold" benchmark can be 10–100×
      slower than a "warm" one).
    - GPU warm-up, memory allocation patterns, and XLA autotuning.
    - Comparison fairness: QL's native engine uses a single CPU core; a
      naive "VALAX is 50× faster on GPU" claim requires also reporting
      per-core CPU performance to be meaningful.

    The current priority is to finish foundational features (calendars,
    cashflow engine, short-rate calibration) before producing benchmark
    marketing material. The numerical-validation tests above are the
    evidence of correctness that has to come first.

## 6. Running the Comparison Suite

```bash
# Install QuantLib (not included in [dev]).
pip install QuantLib

# Run the full comparison suite.
pytest tests/test_quantlib_comparison -v

# Run a single file.
pytest tests/test_quantlib_comparison/test_european_options_ql.py -v

# Run a single comparison.
pytest tests/test_quantlib_comparison/test_european_options_ql.py::test_call_price_matches -v

# Exit non-zero on any disagreement exceeding tolerance.
pytest tests/test_quantlib_comparison --maxfail=1
```

The suite also serves as an executable specification: each comparison row in
the matrix above corresponds to a runnable pytest that any contributor can
execute locally.
