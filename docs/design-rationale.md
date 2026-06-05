# Design Rationale: What VALAX Is Optimised For

This document explains *why* VALAX makes the architectural choices it does. It sits beside [Vision](vision.md) (which describes the destination) and [Roadmap](roadmap.md) (which describes the order). This page is about the **logic that connects the two** — what is actually hard about quant systems at scale, and which pieces of that hardness VALAX is built to attack.

It is deliberately opinionated. It is also deliberately honest about what VALAX does **not** fix.

---

## 1. The thesis in one paragraph

The hardest part of building a bank-scale quantitative finance system is **not** the maths in any individual component. The maths is mostly solved and the components are individually tractable. The hardest part is **keeping ~500 mutually-aware components consistent over twenty years while every one of them is being changed by a different team under regulatory pressure**.

VALAX's architecture is designed to minimise the *coordination tax* that this imposes — the half of a bank quant's day that goes to reconciling identifiers, calendars, sign conventions, snapshot times, and bump conventions across systems. The pricing maths is the easy part. The consistency between pricing, risk, capital, audit, and trader views is the hard part. Every design choice in the library is in service of that consistency.

---

## 2. What actually kills bank programmes, ranked

When a quant systems programme overruns by years or fails to land, the post-mortem almost never reads "the Heston calibrator wasn't accurate enough." It almost always reads one of the things below. They are ordered by how often they sink a programme, not by how intellectually deep they are.

### 2.1 Data and identifier consistency — the silent killer

Not "data infrastructure" (storage, throughput — those are solved with money). **Semantic consistency across the firm.**

Pick a single trade: a 5Y USD payer swap with JPM. Now ask, for each system that touches it:

- What's its ID in the front-office booking system?
- In the firm's golden-source trade store?
- In the collateral / margin system?
- In the regulatory reporting feed?
- In the CVA engine's netting-set definition?
- In the general ledger?

The answer is typically six different IDs, mapped through six different reference tables, owned by six different teams, each with its own change-control board. Multiply by tens of millions of live trades and two decades of history. Now add: reference data on issuers, calendars per market, day-count conventions per leg, fixing sources, corporate-action handling, bitemporal versioning of every overrideable value.

The pricing maths for that swap is 200 lines of code. The identifier plumbing is a 30-person team for a decade.

### 2.2 Model governance and regulatory permission — the second killer

A bank cannot deploy a new pricer the way a startup ships a service. Under SR 11-7 (US), SS1/23 (UK), TRIM (EU), and FRTB IMA, every model carries an inventory entry, a tier classification, independent validation by a team that did not build it, sign-off by a model risk committee, documented limitations and approved use perimeter, annual re-validation, and re-validation on every material change. IMA capital approval for a new product class is typically an 18–36 month project.

You can build a state-of-the-art calibrator in a week. Getting it into production for capital use takes years. That asymmetry is the single biggest constraint on a bank's ability to ship quant code.

### 2.3 Joint factor consistency, XVA, and what-if at trader speed

Risk *is* hard, but in a specific way. The univariate maths is mostly known. What's hard at scale is:

- **Joint factor dynamics across asset classes.** Equity surfaces, IR curves, credit curves, FX surfaces all evolving under *one* consistent joint distribution for CVA. A 10,000-factor joint covariance that is PSD, calibrated to vanillas, and stable under shrinkage is a real research problem.
- **XVA** (CVA / FVA / KVA / MVA / ColVA). Each is a portfolio-level Monte Carlo on the *entire* derivatives book under joint market+credit scenarios. Computationally savage; the governance of which CSAs, netting sets, funding curves, recoveries, and wrong-way-risk correlations to use is harder than the computation.
- **What-if at trader speed.** Pre-trade: "if I add this hedge, what does it do to my desk's IMA capital?" needs full-revaluation risk on a *hypothetical* book in seconds.

Hard, but a known playbook. Bounded scope.

### 2.4 Pricing engines

Lower than people expect. 90% of the trading book is vanilla products with mature, 30-year-old pricers. The hard 10% (autocallables, Bermudans, callable range accruals, SLV calibration, pathwise Greeks for barriers) is known research with bounded scope.

Where it remains hard:
- Calibration robustness across regimes (Heston blowing up in crisis vol).
- Greek stability day-over-day for hedge ratios that don't jitter.
- Cross-desk model consistency (swaption Bachelier vol reconcilable with cap Black vol).

All intellectually tractable. Hire the right four PhDs and ship.

### 2.5 Execution algorithms

Technically intense (microstructure, queue position, adverse selection, microsecond latency) but **organisationally the simplest** thing on the list — localised to one desk, independently testable via historical replay, self-policing economics (bad algos lose money visibly and are killed in days). The smartest people on a bank's tech floor work here, but the *programme management* is light compared to risk, capital, or finance integration.

### 2.6 The meta-point

The pattern is unmistakeable: **the hard things are not the components, they are the contracts between the components.** Every arrow between two systems is a bilateral agreement about identifiers, timestamps, calendars, sign conventions, currency precision, settlement, fixings, and corporate-action handling. There are typically 200–500 such contracts in a tier-1 bank. Each is owned by a different team with a different release cadence. The cost of changing any one of them grows faster than linearly in the number of others it touches.

That web of contracts is the moat that protects incumbent bank software. It is also the cost that grinds new programmes to a halt.

---

## 3. The seven choices VALAX commits to, and what each one buys

The architecture is not "JAX because JAX is cool." Each choice is a targeted response to one of the pain points in §2. The table below is the load-bearing claim of this entire document.

| Architectural choice | Pain it targets | What it buys you |
|---|---|---|
| **Pure functional, no mutable state.** Every pricing call is `(instrument, market) → price` with no side effects. | §2.1 (reproducibility / lineage / audit) and §2.2 (model governance) | Determinism by construction. Yesterday's risk number can be replayed bit-exactly from yesterday's inputs. The auditor's reconciliation question becomes trivial. The model risk team's "reproduce the validation pack" request becomes a function call. |
| **Single `MarketData` pytree.** One object carries every risk factor's level. | §2.1 (cross-system contracts) | Replaces dozens of bilateral "which version of which curve at which timestamp" contracts with a single typed object. Any downstream consumer (pricer, risk, capital, what-if) sees the *same* market state by construction. |
| **Autodiff Greeks via `jax.grad`.** No bump-and-reprice framework. | §2.1 (drift between pricing and risk) and §2.3 (XVA Greek consistency) | The single largest source of front-office vs middle-office disagreement is bump frameworks drifting out of sync with the pricer they shadow. With autodiff there is no separate framework — the pricer *is* the Greek engine. The trader's delta and the regulator's DV01 come from the same line of code. |
| **One JIT-compiled code path for the trader's hot price and the overnight risk run.** | §2.3 (what-if at trader speed) | Eliminates the "research vs production" duality. A pre-trade what-if uses the same compiled pricing function as the EOD risk batch; the only difference is the input book. Whatever speed-up you buy on the batch flows straight into the trader's quote latency. |
| **Integer ordinals + static day-count conventions in JAX-traced code.** | §2.1 (calendar / timestamp ambiguity) | Calendar bugs are the #2 source of cross-system inconsistency after identifier bugs. By forcing dates into JIT-safe integer ordinals with day-count conventions as static module fields, every pricing function declares its date semantics structurally — and the type system catches mismatches before they reach production. |
| **`equinox.Module` pytrees for every domain type** — curves, surfaces, models, scenarios, ladders, bucket maps. | §2.3 (joint factor consistency) | Uniform composability. `jit` / `vmap` / `grad` work without ceremony across the entire stack. Risk types and pricing types and calibration types all participate in the same pytree machinery, so a joint scenario across asset classes is one `MarketScenario` pytree, not a hand-rolled struct per asset class. |
| **GPU-first `vmap` over scenarios and instruments.** | §2.3 (XVA, full-revaluation what-if) | A 10,000-scenario × 10,000-instrument VaR is one JIT-compiled call, not a Python loop. The CVA nested-MC inner loop that takes a bank's grid overnight collapses to a daytime run on a single GPU. This is the single largest *quantitative* advantage VALAX has over QuantLib-style stacks. |

The crucial property is that these choices **compose multiplicatively, not additively**. Determinism + autodiff + single JIT path means an audit trail of a stress scenario is identical to a trader's pre-trade what-if is identical to the regulator's IMA backtest. They are not three different systems with reconciliation jobs between them — they are three call sites of the same function.

---

## 4. The compound effect

In a traditional bank stack:

```
Trader's price        =  pricer_v3.cpp + cache_v7 + fixings_db_at_now
EOD risk number       =  pricer_v3.cpp + bump_framework_v12 + curves_at_4pm
Regulatory capital    =  pricer_v3_archived.cpp + frtb_engine + curves_at_eod_t-1
What-if               =  separate "what-if" service with its own model loader
Audit replay          =  best-effort reconstruction from logs
```

Each row is a different system, with a different snapshot mechanism, a different cache, and a different reason it might disagree with the others. Most of the firm's quant headcount goes into making them agree. The reconciliation cost dominates the build cost over the lifetime of the system.

In VALAX, every row is *the same call*:

```python
# Trader hot price
price = V(instrument, market_now)

# EOD risk
ladder = compute_ladder(V, book, market_eod)

# Regulatory capital
bucketed = bucket_sensitivity_ladder(ladder, rate_bucket=frtb_vertices, ...)

# What-if
price_after = V(instrument, apply_scenario(market_now, hypothetical_shock))

# Audit replay
price_then = V(instrument, snapshot_at("2024-03-14T16:00Z"))
```

Same `V`. Same `market` type. Same JIT-compiled binary. The reconciliation problem doesn't go away in the real world — there are still booking systems and golden sources to reconcile — but the *internal* reconciliation surface area is reduced to a single boundary: did the right `MarketData` arrive at the input? Everything downstream is mechanical.

This is the architectural payoff in one sentence: **VALAX is designed so that the consistency tax inside the engine is zero, leaving the firm free to spend its quant budget on the consistency tax at its boundaries — where it actually has to be paid.**

---

## 5. What VALAX does not fix

Honest scoping. The following are real, important, and **outside the scope of any pricing library, including this one**:

| Problem | Why VALAX does not fix it |
|---|---|
| Reference-data identifier mapping | Lives upstream of any pricer; the firm needs a golden-source identifier service regardless of what library prices the trade. |
| Model risk governance (SR 11-7, SS1/23, TRIM) | Process, not code. VALAX makes validation cheaper (pure functions, deterministic outputs) but does not replace MRM. |
| Regulatory approval cycles for IMA / new product classes | Calendar-driven and human-driven. The most VALAX can do is shorten the *internal* preparation: validation packs, backtesting evidence, PLA tracking. |
| Trade lifecycle, settlement, corporate actions, collateral movements | VALAX values positions and computes their risk. It does not manage the trade after the value is computed. |
| Order management, execution, market-making policy | Out of scope by design (see Vision § Non-goals). VALAX is the pricing-and-risk layer underneath such systems, not a replacement. |
| Joint factor calibration across the entire firm | The *primitives* are in place (autodiff, joint pytree scenarios, PCA bucketing). The actual joint model — what factors, what dependence, what stress library — is a multi-year quant programme that consumes the library, it doesn't ship with it. |

The point of being explicit here is that the design rationale in §3 is a *targeted* claim, not a universal one. VALAX wins on the half of the problem that is amenable to architectural attack. The other half still requires a bank, with bank people, on a bank's timeline.

---

## 6. Implications for contributors

If you are extending VALAX, the design rationale above implies a small set of acceptance criteria for every new module. These are not style guidelines — they are the load-bearing properties:

1. **Every pricing function must be `jax.grad`-able.** If you need a workaround (`stop_gradient`, finite-difference fallback), the Greek consistency property in §3 breaks and the change should not land.
2. **Every domain type must be an `equinox.Module` pytree.** Anything that does not flatten / unflatten under `jax.tree_util` is incompatible with `vmap` across scenarios and instruments — i.e. with the compound-effect claim in §4.
3. **No global state, no caches that survive between calls, no observer pattern.** Determinism is by construction; any leak of state means yesterday's risk run might not replay tomorrow.
4. **Dates inside traced code are integer ordinals.** `datetime.date` may appear at the user-facing boundary and only there.
5. **No `scipy` inside JIT-traced code.** Use `optimistix` / `lineax` / `jax.scipy`. Otherwise the trader-hot-path / batch-risk equivalence in §4 silently breaks.
6. **New instruments must declare their risk-factor consumption** in [Risk Factors § 5](risk-factors.md#5-instrument-factor-matrix). This is what keeps the bucketing layer in [§ 7.8 of Theory](theory.md#78-risk-bucketing-linear-and-jacobian-transformations) honest at scale.

If any of these properties is hard to satisfy for a new module, that is a strong signal the module is fighting the architecture, and the right next step is to redesign the module rather than weaken the architecture.

---

## 7. Synthetic-first testing and non-tautological validation

A library whose architectural promise is "the trader's price, the EOD
risk number, and the audit replay all come from the same JIT-compiled
function" is only as good as the tests that defend that promise.
Two design decisions in the test layer are non-obvious and worth
documenting alongside the architectural ones above.

### 7.1 Tests run on synthetic data, not on real market data

Every dependency on Bloomberg / Reuters / a CSV checked into the repo
is a dependency that (a) eventually rots, (b) cannot be sampled in
sufficient density to catch corner cases, and (c) cannot be replayed
on a CI runner without leaking proprietary data.

The synthetic-market subpackage (`valax.market.synthetic`) produces
every input the library consumes — spots, vols, dividends, discount
curves, correlations, ground-truth model parameters, noisy quotes,
portfolios, market tapes, scenario sets — from a single `master_seed`.
Two registries with identical `(master_seed, library_version)` produce
identical bytes on every machine. The contract is **reproducibility
without network or filesystem access**.

This unlocks two things real market data cannot:

1. **Bit-exact reproducibility across CI runners, contributors, and
   regulators.** A test failure today reproduces tomorrow on a
   different laptop.
2. **Deliberately broken inputs.** With real data we can only see
   what the market produced; with synthetic data we can inject
   exactly the static-arbitrage violations the library should one day
   detect (`inject_non_psd_correlation`, `inject_calendar_arb`, etc.)
   and write the corresponding tests *before* the detector exists.

### 7.2 Non-tautological validation, not "round-trip the same model"

The most common test pattern in quant libraries is "generate data
with model X, calibrate model X to it, assert recovered parameters
match the originals." This pattern is **tautological**: it only
exercises the optimiser. The pricer and the calibrator share so much
code that a class of bugs (sign convention, time-scale convention,
missing factor of two) breaks both symmetrically and silently
satisfies the assertion.

VALAX deliberately avoids this pattern. The test layer is built
around four invariants that any sensible quant library must satisfy
*regardless of model choice*:

| Invariant | What it tests | What a failure means |
|---|---|---|
| **Pricer × implied-vol round-trip.** `vol_iv(price(σ_true)) ≈ σ_true`. | Both `black_scholes_price` and `black_scholes_implied_vol` in one shot. | One of the two drifted; the other can't compensate. |
| **Engine cross-consistency.** Analytic price vs MC price on the same `MarketData` agree to within `3·stderr`. | Two independent code paths to the same number. | A bug specific to one engine cannot hide behind the other. |
| **Autodiff Greeks vs central FD.** Same pricer, two independent gradient methods. | Decouples "we computed a Greek" from "we computed the right Greek". | Autodiff plumbing or pricer composition broke. |
| **Calibration residual at the noise floor.** Fit RMSE ≤ `1.5 ×` empirical observation noise RMSE. | The optimiser is at the noise floor, not under- or over-fitting. | Either calibrator convergence regressed or the pricer/vol formula drifted (because if either drifted the *clean* residual would no longer be near zero). |

The fifth pattern — **bootstrap non-self-roundtrip** — deserves its
own line: synthesise a truth NSS curve, read par-deposit rates at
tenors that *do not coincide with the NSS pillars*, bootstrap a new
curve from those quotes, and check the truth's zero rates are
recovered at a *third* set of dates. Three disjoint date grids
guarantee any passing test is real interpolation evidence, not a
self-roundtrip on the same coordinates.

### 7.3 Reserved exception types and the public detection backlog

Because static-arbitrage detection is not yet implemented inside the
library, the synthetic test layer references **reserved exception
types** in `valax.core.diagnostics` (`NonPSDCorrelationError`,
`ButterflyArbError`, `CalendarArbError`, …). None of them is raised
today. Five `@pytest.mark.xfail(strict=True)` markers in
`tests/test_market/test_arbitrage_handling.py` each expect one of
these errors; the day the corresponding constructor or validator
raises, the test flips from `xfail` to `pass` with no test change.

The xfail set is the **machine-readable, public backlog of missing
safety checks** — see the
[Arbitrage Detection — Session Backlog](roadmap.md#arbitrage-detection--session-backlog)
in the roadmap. Removing one xfail is a real engineering deliverable.

### 7.4 Versioned golden datasets, not implicit allclose

Numerical contracts that matter are pinned on disk under
`tests/golden/v{version}/` and indexed by
`tests/golden/golden_manifest.json` (sha256, library version, jax
version, master seed). The helper `assert_matches_golden(name, value,
*, version, rtol, atol)` refuses silent drift: an unintentional
change to a pricer's output requires either a version bump (a
deliberate, reviewable act) or a `REGEN_GOLDEN=1` regeneration run
(also deliberate). The model is "numerical contracts change in
PRs, not in git stash".

### 7.5 Acceptance criterion implied by the above

A contribution that adds a new pricer, model, calibrator, or
infrastructure component is expected to:

1. Drive its happy path through `valax.market.synthetic` rather than
   adding hard-coded numbers.
2. Add at least one non-tautological assertion from §7.2 (cross-engine
   or AD-vs-FD is usually the cheapest).
3. If it introduces a numerical contract worth pinning, register a
   golden via `assert_matches_golden` and ship the artifact + manifest
   entry in the same PR.
4. If it should *eventually* detect an arbitrage condition, add the
   corresponding `xfail` test now, using one of the reserved
   exception types — so the day the detector lands, the test is
   already there.

These properties are the test-layer analogue of the acceptance
criteria in §6 and serve the same goal: the architectural promise
holds in practice, not just in design.

---

## 8. Where to read next

- **The destination** these choices serve → [Vision](vision.md).
- **The time-ordered build-out** → [Roadmap](roadmap.md).
- **The production scaffolding** that operationalises this rationale → [Architecture: Production Design](architecture/production.md).
- **How the rationale plays out in the risk pipeline specifically** → [Risk: End-to-End § 6 *Why VALAX is built the way it is*](risk-overview.md#6-why-valax-is-built-the-way-it-is).
- **Concrete examples** of what the autodiff + pytree + vmap combination buys you → [Examples](examples.md).
- **The synthetic data + reproducibility + arbitrage test layer** in detail → [User Guide → Synthetic Market Data](guide/synthetic_market.md), [User Guide → Reproducibility & Arbitrage Tests](guide/reproducibility_and_arbitrage_tests.md), and [API → Market Data](api/market.md).

---

*This document changes slowly. It updates when the underlying claim about what is and is not hard about bank-scale quant systems changes — which, on the evidence of the last twenty years, is rarely.*
