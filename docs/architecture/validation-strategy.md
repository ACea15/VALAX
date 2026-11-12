# Validation Strategy

*Why the test layer is built the way it is: synthetic data instead of real market data, and checks that cannot pass by construction. This page is the rationale; the pages under **Validation** are the evidence.*

A library whose architectural promise is "the trader's price, the EOD risk number, and the audit replay all come from the same JIT-compiled function" ([Positioning § 4](../positioning/index.md#4-the-seven-choices-valax-commits-to-and-what-each-one-buys)) is only as good as the tests that defend that promise. Two design decisions in the test layer are non-obvious and worth stating alongside the architectural ones.

## 1. Tests run on synthetic data, not on real market data

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

How to drive it: [User Guide → Synthetic Market Data](../guide/synthetic_market.md).

## 2. Non-tautological validation, not "round-trip the same model"

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
| **Engine cross-consistency.** Analytic price vs MC price on the same `MarketState` agree to within `3·stderr`. | Two independent code paths to the same number. | A bug specific to one engine cannot hide behind the other. |
| **Autodiff Greeks vs central FD.** Same pricer, two independent gradient methods. | Decouples "we computed a Greek" from "we computed the right Greek". | Autodiff plumbing or pricer composition broke. |
| **Calibration residual at the noise floor.** Fit RMSE ≤ `1.5 ×` empirical observation noise RMSE. | The optimiser is at the noise floor, not under- or over-fitting. | Either calibrator convergence regressed or the pricer/vol formula drifted (because if either drifted the *clean* residual would no longer be near zero). |

The fifth pattern — **bootstrap non-self-roundtrip** — deserves its
own line: synthesise a truth NSS curve, read par-deposit rates at
tenors that *do not coincide with the NSS pillars*, bootstrap a new
curve from those quotes, and check the truth's zero rates are
recovered at a *third* set of dates. Three disjoint date grids
guarantee any passing test is real interpolation evidence, not a
self-roundtrip on the same coordinates.

The calibration-residual and bootstrap checks are worked through, with code, in [Reproducibility & Arbitrage Tests § Calibration-residual closed loops](../guide/reproducibility_and_arbitrage_tests.md#calibration-residual-closed-loops).

## 3. Guard rails: golden datasets and the detection backlog

Two further pieces of test infrastructure make drift and missing checks visible rather than silent. Both are documented in [Reproducibility & Arbitrage Tests](../guide/reproducibility_and_arbitrage_tests.md):

- **Versioned golden datasets.** Numerical contracts are pinned on disk under `tests/golden/v{version}/`, indexed by a manifest (sha256, library and jax versions, master seed), and checked with `assert_matches_golden`. An output change needs a deliberate version bump or `REGEN_GOLDEN=1` run: numerical contracts change in PRs, not in `git stash`. See [§ Golden manifest](../guide/reproducibility_and_arbitrage_tests.md#golden-manifest).
- **Reserved exception types as a public backlog.** Static-arbitrage detection is not yet implemented, so `valax.core.diagnostics` reserves exception types (`NonPSDCorrelationError`, `ButterflyArbError`, `CalendarArbError`, …) that nothing raises yet, each paired with an `@pytest.mark.xfail(strict=True)` test. The day a detector lands, its test flips to pass with no test change; removing an xfail is a real deliverable. See [§ Reserved exception types](../guide/reproducibility_and_arbitrage_tests.md#reserved-exception-types).

## 4. Acceptance criteria for contributions

A contribution that adds a new pricer, model, calibrator, or
infrastructure component is expected to:

1. Drive its happy path through `valax.market.synthetic` rather than
   adding hard-coded numbers.
2. Add at least one non-tautological assertion from § 2 (cross-engine
   or AD-vs-FD is usually the cheapest).
3. If it introduces a numerical contract worth pinning, register a
   golden via `assert_matches_golden` and ship the artifact + manifest
   entry in the same PR.
4. If it should *eventually* detect an arbitrage condition, add the
   corresponding `xfail` test now, using one of the reserved
   exception types — so the day the detector lands, the test is
   already there.

These are the test-layer analogue of the [invariants for contributors](overview.md#invariants-for-contributors), and serve the same goal: the architectural promise holds in practice, not just in design. The broader evidence base — the QuantLib comparison suite, the AAD audit and the benchmarks — is in the other pages of this section, starting with the [QuantLib Validation Pyramid](quantlib-validation-pyramid.md).
