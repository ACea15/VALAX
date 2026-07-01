# Why the Multi-Curve Framework

This page answers a question the API reference cannot: **why does VALAX
carry a whole multi-curve solver (`bootstrap_curve_graph`, `CurveSpec`,
`CurveGraph`, `quote_jacobian`) instead of a single discount curve?**

Short answer: because the post-2008 rates market has instruments whose
no-arbitrage relations couple **multiple curves in a single equation**.
No ordering of one-curve bootstraps can honour those constraints. The
joint Newton solve is also the natural home for the implicit-adjoint
quote-Jacobian that a modern desk needs to hedge.

Long answer follows.

## The single-curve world (pre-2008)

Before 2008, a bank ran **one curve per currency**. USD LIBOR was
simultaneously:

- the *funding* rate you discounted at, and
- the *reference* rate on floating legs.

One curve, one bootstrap, done. `PV_{\text{float}} = DF(\text{start}) -
DF(\text{end})$ was an exact identity — floating and discounting share
the same curve so the sum telescopes.

## What changed in 2008

Three empirical forces broke the single-curve model. See
[`theory.md` §3.2](../theory.md#32-single-curve-vs-multi-curve-framework)
for the arbitrage-theoretic version of the same story.

### 1. LIBOR credit spread blew out

A 3M unsecured interbank loan is not risk-free — it carries bank credit
and liquidity premia that widen during stress. So LIBOR (and its
successors on term rates) is no longer a good proxy for the risk-free
rate. Empirically, the 3M-vs-1M compounded LIBOR basis stopped being
zero: it hit ~50 bp in late 2008 and stays 5–20 bp in normal times.

### 2. CSAs became universal

Almost every dealer trade is now collateralised daily under a Credit
Support Annex (CSA). The rate paid on posted collateral is the
**overnight index rate** — SOFR (USD), €STR (EUR), SONIA (GBP), TONA
(JPY). A replication argument (Piterbarg 2010) then shows the
arbitrage-free discount rate for the trade must equal the collateral
remuneration rate. So the discount curve is *OIS*, not LIBOR/EURIBOR.

### 3. Tenor basis is now a market instrument

Because different LIBOR/EURIBOR tenors carry different credit-liquidity
premia per unit time, the market quotes a non-zero **basis** between
them. Under a single-curve model that basis is zero by construction —
which is *arbitrage* in the real market. VALAX has to represent it.

## What a real desk runs today

A minimal USD desk carries **three** curves:

- `USD.SOFR.OIS` — the discount curve.
- `USD.SOFR.3M` — the 3M-projection curve.
- `USD.SOFR.6M` — the 6M-projection curve.

A EUR/USD desk carries at least four, plus FX links:

- `USD.SOFR.OIS`, `USD.SOFR.3M`
- `EUR.ESTR.OIS`, `EUR.EURIBOR.6M`
- FX spot + covered-interest-parity ties on the short end (via
  [`FXForward`](../api/curves.md#fxforward), [`FXSwap`](../api/curves.md#fxswap))
- cross-currency basis on the long end (via
  [`CrossCurrencyBasisSwap`](../api/curves.md#crosscurrencybasisswap-ccbs))

This collection of curves plus the identifier-keyed lookup is what
[`CurveGraph`](../api/curves.md#curvegraph) represents.

## Why sequential bootstrap breaks down

The pre-MC-Curves-2 code (`bootstrap_multi_curve`, now deprecated)
proceeds in a fixed order:

```
Step 1: Bootstrap OIS from OIS instruments               (one-curve solve)
Step 2: With OIS fixed, bootstrap 3M from IBOR swaps     (one-curve solve)
Step 3: With OIS+3M fixed, bootstrap 6M from 6M IBOR ... (one-curve solve)
```

This works **only when each new curve is constrained by instruments
that touch it plus previously-solved curves**. Two structural cases
break the ordering.

### Structural failure 1 — tenor-basis swaps

A 3M-vs-6M same-currency basis swap has this par condition (see
[`theory.md` §3.7](../theory.md#37-no-arbitrage-relations-across-curves)
for the derivation):

$$
\sum_i F^{\text{3M}}_i\,\tau^{\text{3M}}_i\,DF^{\text{OIS}}(T^{\text{3M}}_i)
\;=\;
\sum_j (F^{\text{6M}}_j + s)\,\tau^{\text{6M}}_j\,DF^{\text{OIS}}(T^{\text{6M}}_j)
$$

The forwards $F^{\text{3M}}_i$ come from the 3M curve, $F^{\text{6M}}_j$
from the 6M curve. This *one equation* constrains *both* forward curves
simultaneously. There is no ordering that recovers the right answer:

- **Build 3M first?** You have to guess where 6M sits.
- **Build 6M first?** You have to guess where 3M sits.
- **Build both from IBOR swaps first, cross-check basis later?**
  The basis quote is not respected — arbitrage in your model.

The only correct answer is a **joint solve** — put the 3M pillars, the
6M pillars, and the basis-swap residual into one Newton system.

### Structural failure 2 — cross-currency

A EUR-USD cross-currency basis swap touches **four** unknown curves at
once — `USD.SOFR.OIS`, `USD.SOFR.3M`, `EUR.ESTR.OIS`,
`EUR.EURIBOR.6M` — plus FX spot. See [`theory.md` §3.7 "Cross-currency
basis"](../theory.md#37-no-arbitrage-relations-across-curves) for the
XCCY equation.

There is **no** ordering that lets you finish one currency then the
other, because the CCBS quote *is* the market's statement about the
relationship between the two currencies' curves. A sequential pipeline
has to either

- ignore the CCBS quote (then EUR is inconsistent with USD in your
  model, and every EUR-USD quanto/XCCY trade is mispriced), or
- hard-code an FX assumption (then you're making up the answer).

The FX forward has the same structural issue on the short end: it
couples both currencies' OIS curves through covered interest parity in
a single equation.

## Enter the joint Newton solve

[`bootstrap_curve_graph`](../api/curves.md#bootstrap_curve_graph)
concatenates the log-DFs of every curve into one flat state vector
$\mathbf{x} \in \mathbb{R}^N$ and runs `optimistix.Newton` to zero every
instrument's residual simultaneously:

$$
\mathbf{R}(\mathbf{x}) = \mathbf{0}
\quad\text{where}\quad
R_i(\mathbf{x}) = \text{Pricing}_i\big(\{C^{(k)}\}_{k\in\mathcal{T}(i)}\big) - \text{Quote}_i
$$

$\mathcal{T}(i)$ is the set of curves instrument $i$ touches, declared
via its static `curves_touched` tuple. The solver does *not* branch on
instrument type — it just asks each instrument for its residual, which
is what the [`BootstrapInstrument` protocol](../api/curves.md#bootstrapinstrument-protocol)
requires.

Because each instrument owns its residual function, the solver's code
path is identical for `DepositRate`, `TenorBasisSwap`, and
`CrossCurrencyBasisSwap`. New quote types register by conforming to the
protocol; the solver never learns about them.

The joint residual system is derived in
[`theory.md` §3.8](../theory.md#38-joint-multi-curve-calibration) with
a worked EUR-USD 28-unknown example.

## Bonus: implicit-adjoint quote-Jacobian

Wrapping the Newton solve in `optimistix.root_find` gives you
`ImplicitAdjoint` for free: `jax.grad` and `jax.jacrev` propagate
through the calibrated graph via the implicit function theorem, using
one linear solve of the same Jacobian the Newton iteration already
factorised:

$$
\frac{\partial \mathbf{x}^*}{\partial \text{Quote}_i}
= -\,\mathbf{J}^{-1}\,\frac{\partial \mathbf{R}}{\partial \text{Quote}_i}
$$

The consequence is that
[`quote_jacobian`](../api/curves.md#quote_jacobian) delivers the full

$$
J_{ki} = \frac{\partial (\text{DF at pillar } k)}{\partial (\text{market quote } i)}
$$

matrix in *one linear solve per output* — independent of Newton
iteration count, and *linear* rather than *quadratic* in the graph size.

This is the matrix a rates desk uses to convert a model-computed DV01
on an exotic into a **hedge portfolio of liquid instruments**. Cheap
via autodiff, prohibitively expensive via finite differences (each FD
column would need one full re-bootstrap of every curve in the graph).

## Concrete evidence from the test suite

`tests/test_curves/test_bootstrap_graph.py` demonstrates each case
end-to-end. Verdicts:

| Case | Sequential pipeline | Joint solver |
|---|---|---|
| Single-curve deposit strip | Works | Agrees to `0.0` (machine precision) |
| Dual-curve OIS + 3M IBOR swap | Works (was the original `MultiCurveSet` use case) | `max_abs_residual ≈ 2.4e-16` |
| **Three-curve tenor-basis** (OIS + 3M + 6M) | **Cannot represent the joint constraint** | Converges to ≤ 1e-10 |
| **Four-curve EUR/USD via CCBS + FXForward** | **Cannot even be attempted** | Converges to ≤ 1e-10 |
| `jax.grad` through the solver | Would need FD (one full rebootstrap per quote) | One linear solve via `ImplicitAdjoint` |

The last two rows are the entire justification for shipping
MC-Curves-2. Without them, VALAX would systematically misprice every
tenor-basis and every cross-currency trade a desk carries.

## What MC-Curves-2 shipped

- [`CurveSpec`](../api/curves.md#curvespec) — declarative curve descriptor
  with alphabet-validated identifiers.
- [`bootstrap_curve_graph`](../api/curves.md#bootstrap_curve_graph) —
  joint Newton solve, `optimistix.Newton` + `ImplicitAdjoint`.
- [`CurveBuildDiagnostics`](../api/curves.md#curvebuilddiagnostics) —
  residuals, iteration count, converged flag.
- [`quote_jacobian`](../api/curves.md#quote_jacobian) — implicit-adjoint
  quote-sensitivity matrix in three parametrisations
  (`log_df` / `df` / `zero_rate`).
- `bootstrap_multi_curve` retained one deprecation cycle with a
  `DeprecationWarning`; `bootstrap_simultaneous` migrated to a thin
  wrapper over the joint solver.
- Curve-identifier alphabet frozen as
  `<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]` (see
  [`production.md` §11.3](../architecture/production.md#113-curve-graph-data-model)).

## Where to go next

- **Recipes** — [`curves.md` §5](curves.md#5-multi-curve-bootstrap)
  walks through a USD dual-curve build with `bootstrap_curve_graph`.
- **API** — [`api/curves.md`](../api/curves.md#curvespec) documents every
  argument and return type of the joint solver.
- **Math** — [`theory.md` §3.7](../theory.md#37-no-arbitrage-relations-across-curves)
  derives every residual (CIP, tenor basis, XCCY).
  [`theory.md` §3.8](../theory.md#38-joint-multi-curve-calibration)
  formalises the joint system.
- **Design** — [`production.md` §11](../architecture/production.md#11-multi-curve-framework)
  documents the data model, joint-solver interface, and phased delivery.
- **Roadmap** — [`roadmap.md` §1.1](../roadmap.md#11-multi-curve-bootstrapping)
  tracks what's shipped and what's next (MC-Curves-3: extended diagnostics,
  monotone-convex interpolation, HW convexity, CSA).
