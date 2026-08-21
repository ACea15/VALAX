# Rates Session Guide (TEMPORARY — delete or promote when the session lands)

> **Status: temporary working document.** This is a scratch plan to open a
> focused interest-rates development session cold. It is deliberately **not**
> wired into the mkdocs site. When the work it describes is done, either delete
> this file or promote the durable parts into `docs/roadmap.md` /
> `docs/architecture/`.

Companion reading (durable docs): the interest-rate *theory* lives in
`docs/theory.md` (§2.6 LMM, §2.8 Hull-White, §3 Curve Framework, and the new
Interest-Rate Derivatives section); the roadmap rates tiers are in
`docs/roadmap.md`.

---

## 1. Where rates stands today (map)

Maturity: **production** (validated, differentiable, tested) / **partial** /
**stub** / **missing**.

| Area | State | Where |
|---|---|---|
| **Multi-curve bootstrap** (joint Newton solver, OIS/tenor-basis/XCCY/FX/futures, quote Jacobians) | **production** | `valax/curves/bootstrap_graph.py`, `graph.py`, `instruments.py` |
| Single-curve bootstrap | production | `valax/curves/bootstrap.py` |
| `bootstrap_multi_curve` (sequential) | **deprecated** | `valax/curves/multi_curve.py` |
| **Analytic caplets/caps** (Black-76 + Bachelier, dual-curve) | **production** | `valax/pricing/analytic/caplets.py` |
| **Analytic swaptions** (Black-76 + Bachelier, dual-curve annuity) | **production** | `valax/pricing/analytic/swaptions.py` |
| **Bonds** (YTM, duration, convexity, KRDs, floating, OIS) | production | `valax/pricing/analytic/{bonds,floating}.py` |
| **Hull-White 1F** analytic exact-fit ZCB | production | `valax/models/hull_white.py` |
| **Hull-White trinomial tree** (callable/puttable bonds) | **production** | `valax/pricing/lattice/hull_white_tree.py` |
| **LMM/BGM** MC (PCA, Rebonato vol) | partial (MC only) | `valax/models/lmm.py`, `valax/pricing/mc/lmm_paths.py` |
| Bermudan swaption (LSM on LMM) | partial | `valax/pricing/mc/bermudan.py` |
| CMS / range-accrual analytic | **partial (no convexity adj)** | `valax/pricing/analytic/rates_exotics.py` |
| Inflation (ZC/YoY swaps, caps) | partial | `valax/pricing/analytic/inflation.py`, `valax/curves/inflation.py` |

## 2. The gaps that define the session

1. **No rate-model calibration at all.** `valax/calibration/` is equity-only
   (SABR/Heston/SLV). No Hull-White→swaptions (Jamshidian), no LMM calibration.
2. **No Hull-White MC** (`generate_hull_white_paths`) — roadmap MC-B1, unblocked.
3. **No rates PDE.** Roadmap "PR-3: HW short-rate PDE for callable bonds +
   Bermudan swaptions" is entirely future. *(The equity 2-D ADI substrate in
   `valax/pricing/pde/` is a natural springboard — `schemes.py`, `operators.py`,
   dispatch are mature.)*
4. **No G2++/HW-2F**, no Vasicek/CIR/Cheyette.
5. **Thin QuantLib validation.** Swaptions, caps, and the HW callable tree have
   **no QL cross-check** despite QL engines existing.
   `tests/test_quantlib_comparison/test_cap_strip_on_caplet_vols_ql.py` is a
   skipped placeholder.
6. **No rates vol-surface objects** (swaption cube / cap-floor optionlet surface;
   normal-vs-lognormal quoting abstraction).
7. **CMS/range-accrual ship without convexity adjustment** (documented caveats).

**Cross-cutting production prerequisite (not required for prototypes):** business
calendars + a stub/compounding-aware cashflow engine (roadmap P1.1/P1.2).

## 3. Recommended session sequence

Ordered to keep the **clean-oracle discipline** that caught the 33% Heston bug:
pin what exists to a reference *before* building on it.

1. **QuantLib validation harness for the existing rate pricers** *(highest
   readiness, no new pricing code)*. Add QL cross-checks:
   - swaptions vs `ql.Swaption` + `ql.BlackSwaptionEngine` / `BachelierSwaptionEngine`
   - caps/floors vs `ql.CapFloor` + Black/Bachelier engines
   - HW callable/puttable tree vs `ql.TreeCallableFixedRateBondEngine`
   Reuse `tests/test_quantlib_comparison/_ql_adapters.py` conventions.
   **Oracle:** QuantLib directly. De-risks everything below.

2. **Hull-White short-rate MC** (`generate_hull_white_paths`, ~60 LOC) + register
   `(FixedRateBond/FloatingRateBond/CallableBond/PuttableBond, HullWhiteModel)`
   MC recipes. **Oracle:** the existing HW trinomial tree + analytic ZCB
   (MC ↔ tree ↔ analytic triangulation).

3. **Hull-White swaption analytic pricer (Jamshidian) + calibration to the ATM
   swaption surface.** Add `hw_swaption_price` (Jamshidian decomposition on the
   affine ZCB), then `calibrate_hull_white` via `optimistix`.
   **Oracle:** the repo's Black-76 swaption pricer for the market leg + QL
   `JamshidianSwaptionEngine`. Depends on (1) and a swaption-vol quoting object.

4. **PR-3: Hull-White short-rate PDE** for callable/puttable bonds and Bermudan
   swaptions, built on the existing PDE substrate. **Oracle:** the HW tree and
   LSM. Sequence *after* (2)/(3) so MC, tree, and PDE cross-validate — exactly
   the multi-route pattern that made Heston smooth.

Later / lower priority: CMS convexity (Hagan replication; oracle QL
`AnalyticHaganPricer`); rates vol-surface objects (unblocks the skipped cap-strip
test); G2++/HW-2F.

## 4. Reusable machinery already in place

- **Curves + Jacobians:** `bootstrap_curve_graph`, `quote_jacobian` — discount
  factors, forwards, and their sensitivities are differentiable and QL-validated.
- **PDE substrate:** `valax/pricing/pde/{grids,operators,schemes,boundary,`
  `terminal,exercise,dispatch,recipes}.py` — 1-D theta-scheme + Rannacher +
  penalty/projection exercise seam, plus the 2-D ADI layer from PR-2.
- **Calibration substrate:** `valax/calibration/{loss,transforms}.py` +
  `optimistix` least-squares (used by SABR/Heston) — directly reusable for HW/LMM.
- **QL harness:** `tests/test_quantlib_comparison/_ql_adapters.py` (date snapping,
  flat curves, Act/365, `NullCalendar`).
- **Numerical debugging discipline:** `docs/architecture/numerical-pitfalls.md`.
