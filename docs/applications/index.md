# Applications

*A map of who inside a financial institution actually uses VALAX, why they use it, and where in the library to find the tools they need — organised by audience rather than by module.*

The rest of the documentation is organised by *what* the library does — pricing, curves, risk, calibration. This section is organised by *who* uses it. If you are evaluating VALAX for a specific function inside a bank, an asset manager, a hedge fund, or an academic group, start here and jump to the sub-page that fits your role.

---

## 1. The sixty-second story: one pipeline, many audiences

Every function of the library reduces to one primitive:

```
V(instrument, MarketData) → price
```

— a pure JAX function you can differentiate, JIT-compile, and `vmap`. Every downstream capability is a derived quantity on top of that primitive:

- **Greeks** are `jax.grad(V)`.
- **Portfolios** are `jax.vmap(V)`.
- **VaR / ES** are sample statistics on a P&L vector produced by scenario `vmap`.
- **Regulatory capital** is a bucketed contraction of the sensitivity ladder.
- **Neural surrogates** train against the pricer's own gradient as ground truth.
- **Deep-hedging policies** learn against the gradient of P&L through the SDE simulation.

*The same code path* serves the trader's morning delta, the risk manager's VaR, the model validator's challenger benchmark, the treasurer's IRRBB report, the regulator's FRTB submission, and the researcher's neural-SDE calibration. That "one pipeline, many audiences" property is what these application pages document.

For the systems around VALAX in the bank stack (order-management, market data, settlement), see [Where VALAX Fits](../landscape.md). For the risk engine's internal architecture, see [Risk: End-to-End](../risk-overview.md). For the deep motivation behind each design decision, see [Design Rationale](../design-rationale.md).

---

## 2. The audience map — ranked by fit today

Not every function in a financial institution is a good fit for VALAX today. Some are 🟢 immediate wins; some are 🟡 near-term with a roadmap dependency; some are 🔴 out of scope by design. The honest ranking:

| Rank | Audience | Fit today | Why | Sub-page |
|---|---|---|---|---|
| 🥇 | **Market Risk & Model Validation** | 🟢🟢🟢 | Regulator-mandated need; no incumbent lock-in; batch cadence; `valax/risk/` is the most complete module in the library | [Market Risk & Model Validation](market-risk.md) |
| 🥇 | **Quant Research / Structuring / Model R&D** | 🟢🟢🟢 | Differentiable pricing is a categorical advantage — enables deep hedging, neural surrogates, gradient-based calibration that C++ analytics libraries structurally cannot support | [Quant Research](quant-research.md) |
| 🥈 | **Treasury / ALM** | 🟢🟢 | IRRBB Standardised Approach is a near-perfect fit; HQLA pricing is turnkey; multi-curve is the roadmap gap | [Treasury & ALM](treasury.md) |
| 🥈 | **Regulatory reporting (FRTB / CCAR)** | 🟢🟢 | Bucketed sensitivities, PLA test, VaR backtesting all shipped; SBA aggregation on roadmap | [Regulatory: FRTB & CCAR](regulatory.md) |
| 3 | **Front-office trading (execution)** | 🟡 | Latency-sensitive; incumbent-locked; needs a service layer + streaming market data that are Vision-tier items | — |
| 3 | **Back office / operations** | 🔴 | Wrong domain — not what the library does, by design | — |

There are two rank-1 audiences, and they represent two different theses about the library:

- **Market Risk & Model Validation** is the strongest **commercial** adoption case — regulator-mandated need, no incumbent lock-in, budget in hand.
- **Quant Research** is the strongest **technical** adoption case — differentiable pricing enables research directions that are structurally out of reach for C++ analytics libraries.

Both matter. A bank considering VALAX seriously should adopt in both simultaneously (see [§ 4](#4-the-strategic-order-of-operations) below).

---

## 3. The four sub-pages

Each sub-page below is written for a specific audience — role, cadence, KPIs, and the buyer-tailored pitch. Read the one that matches your function first; skim the others for context.

### 🥇 [Market Risk & Model Validation](market-risk.md)

**For:** Middle-office Market Risk Management (MRM) and second-line Model Validation (SR 11-7 / TRIM).

The daily MRM production loop — sensitivity ladders, waterfall P&L explain, full-revaluation VaR/ES, Basel backtesting, limits monitoring, FRTB PLA — all in one code path. Plus the Model Validation angle: an independent-by-construction challenger pricer in Python + JAX rather than C++, with a QuantLib comparison suite already in the tree covering options, fixed income, Heston, SABR, caps, swaptions, MC, PDE, lattice, SLV, and Greeks. Deterministic replay for supervisory inspections.

### 🥇 [Quant Research](quant-research.md)

**For:** Front-office quants, structuring desks, model R&D teams, hedge-fund systematic PMs, academic researchers.

The argument that differentiable pricing is not a "nice implementation choice" but the primary source of research-productivity leverage: every new pricer ships with all Greeks; a new payoff is a Python function, not a class subtree; calibration is `optimistix.least_squares` on the autodiff Jacobian; neural surrogates and deep hedging become one-file prototypes rather than multi-quarter engineering projects. The paper-to-calibrated-prototype loop collapses from months to hours. This is the innovation-track adoption case and — arguably — the strongest technical fit of any audience.

### 🥈 [Treasury & ALM](treasury.md)

**For:** Head of ALM, Treasurers, banking-book risk teams, LCR/NSFR reporting teams, HQLA portfolio managers.

The IRRBB Standardised Approach walkthrough — the six BCBS 368 scenarios map 1-to-1 to VALAX primitives (`parallel_shift`, `steepener`, `flattener`, `key_rate_bump`) and ΔEVE + SOT + KRD bucketing are one call each. Plus HQLA / investment-portfolio pricing, cash-flow bucketing for LCR / NSFR, and the honest gap on behavioural models (deposit pass-through, prepayment) — which are the natural home for the JAX moat once someone builds them.

### 🥈 [Regulatory: FRTB & CCAR](regulatory.md)

**For:** Regulatory reporting teams, capital planning, CROs, supervisory dialogue.

The two flagship trading-book regulatory workflows — Basel III.1 FRTB (SBA and IMA, PLA, backtesting) and the Fed's annual CCAR / DFAST stress tests. Both start from the same VALAX pipeline; they differ only in how bucketing and scenarios are configured. Coverage-vs-roadmap tables are explicit about what ships today (autodiff Greeks, PLA test, bucketing, VaR backtest, ES) and what is on the roadmap (SBA correlation aggregation, multi-curve `MarketData`, automated IMA stress-period selection).

---

## 4. The strategic order-of-operations

For a bank considering serious adoption, the sequence that maximises return and minimises risk:

1. **Model Validation first.** SR 11-7 / TRIM already require an independent challenger; VALAX is architected to be one. The QuantLib comparison suite is a shipped evidence pack. Adoption here is the lowest-friction, highest-defensibility entry point.
2. **Simultaneously: Quant Research adoption** — different budget, different buyer, provides the frontier-capability story (neural surrogates, deep hedging, differentiable calibration). The two teams cross-pollinate: research prototypes graduate into validated production without a rewrite because it is the same library.
3. **Expand into MRM** once Model Validation is a happy internal reference. Same code path, adjacent workflow, natural cross-sell to the CRO.
4. **Treasury / ALM** once multi-curve `MarketData` ships (roadmap top priority).
5. **Regulatory reporting** falls out as an *output* of steps 1–4 — the FRTB PLA, VaR backtest, and CCAR waterfall are computations on top of the same pipeline.
6. **Front-office trading** — Vision-tier. Pursue only after the service layer, market-data adapters, and JIT warm-up strategy described in [architecture/production.md](../architecture/production.md) are in place.
7. **Back office** — never. Out of scope by design.

The through-line: **do not start with trading, and do not start with regulatory reporting**. Start with the two functions that are structurally set up to benefit (Model Validation, Quant Research) and let the rest of the stack pull adoption naturally as the library becomes an internal reference.

---

## 5. Where to read next

- **The systems around VALAX in a bank stack** → [Where VALAX Fits](../landscape.md).
- **The end-to-end risk pipeline these applications sit on** → [Risk: End-to-End](../risk-overview.md).
- **The mathematical foundations** → [Models & Theory](../theory.md).
- **The concrete code walkthroughs** → [User Guide](../guide/tutorial-rates.md).
- **The forward direction** → [Vision](../vision.md), [Roadmap](../roadmap.md).
- **Why the library makes the architectural choices it does** → [Design Rationale](../design-rationale.md).
