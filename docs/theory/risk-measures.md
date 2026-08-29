[← Models & Theory index](index.md)

# 7. Risk Measures

### 7.1 Value at Risk (VaR)

**Definition:** The $\alpha$-level VaR over holding period $h$ is the loss $l$ such that:

$$
\mathbb{P}(\text{Loss} > l) = 1 - \alpha
$$

For example, 99% 10-day VaR = $10M means there is a 1% chance of losing more than $10M over 10 days.

**Parametric (delta-normal) VaR** (implemented in `valax/risk/var.py`, `parametric_var`):

Assume P&L is linear in risk factors and risk factors are jointly normal:

$$
\text{VaR}_\alpha = z_\alpha \sqrt{\boldsymbol{\delta}^T \boldsymbol{\Sigma} \boldsymbol{\delta}}
$$

where $\boldsymbol{\delta}$ is the vector of portfolio sensitivities (from autodiff), $\boldsymbol{\Sigma}$ is the covariance matrix of risk factor changes, and $z_\alpha$ is the normal quantile.

**Assumptions:** Linearity (no gamma), normality (no fat tails), static portfolio. These are strong assumptions that break for options portfolios (significant gamma) and in stressed markets (fat tails).

**Full-revaluation VaR** (implemented in `valax/risk/var.py`, `value_at_risk`):

Reprice the portfolio under each scenario (historical or simulated), sort the P&L distribution, and read off the quantile. No linearity or normality assumption. VALAX uses `jax.vmap` to reprice across all scenarios in parallel.

**VaR is not coherent:** VaR violates the **subadditivity** axiom: $\text{VaR}(A + B)$ can exceed $\text{VaR}(A) + \text{VaR}(B)$. This means diversification can appear to *increase* risk under VaR, which is economically nonsensical. This is why regulators (Basel III.1) are shifting to Expected Shortfall.

### 7.2 Expected Shortfall (CVaR)

**Definition:** The expected loss conditional on exceeding VaR:

$$
\text{ES}_\alpha = \mathbb{E}[\text{Loss} \mid \text{Loss} > \text{VaR}_\alpha]
$$

Equivalently, it is the average of all losses in the worst $(1-\alpha)$ tail.

**Properties:**

- **Coherent risk measure** — satisfies subadditivity, monotonicity, positive homogeneity, and translation invariance
- Always $\geq$ VaR — it captures tail severity, not just tail probability
- More sensitive to the shape of the tail distribution
- Now required by Basel III.1 FRTB for market risk capital calculation

**VALAX implementation:** `valax/risk/var.py`, `expected_shortfall`. Computed from the same P&L distribution as VaR — sort, take the mean of the worst $(1-\alpha)$ fraction.

### 7.3 P&L Attribution

**Second-order Taylor expansion** (implemented in `valax/risk/var.py`, `pnl_attribution`):

$$
\Delta V \approx \sum_i \frac{\partial V}{\partial x_i}\Delta x_i + \frac{1}{2}\sum_{i,j}\frac{\partial^2 V}{\partial x_i \partial x_j}\Delta x_i \Delta x_j
$$

This decomposes P&L into contributions from each risk factor (delta P&L) and second-order effects (gamma/cross-gamma P&L). The residual (actual P&L minus Taylor approximation) is the **unexplained P&L** — it should be small for well-understood portfolios.

VALAX computes all first and second derivatives via `jax.grad` and `jax.hessian`, making P&L attribution exact to second order with no finite-difference noise.

### 7.4 Sensitivity Ladders

A **sensitivity ladder** is a bucketed grid of Greeks across risk factor dimensions — tenor, maturity, moneyness, or curve index. Each bucket carries both first-order (delta/vega/rho) and second-order (gamma/vanna/volga) terms, enabling a multi-rung P&L decomposition that captures nonlinear effects missed by a scalar delta-only explain.

The term "ladder" comes from how each sensitivity is laid out on a grid — a delta ladder, for example, might show $\partial V / \partial r_i$ at each of the 12 standard tenor buckets (2W, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y), forming a "rung" at each bucket. Stacking delta, gamma, vega, vanna, and volga ladders produces the full risk picture.

#### Why Banks Use Ladders

A scalar P&L explain — "you lost \$2M on vega" — is insufficient for a trading desk. Ladders provide:

1. **Granular attribution**: Not just vega P&L, but *which expiry/tenor buckets* drove the loss
2. **Nonlinear accuracy**: As options approach expiry, the gamma P&L rung dominates; for large vol moves, vanna and volga rungs become material
3. **Regulatory alignment**: ISDA SIMM and FRTB SA require bucketed sensitivities at standard vertices, separately bucketed by currency, tenor, and curve index

#### The Waterfall Decomposition

The waterfall decomposes a scenario's P&L into successive rungs of increasing refinement. Given risk factor changes $\Delta S$ (spot), $\Delta\sigma$ (vol), $\Delta r$ (rates), and $\Delta q$ (dividends):

$$
\Delta V \approx \underbrace{\sum_i \frac{\partial V}{\partial S_i} \Delta S_i}_{\text{Rung 1: Delta (spot)}} + \underbrace{\sum_j \frac{\partial V}{\partial \sigma_j} \Delta \sigma_j}_{\text{Rung 2: Vega}} + \underbrace{\sum_k \frac{\partial V}{\partial r_k} \Delta r_k}_{\text{Rung 3: Rho / DV01}} + \underbrace{\sum_i \frac{\partial V}{\partial q_i} \Delta q_i}_{\text{Rung 4: Dividend}}
$$

$$
+ \underbrace{\frac{1}{2}\sum_i \frac{\partial^2 V}{\partial S_i^2} \Delta S_i^2}_{\text{Rung 5: Gamma (spot)}}
+ \underbrace{\frac{1}{2}\sum_k \frac{\partial^2 V}{\partial r_k^2} \Delta r_k^2}_{\text{Rung 6: Rate convexity}}
$$

$$
+ \underbrace{\sum_i \frac{\partial^2 V}{\partial S_i \partial \sigma_i} \Delta S_i \Delta \sigma_i}_{\text{Rung 7: Vanna}}
+ \underbrace{\frac{1}{2}\sum_j \frac{\partial^2 V}{\partial \sigma_j^2} \Delta \sigma_j^2}_{\text{Rung 8: Volga}}
$$

$$
+ \underbrace{\sum_{i,k} \frac{\partial^2 V}{\partial S_i \partial r_k} \Delta S_i \Delta r_k}_{\text{Rung 9: Cross spot×rate}}
+ \underbrace{\sum_{j,k} \frac{\partial^2 V}{\partial \sigma_j \partial r_k} \Delta \sigma_j \Delta r_k}_{\text{Rung 10: Cross vol×rate}}
$$

The **predicted P&L** is the sum of all 10 rungs. The **unexplained** is the residual between the predicted and actual (full-repricing) P&L. For small moves, the unexplained is dominated by third-order terms and is typically less than 1% of the actual P&L. For large moves (crashes, rate shocks), the unexplained grows, signaling that the Taylor approximation is breaking down — this itself is a useful risk signal.

#### Computational Cost with Autodiff

In a traditional bump-and-reprice system:

- **First-order ladder** ($N$ risk factors): $2N$ repricings (central differences)
- **Second-order diagonal** ($N$ gammas): $3N$ repricings
- **Full cross-gamma matrix** ($N \times N$): $N^2$ repricings
- **Total for a 50-factor portfolio**: ~2,500+ repricings

In VALAX:

- **First-order ladder**: 1 reverse-mode `jax.grad` pass ($\approx 3\times$ one pricing)
- **Full Hessian** (all second-order terms): 1 `jax.hessian` pass ($\approx 3N\times$ one pricing)
- **Total for a 50-factor portfolio**: ~150× one pricing

This is a structural advantage that compounds with portfolio size. The ladder that takes a traditional system thousands of bump-and-reprice calls falls out of two autodiff passes.

**VALAX implementation:** `valax/risk/ladders.py` provides `SensitivityLadder` (bucketed pytree of all sensitivities), `compute_ladder()` (Jacobian + Hessian computation), `waterfall_pnl()` (arithmetic decomposition from a precomputed ladder), and `waterfall_pnl_report()` (full decomposition including repricing for the unexplained). The ladder can be computed once and reused across many scenarios — the waterfall arithmetic is instantaneous.

### 7.5 P&L Vectors: Hypothetical, Risk-Theoretical, Actual

VaR, ES, capital, and every backtest in this section all collapse to the same primitive: a **P&L vector** $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_N) \in \mathbb{R}^N$ representing portfolio P&L under $N$ scenarios or $N$ historical observation periods. Once the vector exists, the risk numbers are just simple sample statistics over it.

#### Three flavours of P&L

Regulators (FRTB / BCBS d352, d457, d558) distinguish three P&L series produced by the same portfolio under the same set of risk-factor moves:

| Series | Symbol | How it is computed | What it measures |
|---|---|---|---|
| **Actual P&L (APL)** | $\Delta V^{\text{APL}}$ | End-of-day P&L from the official books: includes intraday trading, new deals, fees, valuation adjustments | What the desk actually made/lost |
| **Hypothetical P&L (HPL)** | $\Delta V^{\text{HPL}}$ | Reprice **today's** portfolio under tomorrow's market data; *no* intraday flow, *no* fees, *no* CVA/FVA noise | Clean P&L caused purely by market moves |
| **Risk-Theoretical P&L (RTPL)** | $\Delta V^{\text{RTPL}}$ | Predicted P&L from the risk engine: ladder Greeks $\times$ risk-factor changes | What the risk model *thinks* should have happened |

The hierarchy goes APL → HPL by stripping non-market noise; HPL → RTPL by replacing full revaluation with a Taylor expansion. The differences between the three series are themselves diagnostic signals:

$$
\underbrace{\Delta V^{\text{APL}} - \Delta V^{\text{HPL}}}_{\text{trading / new-trade P&L}}
\qquad
\underbrace{\Delta V^{\text{HPL}} - \Delta V^{\text{RTPL}}}_{\text{unexplained = model error}}
$$

A persistently large unexplained signals missing risk factors, mis-specified shocks, or a portfolio with material third-order convexity.

#### Building HPL vectors

Given a base market state $M_0$ and $N$ scenarios $\{s_i\}_{i=1}^N$ — either historical observed daily changes or Monte Carlo simulated ones — HPL is obtained by full revaluation:

$$
\pi_i^{\text{HPL}} = V\!\left(M_0 \oplus s_i\right) - V(M_0)
$$

where $M_0 \oplus s_i$ is the shocked market and $V(\cdot)$ is the full pricing function. **VALAX implementation:** `portfolio_pnl()` and the alias `hypothetical_pnl_vector()` in `valax/risk/pnl_vectors.py` compute this in one `jax.vmap` pass over the scenario axis.

#### Building RTPL vectors from a sensitivity ladder

The waterfall decomposition (§7.4) gives the predicted P&L for a single scenario. Stacking the same ten-rung formula across all scenarios produces the RTPL vector:

$$
\pi_i^{\text{RTPL}} = \sum_{k=1}^{10} R_k(s_i; \mathcal{L})
$$

where $\mathcal{L}$ is the precomputed sensitivity ladder and $R_k$ is rung $k$. Crucially, **the ladder is computed once** (one Jacobian + one Hessian pass on the base market), then reused across all $N$ scenarios — each scenario costs only a contraction of ladder arrays with the scenario's shock vectors. For $N = 10\,000$ scenarios and a portfolio with $F$ risk factors, the cost is

$$
\text{Ladder build:}\quad \mathcal{O}(F^2)\cdot C_{\text{pricing}}
\qquad
\text{Vector fill:}\quad \mathcal{O}(N\cdot F^2)
$$

compared with $\mathcal{O}(N \cdot C_{\text{pricing}})$ for HPL — typically two to three orders of magnitude cheaper for option-heavy portfolios where $C_{\text{pricing}}$ is dominated by special-function evaluations. **VALAX implementation:** `risk_theoretical_pnl_vector()` in `valax/risk/pnl_vectors.py`.

#### From P&L vector to risk metrics

Every risk measure is a sample statistic on the P&L vector:

$$
\text{VaR}_\alpha = -\hat{F}^{-1}(1-\alpha),\qquad
\text{ES}_\alpha = -\hat{\mathbb{E}}[\Delta V \mid \Delta V \le -\text{VaR}_\alpha]
$$

with $\hat F$ the empirical CDF of the P&L vector. The choice of vector type drives the interpretation: VaR from an HPL vector built on the past 250 trading days is **historical-simulation VaR**; from a parametric MC vector it is **Monte Carlo VaR**; from an RTPL vector it is the **risk-engine's prediction** of either, used in FRTB backtesting (§7.7).

In the FRTB Internal Models Approach, both ES and the scaled-up stress ES are produced from RTPL vectors (the desk's risk engine is responsible for the prediction), while the daily backtest compares HPL to model VaR (§7.6) and the PLA test compares HPL to RTPL (§7.7) — three different uses of the same vector primitive.

### 7.6 VaR Backtesting

VaR is a forecast. A forecast that is never compared to outcomes is, in regulatory language, a model not yet validated. **Backtesting** is the systematic comparison of VaR forecasts to realized P&L. The Basel framework requires daily 99% one-day VaR to be backtested over a rolling window of 250 trading days, with capital multipliers tied to the number of breaches.

#### Breaches and unconditional coverage

A **breach** (or exception) on day $t$ occurs when the loss exceeds the VaR forecast:

$$
I_t = \mathbf{1}\!\left\{ -\Delta V_t^{\text{HPL}} > \text{VaR}_t \right\}
$$

Under the null hypothesis that the model is correctly specified at the $\alpha$ confidence level, breaches form an i.i.d. Bernoulli sequence with $p = 1 - \alpha$. For 99% VaR over $n$ days, the expected number of breaches is $np$ — e.g., 2.5 breaches per 250-day window.

**Kupiec's proportion-of-failures (POF) test** (Kupiec 1995) is the likelihood-ratio test for unconditional coverage:

$$
\text{LR}_{uc} = -2\ln\!\frac{(1-p)^{n-x}\,p^{\,x}}{(1-\hat p)^{n-x}\,\hat p^{\,x}}
\;\;\overset{a}{\sim}\;\; \chi^2_1
$$

where $x = \sum_t I_t$ is the observed breach count and $\hat p = x/n$ is the empirical breach rate. Reject the model if $\text{LR}_{uc}$ exceeds the $\chi^2_1$ critical value at the desired significance level (3.84 at 5%).

The POF test is unconditional: it cares only about the *count* of breaches, not their timing. A model that produces 2 breaches per year, both in the same week, passes the POF test but is clearly mis-specified — breaches should not cluster.

#### Conditional coverage and independence

**Christoffersen's independence test** (Christoffersen 1998) detects breach clustering by treating $I_t$ as a first-order Markov chain with transition matrix

$$
\Pi = \begin{pmatrix} 1-\pi_{01} & \pi_{01} \\ 1-\pi_{11} & \pi_{11} \end{pmatrix}
$$

where $\pi_{ij} = \Pr(I_t = j \mid I_{t-1} = i)$. Under the null of independence, $\pi_{01} = \pi_{11} = \hat p$ (no memory). The LR statistic is

$$
\text{LR}_{ind} = -2\ln\!\frac{(1-\hat\pi)^{n_{00}+n_{10}}\,\hat\pi^{n_{01}+n_{11}}}{(1-\hat\pi_{01})^{n_{00}}\,\hat\pi_{01}^{n_{01}}\,(1-\hat\pi_{11})^{n_{10}}\,\hat\pi_{11}^{n_{11}}}
\;\;\overset{a}{\sim}\;\; \chi^2_1
$$

with $n_{ij}$ the count of $(I_{t-1}, I_t) = (i, j)$ transitions and $\hat\pi_{ij} = n_{ij} / (n_{i0} + n_{i1})$. Christoffersen combines this with Kupiec into the **conditional coverage** test:

$$
\text{LR}_{cc} = \text{LR}_{uc} + \text{LR}_{ind}\;\;\overset{a}{\sim}\;\;\chi^2_2
$$

which is the joint test of correct breach rate *and* independent breaches.

#### Basel traffic light

For regulatory capital, Basel ignores the LR machinery and uses a simple count-based zoning over 250 days at the 99% level (BCBS 1996; carried forward into FRTB BCBS d352 for the standardized backtest):

| Zone | Breaches in 250 days | Interpretation | Capital multiplier |
|---|---|---|---|
| Green | 0–4 | Acceptable | $\times 3.00$ |
| Yellow | 5–9 | Investigation required | $3.00 \to 3.85$ (sliding scale) |
| Red | ≥10 | Model rejected | $\times 4.00$ |

The thresholds come from the cumulative binomial distribution: under the null $p = 0.01$, the probability of observing at most $x$ breaches in 250 trials crosses the 95% point between 4 and 5 breaches, and the 99.99% point between 9 and 10 breaches.

#### What backtesting cannot do

- **Power is low** for short windows. Distinguishing a 0.5%-tail model from a 1.5%-tail model with only 250 observations requires substantial mis-specification before the LR test rejects.
- **Breach data is censored** by trading: end-of-day VaR vs end-of-day P&L misses intraday breaches that were unwound before the close.
- **Tail severity is ignored**. Five breaches that average $-1.1 \times \text{VaR}$ count the same as five at $-10 \times \text{VaR}$, but the second portfolio is far riskier. This is one of the motivations for backtesting ES as well, and for the FRTB PLA test below.

**VALAX implementation:** `valax/risk/backtesting.py` provides `var_breaches()`, `kupiec_pof()`, `christoffersen_independence()`, `christoffersen_conditional_coverage()`, and `basel_traffic_light()`.

### 7.7 FRTB P&L Attribution Test

The Basel III.1 / FRTB Internal Models Approach (IMA) introduces a **second** model-validation requirement on top of the VaR backtest: each trading desk must demonstrate that its risk model's predicted P&L (RTPL) tracks the desk's clean realized P&L (HPL) on a daily basis. This is the **P&L Attribution (PLA) test**, specified in BCBS d352 and refined in d457 and d558.

The intuition: a desk whose RTPL closely tracks HPL has a risk model that captures the actual drivers of P&L. A desk whose RTPL and HPL diverge — even if VaR passes the backtest — is using a model whose Greeks and risk factors do not reflect reality, and is barred from using the IMA for capital.

#### The two test statistics

BCBS d558 prescribes two complementary statistics computed over a 250-day window of paired (RTPL, HPL) observations:

**1. Spearman rank correlation.** Captures monotonic agreement, robust to scale differences and outliers:

$$
\rho_S = \mathrm{Corr}\!\left( \mathrm{rank}(\boldsymbol{\pi}^{\text{RTPL}}),\; \mathrm{rank}(\boldsymbol{\pi}^{\text{HPL}}) \right)
$$

A high $\rho_S$ means good days and bad days line up between the two series.

**2. Kolmogorov–Smirnov statistic.** Captures distributional agreement:

$$
D_{KS} = \sup_{x} \left| \hat F_{\text{RTPL}}(x) - \hat F_{\text{HPL}}(x) \right|
$$

where $\hat F$ is the empirical CDF. A small $D_{KS}$ means the two P&L distributions have similar shape — fat tails and skews line up.

Spearman alone is insufficient because two perfectly rank-correlated series can have wildly different magnitudes (RTPL systematically half of HPL). KS alone is insufficient because two series with the same marginal distribution can be daily-shuffled relative to each other. Both must pass.

#### BCBS d558 traffic-light thresholds

The final FRTB rules (BCBS d558, §MAR32) specify the following zones:

| Test | Green | Amber | Red |
|---|---|---|---|
| Spearman $\rho_S$ | $\ge 0.80$ | $\ge 0.70$ | $< 0.70$ |
| KS test ($p$-value) | $\ge 0.264$ | $\ge 0.055$ | $< 0.055$ |

The overall PLA zone is the worse of the two test zones. A red zone disqualifies the desk from IMA capital treatment for the next quarter, forcing it to the standardized approach (typically more punitive). Two consecutive amber quarters also force a fallback.

For $n = 250$ observations, the KS $p$-value thresholds correspond approximately to critical statistics $D^*_{0.264} \approx 0.063$ (green) and $D^*_{0.055} \approx 0.085$ (amber), computed from the Kolmogorov distribution.

#### Why PLA is harder than backtesting VaR

The VaR backtest only requires that **tail counts** match — 99% accuracy on 1% of days. The PLA test requires that the **entire distribution** of daily P&L matches and that the **time ordering** matches. A model with the right unconditional tail but wrong day-to-day dynamics passes the backtest and fails PLA. This is by design: FRTB targets risk factor coverage and Greek quality, not just tail calibration.

The most common PLA failures come from:

1. **Missing risk factors.** Vol-surface risk modelled as one ATM vol per asset (instead of a smile) gives systematic RTPL−HPL bias on smile-twist days.
2. **Linearisation of strongly convex products.** A delta-only RTPL on a heavy-gamma book fails KS even when Spearman is fine.
3. **Stale Greeks.** Computing the ladder once a week and reusing it makes RTPL track HPL well in calm markets but explode during regime shifts.

VALAX's autodiff ladder addresses (2) and (3) directly: a full second-order ladder (10 rungs including vanna, volga, cross-gamma) is cheap enough to recompute daily, eliminating the linearisation bias and Greek-staleness failure modes.

**VALAX implementation:** `valax/risk/backtesting.py` provides `pla_spearman()`, `pla_ks()`, `ks_statistic()`, and `pla_traffic_light()`. The companion guide section (`docs/guide/risk.md` § *FRTB PLA Test*) shows the end-to-end workflow on a 250-day window.

### 7.8 Risk Bucketing: Linear and Jacobian Transformations

So far the risk engine lives in **raw factor space**: one sensitivity per `MarketData` leaf (one DV01 per pillar, one vega per asset, one credit delta per hazard pillar). For reporting, capital, and hedging, the same sensitivities have to be expressed in different coordinate systems:

| Purpose | Target space | Why |
|---|---|---|
| Regulatory reporting (FRTB SBA, ISDA SIMM) | Standard buckets (10 IR tenors, 11 equity sectors, …) | Mandated by BCBS d457 §MAR21 and ISDA SIMM v2.6 — capital is computed on bucketed sensitivities, not raw pillars. |
| Risk explain to traders | Coarse curve regions: short, belly, wings | "DV01 by tenor block" is more actionable than a 30-pillar vector. |
| Stable VaR | PCA factor scores (level, slope, curvature) | Yield-curve covariance is rank-deficient on raw pillars; PCA gives a well-conditioned 3–5 factor basis. |
| Vol-surface explain | SABR / SVI parameters | Trader intuition lives in $(\alpha, \rho, \nu)$ space, not per-knot bumps. |

All four reduce to the same mathematical operation — a change of coordinates — but two distinct *flavours* arise depending on whether the new coordinates are defined by aggregation or by a nonlinear reparameterization.

#### Notation

Let $x \in \mathbb{R}^n$ be the raw risk factors and $b \in \mathbb{R}^m$ the bucket/coarse coordinates, with $m \le n$ (usually $m \ll n$). Let

$$
\delta_x \;=\; \nabla_x V \in \mathbb{R}^n, \qquad
\Delta x \in \mathbb{R}^n,
\qquad
\Sigma_x \in \mathbb{R}^{n\times n}
$$

denote the raw sensitivities, factor shocks, and factor covariance respectively. We want the analogous quantities $\delta_b$, $\Delta b$, $\Sigma_b$ in bucket space.

#### Flavour 1: Linear aggregation

A **linear bucketing** is defined by an aggregation matrix

$$
A \in \mathbb{R}^{m\times n}, \qquad b \;=\; A\,x
$$

where each row of $A$ specifies one bucket as a linear combination of raw factors. Two common conventions are:

| Convention | Definition of $A_{ij}$ | Use |
|---|---|---|
| **Indicator** (FRTB tenor vertices) | $1$ if pillar $j$ is nearest to bucket $i$, else $0$ | Standard regulatory tenor buckets {0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30}. |
| **Linear distribution** | Piecewise-linear weights summing to 1 per pillar | Smooth re-binning of one tenor grid onto a coarser one; avoids step discontinuities in bucketed DV01. |
| **Equal-weight group** | $1$ if factor $j$ belongs to group $i$, else $0$ | Sector / currency / rating buckets where each factor lives in exactly one group. |

The induced transformations follow from a single rule — preserve dot products — which is just the statement that **the P&L from a shock must not depend on the coordinate system**:

$$
\delta_b \cdot \Delta b \;=\; \delta_x \cdot \Delta x \quad \text{for all shocks.}
$$

Combining with $b = Ax$ ⇒ $\Delta b = A\,\Delta x$ on the *factor* side gives the **dual relation** on the *shock* side:

$$
\boxed{\;\;\Delta x \;=\; A^{\!\top}\Delta b, \qquad \delta_b \;=\; A\,\delta_x, \qquad \Sigma_b \;=\; A\,\Sigma_x\,A^{\!\top}\;\;}
$$

The shock relation $\Delta x = A^\top \Delta b$ should be read carefully: it is the **minimum-norm** factor shock that produces the given bucket move; it is not unique (any shock in $\ker A$ leaves $\Delta b$ untouched). The choice $A^\top$ is the canonical "spread the bucket shock evenly across its constituent factors" rule, and it is the unique choice that makes the PnL identity above hold automatically for **every** $\delta_x$.

The covariance transformation $\Sigma_b = A\,\Sigma_x\,A^\top$ is exact when bucket factors are linear in raw factors and PSD is preserved (a sum of outer products of $A$-rows weighted by PSD $\Sigma_x$ is PSD). A 30 × 30 yield-curve covariance compresses to a 10 × 10 bucket covariance with no approximation error.

#### Flavour 2: Jacobian reparameterization

When the bucket coordinates are **nonlinear** in the raw factors — or, equivalently, when the raw factors are a smooth function of the buckets

$$
x \;=\; g(b),
$$

the chain rule replaces $A^\top$ with the Jacobian:

$$
J \;=\; \frac{\partial x}{\partial b} \in \mathbb{R}^{n\times m}.
$$

The full transformation pair becomes

$$
\boxed{\;\;\Delta x \;=\; J\,\Delta b \;+\; \mathcal{O}(\|\Delta b\|^2), \qquad
\delta_b \;=\; J^{\!\top}\,\delta_x, \qquad
\Sigma_b \;=\; J^{\!\top}\,\Sigma_x\,J\;\;}
$$

Linear aggregation is recovered as the special case $J = A^\top$ (constant Jacobian). Two important examples:

- **PCA on yield-curve returns.** Compute the eigendecomposition $\Sigma_x = V\Lambda V^\top$ and keep the top $m$ columns as $J = V_{:,1:m}$. The buckets $b = J^\top x$ are the principal-component scores; for a USD curve, the first three are virtually always interpretable as **level**, **slope**, **curvature** (Litterman & Scheinkman 1991). PCA bucketing typically explains ≥99% of yield-curve variance with $m=3$ and is the textbook way to make a 30-pillar covariance VaR practical.

- **Level / slope / curvature factors.** A hand-picked, fixed Jacobian $J = [\mathbf{1},\,t,\,t^2 - \bar{t^2}]$ (or any orthogonal basis on $[t_{\min}, t_{\max}]$) gives three interpretable buckets with no calibration. Useful for stress design ("what's the +25 bp parallel + −10 bp twist scenario worth?").

- **SVI / SABR slice parameters.** For each vol-slice expiry, the Jacobian $J = \partial \sigma_{\text{grid}}/\partial (\alpha, \rho, \nu)$ (or the SVI five-parameter analogue) converts a per-strike vol ladder into a 3- or 5-parameter sensitivity vector. This is exactly the autodiff Jacobian of the SVI/SABR vol function at the calibrated parameters and is the natural "vol-shape Greek" — vega-of-skew, vega-of-convexity, vega-of-wing.

In all three cases the Jacobian is computed once on the base market (analytically for PCA and L/S/C, autodiff for SVI/SABR) and reused for every scenario. **VALAX provides a generic `jacobian_from_fn(b_to_x, b_base)` wrapper around `jax.jacobian` so any smooth reparameterization plugs in.**

#### Pulling sensitivities through a bucketing chain

Because both transformations are linear maps on sensitivity space, they compose: aggregating yield-curve PCA scores into a single "rates" bucket alongside an equity bucket is just $A \cdot J^\top$ applied to $\delta_x$. This makes it natural to build risk reports as a *pipeline*:

```
raw factors  ─ autodiff ─►  δ_x
δ_x          ─ J^⊤      ─►  PCA scores  (factor-reduction)
PCA scores   ─ A        ─►  level / belly / wings  (regulatory aggregation)
```

Every stage preserves the PnL identity, so the bottom-line bucket P&L from a bucket scenario always equals the raw P&L from the implied raw shock — no matter how many transformations sit between them.

#### Covariance shrinkage and conditioning

A practical reason to bucket: the raw covariance $\Sigma_x$ estimated from $T$ historical observations is rank-min$(T,n)$; for a 30-pillar curve with $T = 250$ daily observations it is full rank but poorly conditioned, and parametric VaR $\sqrt{\delta^\top \Sigma \delta}$ becomes numerically unstable. After bucketing to 5 PCA factors, $\Sigma_b \in \mathbb{R}^{5\times 5}$ is rock-solid. The trade-off is fidelity: bucketing throws away the $n - m$ directions of least variance, by construction. The eigenvalue ratio $\lambda_{m+1}/\lambda_1$ is the standard diagnostic for whether further dimensions are worth keeping.

**VALAX implementation:** `valax/risk/bucketing.py` provides

- `BucketMap` — the $A$ matrix as an `eqx.Module` with labels;
- linear ops: `aggregate`, `pushforward_scenario`, `aggregate_covariance`, `aggregate_matrix`;
- Jacobian ops: `pushforward_sensitivities`, `pullback_shocks`, `reparameterize_covariance`, `jacobian_from_fn`;
- builders: `tenor_bucket_map` (indicator and linear), `equal_weight_bucket_map`, `level_slope_curvature_jacobian`, `pca_jacobian`;
- ladder convenience: `bucket_sensitivity_ladder` applies independent bucketing to each component of a `SensitivityLadder`, including bilateral aggregation of cross-gamma blocks.
