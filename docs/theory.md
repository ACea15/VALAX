# Models & Theory

This document links the mathematical foundations of quantitative finance to the specific models and implementations in VALAX. The [User Guide](guide/analytical.md) shows *how* to use each pricing function; this document explains *why* the formulas hold, what assumptions they rest on, and where those assumptions break down.

Throughout, we reference VALAX modules by path (e.g., `valax/pricing/analytic/black_scholes.py`) so you can trace each formula to its implementation.

---

## 1. Foundational Framework

Every pricing formula in VALAX rests on three pillars: no-arbitrage, risk-neutral valuation, and the connection between stochastic processes and partial differential equations. Understanding these unlocks the entire library.

### 1.1 No-Arbitrage and Risk-Neutral Pricing

An **arbitrage** is a trading strategy that costs nothing, has no chance of loss, and has a positive probability of profit. The Fundamental Theorem of Asset Pricing states:

> A market is arbitrage-free if and only if there exists at least one **equivalent martingale measure** $\mathbb{Q}$ under which all discounted asset prices are martingales.

Under the physical (real-world) measure $\mathbb{P}$, assets have expected returns that reflect risk premia. Under the risk-neutral measure $\mathbb{Q}$, all assets earn the risk-free rate in expectation. The price of any derivative with payoff $V_T$ at time $T$ is:

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[V_T]
$$

or more generally, with stochastic rates:

$$
V_0 = \mathbb{E}^{\mathbb{Q}}\!\left[\exp\!\left(-\int_0^T r_s\,ds\right) V_T\right]
$$

This is the master equation behind every pricing function in VALAX. Analytical formulas evaluate this expectation in closed form. Monte Carlo (`valax/pricing/mc/`) estimates it by simulation. PDE methods (`valax/pricing/pde/`) solve the equivalent differential equation. Lattice methods (`valax/pricing/lattice/`) approximate it on a discrete tree.

**VALAX convention:** All models are specified under $\mathbb{Q}$. Drift terms in SDEs (e.g., $r - q$ in Black-Scholes) are risk-neutral drifts, not real-world expected returns.

### 1.2 Itô's Lemma

If $S_t$ follows the SDE:

$$
dS_t = \mu_t\,dt + \sigma_t\,dW_t
$$

and $f(t, S)$ is a twice-differentiable function, then:

$$
df = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial S} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial S^2}\right)dt + \sigma_t \frac{\partial f}{\partial S}\,dW_t
$$

This is the workhorse of derivatives pricing. It tells us how the value of any function of $S$ evolves, and it gives rise to the Black-Scholes PDE (Section 5.2) when we impose the no-arbitrage hedging argument.

### 1.3 Girsanov's Theorem and Measure Change

**Girsanov's theorem** provides the mechanism for switching from $\mathbb{P}$ to $\mathbb{Q}$. If $W_t^{\mathbb{P}}$ is a Brownian motion under $\mathbb{P}$, then:

$$
W_t^{\mathbb{Q}} = W_t^{\mathbb{P}} + \int_0^t \lambda_s\,ds
$$

is a Brownian motion under $\mathbb{Q}$, where $\lambda_t = (\mu_t - r_t) / \sigma_t$ is the **market price of risk**. The measure change absorbs the risk premium into the drift, converting the physical drift $\mu$ into the risk-neutral drift $r$.

In VALAX, this appears implicitly: all model SDEs are written directly under $\mathbb{Q}$ with risk-neutral drifts. For example, `BlackScholesModel` in `valax/models/black_scholes.py` defines `GBMDrift` as $r - q$, not the real-world expected return.

### 1.4 The Feynman-Kac Theorem

The **Feynman-Kac theorem** connects the risk-neutral expectation to a PDE:

If $V(t, S) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-r(T-t)} g(S_T) \mid S_t = S\right]$, then $V$ satisfies:

$$
\frac{\partial V}{\partial t} + (r - q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0
$$

with terminal condition $V(T, S) = g(S)$.

This is why Monte Carlo and PDE methods give the same answer: they are computing the same object (the risk-neutral expectation) via different routes. MC samples paths and averages the discounted payoff. PDE solves the differential equation backward from the terminal condition.

**In VALAX:** `valax/pricing/mc/engine.py` computes $\mathbb{E}^{\mathbb{Q}}[e^{-rT}g(S_T)]$ by simulation. `valax/pricing/pde/solvers.py` solves the Feynman-Kac PDE via Crank-Nicolson. For Black-Scholes models with European payoffs, both converge to the closed-form answer in `valax/pricing/analytic/black_scholes.py`. The QuantLib comparison tests (`tests/test_quantlib_comparison/`) verify this convergence.

---

## 2. Stochastic Models

Each model in VALAX defines a stochastic process for one or more state variables under $\mathbb{Q}$. The choice of model determines what dynamics are captured (constant vol, stochastic vol, smile, term structure).

### 2.1 Black-Scholes / Geometric Brownian Motion

**SDE** (under $\mathbb{Q}$):

$$
dS_t = (r - q)S_t\,dt + \sigma S_t\,dW_t
$$

where $r$ is the risk-free rate, $q$ is the continuous dividend yield, and $\sigma$ is the constant volatility.

**Exact solution** (used for path generation in `valax/pricing/mc/paths.py`):

$$
S_T = S_0 \exp\!\left[\left(r - q - \tfrac{1}{2}\sigma^2\right)T + \sigma W_T\right]
$$

**Closed-form price** (Black-Scholes formula, implemented in `valax/pricing/analytic/black_scholes.py`):

$$
C = S_0 e^{-qT}\Phi(d_1) - Ke^{-rT}\Phi(d_2)
$$

$$
d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

**Assumptions and limitations:**

- Constant volatility — markets exhibit smiles and skews, violating this
- Log-normal returns — cannot accommodate negative prices (problematic for rates)
- Continuous hedging — assumes no transaction costs or discrete rebalancing
- No jumps — short-dated options show steeper smiles than diffusion can produce

**When to use:** Vanilla European equity/FX options where the smile effect is secondary, or as a benchmark for testing other methods. The BSM formula remains the industry standard for quoting implied volatility, even when pricing uses more complex models.

**VALAX implementation:** Model in `valax/models/black_scholes.py` (`BlackScholesModel`, `GBMDrift`, `GBMDiffusion`). Analytic pricing in `valax/pricing/analytic/black_scholes.py`. MC paths via diffrax in `valax/pricing/mc/paths.py` (`generate_gbm_paths`). PDE solver in `valax/pricing/pde/solvers.py`. Binomial tree in `valax/pricing/lattice/binomial.py`.

### 2.2 Black-76 (Futures / Forwards)

**SDE** (forward price under the $T$-forward measure):

$$
dF_t = \sigma F_t\,dW_t^T
$$

The forward price is a martingale under the $T$-forward measure — no drift term. This is the natural setting for options on futures, forward rates, and swap rates.

**Closed-form price** (implemented in `valax/pricing/analytic/black76.py`):

$$
C = e^{-rT}\left[F\Phi(d_1) - K\Phi(d_2)\right]
$$

$$
d_1 = \frac{\ln(F/K) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

**Relationship to Black-Scholes:** Black-76 is Black-Scholes applied to a forward price rather than a spot price. Setting $F = S_0 e^{(r-q)T}$ recovers the BSM formula.

**When to use:** Caplets, floorlets, and swaptions (`valax/pricing/analytic/caplets.py`, `valax/pricing/analytic/swaptions.py`). This is the market-standard model for quoting interest rate option volatilities.

### 2.3 Bachelier (Normal Model)

**SDE:**

$$
dF_t = \sigma_n\,dW_t
$$

The forward price follows arithmetic Brownian motion — it can go negative.

**Closed-form price** (implemented in `valax/pricing/analytic/bachelier.py`):

$$
C = e^{-rT}\left[(F - K)\Phi(d) + \sigma_n\sqrt{T}\,\phi(d)\right], \qquad d = \frac{F - K}{\sigma_n\sqrt{T}}
$$

where $\phi$ is the standard normal density.

**Assumptions and differences from Black-76:**

- $\sigma_n$ is a **normal volatility** (units of rate, e.g., 80 bps/yr), not a lognormal volatility (dimensionless)
- Returns are normally distributed, not log-normally
- Well-behaved at zero and negative rates — no $\ln(F/K)$ singularity
- Underprices deep OTM options relative to lognormal models (thinner tails)

**When to use:** Interest rate options in low/negative rate environments (EUR, JPY, CHF post-2014). Also used in `valax/pricing/analytic/caplets.py` and `valax/pricing/analytic/swaptions.py` as `caplet_price_bachelier` and `swaption_price_bachelier`.

### 2.4 Heston Stochastic Volatility

**SDE** (two-factor system, implemented in `valax/models/heston.py`):

$$
dS_t = (r - q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v
$$

$$
dW_t^S\,dW_t^v = \rho\,dt
$$

| Parameter | Meaning | Typical range |
|-----------|---------|---------------|
| $v_0$ | Initial variance | 0.01–0.09 |
| $\theta$ | Long-run variance | 0.01–0.09 |
| $\kappa$ | Mean-reversion speed | 0.5–5.0 |
| $\xi$ | Vol-of-vol | 0.1–1.0 |
| $\rho$ | Spot-vol correlation | −0.9 to −0.3 (equity) |

**The Feller condition:** If $2\kappa\theta > \xi^2$, the variance process $v_t$ is strictly positive almost surely. When violated, $v_t$ can hit zero, requiring careful numerical treatment. VALAX's diffrax-based simulation handles this via adaptive step-size SDE solvers (`valax/pricing/mc/paths.py`, `generate_heston_paths`), but extreme parameter sets near the Feller boundary may still require absorption or reflection schemes.

**Characteristic function** (not yet implemented — see Roadmap P2.1):

$$
\phi(\omega) = \mathbb{E}^{\mathbb{Q}}[e^{i\omega \ln S_T}] = \exp\!\big(C(\omega, T) + D(\omega, T)\,v_0 + i\omega \ln S_0\big)
$$

where $C$ and $D$ satisfy Riccati ODEs with known closed-form solutions. This enables semi-analytic pricing via the COS method (Fang-Oosterlee) or Carr-Madan FFT, giving prices in microseconds rather than the seconds required by MC.

**What Heston captures that Black-Scholes cannot:**

- Volatility smile (convexity in the implied vol curve) — driven by vol-of-vol $\xi$
- Volatility skew (asymmetry) — driven by spot-vol correlation $\rho$
- Volatility mean-reversion — term structure of smile flattens at long maturities
- Fat tails in return distributions

**Limitations:**

- Cannot produce short-dated smile steepness seen in practice (no jumps)
- Five parameters to calibrate — potential overfitting or flat directions in the loss surface
- MC-only in VALAX currently — too slow for real-time calibration (COS method will fix this)

**VALAX implementation:** Model definition in `valax/models/heston.py` (`HestonModel`, `HestonDrift`, `HestonDiffusion`). MC simulation in `valax/pricing/mc/paths.py`. Calibration in `valax/calibration/heston.py`. The simulation works in $(\\ln S, v)$ space with correlated Brownian motions via Cholesky decomposition.

### 2.5 SABR

**SDE:**

$$
dF_t = \alpha_t F_t^\beta\,dW_t^F
$$

$$
d\alpha_t = \nu\,\alpha_t\,dW_t^\alpha
$$

$$
dW_t^F\,dW_t^\alpha = \rho\,dt
$$

| Parameter | Meaning | Effect on smile |
|-----------|---------|-----------------|
| $\alpha$ | Initial vol level | Shifts ATM vol up/down |
| $\beta$ | CEV exponent ($0 \leq \beta \leq 1$) | Controls backbone: $\beta=1$ lognormal, $\beta=0$ normal |
| $\rho$ | Forward-vol correlation | Controls skew (negative $\rho$ → downward skew) |
| $\nu$ | Vol-of-vol | Controls smile curvature (wings) |

**Hagan's approximation** (implemented in `valax/pricing/analytic/sabr.py`):

SABR does not have a closed-form option price. Instead, Hagan et al. (2002) derived an asymptotic expansion for the **implied Black-76 volatility** $\sigma_B(K, F)$, which is then fed into the Black-76 formula. This two-step approach (SABR → implied vol → Black-76 → price) is the market standard for rates options.

The approximation is accurate for:

- Strikes not too far from ATM (within ~2–3 standard deviations)
- Non-zero forward and strike (breaks down as $F \to 0$ or $K \to 0$)
- Short to medium expiries

**Known limitations of Hagan's formula:**

- **Probability mass leakage:** For $\beta < 1$ and low rates, the formula can imply negative densities at low strikes. The "free boundary" SABR or "arbitrage-free SABR" (Hagan-Lesniewski 2014) fixes this but is more complex.
- **Extrapolation:** Wings can blow up or go negative far from ATM. Production systems typically cap/floor the extrapolation.
- **Smile dynamics:** SABR is calibrated per-expiry — it does not produce a consistent dynamic model across time.

**Why SABR dominates rates markets:** It has exactly the right number of parameters (4) to fit the smile at a single expiry with intuitive controls. $\beta$ is typically fixed (0, 0.5, or 1) from market convention, leaving 3 free parameters to fit. Per-expiry calibration matches the market practice of quoting swaption vols on a grid of (expiry, tenor) pairs.

**VALAX implementation:** Analytic implied vol in `valax/pricing/analytic/sabr.py`. MC simulation in `valax/pricing/mc/sabr_paths.py` via diffrax. Per-expiry calibration in `valax/calibration/sabr.py`. Vol surface construction in `valax/surfaces/sabr_surface.py`.

### 2.6 LIBOR Market Model (LMM / BGM)

**SDE** (under the spot measure, implemented in `valax/models/lmm.py`):

$$
\frac{dF_i(t)}{F_i(t)} = \mu_i(t)\,dt + \sigma_i(t) \cdot dW_t
$$

where $F_i$ is the simply-compounded forward rate for the period $[T_i, T_{i+1}]$ and the drift $\mu_i(t)$ is determined by the no-arbitrage condition (it depends on all forward rates $F_j$ for $j \leq i$ and the volatility/correlation structure).

**Key features:**

- Models the **entire forward rate curve** simultaneously — $N$ correlated forward rates
- Each forward rate has its own volatility function $\sigma_i(t)$
- Forward rate correlations are parameterized (exponential, two-parameter) and can be reduced via PCA factor loading
- Natural model for caps/floors (each caplet sees a single forward rate) and swaptions (swap rates are functions of forward rates)

**Volatility specifications** (in `valax/models/lmm.py`):

- `PiecewiseConstantVol`: Flat vol per forward rate per time period
- `RebonatoVol`: $(a + b\tau) e^{-c\tau} + d$ parametric form — captures the hump shape of cap vol term structures

**Correlation specifications:**

- Exponential: $\rho_{ij} = e^{-\beta|T_i - T_j|}$
- Two-parameter: $\rho_{ij} = \rho_\infty + (1 - \rho_\infty) e^{-\beta|T_i - T_j|}$
- PCA factor loading: Eigendecomposition of the correlation matrix, retaining the top $k$ factors

**Drift correction under the spot measure:** The no-arbitrage drift of $F_i$ under the spot (discretely-compounded bank account) measure is:

$$
\mu_i(t) = \sigma_i(t) \cdot \sum_{j=\eta(t)}^{i} \frac{\delta_j F_j(t)}{1 + \delta_j F_j(t)} \sigma_j(t)
$$

where $\eta(t)$ is the index of the first alive forward rate and $\delta_j$ is the accrual fraction. This drift is path-dependent — it must be computed at each simulation step.

**When to use:** Bermudan swaptions, CMS products, callable rate exotics — any product where the payoff depends on multiple points of the forward rate curve and/or has early exercise features.

**VALAX implementation:** Full model in `valax/models/lmm.py`. Path generation in `valax/pricing/mc/lmm_paths.py`. Rate payoffs (caplet, cap, swaption) in `valax/pricing/mc/rate_payoffs.py`. Bermudan swaption pricing via Longstaff-Schwartz in `valax/pricing/mc/bermudan.py`.

### 2.7 Garman-Kohlhagen (FX Options)

**SDE** (under the domestic risk-neutral measure):

$$
dS_t = (r_d - r_f)\,S_t\,dt + \sigma\,S_t\,dW_t
$$

where $S_t$ is the spot FX rate (domestic per foreign), $r_d$ is the domestic risk-free rate, $r_f$ is the foreign risk-free rate, and $\sigma$ is the FX volatility. The foreign rate plays exactly the role of a continuous dividend yield — holding foreign currency earns the foreign risk-free rate, just as holding a stock earns its dividend yield.

**FX forward rate** (from covered interest rate parity):

$$
F = S \cdot e^{(r_d - r_f)T}
$$

When $r_d > r_f$, the forward rate is above spot (domestic currency trades at a forward discount). This arbitrage-free relationship links the FX forward market to the interest rate differential.

**Closed-form price** (Garman-Kohlhagen 1983, implemented in `valax/pricing/analytic/fx.py`):

$$
C = N \left[S\,e^{-r_f T}\,\Phi(d_1) - K\,e^{-r_d T}\,\Phi(d_2)\right]
$$

$$
d_1 = \frac{\ln(S/K) + (r_d - r_f + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

where $N$ is the foreign notional. This is algebraically identical to Black-Scholes with $q = r_f$. The put price follows from put-call parity: $C - P = N\,(S\,e^{-r_f T} - K\,e^{-r_d T})$.

**Delta conventions — what makes FX unique:**

FX options are quoted in **delta space**, not strike space. The standard quoting points are 10Δ put, 25Δ put, ATM (delta-neutral straddle), 25Δ call, 10Δ call. This reflects the fact that FX traders think in terms of hedge ratios, not absolute price levels.

Three delta conventions coexist:

| Convention | Call delta | When used |
|------------|-----------|-----------|
| **Spot delta** | $\Delta = e^{-r_f T}\,\Phi(d_1)$ | G10 pairs (EUR/USD, USD/JPY) |
| **Forward delta** | $\Delta = \Phi(d_1)$ | Some interbank markets |
| **Premium-adjusted** | $\Delta = e^{-r_f T}\,\Phi(d_1) - V/(S \cdot N)$ | EM pairs where premium is paid in foreign currency |

The **premium-adjusted delta** accounts for the fact that when the option premium is paid in foreign currency, the premium itself has FX exposure. Buying a call and paying the premium in foreign currency requires selling foreign to fund the premium, reducing the net delta. This adjustment matters most for deep ITM options and long-dated trades.

**ATM conventions:** "ATM" in FX does not mean $S = K$. The standard is **delta-neutral straddle (DNS)**: the strike where the call delta equals the absolute put delta, so the straddle has zero delta. This strike is close to (but not exactly) the forward rate.

**Premium currency:** FX option premiums can be paid in either domestic or foreign currency. The `premium_currency` field on `FXVanillaOption` tracks this. When the premium is in foreign currency, the premium-adjusted delta convention applies.

**VALAX implementation:** Instruments in `valax/instruments/fx.py` (`FXForward`, `FXVanillaOption`, `FXBarrierOption`). Pricing and delta utilities in `valax/pricing/analytic/fx.py`: `garman_kohlhagen_price`, `fx_forward_price`, `fx_delta` (all three conventions), `strike_to_delta`, `delta_to_strike` (Newton-Raphson inversion for vol surface construction from delta quotes). All functions are differentiable — `jax.grad` gives the full set of FX Greeks including domestic rho, foreign rho, vanna, and volga.

### 2.8 Hull-White One-Factor Short-Rate Model

The Hull-White (1990) model is the **extended Vasicek** process, designed so that the initial discount curve is fit exactly. It is the workhorse of rates desks for callable bonds, puttable bonds, Bermudan swaptions, and any rate instrument with embedded optionality where an exact match to the initial curve matters.

**SDE** (under the risk-neutral measure, implemented in `valax/models/hull_white.py`):

$$
dr_t = \left[\theta(t) - a\,r_t\right]dt + \sigma\,dW_t
$$

| Parameter | Meaning | Typical range |
|-----------|---------|---------------|
| $a$ | Mean-reversion speed | 0.01–0.10 |
| $\sigma$ | Short-rate volatility | 0.005–0.02 |
| $\theta(t)$ | Time-dependent drift | Calibrated to initial curve |

The key feature is that $\theta(t)$ is a **free function**, not a scalar parameter. It is chosen to make the model-implied zero-coupon bond prices $P(0, T)$ match the initial market curve $P^M(0, T)$ exactly:

$$
\theta(t) = \frac{\partial f^M(0, t)}{\partial t} + a\,f^M(0, t) + \frac{\sigma^2}{2a}\left(1 - e^{-2at}\right)
$$

where $f^M(0, t) = -\partial \ln P^M(0, t)/\partial t$ is the instantaneous forward rate. In practice $\theta(t)$ is never computed explicitly — its effect is absorbed directly into the analytic bond price (below) and the trinomial tree shifts $\alpha_i$.

**Affine term structure** (closed-form zero-coupon bond prices):

$$
P(t, T \mid r_t) = A(t, T)\,e^{-B(t, T)\,r_t}
$$

$$
B(t, T) = \frac{1 - e^{-a(T - t)}}{a}
$$

$$
\ln A(t, T) = \ln\frac{P^M(0, T)}{P^M(0, t)} + B(t, T)\,f^M(0, t) - \frac{\sigma^2}{4a}\left(1 - e^{-2at}\right)B(t, T)^2
$$

At $t = 0$ and $r_0 = f^M(0, 0)$, these formulas recover $P^M(0, T)$ exactly — this is the **exact-fit property**. Implemented in `valax/models/hull_white.py` as `hw_bond_price`. VALAX computes the instantaneous forward $f^M(0, t)$ via `jax.grad` through the curve's log-DF interpolation, giving piecewise-constant forwards for a log-linear curve with no manual finite differences.

**Short-rate distribution:**

$$
r_t \sim \mathcal{N}\!\left(\mathbb{E}[r_t],\;\frac{\sigma^2}{2a}\!\left(1 - e^{-2at}\right)\right)
$$

The unconditional short-rate variance is implemented as `hw_short_rate_variance`. One uncomfortable consequence: $r_t$ can be **negative** with non-zero probability. For rates that stayed negative in EUR/JPY/CHF markets post-2014 this was a feature; for USD pre-2008 it was a known limitation. Squared-Gaussian and shifted-lognormal extensions exist but are out of scope.

**Jamshidian decomposition for European swaption pricing** (not yet implemented in VALAX):

Because the short rate is the single state variable, a European swaption can be decomposed into a portfolio of options on individual zero-coupon bonds. The decomposition hinges on the monotone dependence of each $P(T_0, T_i)$ on $r_{T_0}$, allowing the swaption strike to be converted into a single critical rate $r^*$ such that:

$$
\text{Swaption} = \sum_{i=1}^{N} c_i \cdot \text{ZBO}(K_i^*)
$$

where each $\text{ZBO}(K_i^*)$ is a zero-coupon bond option priced by the Black-76 formula (with a maturity-dependent volatility coming from the integrated short-rate variance). This is the standard fast calibration route for HW parameters against a swaption grid.

**Trinomial tree** (Hull & White 1994, implemented in `valax/pricing/lattice/hull_white_tree.py`):

For products with early exercise (callable/puttable bonds, Bermudan swaptions), VALAX builds a **recombining trinomial tree** via a two-pass construction:

1. **Symmetric *x*-tree:** First build a tree on the auxiliary process $x_t = r_t - \alpha_t$, which has zero drift. The tree has time step $\Delta t$, state step $\Delta x = \sigma\sqrt{3\Delta t}$, and three branch types: normal (up/mid/down), up-branching at the bottom of the tree, and down-branching at the top. The truncation level $j_{\max} \approx \lceil 0.1835 / (a\,\Delta t)\rceil$ is chosen so transition probabilities stay non-negative. Branching probabilities depend only on $\eta_j = -a\,j\,\Delta t$ and sum to one per state.
2. **Arrow-Debreu forward induction:** Sweep forward in time solving for each $\alpha_i$ such that the tree-implied discount factor $P(0, t_{i+1})$ matches the initial curve. This is a one-dimensional equation per step — VALAX solves it in closed form using the Arrow-Debreu state prices. The resulting tree **exactly reprices the initial curve by construction**, just like the analytic bond formula.

For callable and puttable bonds, backward induction rolls the bond cashflows back through the tree. At each call date, the value at node $(i, j)$ becomes $\min(\text{continuation}, \text{call price})$ (issuer-optimal). At each put date, it becomes $\max(\text{continuation}, \text{put price})$ (holder-optimal). Both operations are smooth enough for `jax.grad` to flow through, so **Greeks of callable bonds come from autodiff through the tree** — no bumping the curve and rebuilding.

**When to use Hull-White:**

- Callable and puttable bonds (the primary driver — implemented in VALAX)
- Bermudan swaptions (on the roadmap)
- Any IR exotic where the initial curve must be matched exactly

**When Hull-White is insufficient:**

- Smile-sensitive products — HW has a single vol parameter, so the entire swaption vol grid cannot be fitted. G2++ (two-factor) partially addresses this. SABR or LMM with a vol surface is the smile-aware alternative.
- Products sensitive to the distribution of forward rate curves, not just the short rate. LMM wins here.

**VALAX implementation:** Model in `valax/models/hull_white.py` (`HullWhiteModel`, `hw_B`, `hw_bond_price`, `hw_short_rate_variance`). Trinomial tree construction and pricing in `valax/pricing/lattice/hull_white_tree.py` (`HullWhiteTree`, `build_hull_white_tree`, `price_callable_bond`, `price_puttable_bond`). Calibration to swaption surface (Jamshidian) is on the roadmap (P1.4 follow-up).

### 2.9 Two-Asset Correlated BSM and Spread Options

Spread options pay on the difference of two underlyings — a standard structure across energy (heat rate, crack spread), equities (pairs), and commodities (calendar spread). VALAX implements two complementary closed-form methods under correlated BSM dynamics.

**Model** (two assets under the risk-neutral measure):

$$
dS_1 = (r - q_1)S_1\,dt + \sigma_1 S_1\,dW_1, \qquad dS_2 = (r - q_2)S_2\,dt + \sigma_2 S_2\,dW_2
$$

$$
dW_1\,dW_2 = \rho\,dt
$$

**Spread call payoff:**

$$
V_T = N \cdot \max\!\left(S_1(T) - S_2(T) - K,\; 0\right)
$$

#### Margrabe's Formula (Exact for $K = 0$)

When the strike is zero, a spread call is an **exchange option**: the right to deliver $S_2$ and receive $S_1$. Margrabe (1978) observed that changing numéraire from the money-market account to $S_2$ itself makes the problem one-dimensional. Under the $S_2$-forward measure, the ratio $X_t = S_1(t)/S_2(t)$ is a martingale following geometric Brownian motion with volatility:

$$
\sigma_s = \sqrt{\sigma_1^2 - 2\rho\,\sigma_1\sigma_2 + \sigma_2^2}
$$

which is precisely the standard deviation of the log-return of the ratio. The exchange-option price is then Black-Scholes on $X$ with "strike" 1, re-expressed in original coordinates:

$$
C = N\left[S_1\,e^{-q_1 T}\,\Phi(d_1) - S_2\,e^{-q_2 T}\,\Phi(d_2)\right]
$$

$$
d_1 = \frac{\ln(S_1/S_2) + (q_2 - q_1 + \tfrac{1}{2}\sigma_s^2)T}{\sigma_s\sqrt{T}}, \qquad d_2 = d_1 - \sigma_s\sqrt{T}
$$

This is **exact** — no approximation error. Note that the risk-free rate does not appear: discounting cancels because the payoff is a ratio of traded assets. Margrabe's formula is implemented in `valax/pricing/analytic/spread.py` as `margrabe_price`.

#### Kirk's Approximation (for $K \neq 0$)

For general spread options with $K \neq 0$, no closed form exists — the sum $S_2(T) + K$ is not lognormal. Kirk (1995) proposed the industry-standard approximation: treat $S_2(T) + K$ **as if** it were a single lognormal asset with an adjusted volatility. Defining forwards $F_i = S_i e^{(r - q_i)T}$ and the moneyness ratio $\lambda = F_2 / (F_2 + K)$:

$$
\sigma_{\text{kirk}} = \sqrt{\sigma_1^2 - 2\rho\,\sigma_1\sigma_2\,\lambda + \sigma_2^2\,\lambda^2}
$$

$$
C = N\,e^{-rT}\left[F_1\,\Phi(d_1) - (F_2 + K)\,\Phi(d_2)\right]
$$

$$
d_1 = \frac{\ln\!\big(F_1 / (F_2 + K)\big) + \tfrac{1}{2}\sigma_{\text{kirk}}^2 T}{\sigma_{\text{kirk}}\sqrt{T}}, \qquad d_2 = d_1 - \sigma_{\text{kirk}}\sqrt{T}
$$

The intuition: as $K \to 0$, $\lambda \to 1$ and $\sigma_{\text{kirk}} \to \sigma_s$, recovering Margrabe. For $K$ small relative to $F_2$, the approximation is extremely accurate (typically better than 10 bps on a 20% vol, 6-month option). It deteriorates when $|K|$ is large relative to $F_2$, when the correlation is highly negative, or when $\sigma_2$ is much larger than $\sigma_1$. Carmona & Durrleman (2003) give tighter bounds and alternative approximations.

Kirk's formula is implemented in `valax/pricing/analytic/spread.py` as `kirk_price`. A convenience dispatcher `spread_option_price` routes to `margrabe_price` when the strike is zero, though for `jax.jit` compatibility call `kirk_price` directly — it handles $K = 0$ gracefully.

**Greeks via autodiff:** Because both formulas are pure JAX, the full set of spread-option Greeks — delta 1, delta 2, gamma 1, gamma 2, cross-gamma ($\partial^2 V / \partial S_1 \partial S_2$), correlation vega, and cross-vega — come from `jax.grad` with zero additional code. Cross-gamma in particular is the sensitivity that traders use to size the correlation hedge; it is notoriously expensive to compute via finite differences on a 2D spot grid but free in VALAX.

**When MC is still required:** Path-dependent spread options (Asian spread, spread barrier, Bermudan spread) have no closed form and require correlated 2D Monte Carlo. VALAX's diffrax integration supports correlated Brownian motions directly via Cholesky factorization, though a packaged multi-asset MC payoff library is roadmap P2.2/P5.x.

**VALAX implementation:** Instrument in `valax/instruments/options.py` (`SpreadOption`). Closed-form pricers in `valax/pricing/analytic/spread.py` (`margrabe_price`, `kirk_price`, `spread_option_price`).

---

## 3. Curve Framework

The yield curve is the most fundamental piece of market data. It determines the time value of money and drives the pricing of every fixed income product.

### 3.1 Discount Factors, Zero Rates, and Forward Rates

Three equivalent representations of the term structure:

**Discount factor** $DF(T)$: the price today of $1 received at time $T$.

$$
DF(T) = e^{-r(T) \cdot T}
$$

**Continuously-compounded zero rate** $r(T)$: the constant rate that, compounded continuously, gives the discount factor.

$$
r(T) = -\frac{\ln DF(T)}{T}
$$

**Simply-compounded forward rate** $F(T_1, T_2)$: the rate implied by the curve for the period $[T_1, T_2]$.

$$
F(T_1, T_2) = \frac{1}{\tau(T_1, T_2)}\left(\frac{DF(T_1)}{DF(T_2)} - 1\right)
$$

where $\tau$ is the day count fraction (see Section 3.4).

**VALAX implementation:** `DiscountCurve` in `valax/curves/discount.py` stores $(T_i, DF_i)$ pairs and interpolates. `forward_rate()` and `zero_rate()` compute the derived quantities. All three representations are linked via autodiff — differentiating a price w.r.t. `discount_factors` gives key-rate durations automatically.

### 3.2 Single-Curve vs. Multi-Curve Framework

**Pre-2008 (single-curve):** One curve did everything. A 3M LIBOR swap curve was bootstrapped from deposits and swaps, then used for both discounting and forward rate projection. The implicit assumption: interbank lending is risk-free.

**Post-2008 (multi-curve):** The credit crisis revealed that LIBOR tenors carry different credit/liquidity premia. A 3M rate is *not* three compounded 1M rates — the basis spread between them can be 20+ bps.

The modern framework uses **separate curves for separate roles**:

| Curve | Role | Typical index |
|-------|------|---------------|
| **Discount curve** | Present-value discounting of all cashflows | OIS (SOFR in USD, €STR in EUR) |
| **Forward curve (3M)** | Projection of 3M floating rates | 3M SOFR / 3M EURIBOR |
| **Forward curve (6M)** | Projection of 6M floating rates | 6M EURIBOR |

**Why OIS for discounting?** Under CSA (Credit Support Annex) agreements, collateralized derivatives earn the OIS rate on their posted margin. The OIS rate is therefore the correct funding rate for discounted cashflows. Uncollateralized trades theoretically require a different (higher) discount rate, which leads to FVA (Funding Valuation Adjustment — see Roadmap P3.1).

**Dual-curve pricing of a swap:**

For a swap where the floating leg pays 3M SOFR:

1. **Project floating cashflows** using the 3M forward curve: $F_i = \frac{1}{\tau_i}\left(\frac{DF_{\text{3M}}(T_{i-1})}{DF_{\text{3M}}(T_i)} - 1\right)$

2. **Discount all cashflows** (both fixed and floating) using the OIS curve: $PV = \sum_i \text{amount}_i \times DF_{\text{OIS}}(T_i^{\text{pay}})$

In the single-curve world, the floating leg of a par swap has a clean identity: $PV_{\text{float}} = N \times (DF_{\text{start}} - DF_{\text{end}})$. This breaks under dual-curve — you must project each forward rate individually from the forward curve and discount with the OIS curve.

**VALAX implementation:** `MultiCurveSet` in `valax/curves/multi_curve.py` holds one `discount_curve` and a dictionary of `forward_curves` keyed by tenor. `bootstrap_multi_curve()` first builds the OIS curve, then bootstraps each forward curve using OIS discounting. The dual-curve swap residual function `_dual_curve_swap_residuals` explicitly separates forward projection from discounting.

### 3.3 Curve Bootstrapping

**The bootstrap problem:** Given $N$ market quotes (deposit rates, FRA rates, par swap rates), find the $N$ discount factors at the corresponding pillar dates such that all instruments reprice exactly.

**Sequential bootstrap** (implemented in `valax/curves/bootstrap.py`, `bootstrap_sequential`):

Process instruments in maturity order. Each instrument provides one equation with one unknown (the discount factor at its maturity), assuming all shorter-maturity DFs are already known.

- **Deposit:** $DF(T) = \frac{DF(T_{\text{start}})}{1 + r \cdot \tau}$
- **FRA:** Same formula as deposit, but $T_{\text{start}} > t_0$, so $DF(T_{\text{start}})$ comes from a prior instrument
- **Swap:** $DF(T_n) = \frac{DF(T_{\text{start}}) - r_{\text{swap}} \sum_{i=1}^{n-1} \tau_i \cdot DF(T_i)}{1 + r_{\text{swap}} \cdot \tau_n}$

This works when instruments are non-overlapping and ordered by maturity.

**Simultaneous bootstrap** (implemented in `valax/curves/bootstrap.py`, `bootstrap_simultaneous`):

When instruments overlap (e.g., swaps sharing intermediate payment dates), the sequential method is insufficient. Instead, solve the full system simultaneously:

$$
\mathbf{R}(\mathbf{x}) = \mathbf{0}
$$

where $\mathbf{x} = (\ln DF_1, \ldots, \ln DF_N)$ (log-space ensures positivity) and $R_i$ is the repricing residual for instrument $i$. VALAX uses `optimistix.root_find` with Newton's method. The Jacobian $\partial R_i / \partial x_j$ is computed automatically via `jax.jacobian` — no finite differences needed.

### 3.4 Interpolation Methods

Between bootstrap pillars, the curve must be interpolated. The choice of interpolation determines the smoothness of forward rates.

**Log-linear on discount factors** (VALAX's current method in `DiscountCurve.__call__`):

$$
\ln DF(t) = \text{linear\_interp}(t; t_i, \ln DF_i)
$$

This is equivalent to **piecewise-constant continuously-compounded forward rates** between pillars. It is simple, monotone (DFs are always decreasing), and stable. The drawback: forward rates have jumps at pillar dates, which can cause unrealistic hedging behavior for instruments sensitive to the forward rate curve shape.

**Alternatives not yet implemented:**

| Method | Forward rate behavior | Pros | Cons |
|--------|----------------------|------|------|
| Linear on zero rates | Piecewise linear forwards | Simple | Non-monotone DFs possible |
| Cubic spline on log-DF | Smooth forwards | Beautiful curves | Can oscillate, negative forwards |
| Monotone convex | Positive, continuous forwards | Market standard | Complex implementation |
| Tension splines | Controllable smoothness | Flexible | Extra parameter |

**Why interpolation matters:** A cap prices each caplet using the forward rate over its accrual period. If the forward curve has artificial jumps at pillar dates, caplet prices will have artifacts where accrual periods straddle pillars. Smooth forward rates produce smoother caplet prices and more stable hedging.

### 3.5 Day Count Conventions

Day count conventions determine the year fraction $\tau(T_1, T_2)$ used in rate calculations. Different conventions exist for historical and market-practice reasons.

| Convention | Formula | Typical use |
|------------|---------|-------------|
| **Act/360** | $\frac{d}{360}$ | USD money market, SOFR, EURIBOR swaps |
| **Act/365 Fixed** | $\frac{d}{365}$ | GBP swaps, AUD swaps, many curve internals |
| **Act/Act (ISDA)** | $\frac{d_1}{D_1} + \frac{d_2}{D_2}$ | US Treasuries, government bonds |
| **30/360** | $\frac{360(Y_2-Y_1)+30(M_2-M_1)+(D_2^*-D_1^*)}{360}$ | US corporate bonds, EUR fixed legs |

where $d$ is the actual number of days and $D$ is the denominator (360 or 365).

**VALAX implementation:** `valax/dates/daycounts.py` implements all four conventions as pure functions on integer ordinals. The `year_fraction(start, end, convention)` dispatcher selects the appropriate function. All are JIT-compatible.

**Practical impact:** Using Act/360 instead of Act/365 on a $100M 5Y swap changes the PV by thousands of dollars. Getting the day count wrong is one of the most common sources of pricing discrepancies between systems.

### 3.6 Inflation Curves and Breakeven Pricing

Inflation derivatives link two different economies: the **nominal** economy (where cashflows are paid and discounted) and the **real** economy (where index-linked payoffs are determined). Pricing any inflation instrument requires two curves:

- The **nominal discount curve** $P^N(0, T)$ — same object used for every other fixed income product.
- The **inflation curve** — a term structure of **forward CPI levels** $\text{CPI}(T)$ derived from inflation swap quotes.

#### Forward CPI vs. Real Rates

There are two equivalent representations of the inflation curve:

**Forward CPI representation** (used in VALAX):

$$
\text{CPI}(T) = \mathbb{E}^{\mathbb{Q}_N}\!\left[\text{CPI}_T\right]
$$

i.e. the expected future CPI index under the nominal risk-neutral measure. From the forward CPI curve, two derived rates are used in pricing:

**Zero-coupon (breakeven) inflation rate** $z(T)$ — the annually-compounded rate such that:

$$
\text{CPI}(T) = \text{CPI}(0) \cdot (1 + z(T))^T \quad\Longleftrightarrow\quad z(T) = \left(\frac{\text{CPI}(T)}{\text{CPI}(0)}\right)^{1/T} - 1
$$

**Year-on-year forward inflation rate** — the single-period rate between two pillars:

$$
\text{YoY}(T_{i-1}, T_i) = \frac{\text{CPI}(T_i)}{\text{CPI}(T_{i-1})} - 1
$$

**Real-rate representation** (equivalent, not directly stored in VALAX):

$$
\text{CPI}(T) = \text{CPI}(0) \cdot \frac{P^R(0, T)}{P^N(0, T)}
$$

where $P^R(0, T)$ is the real-curve discount factor. This is a **Fisher-identity-in-expectation** statement: the forward CPI ratio equals the ratio of real to nominal discount factors. VALAX stores the forward CPI levels directly, which is more numerically stable (no division of two small discount factors) and maps cleanly to the quoted ZCIS breakeven market.

**Interpolation:** VALAX interpolates the inflation curve in **log-CPI space**, giving piecewise-constant instantaneous forward inflation rates between pillars — the direct analogue of log-linear discount factor interpolation. This keeps forward CPI levels positive and produces smooth year-on-year forward rates.

#### Zero-Coupon Inflation Swaps (ZCIS)

The ZCIS is the liquid benchmark of the inflation market — it defines the curve. One party pays a fixed rate compounded over the maturity; the counterparty pays the realized CPI ratio. Both settle as a **single cashflow at maturity**:

$$
\text{PV}^{\text{fix}} = N\,DF^N(T)\left[(1 + K)^T - 1\right]
$$

$$
\text{PV}^{\text{inf}} = N\,DF^N(T)\left[\frac{\text{CPI}(T)}{\text{CPI}(0)} - 1\right]
$$

The **breakeven rate** $K^*$ is the fixed rate that makes the NPV zero. Because both legs discount at the same $DF^N(T)$, the discount factor cancels and the breakeven is a pure statement about the forward CPI curve:

$$
K^*(T) = \left(\frac{\text{CPI}(T)}{\text{CPI}(0)}\right)^{1/T} - 1
$$

This is algebraically identical to the zero-coupon inflation rate $z(T)$ — the ZCIS breakeven *is* the quoted point on the inflation curve. Implemented in `valax/pricing/analytic/inflation.py` as `zcis_price` and `zcis_breakeven_rate`.

#### Year-on-Year Inflation Swaps (YYIS)

The YYIS pays the one-period YoY CPI ratio at each coupon date rather than the cumulative ratio at maturity:

$$
\text{PV}^{\text{inf}} = N \sum_{i=1}^{n}\left(\frac{\text{CPI}(t_i)}{\text{CPI}(t_{i-1})} - 1\right)DF^N(t_i)
$$

$$
\text{PV}^{\text{fix}} = N\,K \sum_{i=1}^{n} \tau_i\,DF^N(t_i)
$$

**Convexity adjustment** (not applied in VALAX's baseline pricer — documented as a known limitation):

The true expected YoY CPI ratio under the $t_i$-forward measure differs from the ratio of forward CPIs:

$$
\mathbb{E}^{\mathbb{Q}_{t_i}}\!\left[\frac{\text{CPI}(t_i)}{\text{CPI}(t_{i-1})}\right] \neq \frac{\text{CPI}(t_i)}{\text{CPI}(t_{i-1})}
$$

The difference is a **convexity correction** that depends on:

- The volatility of the forward CPI ratio (inflation vol)
- The correlation between real and nominal short rates

Under the Jarrow-Yildirim model (a three-factor Heath-Jarrow-Morton framework with nominal rates, real rates, and CPI), the convexity adjustment has a closed form involving the covariance of the three Brownian motions. VALAX's YYIS pricer (`yyis_price`) uses the forward ratio directly — this is the standard **baseline** practice and is accurate to within a few basis points for typical inflation volatilities and rate correlations. A Jarrow-Yildirim extension with convexity adjustment is on the roadmap.

#### Inflation Caps and Floors

An inflation cap is a strip of caplets, each paying:

$$
\text{Caplet}_i = N \cdot \max\!\left(\text{YoY}_i - K,\; 0\right)
$$

where $\text{YoY}_i$ is the realized year-on-year CPI ratio. Market practice is to price each caplet via **Black-76** on the forward YoY rate $F_i$ (treated as lognormal) with a market-quoted inflation volatility:

$$
\text{Caplet}_i = N\,DF^N(t_i)\left[F_i\,\Phi(d_1) - K\,\Phi(d_2)\right]
$$

$$
d_1 = \frac{\ln(F_i/K) + \tfrac{1}{2}\sigma^2 T_i}{\sigma\sqrt{T_i}}, \qquad d_2 = d_1 - \sigma\sqrt{T_i}
$$

Floors follow by put-call parity. Implemented in `valax/pricing/analytic/inflation.py` as `inflation_cap_floor_price_black76`. The same caveat on the convexity adjustment applies: using the forward YoY rate directly is the baseline; a full Jarrow-Yildirim treatment would add a convexity term.

#### Seasonality

Published CPI indices exhibit strong **monthly seasonality** — energy in winter, retail around holidays. Market practice is to bootstrap a seasonally-adjusted CPI curve from ZCIS quotes (which span annual periods so seasonality averages out), then overlay a monthly seasonality factor pattern to produce month-end CPI projections. VALAX does not yet model seasonality — the `InflationCurve` stores forward CPI levels at whatever pillar dates are provided and interpolates smoothly between them. For instruments that settle on ZCIS anniversary dates, this is exact; for month-end indexed instruments, a seasonality overlay is a roadmap item.

**VALAX implementation:** Curve in `valax/curves/inflation.py` (`InflationCurve`, `forward_cpi`, `zc_inflation_rate`, `yoy_forward_rate`, `from_zc_rates`). Pricers in `valax/pricing/analytic/inflation.py` (`zcis_price`, `zcis_breakeven_rate`, `yyis_price`, `inflation_cap_floor_price_black76`). Instruments in `valax/instruments/inflation.py`. Because the inflation curve is an `equinox.Module` pytree, `jax.grad` of any price w.r.t. the curve's `forward_cpis` gives **inflation key-rate sensitivities (IE01)** for free.

---

## 4. Volatility

Volatility is the most complex and contested area of quantitative finance. Every option-dependent pricing function in VALAX requires a volatility input. Understanding the hierarchy of volatility concepts is essential.

### 4.1 Implied Volatility

**Implied volatility** $\sigma_{\text{imp}}(K, T)$ is the volatility that, when plugged into the Black-Scholes (or Black-76 or Bachelier) formula, reproduces the market price of an option with strike $K$ and expiry $T$.

It is a **quoting convention**, not a model parameter. The market trades in implied vol because:

1. Vol is more stable than price (an option's price changes with spot; its implied vol is relatively sticky)
2. Vol is comparable across strikes and expiries (prices are not)
3. It normalizes for moneyness and time to expiry

**Implied vol inversion** (implemented in `valax/pricing/analytic/black_scholes.py`, `black_scholes_implied_vol`):

Given a market price $V_{\text{mkt}}$, find $\sigma$ such that $\text{BS}(S, K, T, r, q, \sigma) = V_{\text{mkt}}$. VALAX uses Newton-Raphson with $\text{vega} = \partial V / \partial \sigma$ as the Jacobian (obtained via `jax.grad`).

### 4.2 The Volatility Surface

In the real market, implied vol varies with both strike and expiry — the **volatility surface** $\sigma_{\text{imp}}(K, T)$.

**Smile:** At a fixed expiry, implied vol as a function of strike is U-shaped (or skewed). This contradicts Black-Scholes (which would produce a flat line). The smile reflects:

- Fat tails in the return distribution (deep OTM options are more expensive than BSM predicts)
- Skewness (puts are more expensive than equidistant calls — the "skew")
- Supply/demand for downside protection

**Term structure:** ATM implied vol varies with expiry — typically higher for short dates (event risk) and reverting to a long-run level.

**VALAX surface implementations** (`valax/surfaces/`):

| Surface | Parameterization | Strengths | Limitations |
|---------|-----------------|-----------|-------------|
| `GridVolSurface` | Bilinear interpolation on $(K, T)$ grid | No assumptions, matches inputs exactly | Not smooth, no extrapolation control |
| `SABRVolSurface` | Per-expiry SABR calibration with parameter interpolation across expiries | Industry standard for rates, intuitive parameters | Per-expiry only, no global arbitrage constraint |
| `SVIVolSurface` | Gatheral's SVI parameterization per expiry | Matches wings correctly (Roger Lee), parsimonious | Per-expiry calibration, SSVI needed for global fit |

### 4.3 SVI Parameterization

Gatheral's **Stochastic Volatility Inspired** (SVI) parameterizes the implied total variance $w(k) = \sigma^2_{\text{imp}}(k) \cdot T$ as a function of log-moneyness $k = \ln(K/F)$:

$$
w(k) = a + b\left(\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right)
$$

| Parameter | Meaning |
|-----------|---------|
| $a$ | Overall variance level |
| $b$ | Slope of the wings (must be $\geq 0$) |
| $\rho$ | Rotation/skew ($-1 < \rho < 1$) |
| $m$ | Translation (horizontal shift of the minimum) |
| $\sigma$ | Smoothing of the ATM vertex ($\sigma > 0$) |

**Why SVI?** For large $|k|$, $w(k) \approx a + b(1 \pm \rho)(|k - m|)$, which is linear in log-moneyness. The **Roger Lee moment formula** proves that implied variance must be asymptotically linear in $|k|$ for any arbitrage-free model. SVI satisfies this by construction — parametric models like polynomial fits do not.

**Arbitrage constraints on the surface:**

- **Calendar spread:** Total variance must be non-decreasing in $T$: $w(k, T_1) \leq w(k, T_2)$ for $T_1 < T_2$. Violated if nearby expiry smiles cross.
- **Butterfly:** The risk-neutral density must be non-negative everywhere: $\frac{\partial^2 C}{\partial K^2} \geq 0$. Violated if the smile curves too sharply.
- **Call spread monotonicity:** $\frac{\partial C}{\partial K} \leq 0$. Violated if the smile is too steep.

SSVI (Surface SVI) by Gatheral and Jacquier enforces calendar-spread arbitrage globally across expiries — this is on the roadmap.

**VALAX implementation:** `valax/surfaces/svi.py` (`SVIVolSurface`). Calibration uses `optimistix.least_squares` (Levenberg-Marquardt). The surface is a differentiable pytree — `jax.grad` through the surface gives vega and other vol sensitivities.

### 4.4 Local Volatility (Dupire)

**Concept:** Local volatility $\sigma_{\text{loc}}(S, t)$ is the unique deterministic volatility function such that:

$$
dS_t = (r - q)S_t\,dt + \sigma_{\text{loc}}(S_t, t)\,S_t\,dW_t
$$

matches all European option prices observed in the market.

**Dupire's formula** extracts local vol from the implied vol surface:

$$
\sigma_{\text{loc}}^2(K, T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}
$$

**Why it matters:** Local vol is the *minimum-complexity model* that fits the entire vol surface. Any model that matches all European option prices must produce the same local vol surface. This makes it the baseline for exotic pricing.

**Limitations:**

- Smile dynamics are wrong: local vol predicts that the smile flattens as spot moves (sticky local vol), whereas the market smile tends to follow spot (closer to sticky strike)
- This leads to underpricing of forward-starting and barrier options
- Production systems use SLV (stochastic-local vol) to combine local vol's surface-matching with stochastic vol's realistic dynamics

**VALAX status:** Not yet implemented. Dupire's formula requires $\partial C/\partial T$ and $\partial^2 C/\partial K^2$ — both obtainable via `jax.grad` through the vol surface and pricing function. This makes VALAX particularly well-suited for local vol extraction (see Roadmap P2.2).

---

## 5. Pricing Methods

VALAX implements four pricing methods. They compute the same risk-neutral expectation via different mathematical routes. Understanding when to use each is critical for production systems.

### 5.1 Analytical (Closed-Form)

**When available:** For specific model-payoff combinations where the expectation $\mathbb{E}^{\mathbb{Q}}[e^{-rT}g(S_T)]$ can be evaluated in closed form.

| Model | Payoff | Formula | VALAX function |
|-------|--------|---------|----------------|
| BSM | European call/put | Black-Scholes | `black_scholes_price` |
| BSM (forward) | European on forward | Black-76 | `black76_price` |
| Normal | European on forward | Bachelier | `bachelier_price` |
| SABR | European (via implied vol) | Hagan + Black-76 | `sabr_price` |
| Any | ZCB, coupon bond | Discounted cashflows | `fixed_rate_bond_price` |
| Any | Swap | Cashflow identity | `swap_price` |

**Advantages:** Exact (to machine precision), instantaneous computation, trivially differentiable.

**Limitations:** Only available for simple payoffs under specific models. No closed form for Asian, barrier, Bermudan, or any path-dependent payoff under stochastic vol.

### 5.2 PDE (Finite Differences)

**Derivation:** Apply Itô's lemma to the option value $V(t, S)$ and construct a hedged portfolio (long option, short $\Delta$ units of stock). The no-arbitrage condition forces the portfolio to earn the risk-free rate, yielding the **Black-Scholes PDE**:

$$
\frac{\partial V}{\partial t} + (r-q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} = rV
$$

with terminal condition $V(T, S) = g(S)$.

**Log-spot transformation:** VALAX transforms to $x = \ln S$, giving:

$$
\frac{\partial V}{\partial t} + \left(r - q - \frac{\sigma^2}{2}\right)\frac{\partial V}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 V}{\partial x^2} = rV
$$

This removes the $S^2$ coefficient, making the grid uniform in moneyness rather than price — much better numerical behavior.

**Crank-Nicolson scheme** (implemented in `valax/pricing/pde/solvers.py`):

Time-stepping uses the average of explicit and implicit Euler:

$$
\frac{V^{n+1} - V^n}{\Delta t} = \frac{1}{2}\left(\mathcal{L}V^{n+1} + \mathcal{L}V^n\right)
$$

where $\mathcal{L}$ is the spatial differential operator. This gives:

- **Second-order accuracy** in both time ($O(\Delta t^2)$) and space ($O(\Delta x^2)$)
- **Unconditional stability** — no CFL restriction on $\Delta t / \Delta x^2$

Each time step requires solving a **tridiagonal linear system**, handled efficiently by `lineax` in VALAX.

**Rannacher smoothing** (not yet implemented): Crank-Nicolson can produce spurious oscillations near the strike at expiry (where the payoff has a kink). Running 2–4 fully implicit steps before switching to CN eliminates these oscillations. This is a planned improvement.

**When to use:** 1D problems (single underlying) where you need prices on the full $(S, t)$ grid — useful for American/Bermudan exercise (backward induction through the grid). Faster than MC for low-dimensional problems, but curse-of-dimensionality makes it impractical for $d > 3$.

**VALAX implementation:** `valax/pricing/pde/solvers.py` (`pde_price`). Uses `jax.lax.scan` for the backward time loop and `lineax` for the tridiagonal solve. Grid boundary conditions use Black-Scholes asymptotics.

### 5.3 Monte Carlo Simulation

**Mathematical basis:** Directly estimate the risk-neutral expectation:

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[g(S_T)] \approx \frac{e^{-rT}}{N}\sum_{i=1}^{N} g(S_T^{(i)})
$$

where $S_T^{(i)}$ are sample paths generated from the SDE under $\mathbb{Q}$.

**Convergence rate:** The standard error of the MC estimate is:

$$
\text{SE} = \frac{\sigma_g}{\sqrt{N}}
$$

where $\sigma_g$ is the standard deviation of the payoff. This is **independent of dimension** — the key advantage of MC over PDE/lattice methods. To halve the error, quadruple the number of paths.

**SDE discretization** (handled by diffrax in VALAX):

For $dS = \mu(S)\,dt + \sigma(S)\,dW$:

| Scheme | Strong order | Weak order | Notes |
|--------|-------------|------------|-------|
| Euler-Maruyama | 0.5 | 1.0 | Simplest, may need small $\Delta t$ |
| Milstein | 1.0 | 1.0 | Adds $\frac{1}{2}\sigma\sigma'(\Delta W^2 - \Delta t)$ correction |
| SRA (Splitting) | 1.5 | 2.0 | Used by diffrax for higher accuracy |

VALAX delegates to diffrax's adaptive SDE solvers rather than hand-coding Euler steps. This is more accurate and handles stiff problems (e.g., Heston near the Feller boundary) automatically.

**Variance reduction** (not yet implemented — planned):

| Technique | Idea | Typical improvement |
|-----------|------|---------------------|
| Antithetic variates | Use $W$ and $-W$ paths | 2x for smooth payoffs |
| Control variates | Subtract a known-mean quantity | 5–50x if good control exists |
| Importance sampling | Sample more from important regions | Problem-specific |
| Stratified sampling | Force even coverage of $W$ distribution | $\sqrt{N}$ improvement |

**When to use:** High-dimensional problems (multi-asset, path-dependent, stochastic vol), exotic payoffs (Asian, barrier, autocallable, Bermudan via LSM). MC is the only feasible method for dimensions $> 3$.

**VALAX implementation:** Path generation in `valax/pricing/mc/paths.py` (GBM, Heston), `valax/pricing/mc/sabr_paths.py` (SABR), `valax/pricing/mc/lmm_paths.py` (LMM). Payoffs in `valax/pricing/mc/payoffs.py` and `valax/pricing/mc/rate_payoffs.py`. Engine in `valax/pricing/mc/engine.py`. Bermudan (LSM) in `valax/pricing/mc/bermudan.py`.

**Differentiability note:** VALAX computes MC Greeks via the **pathwise method** — `jax.grad` differentiates through the entire simulation. This works when the payoff is continuous (or smoothed). For discontinuous payoffs (digital, barrier), the paths must be smoothed (e.g., sigmoid approximation to indicator functions in `valax/pricing/mc/payoffs.py`) or the **likelihood ratio method** (score function estimator) should be used instead. The likelihood ratio method is not yet implemented.

### 5.4 Lattice (Binomial Trees)

**CRR (Cox-Ross-Rubinstein) parameterization** (implemented in `valax/pricing/lattice/binomial.py`):

At each time step $\Delta t$, the stock moves up by factor $u$ or down by factor $d$:

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u} = e^{-\sigma\sqrt{\Delta t}}
$$

The risk-neutral probability of an up move:

$$
p = \frac{e^{(r-q)\Delta t} - d}{u - d}
$$

**Derivation:** $u$ and $d$ are chosen so that the binomial distribution matches the first two moments of GBM over each step:

- Mean: $\mathbb{E}^{\mathbb{Q}}[S_{t+\Delta t}/S_t] = pu + (1-p)d = e^{(r-q)\Delta t}$ ✓
- Variance: $\text{Var}[\ln(S_{t+\Delta t}/S_t)] = \sigma^2 \Delta t + O(\Delta t^2)$ ✓

**Backward induction:** Starting from the terminal payoff at expiry, work backward:

$$
V_{i,j} = e^{-r\Delta t}\left[p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}\right]
$$

For **American options**, at each node compare the continuation value with the immediate exercise value:

$$
V_{i,j} = \max\!\left(g(S_{i,j}),\; e^{-r\Delta t}[p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}]\right)
$$

**Convergence:** The CRR tree converges to the Black-Scholes price as $n \to \infty$, with rate $O(1/n)$. However, convergence oscillates between even and odd $n$ (because the strike alignment with the grid alternates). Odd/even averaging or the Leisen-Reimer parameterization smooths this.

**Connection to PDE:** The binomial tree is mathematically equivalent to an **explicit finite difference scheme** for the Black-Scholes PDE on a $(t, \ln S)$ grid. The CRR parameters correspond to specific grid spacings. This explains why trees and PDE methods give the same answer.

**When to use:** American/Bermudan options on a single underlying (natural backward induction). Pedagogically clear. Limited to low dimensions (multi-asset trees have exponential node growth).

**VALAX implementation:** `valax/pricing/lattice/binomial.py` (`binomial_price`). Supports European and American exercise. Uses `jax.lax.scan` for backward induction. Greeks via `jax.grad` through the entire tree computation.

---

## 6. Greeks and Automatic Differentiation

### 6.1 Greeks as Derivatives

Greeks measure the sensitivity of an option's price to changes in inputs:

| Greek | Definition | What it measures |
|-------|-----------|------------------|
| Delta ($\Delta$) | $\frac{\partial V}{\partial S}$ | Sensitivity to spot price |
| Gamma ($\Gamma$) | $\frac{\partial^2 V}{\partial S^2}$ | Curvature of delta (hedging cost) |
| Vega ($\mathcal{V}$) | $\frac{\partial V}{\partial \sigma}$ | Sensitivity to implied volatility |
| Theta ($\Theta$) | $\frac{\partial V}{\partial t}$ | Time decay |
| Rho ($\rho$) | $\frac{\partial V}{\partial r}$ | Sensitivity to interest rates |
| Vanna | $\frac{\partial^2 V}{\partial S \,\partial \sigma}$ | Cross-sensitivity of delta to vol |
| Volga | $\frac{\partial^2 V}{\partial \sigma^2}$ | Sensitivity of vega to vol |

### 6.2 Automatic Differentiation

Traditional libraries (QuantLib, etc.) compute Greeks via **finite differences** (bump-and-reprice):

$$
\Delta \approx \frac{V(S + h) - V(S - h)}{2h}
$$

This requires choosing $h$ (too large = truncation error; too small = floating-point cancellation), computing the price twice per Greek, and scales linearly with the number of risk factors.

VALAX uses **automatic differentiation** via `jax.grad`, which computes exact derivatives by applying the chain rule through the computational graph of the pricing function.

**Forward mode AD:** Propagates derivatives forward through the computation. Computes $\partial V / \partial x_i$ for a single input $x_i$ in one pass. Efficient when there are few inputs and many outputs.

**Reverse mode AD (backpropagation):** Propagates derivatives backward from the output. Computes $\partial V / \partial x_i$ for *all* inputs in one pass. Efficient when there is one output (a price) and many inputs (all risk factors). This is what `jax.grad` uses by default.

**Cost:** One reverse-mode pass costs approximately 2–4x the cost of the forward evaluation. This gives *all* first-order Greeks simultaneously — versus $2N$ evaluations for $N$ Greeks via central finite differences.

**Higher-order Greeks** use nested differentiation: `jax.grad(jax.grad(price_fn))` gives gamma. The computational cost grows linearly with the nesting depth, but each level is exact.

**VALAX implementation:** `valax/greeks/autodiff.py` provides `greeks()` (all Greeks at once) and `greek()` (single Greek by name). These are thin wrappers around `jax.grad` with appropriate `argnums` selection. Because every VALAX data structure is a JAX pytree, differentiation works through curves (`DiscountCurve`), surfaces, and model parameters — giving key-rate durations, surface sensitivities, and model parameter Greeks automatically.

### 6.3 Pathwise Method for MC Greeks

When computing Greeks of MC prices, `jax.grad` differentiates through the entire path simulation:

$$
\frac{\partial}{\partial \theta}\mathbb{E}[g(S_T(\theta))] = \mathbb{E}\!\left[\frac{\partial g}{\partial S_T} \cdot \frac{\partial S_T}{\partial \theta}\right]
$$

This is the **pathwise (infinitesimal perturbation analysis)** estimator. It works when:

- The payoff $g$ is differentiable w.r.t. $S_T$ (or smoothed to be so)
- The path $S_T(\theta)$ is differentiable w.r.t. the parameter $\theta$

**When pathwise fails:** Discontinuous payoffs (digital options, barrier knock-in/out). The derivative of an indicator function is zero almost everywhere and infinite at the barrier — the estimator has zero variance but is biased (always returns zero). VALAX addresses this via smooth sigmoid approximations to barriers in `valax/pricing/mc/payoffs.py`. The alternative **likelihood ratio method** differentiates the probability density instead of the payoff, but is not yet implemented.

---

## 7. Risk Measures

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
\Delta V \approx \underbrace{\sum_i \frac{\partial V}{\partial S_i} \Delta S_i}_{\text{Rung 1: Delta (spot)}}
+ \underbrace{\sum_j \frac{\partial V}{\partial \sigma_j} \Delta \sigma_j}_{\text{Rung 2: Vega}}
+ \underbrace{\sum_k \frac{\partial V}{\partial r_k} \Delta r_k}_{\text{Rung 3: Rho / DV01}}
+ \underbrace{\sum_i \frac{\partial V}{\partial q_i} \Delta q_i}_{\text{Rung 4: Dividend}}
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

---

## 8. Calibration Theory

### 8.1 The Calibration Problem

Model calibration finds parameters $\boldsymbol{\theta}$ such that model prices match market prices:

$$
\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^{N} w_i \left(\sigma_{\text{model}}(K_i, T_i; \boldsymbol{\theta}) - \sigma_{\text{market}}(K_i, T_i)\right)^2
$$

Calibration is typically done in **implied volatility space** rather than price space because:

- Vols are more homogeneous in scale (a 0.1% vol error has consistent meaning across strikes)
- Price errors are dominated by ATM options (highest vega), under-weighting wings
- Vol-space residuals produce a better-conditioned Jacobian

### 8.2 Levenberg-Marquardt Algorithm

For least-squares problems $\min \|\mathbf{r}(\boldsymbol{\theta})\|^2$, the Levenberg-Marquardt (LM) algorithm interpolates between Gauss-Newton and gradient descent:

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - (\mathbf{J}^T\mathbf{J} + \lambda \mathbf{I})^{-1} \mathbf{J}^T \mathbf{r}
$$

where $\mathbf{J} = \partial \mathbf{r} / \partial \boldsymbol{\theta}$ is the Jacobian of residuals. When $\lambda \to 0$, this is Gauss-Newton (fast near the solution). When $\lambda \to \infty$, this is gradient descent (robust far from the solution). $\lambda$ is adapted automatically.

**VALAX advantage:** The Jacobian $\mathbf{J}$ is computed exactly via `jax.jacobian` — no finite differences. This gives faster convergence (accurate search directions) and is cheaper for models with many market quotes (one reverse-mode pass per residual, vs. $2p$ evaluations for $p$-parameter finite differences).

**VALAX implementation:** `valax/calibration/sabr.py` and `valax/calibration/heston.py` use `optimistix.least_squares` (which implements LM). Alternative solvers: BFGS via `optimistix.minimise`, Adam via `optax`.

### 8.3 Parameter Constraints and Transforms

Many model parameters have natural bounds:

| Parameter | Constraint | Transform |
|-----------|-----------|-----------|
| Volatility $\sigma$ | $> 0$ | $\sigma = \text{softplus}(x) = \ln(1 + e^x)$ |
| Correlation $\rho$ | $(-1, 1)$ | $\rho = \tanh(x)$ |
| CEV exponent $\beta$ | $[0, 1]$ | $\beta = \text{sigmoid}(x)$ |
| Mean-reversion $\kappa$ | $> 0$ | $\kappa = \text{softplus}(x)$ |

VALAX optimizes over the **unconstrained** variable $x$ and applies the transform to get the **constrained** parameter. The transforms are smooth and differentiable — autodiff flows through them seamlessly.

**VALAX implementation:** `valax/calibration/transforms.py` defines `to_unconstrained` and `to_constrained` for each transform type.

### 8.4 Identifiability and Ill-Conditioning

**Identifiability:** A model is identifiable if different parameter values produce different prices. If parameters are non-identifiable (or nearly so), the calibration problem has multiple solutions and the optimizer may find any of them — producing unstable Greeks.

**Known issues:**

- **Heston:** $\kappa$ and $\theta$ are weakly identified from vanilla options alone (the term structure of smile is needed to separate them). This often manifests as a flat direction in the loss surface.
- **SABR:** With $\beta$ fixed (standard practice), the remaining three parameters ($\alpha$, $\rho$, $\nu$) are well-identified from three or more strikes at a single expiry.
- **SVI:** Five parameters for a single-expiry smile can be over-parameterized when only a few strikes are liquid.

**Regularization** (not yet implemented): Adding a penalty term $\lambda \|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{prior}}\|^2$ to the loss function biases the solution toward a prior (e.g., yesterday's parameters), improving stability at the cost of fit quality. This is standard practice for Heston calibration in production.

---

## References

### Foundational

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*.
- Merton, R. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics*.
- Harrison, J. and Pliska, S. (1981). "Martingales and Stochastic Integrals in the Theory of Continuous Trading." *Stochastic Processes and their Applications*.

### Models

- Black, F. (1976). "The Pricing of Commodity Contracts." *Journal of Financial Economics*.
- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*.
- Hagan, P. et al. (2002). "Managing Smile Risk." *Wilmott Magazine*.
- Brace, A., Gatarek, D., and Musiela, M. (1997). "The Market Model of Interest Rate Dynamics." *Mathematical Finance*.
- Hull, J. and White, A. (1990). "Pricing Interest-Rate-Derivative Securities." *Review of Financial Studies*.
- Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term Structure Models I: Single-Factor Models." *Journal of Derivatives*.
- Jamshidian, F. (1989). "An Exact Bond Option Formula." *Journal of Finance*.
- Margrabe, W. (1978). "The Value of an Option to Exchange One Asset for Another." *Journal of Finance*.
- Kirk, E. (1995). "Correlation in the Energy Markets." In *Managing Energy Price Risk*, Risk Books.
- Carmona, R. and Durrleman, V. (2003). "Pricing and Hedging Spread Options." *SIAM Review*.
- Garman, M. and Kohlhagen, S. (1983). "Foreign Currency Option Values." *Journal of International Money and Finance*.

### Inflation

- Jarrow, R. and Yildirim, Y. (2003). "Pricing Treasury Inflation Protected Securities and Related Derivatives using an HJM Model." *Journal of Financial and Quantitative Analysis*.
- Kerkhof, J. (2005). "Inflation Derivatives Explained." Lehman Brothers Fixed Income Quantitative Research.
- Brigo, D. and Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*, ch. 15 (Inflation).

### Volatility Surfaces

- Dupire, B. (1994). "Pricing with a Smile." *Risk Magazine*.
- Gatheral, J. (2004). "A Parsimonious Arbitrage-Free Implied Volatility Parameterization." Presentation at Global Derivatives.
- Lee, R. (2004). "The Moment Formula for Implied Volatility at Extreme Strikes." *Mathematical Finance*.
- Gatheral, J. and Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces." *Quantitative Finance*.

### Numerical Methods

- Cox, J., Ross, S., and Rubinstein, M. (1979). "Option Pricing: A Simplified Approach." *Journal of Financial Economics*.
- Longstaff, F. and Schwartz, E. (2001). "Valuing American Options by Simulation." *Review of Financial Studies*.
- Fang, F. and Oosterlee, C. (2008). "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on Scientific Computing*.

### Curves and Calibration

- Brigo, D. and Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*. Springer.
- Hagan, P. and West, G. (2006). "Interpolation Methods for Curve Construction." *Applied Mathematical Finance*.
- Rebonato, R. (2002). *Modern Pricing of Interest-Rate Derivatives*. Princeton University Press.

### Risk

- Artzner, P. et al. (1999). "Coherent Measures of Risk." *Mathematical Finance*.
- McNeil, A., Frey, R., and Embrechts, P. (2005). *Quantitative Risk Management*. Princeton University Press.
