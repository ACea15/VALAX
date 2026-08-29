[← Models & Theory index](index.md)

# 2. Stochastic Models

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

**The Feller condition:** If $2\kappa\theta > \xi^2$, the variance process $v_t$ is strictly positive almost surely. When violated, $v_t$ can hit zero — a regime that defeats naïve Euler-with-reflection schemes, which acquire $O(1/\sqrt{n_{\text{steps}}})$ bias at the absorbing boundary. VALAX's `generate_heston_paths` (in `valax/pricing/mc/paths.py`) is implemented as **Andersen's (2008) Quadratic-Exponential (QE) scheme**, which is bias-free in distribution at each $\Delta t$ step regardless of Feller compliance. The variance is sampled by exact two-moment matching against either a shifted-squared-normal (quadratic branch, low variance-of-variance) or a Bernoulli–exponential mixture (exponential branch, high variance-of-variance); the log-spot uses Andersen's matching "central" discretisation with trapezoidal weights $\gamma_1 = \gamma_2 = 1/2$. This is the canonical choice for Heston MC and is why the validation pyramid's Stage-3 Heston Asian chain test (`tests/test_quantlib_comparison/test_exotic_on_heston_surface_ql.py`) agrees with QuantLib's `MCDiscreteArithmeticAPHestonEngine` at $3\,\text{SE}$ on Feller-violating calibrated parameter sets.

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

**VALAX implementation:** Model definition in `valax/models/heston.py` (`HestonModel`). MC simulation via Andersen QE in `valax/pricing/mc/paths.py::generate_heston_paths`. Calibration in `valax/calibration/heston.py`. The simulation works in $(\\ln S, v)$ space with the QE-conditional Gaussian for the log-spot update at each step.

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
