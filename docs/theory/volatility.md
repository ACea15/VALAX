[← Models & Theory index](index.md)

# 4. Volatility

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

**Definition.** Local volatility $\sigma_{\text{loc}}(S, t)$ is the *unique deterministic* volatility function such that the diffusion

$$
dS_t = (r - q)\,S_t\,dt + \sigma_{\text{loc}}(S_t, t)\,S_t\,dW_t
$$

matches every European call price observed in the market. Uniqueness is the Dupire–Derman–Kani existence theorem: any model whose risk-neutral marginals reproduce the vanilla surface must produce the *same* $\sigma_{\text{loc}}$.

**Two equivalent extraction formulas.** Dupire's original derivation gave the *price-space* form

$$
\sigma_{\text{loc}}^2(K, T) \;=\; \frac{\dfrac{\partial C}{\partial T} + (r - q)\,K\,\dfrac{\partial C}{\partial K} + q\,C}{\dfrac{1}{2}\,K^2\,\dfrac{\partial^2 C}{\partial K^2}}.
$$

Equivalently, working in total implied variance $w(k, T) = \sigma_{\text{IV}}^2(k, T)\,T$ as a function of log-moneyness $k = \ln(K / F(T))$ (Gatheral 2006), one gets the *IV-space* form

$$
\sigma_{\text{loc}}^2(k, T) \;=\; \frac{\dfrac{\partial w}{\partial T}}{1 - \dfrac{k}{w}\,\dfrac{\partial w}{\partial k} + \dfrac{1}{4}\left(-\dfrac{1}{4} - \dfrac{1}{w} + \dfrac{k^2}{w^2}\right)\!\left(\dfrac{\partial w}{\partial k}\right)^{\!2} + \dfrac{1}{2}\,\dfrac{\partial^2 w}{\partial k^2}}.
$$

VALAX uses the second form. The arithmetic is identical, but the IV-space form has three structural advantages for an autodiff library:

1. **Smooth source data.** A calibrated `SVIVolSurface` gives $w(k, T)$ as a closed-form $C^{\infty}$ function of $k$. Its three partial derivatives are exact via `jax.grad`. The price-space form needs $\partial^2 C / \partial K^2$, which on any discrete strike grid is numerically two finite differences deep — noisy and prone to amplifying arbitrage violations.
2. **Arbitrage diagnostic.** The denominator above equals exactly the no-arbitrage "g-function" of Gatheral & Jacquier (2014); a non-positive value at $(k, T)$ is a butterfly-arbitrage violation in the input surface. We let it propagate as NaN rather than clamping, so a malformed surface fails loudly.
3. **Differentiability through surface parameters.** $\partial \sigma_{\text{loc}}^2 / \partial \text{(SVI } a, b, \rho, m, \sigma\text{)}$ is one more `jax.grad` away — useful for risk attribution and for the leverage-function calibration in §4.5.

**Why the IV-space derivation works.** Substitute $C = C_{\text{BS}}(K, T;\,\sigma_{\text{IV}}(K, T))$ into the price-space Dupire formula and apply the chain rule. The BS price derivatives in $(K, T)$ combine with the implied-vol derivatives in a way that, after reparameterising $K \to k$ and $\sigma_{\text{IV}} \to w$, telescopes to the formula above. The full derivation is in Gatheral (2006, §1.4–1.5).

**Numerical truncation interval is not needed.** Unlike Fourier-inversion pricers (Heston COS), Dupire extraction is purely *local* in $(k, T)$ — one evaluation of $w$ and three of its derivatives. There is no series to truncate, no integral to discretise. The only numerical care points are:

- **Short-maturity boundary.** Standard SVI surfaces flat-extrapolate $w$ below the first calibrated expiry, making $\partial w / \partial T = 0$ and $\sigma_{\text{loc}} = 0$. VALAX's `SVIVolSurface` instead extrapolates $w$ linearly through the origin — i.e. holds implied vol constant as $T \to 0^{+}$, which makes $\partial w / \partial T$ positive and the formula well-posed at the boundary.
- **Precision.** Second derivatives of $w$ in `f32` lose ~3 significant digits near ATM. The Dupire entry point raises `RuntimeError` if `jax_enable_x64` is off; `valax/__init__.py` enables it library-wide by default.

**Monte Carlo simulation under the extracted surface.** Once $\sigma_{\text{loc}}$ is callable, simulation is a state- and time-dependent log-Euler scheme:

$$
\ln S_{t_{n+1}} = \ln S_{t_n} + \!\left(r - q - \tfrac{1}{2}\sigma_n^2\right)\!\Delta t + \sigma_n\sqrt{\Delta t}\,Z_n,
\qquad Z_n \sim \mathcal{N}(0, 1),
$$

where $\sigma_n = \sigma_{\text{loc}}\!\big(S_{t_n},\, t_n + \tfrac{1}{2}\Delta t\big)$ — evaluated at the **midpoint** in time rather than the left endpoint. This single choice does two things at once:

1. It avoids querying $\sigma_{\text{loc}}$ at $T = 0$, where the IV-space formula's $1/w$ terms diverge.
2. It cancels the leading time-direction contribution to the Euler weak-order-1 bias, so the residual bias on a vanilla call is empirically ~10 bp absolute IV at $\Delta t = 1/500$ rather than ~25 bp for left-endpoint Euler at the same step count.

The variance correction $-\tfrac{1}{2}\sigma_n^2 \Delta t$ is *not* the same as Andersen's "central" $K_3, K_4$ scheme for Heston — for local vol the spot SDE has no exchangeable variance-process term to absorb into algebraic constants, so the Itô correction is paid in full each step.

**Milstein scheme (opt-in).** VALAX additionally exposes a strong-order-1 Milstein step via `scheme="milstein"` on `generate_local_vol_paths`. The Milstein correction adds

$$
+\tfrac{1}{2}\,\sigma_n\,\big(\partial \sigma_{\text{loc}} / \partial k\big)\,\Delta t\,\big(Z_n^2 - 1\big),
$$

where $\partial \sigma_{\text{loc}} / \partial k$ is evaluated at $(S_{t_n}, t_n + \tfrac{1}{2}\Delta t)$ via `jax.value_and_grad`, doubling per-step compute. This is **strong-order-1** (Euler is strong-order-0.5) but **weak-order-1** — the same as Euler. In practice this means Milstein has lower path-wise (sample-by-sample) error but the *expectations* it computes — option prices — converge at the same asymptotic rate. Empirically on a moderate equity skew with 4 seeds × 100k paths × {100–1000} steps, Milstein and midpoint-Euler give indistinguishable vanilla-reprice bp errors (both in the 5–25 bp range, MC-noise-dominated). The default is midpoint-Euler because it is cheaper for no measurable accuracy loss on vanilla and SLV-calibration use cases; opt in to Milstein when the binding constraint is *path-wise* accuracy — realized variance, quadratic-variation-style payoffs, or research into the path-distribution structure of the SDE. To tighten the vanilla-reprice gate below the MC-noise floor, use variance reduction (control variates) rather than a higher-order scheme.

**Why local vol matters.** It is the *minimum-complexity model* that reprices the entire vanilla surface by construction. For exotic pricing on equity desks it is the default fallback when calibration of a parametric model (Heston, SABR) leaves a residual smile error larger than the bid-ask. The Dupire surface is also the input to SLV's leverage-function calibration (§4.5) — the leverage $L(S, t)$ that turns Heston into a smile-matching SLV is defined as $L^2 = \sigma_{\text{Dupire}}^2 / \mathbb{E}[V_t \mid S_t = S]$, with the conditional expectation computed by a particle-method MC over Heston-warmstarted SLV paths.

**Limitations.**

- **Smile dynamics are wrong.** Local vol predicts that as spot moves, the implied-vol smile *flattens* (the so-called sticky-local-vol regime). Empirically, equity index smiles move closer to *sticky-strike* — the smile shape stays roughly fixed in $K$-space. The mismatch shows up as a model-vs-market dynamic-hedge P&L on forward-starting payoffs.
- **Forward smile.** Forward-starting and cliquet structures depend on conditional forward smile $\sigma_{\text{IV}}(K, T_2 \mid S_{T_1})$, which under local vol collapses to a near-flat shape independent of $S_{T_1}$ — empirically wrong, and barrier options inherit a similar systematic underpricing.
- **Production systems use SLV.** Stochastic-local volatility combines the surface-matching of local vol with the realistic dynamics of stochastic vol — the industry standard for exotic equity pricing since roughly 2010. The full SLV stack — leverage-function representation, Markovian-projection calibration, joint Heston-QE / log-Euler-Milstein MC, dispatcher recipes — is shipped under §4.5.

**VALAX implementation.** Dupire extraction in `valax/pricing/analytic/dupire.py` (`dupire_local_vol`, `dupire_local_vol_from_strike`). Model wrapper in `valax/models/local_vol.py` (`LocalVolModel`, with a duck-typed `total_variance(k, T)` surface protocol satisfied by all three surface types). MC simulation in `valax/pricing/mc/local_vol_paths.py` (`generate_local_vol_paths`, `jax.lax.scan` over time with midpoint-σ log-Euler). Recipes for European, Asian, equity-barrier, lookback, and variance-swap payoffs are registered against `LocalVolModel` in the unified `mc_price_dispatch`. The validation gates pinned in the test suite are (i) flat-SVI → constant $\sigma_{\text{loc}}$ to $10^{-10}$, (ii) flat-vol LV MC → Black-Scholes at $3\sigma$, (iii) **SVI → Dupire → LV MC reprices the input vanilla IV grid to < 20 bp absolute** at 4 seeds × 100k paths × 500 steps (the headline Dupire-consistency gate), and (iv) QuantLib `LocalVolSurface` flat-limit cross-check to $10^{-6}$.

### 4.5 Stochastic-Local Volatility

**Definition.** A stochastic-local volatility (SLV) model superimposes a deterministic leverage function $L(S, t)$ on top of a stochastic-volatility backbone $V_t$, with the leverage calibrated so that the joint dynamics reproduce a target implied-vol surface by construction. Using the Heston backbone — the production default — the risk-neutral SDE is

$$
\frac{dS_t}{S_t} \;=\; (r - q)\,dt \;+\; L(k_t, t)\,\sqrt{V_t}\,dW_1, \qquad dV_t \;=\; \kappa(\theta - V_t)\,dt \;+\; \xi\,\sqrt{V_t}\,dW_2, \qquad \langle dW_1, dW_2 \rangle \;=\; \rho\,dt,
$$

with $k_t = \log(S_t / F(t))$ the running log-moneyness.

**The Markovian projection identity.** The cornerstone is Gyöngy's (1986) mimicking theorem: for any Itô process with stochastic volatility, there is a Markovian SDE with the *same one-dimensional marginals at every $T$*, whose diffusion coefficient is

$$
\sigma_{\text{proj}}^2(S, t) \;=\; \mathbb{E}\!\left[\sigma^2(S_t, V_t, t) \;\middle|\; S_t = S\right].
$$

For SLV, $\sigma^2(S_t, V_t, t) = L^2(k_t, t)\, V_t$, and since $L$ is a *deterministic* function of $(k, t)$ the conditional expectation factors:

$$
\sigma_{\text{proj}}^2(S, t) \;=\; L^2(k, t) \;\cdot\; \mathbb{E}\!\left[V_t \;\middle|\; k_t = k\right].
$$

If we now demand that this projected diffusion equal Dupire's local volatility (which by §4.4 reproduces the SVI surface exactly):

$$
\boxed{\;L^2(k, t) \;=\; \frac{\sigma_{\text{Dupire}}^2(k, t)}{\mathbb{E}[V_t \mid k_t = k]}\;}
$$

then by Gyöngy the SLV marginals match the SVI marginals at every $T$, i.e. all European options reprice exactly. This is the SLV calibration equation.

**Two-pass calibration.** The identity is a *fixed-point equation*: $L$ is defined in terms of a conditional expectation that itself depends on $L$ (through the SLV dynamics). The Guyon-Henry-Labordère (2012) particle method resolves this in two passes:

1. **Pass 1 — Heston to vanillas.** Calibrate the backbone $(\kappa, \theta, \xi, \rho, v_0)$ to vanilla quotes via VALAX's existing `calibrate_heston`. After pass 1, the Heston marginals are close to but not equal to the SVI marginals — the residual gap is what the leverage will close.
2. **Pass 2 — leverage to Dupire.** Initialise $L \equiv 1$ (pure-Heston limit) and advance an MC particle swarm slice-by-slice through the calibration time grid. At each slice $t_i$, estimate $\hat{\mathbb{E}}[V_{t_i} \mid k_{t_i} = k]$ from the particles via Nadaraya-Watson kernel regression, then set $L(k, t_i) = \sigma_{\text{Dupire}}(k, t_i) / \sqrt{\hat{\mathbb{E}}[V_{t_i} \mid k_{t_i} = k]}$. VALAX wraps this in an *outer fixed-point iteration* (`n_iterations`) — each outer iteration re-simulates the swarm under the previous-iteration leverage, tightening self-consistency.

**Estimator choice.** Two flavours of the Nadaraya-Watson core are exposed (see [SLV guide](../guide/slv.md#choosing-particle-vs-kernel)):

- **`method="particle"`** — pure kernel regression. Asymptotically unbiased; high variance in low-density tails.
- **`method="kernel"`** — adds a Tikhonov ridge that biases the estimator toward the empirical particle mean in tail regions. Small bias in well-populated regions, much smoother in the tails.

**Why SLV beats pure LV.** LV is calibration-perfect on today's vanillas but predicts the wrong *dynamic* smile — the forward smile under LV collapses to a near-flat shape independent of the realised path, while equity markets exhibit a persistent forward skew. SLV by construction matches today's vanillas (via the Markovian-projection identity) *and* preserves Heston's stochastic forward-skew dynamics. The net effect is correct pricing of payoffs sensitive to forward smile (autocallables, cliquets, forward-starting options) where pure LV underprices systematically.

**MC discretisation.** The SLV path generator couples Andersen's (2008) QE scheme on the variance leg (exact in distribution at each $\Delta t$, no Feller-condition penalty) with a log-Euler / Milstein leg on the log-spot:

$$
\ln S_{t_{n+1}} = \ln S_{t_n} + \left(r - q - \tfrac{1}{2}\sigma_n^2\right)\Delta t + \sigma_n \sqrt{\Delta t}\,Z_{1,n} + \underbrace{\tfrac{1}{2}\sigma_n \big(\partial_k L\big)_n \sqrt{V_n}\,\Delta t\,\bigl(Z_{1,n}^2 - 1\bigr)}_{\text{Milstein opt-in}},
$$

with $\sigma_n = L(k_n, t_n + \tfrac{1}{2}\Delta t)\,\sqrt{V_n}$ — the same midpoint-in-time evaluation as the LV path generator (§4.4), for the same reason (avoid the Dupire $T = 0$ singularity at the calibration-grid boundary). Correlation enters via $Z_{1,n} = \rho\,Z_v + \sqrt{1-\rho^2}\,Z_\perp$, where $Z_v$ is the standard normal driving the QE quadratic branch. This is *exact* on the quadratic branch (the typical case for equity-grade parameters) and a controlled approximation on the exponential branch — matching the QuantLib `HestonSLVProcess` convention.

**Limitations.**

- **Particle-method accuracy ceiling.** At moderate MC budgets the calibrated SLV reprices the input vanilla surface to ~100–250 bp absolute IV — an order of magnitude looser than LV's 20 bp gate. The kernel-regression variance in the tails, compounded with the QE-correlation approximation on the exponential branch, sets the floor. Sub-50 bp production accuracy requires Fokker-Planck PDE calibration (QuantLib's `HestonSLVProcess`); this is roadmap item SLV-2.
- **Backbone choice.** Heston is the only backbone shipped. SABR-LV and Bates-LV use the same Markovian-projection machinery but require their own MC generators; not implemented.
- **Calibration is MC-noisy.** Reproducibility across PRNG seeds is at the bp level — pin the seed (or the calibrated `LeverageGrid`) for production pricing.

**VALAX implementation.** Leverage tabulation in `valax/surfaces/leverage.py` (`LeverageGrid`, bilinear on `(k, t)`, differentiable through `values`). Model wrapper in `valax/models/slv.py` (`SLVModel`, x64-asserted `from_heston_and_leverage` factory). Calibration in `valax/calibration/slv.py` (`calibrate_slv_leverage` with `method ∈ {"particle", "kernel"}` and `n_iterations`; end-to-end `calibrate_slv` wraps Pass 1 + Pass 2). MC simulation in `valax/pricing/mc/slv_paths.py` (`generate_slv_paths` returning joint `(S, V)` paths, Andersen-QE variance + log-Euler/Milstein log-spot, midpoint-in-time leverage query). Recipes registered for `EuropeanOption`, `AsianOption`, `EquityBarrierOption`, `LookbackOption`, `VarianceSwap` against `SLVModel` in the unified `mc_price_dispatch`; the dispatcher routes a `slv_scheme=` kwarg through the recipe layer.

The validation gates pinned in the test suite are (i) `LeverageGrid.flat` round-trip + bilinear interpolation contract, (ii) `SLVModel` pytree round-trip + x64 guard + autodiff through `leverage.values` and the Heston block, (iii) flat-leverage SLV MC reduces to Andersen-QE Heston MC within $3 \sigma$ at every moneyness (the key reduction test, since `L \equiv 1` collapses the SLV SDE to pure Heston and Gyöngy's identity simplifies trivially), (iv) Markovian-projection identity holds within 10 % at ATM after calibration (direct algebraic check), (v) particle and kernel methods produce comparable leverages in dense regions and the kernel method is strictly smoother in the wings, (vi) the fixed-point loop contracts in sup-norm, (vii) **SLV MC reprices the input SVI vanilla IV grid to < 250 bp absolute** at 10k calibration particles + 50k pricing paths × 200 steps × 4 seeds (the headline Dupire-consistency gate — see "Limitations" above for why the bar is looser than LV's), and (viii) QuantLib analytic-Heston cross-check at the flat-leverage limit to $3\sigma$ on a 5-moneyness × 5-seed grid.
