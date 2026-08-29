[← Models & Theory index](index.md)

# 9. Interest-Rate Derivatives

Sections [2.6](stochastic-models.md#26-libor-market-model-lmm-bgm) and [2.8](stochastic-models.md#28-hull-white-one-factor-short-rate-model) give the *models* (LMM, Hull-White) and Section [3](curves.md#3-curve-framework) gives the *curves*. This section is about the layer between them: how vanilla rate options — caps/floors and swaptions — are priced by choosing the right **numéraire**, and where the model-dependent hard part (Bermudans, CMS convexity, smile) begins. The central technique is that a well-chosen numéraire turns the relevant rate into a **martingale**, collapsing the price to a Black-76 or Bachelier formula.

### 9.1 Change of Numeraire and the Forward Measure

For any tradeable numéraire $N_t > 0$ there is an equivalent measure $\mathbb{Q}^N$ under which every asset price deflated by $N$ is a martingale:

$$
\frac{V_t}{N_t} = \mathbb{E}^{N}\!\left[\frac{V_T}{N_T}\,\middle|\,\mathcal{F}_t\right].
$$

The risk-neutral measure of Section [1.1](foundations.md#11-no-arbitrage-martingales-and-risk-neutral-pricing) is the special case $N_t = B_t = \exp\!\int_0^t r_s\,ds$ (the money-market account). Rate derivatives are far easier under a different choice.

Taking the **zero-coupon bond** $P(t, T)$ as numéraire defines the **$T$-forward measure** $\mathbb{Q}^T$. Because $P(T, T) = 1$, any payoff $X$ settled at $T$ prices as

$$
V_0 = P(0, T)\,\mathbb{E}^{T}\!\left[X\right],
$$

i.e. **discount once by today's bond, then take a forward-measure expectation** — the stochastic discount factor and the payoff decouple. The pivotal fact: the forward rate for period $[T, T+\tau]$ (below) is a $\mathbb{Q}^{T+\tau}$-**martingale**, so its expectation is just today's forward. This is the engine behind every closed-form rate-option formula.

### 9.2 Caps, Floors, and the Caplet Formula

The simply-compounded **forward rate** seen at $t$ for accrual period $[T_{i-1}, T_i]$ with year fraction $\tau_i$ is, from Section [3.2](curves.md#32-single-curve-vs-multi-curve-framework),

$$
F_i(t) = \frac{1}{\tau_i}\!\left(\frac{P(t, T_{i-1})}{P(t, T_i)} - 1\right)
\quad\text{(forwarding curve)},
$$

and it is a martingale under the $\mathbb{Q}^{T_i}$-forward measure that uses the **discount** (OIS) bond $P(\cdot, T_i)$ as numéraire — the split of the two curves is exactly the multi-curve content of Section [3.7](curves.md#37-no-arbitrage-relations-across-curves).

A **caplet** pays $\tau_i\,(F_i(T_{i-1}) - K)^+$ at $T_i$. Under $\mathbb{Q}^{T_i}$ with $F_i$ a martingale, the price is a Black-76 (Section [2.2](stochastic-models.md#22-black-76-futures-forwards)) expression:

$$
\mathrm{Caplet}_i = P(0, T_i)\,\tau_i\left[F_i(0)\,\Phi(d_1) - K\,\Phi(d_2)\right],\qquad
d_{1,2} = \frac{\ln(F_i(0)/K) \pm \tfrac12\sigma_i^2 T_{i-1}}{\sigma_i\sqrt{T_{i-1}}},
$$

or, for **normal** (Bachelier, Section [2.3](stochastic-models.md#23-bachelier-normal-model)) quoting — the post-2008 market standard for rates because forwards can be negative,

$$
\mathrm{Caplet}_i = P(0, T_i)\,\tau_i\,\sigma_i\sqrt{T_{i-1}}\left[d\,\Phi(d) + \phi(d)\right],\qquad d = \frac{F_i(0) - K}{\sigma_i\sqrt{T_{i-1}}}.
$$

A **cap** is a strip of caplets, $\mathrm{Cap} = \sum_i \mathrm{Caplet}_i$, each with its own accrual and its own optionlet volatility $\sigma_i$. Floors are the put analogue, and **put–call parity is the swap**:

$$
\mathrm{Cap}(K) - \mathrm{Floor}(K) = \text{payer swap struck at } K.
$$

**VALAX implementation:** `valax/pricing/analytic/caplets.py` (`caplet_price_black76`, `caplet_price_bachelier`, `cap_price_*`) is dual-curve aware and takes a per-caplet volatility term structure.

### 9.3 Swaptions and the Annuity Measure

A payer swap over $[T_0, T_N]$ has value $P(t,T_0) - P(t,T_N) - K\,A(t)$, where the **annuity** (present value of a basis point) is

$$
A(t) = \sum_{i=1}^{N} \tau_i\,P(t, T_i).
$$

The **forward swap rate** — the fixed rate making the swap worth zero — is

$$
S(t) = \frac{P(t, T_0) - P(t, T_N)}{A(t)}.
$$

Choosing the **annuity $A(t)$ itself as numéraire** gives the **annuity (swap) measure** $\mathbb{Q}^A$, under which $S(t)$ is a martingale (it is a ratio of tradeables to $A$). A payer **swaption** pays $A(T_0)\,(S(T_0) - K)^+$ at expiry $T_0$, so

$$
\mathrm{Swaption} = A(0)\,\mathbb{E}^{A}\!\left[(S(T_0) - K)^+\right] = A(0)\left[S(0)\,\Phi(d_1) - K\,\Phi(d_2)\right]
$$

(Black-76), with the obvious Bachelier counterpart. The annuity discounts; the forward swap rate is the underlying. **Cash-settled** swaptions replace $A(T_0)$ with a cash annuity evaluated at the single fixing $S(T_0)$, a small model-dependent difference that vanishes at the money.

**VALAX implementation:** `valax/pricing/analytic/swaptions.py` (`swap_rate`, `swap_price`, `swaption_price_black76`, `swaption_price_bachelier`) with a dual-curve annuity and projection.

### 9.4 Bermudan Swaptions and Early Exercise

A **Bermudan swaption** may be exercised into the underlying swap on any of a set of dates, so its value is an **optimal-stopping** problem — no single-numéraire closed form exists, and the price depends on the *joint* dynamics of the curve (in particular the terminal decorrelation of forward rates and the mean-reversion of the short rate). Three routes, all present or enabled in VALAX:

- **Longstaff-Schwartz (LSM)** on simulated LMM paths: regress the continuation value on curve state variables at each exercise date, then exercise when immediate value exceeds the regressed continuation. Implemented in `valax/pricing/mc/bermudan.py` on the LMM of Section [2.6](stochastic-models.md#26-libor-market-model-lmm-bgm).
- **Backward induction on the Hull-White trinomial tree** (Section [2.8](stochastic-models.md#28-hull-white-one-factor-short-rate-model)): the single short-rate state makes the exercise decision a per-node $\max(\text{continuation}, \text{exercise})$ — the same machinery already used for callable bonds.
- **Short-rate PDE** (roadmap PR-3): the exercise projection seam of the finite-difference solver (Section [5.2](pricing-methods.md#52-pde-finite-differences)) carries over directly. Not yet built.

The three routes should agree; building them so they cross-validate is the recommended sequencing (see the roadmap).

### 9.5 CMS and Convexity Adjustments

A **constant-maturity swap (CMS)** pays a swap rate $S(T_p)$ at a single date $T_p$ rather than over the swap's own annuity schedule. The naive "use today's forward swap rate" is **biased**: $S$ is a martingale under the annuity measure $\mathbb{Q}^A$, but a CMS coupon pays under the $T_p$-forward measure $\mathbb{Q}^{T_p}$. The Radon-Nikodym derivative between the two is not constant in $S$, so

$$
\mathbb{E}^{T_p}\!\left[S(T_p)\right] = S(0) + \underbrace{\text{convexity adjustment}}_{\text{depends on the swaption smile}}.
$$

The market-standard evaluation is **Hagan static replication**: express the CMS payoff as an integral over a continuum of payer/receiver swaptions and price each on the swaption smile (typically SABR). The adjustment grows with volatility, maturity, and payment delay.

**VALAX implementation status:** `valax/pricing/analytic/rates_exotics.py` prices `cms_swap_price` and `cms_cap_floor_price_black76` at the **baseline (zero convexity adjustment)** level; the Hagan/SABR replication is a documented gap (see the roadmap and the temporary `RATES_SESSION_GUIDE.md`). This is the cleanest example in the library of a pricer that is *structurally* correct but *not yet market-accurate* — and of why the smile, not just the curve, eventually matters for rates.

### 9.6 Model Choice for Rate Derivatives

| Need | Model | Why | Section |
|------|-------|-----|---------|
| Callable/puttable bonds, exact curve fit | **Hull-White 1F** | Single state, affine ZCB, exact-fit; tree/PDE for exercise | [2.8](stochastic-models.md#28-hull-white-one-factor-short-rate-model) |
| European swaption *level* (fast calibration) | **Hull-White** via Jamshidian | Decompose into a portfolio of ZCB options | [2.8](stochastic-models.md#28-hull-white-one-factor-short-rate-model) |
| Full forward-curve dynamics, Bermudans | **LMM / BGM** | Models each forward directly; PCA-reduced factors | [2.6](stochastic-models.md#26-libor-market-model-lmm-bgm) |
| Swaption/cap **smile** | **SABR** per expiry | Fits the strike dimension the one-factor models cannot | [2.5](stochastic-models.md#25-sabr) |

The recurring limitation of one-factor short-rate models is that a **single volatility parameter cannot fit the swaption grid** — Jamshidian calibration matches a *level*, not a smile. G2++/HW-2F and the market models trade tractability for that flexibility.

**Calibration status.** VALAX has the *pricers* (caps, swaptions, HW ZCB and tree) but **not yet the rate-model calibrators**: Hull-White→swaption-surface (Jamshidian) and LMM caplet/swaption stripping are on the roadmap. The generic SABR calibrator of Section [8](calibration.md#8-calibration-theory) applies directly to a swaption smile once a swaption-vol quoting object exists. The concrete sequencing — validate the existing pricers against QuantLib, add Hull-White Monte-Carlo, then the Jamshidian calibrator, then the short-rate PDE — is captured in the temporary `RATES_SESSION_GUIDE.md` at the repository root.
