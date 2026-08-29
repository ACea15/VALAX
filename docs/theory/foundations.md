[← Models & Theory index](index.md)

# 1. Foundational Framework

Every pricing formula in VALAX rests on three pillars: no-arbitrage, risk-neutral valuation, and the connection between stochastic processes and partial differential equations. Understanding these unlocks the entire library.

### 1.1 No-Arbitrage, Martingales, and Risk-Neutral Pricing

#### What is no-arbitrage?

An **arbitrage** is a trading strategy that costs nothing today, has no chance of loss in any future state, and has a positive probability of profit in at least one state. In plain language: free money with zero risk.

No-arbitrage is the single most powerful assumption in quantitative finance. It is weaker than any specific model — it doesn't require GBM, constant vol, or continuous trading. It simply says: *you cannot design a trading strategy that wins money for free*. If such a strategy existed, every market participant would pile into it, and the resulting price pressure would destroy it. No-arbitrage is an equilibrium condition, not a mathematical convenience.

#### What is a martingale?

A **martingale** is a stochastic process whose expected future value, given all information available today, equals its current value:

$$
\mathbb{E}[X_T \mid \mathcal{F}_t] = X_t \qquad \text{for all } T > t
$$

where $\mathcal{F}_t$ represents the information available at time $t$. In intuitive terms: a martingale is a "fair game" — on average, it doesn't go up or down. Your best forecast of tomorrow's value is today's value.

A simple random walk $X_n = X_0 + Z_1 + Z_2 + \cdots + Z_n$ with $Z_i \sim \mathcal{N}(0,1)$ is a martingale. A stock with a positive expected return is **not** a martingale — it drifts upward on average. A stock's price discounted at the risk-free rate is not a martingale under $\mathbb{P}$ either (because $\mu \neq r$). But as we'll see, no-arbitrage forces the existence of a measure under which the discounted price *is* a martingale.

#### The Fundamental Theorem: why the link exists

The **First Fundamental Theorem of Asset Pricing** (Harrison & Kreps 1979, Harrison & Pliska 1981) establishes:

> A market is arbitrage-free **if and only if** there exists at least one **equivalent martingale measure** $\mathbb{Q}$ — a probability measure equivalent to $\mathbb{P}$ under which all discounted asset prices are martingales.

This is not obvious. Why should "no free money" be the same thing as "there exists a probability measure making discounted prices fair games"? Here is the logic in both directions.

#### Direction 1: Martingale measure ⟹ no arbitrage

This is the easy direction. Suppose $\mathbb{Q}$ exists such that the discounted price $\tilde{S}_t = e^{-rt}S_t$ is a $\mathbb{Q}$-martingale:

$$
\mathbb{E}^{\mathbb{Q}}[\tilde{S}_T \mid \mathcal{F}_t] = \tilde{S}_t
$$

Now consider any self-financing trading strategy with initial cost zero and final value $V_T$. "Self-financing" means no money is injected or withdrawn — every rebalancing is funded from the portfolio itself. The discounted portfolio value $\tilde{V}_t = e^{-rt}V_t$ is a stochastic integral against a $\mathbb{Q}$-martingale, which is itself a $\mathbb{Q}$-martingale (under technical integrability conditions). Therefore:

$$
\mathbb{E}^{\mathbb{Q}}[\tilde{V}_T] = \tilde{V}_0 = 0
$$

If $V_T \geq 0$ in all states (no chance of loss) and $\mathbb{E}^{\mathbb{Q}}[\tilde{V}_T] = 0$, then $V_T = 0$ $\mathbb{Q}$-almost surely. Since $\mathbb{Q}$ is equivalent to $\mathbb{P}$ (same null sets), $V_T = 0$ $\mathbb{P}$-almost surely too. There is no state with positive profit. Hence no arbitrage.

**The key step:** the argument works because $\mathbb{Q}$ is *equivalent* to $\mathbb{P}$ — they agree on which events are possible. If $\mathbb{Q}$ could assign zero probability to a state that $\mathbb{P}$ considers possible, the argument would break: you could have $V_T > 0$ in that state without violating $\mathbb{E}^{\mathbb{Q}}[\tilde{V}_T] = 0$.

#### Direction 2: No arbitrage ⟹ martingale measure exists

This is the deep direction, and it is one of the most beautiful results in mathematical finance. The idea, stripped to its essence:

Think of all possible payoffs you can create from trading. These form a **cone** in the space of random variables — you can combine strategies and scale them up. The "no loss" payoffs (those $\geq 0$ in all states) form another cone — the **positive orthant**.

No-arbitrage says these two cones touch only at zero: the only risk-free, zero-cost portfolio that never loses money is the trivial one that always pays zero. Geometrically, there must be a **separating hyperplane** between the trading cone and the positive orthant (this is the Hahn-Banach theorem, or in finite dimensions, a separating hyperplane theorem from convex analysis).

That separating hyperplane defines a linear functional $\phi$ that is:
- **Positive** on all non-negative payoffs: $\phi(X) > 0$ whenever $X \geq 0$ and $X \neq 0$
- **Zero** on all traded (replicable) payoffs with zero initial cost

A positive linear functional on random variables *is* an expectation under some probability measure. That measure is $\mathbb{Q}$. The fact that $\phi$ is zero on all traded payoffs means traded payoffs have zero expected discounted value under $\mathbb{Q}$ — equivalently, discounted prices are $\mathbb{Q}$-martingales.

**A one-period example to make this concrete:**

Consider a stock that is $S_0 = 100$ today and can go to either $S_u = 120$ or $S_d = 90$ tomorrow. The risk-free rate is $r = 5\%$, so the bond goes from $1$ to $1.05$.

Can we find probabilities $q_u, q_d = 1 - q_u$ such that the discounted stock is a martingale?

$$
\mathbb{E}^{\mathbb{Q}}\!\left[\frac{S_1}{1.05}\right] = S_0 \quad\Longleftrightarrow\quad q_u \cdot \frac{120}{1.05} + (1 - q_u) \cdot \frac{90}{1.05} = 100
$$

Solving: $q_u \cdot 120 + (1 - q_u) \cdot 90 = 105$, so $30\,q_u = 15$, giving $q_u = 0.5$ and $q_d = 0.5$.

Under these "risk-neutral probabilities," the stock's expected return is $0.5 \times 120 + 0.5 \times 90 = 105$, which equals $100 \times 1.05$ — the risk-free return. The stock earns the risk-free rate *in expectation*, not because anyone believes up and down are equally likely, but because these are the probabilities that make the market arbitrage-free.

Why does this work? If $q_u$ didn't exist (for instance, if $S_u = S_d = 120$, so the stock always beats the bond), you could borrow at $r$, buy the stock, and pocket the difference — a riskless profit. The impossibility of arbitrage forces the existence of $q_u \in (0, 1)$ — and that interval $(0, 1)$ is what makes $\mathbb{Q}$ equivalent to $\mathbb{P}$ (both states remain possible).

**When does it break?** If $S_u = 103$ and $S_d = 90$, then $q_u = 15/13 > 1$ — no valid probability measure exists, and indeed there's an arbitrage: the bond dominates the stock in the up state ($105 > 103$) and the stock can be shorted in the down state. The one-period FTAP is: $d < 1 + r < u$ if and only if $q_u \in (0, 1)$ if and only if no arbitrage.

#### The Second Fundamental Theorem: uniqueness and completeness

The **Second Fundamental Theorem** adds:

> The market is **complete** (every contingent claim can be replicated by a trading strategy) if and only if the equivalent martingale measure $\mathbb{Q}$ is **unique**.

In the one-period example above, we had two states and two assets (stock and bond), giving exactly one equation and one unknown ($q_u$) — a unique solution. The market is complete: any payoff $(V_u, V_d)$ can be replicated by holding $\Delta$ shares and $B$ bonds.

If we added a third state (say the stock could also stay at 100), we'd have two equations (the martingale condition has two constraints now, but only one free probability parameter after $q_u + q_d + q_m = 1$) — this is underdetermined, giving infinitely many valid $\mathbb{Q}$ measures. The market is incomplete: some payoffs can't be replicated, and their price is not unique — it lies in a no-arbitrage *interval* $[\inf_{\mathbb{Q}} \mathbb{E}^{\mathbb{Q}}[\tilde{V}_T],\; \sup_{\mathbb{Q}} \mathbb{E}^{\mathbb{Q}}[\tilde{V}_T]]$.

Stochastic volatility models (Heston, SABR) are incomplete: the volatility risk cannot be hedged with the stock alone. This is why Heston has a "market price of volatility risk" that must be specified (or equivalently, why the risk-neutral parameters $\kappa^{\mathbb{Q}}, \theta^{\mathbb{Q}}$ differ from their physical counterparts — the measure is not unique and must be pinned down by calibrating to option prices).

#### From the theorem to the pricing formula

Once we accept that $\mathbb{Q}$ exists and discounted prices are $\mathbb{Q}$-martingales, the pricing formula follows in three lines.

Let $V_T$ be a derivative payoff at time $T$. If the derivative can be replicated by a self-financing strategy, then its discounted value $\tilde{V}_t$ must also be a $\mathbb{Q}$-martingale (it's a portfolio of $\mathbb{Q}$-martingales). In particular:

$$
\tilde{V}_0 = \mathbb{E}^{\mathbb{Q}}[\tilde{V}_T \mid \mathcal{F}_0] \quad\Longrightarrow\quad V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[V_T]
$$

or more generally, with stochastic rates:

$$
V_0 = \mathbb{E}^{\mathbb{Q}}\!\left[\exp\!\left(-\int_0^T r_s\,ds\right) V_T\right]
$$

This is the **master equation** behind every pricing function in VALAX:

- Analytical formulas (e.g., `valax/pricing/analytic/black_scholes.py`) evaluate this expectation in closed form — possible only when the SDE and payoff have special structure.
- Monte Carlo (`valax/pricing/mc/engine.py`) estimates it by simulating paths under $\mathbb{Q}$ and averaging the discounted payoff.
- PDE methods (`valax/pricing/pde/solvers.py`) solve the equivalent differential equation (via Feynman-Kac, Section 1.4).
- Lattice methods (`valax/pricing/lattice/binomial.py`) approximate it on a discrete tree — essentially the one-period argument applied recursively at each node.

In every case, the $\mathbb{Q}$-drift (not the real-world drift $\mu$) appears in the computation. The $\mathbb{Q}$-drift is determined by the martingale condition: discounted prices must have zero drift under $\mathbb{Q}$, which forces the asset's risk-neutral drift to be $r - q$. This is not a choice — it is a consequence of no-arbitrage.

**VALAX convention:** All models are specified under $\mathbb{Q}$. Drift terms in SDEs (e.g., $r - q$ in Black-Scholes) are risk-neutral drifts, not real-world expected returns. The connection between the drift specification and the measure choice is detailed in Section 1.3 (Girsanov).

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

#### What is a "measure"?

A probability measure assigns probabilities to events. When we say a stock follows GBM under the physical measure $\mathbb{P}$:

$$
dS_t = \mu\,S_t\,dt + \sigma\,S_t\,dW_t^{\mathbb{P}}
$$

we mean that in the real world, the stock has expected return $\mu$ (say, 8%/year for equities). The process $W_t^{\mathbb{P}}$ is a Brownian motion under $\mathbb{P}$ — meaning its increments $W_{t+\Delta t}^{\mathbb{P}} - W_t^{\mathbb{P}} \sim \mathcal{N}(0, \Delta t)$ under the real-world probability distribution.

But for pricing, we need a different measure $\mathbb{Q}$ — the risk-neutral measure — under which the stock earns only the risk-free rate $r$. "Changing the measure" means redefining which probability distribution we use to compute expectations, in a way that is consistent (same null sets — events with zero probability stay at zero probability).

The deep question is: **if a Brownian motion is just a sequence of Gaussian random draws, how can we change a probability measure without changing the random draws themselves?**

#### The Radon-Nikodym derivative: reweighting, not resampling

The answer is that we don't redraw the randomness — we **reweight** it. The Radon-Nikodym derivative $\frac{d\mathbb{Q}}{d\mathbb{P}}$ is a random variable $Z_T$ that tells us how to convert $\mathbb{P}$-expectations into $\mathbb{Q}$-expectations:

$$
\mathbb{E}^{\mathbb{Q}}[X] = \mathbb{E}^{\mathbb{P}}[Z_T \cdot X]
$$

for any random variable $X$. The measure $\mathbb{Q}$ is "equivalent" to $\mathbb{P}$ if $Z_T > 0$ almost surely — every scenario that was possible under $\mathbb{P}$ is still possible under $\mathbb{Q}$, just with a different probability weight.

For the GBM case, the Radon-Nikodym derivative is:

$$
Z_T = \frac{d\mathbb{Q}}{d\mathbb{P}}\bigg|_{\mathcal{F}_T} = \exp\!\left(-\lambda\,W_T^{\mathbb{P}} - \frac{1}{2}\lambda^2 T\right)
$$

where $\lambda = (\mu - r)/\sigma$ is the market price of risk (the Sharpe ratio). This is a positive random variable (so $\mathbb{Q} \sim \mathbb{P}$), and its $\mathbb{P}$-expectation is 1 (so $\mathbb{Q}$ is a valid probability measure).

**Intuition:** Paths where the Brownian motion moves upward (positive $W_T^{\mathbb{P}}$) get downweighted by $Z_T$ (because $-\lambda W_T^{\mathbb{P}}$ is negative when $\lambda > 0$ and $W_T > 0$). This is precisely what makes the stock's expected return drop from $\mu$ to $r$ — upward-moving paths are given less weight, reducing the average drift.

#### Girsanov's theorem: the drift-shift identity

**Girsanov's theorem** makes the reweighting concrete. If $W_t^{\mathbb{P}}$ is a Brownian motion under $\mathbb{P}$, define:

$$
W_t^{\mathbb{Q}} = W_t^{\mathbb{P}} + \int_0^t \lambda_s\,ds
$$

Then $W_t^{\mathbb{Q}}$ is a Brownian motion under $\mathbb{Q}$ — its increments are $\mathcal{N}(0, \Delta t)$ under the new measure. The theorem's power is this: **changing the measure is equivalent to shifting the drift of the Brownian motion**.

To see what this does to the stock SDE, substitute $dW_t^{\mathbb{P}} = dW_t^{\mathbb{Q}} - \lambda\,dt$:

$$
dS_t = \mu\,S_t\,dt + \sigma\,S_t\,(dW_t^{\mathbb{Q}} - \lambda\,dt) = (\mu - \sigma\lambda)\,S_t\,dt + \sigma\,S_t\,dW_t^{\mathbb{Q}}
$$

With $\lambda = (\mu - r)/\sigma$, the drift becomes $\mu - \sigma \cdot (\mu - r)/\sigma = r$:

$$
dS_t = r\,S_t\,dt + \sigma\,S_t\,dW_t^{\mathbb{Q}}
$$

The **volatility is unchanged**. Only the drift shifts. This is fundamental: Girsanov changes means but not variances. The diffusion coefficient $\sigma$ is the same under both measures. This is why implied volatility is a meaningful concept — the volatility parameter is measure-invariant.

#### What this means computationally: same random numbers, different drift

Here is the key insight for implementation. Consider simulating 10,000 paths of GBM. In both cases, we draw the **same** Gaussian random numbers $Z_1, Z_2, \ldots, Z_n \sim \mathcal{N}(0,1)$.

**Under $\mathbb{P}$ (physical measure):**

$$
S_{t+\Delta t} = S_t \exp\!\left[(\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]
$$

**Under $\mathbb{Q}$ (risk-neutral measure):**

$$
S_{t+\Delta t} = S_t \exp\!\left[(r - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]
$$

The $Z$ values are identical — they come from the same `jax.random.normal` call or the same `diffrax.VirtualBrownianTree`. The **only difference** is what sits in front of $\Delta t$: $\mu$ vs. $r$. Changing the measure doesn't change the noise; it changes the deterministic drift applied to the noise.

This is exactly what happens in VALAX. In `valax/pricing/mc/paths.py`, the GBM drift is constructed as:

```python
mu = model.rate - model.dividend  # r - q, not the real-world mu
drift = GBMDrift(mu=mu)
```

The `diffrax.VirtualBrownianTree` generates pure Brownian increments — $\mathcal{N}(0, \Delta t)$ samples that are measure-agnostic. The **choice of measure** is encoded entirely in the drift function. If you replaced `model.rate - model.dividend` with a real-world expected return $\mu$, you'd be simulating under $\mathbb{P}$ instead of $\mathbb{Q}$ — using the exact same random number generator.

Similarly, in `valax/pricing/mc/multi_asset_paths.py`, the log-space drift:

```python
log_drift = (model.rate - model.dividends - 0.5 * model.vols**2) * dt
```

encodes the risk-neutral measure through `model.rate`. The correlated Gaussian draws `Z = jax.random.normal(subkey, shape=(n_steps, n_assets))` are pure noise — they don't know which measure they live in.

#### Why this matters: three concrete consequences

**1. Pricing uses $\mathbb{Q}$-drift, never $\mathbb{P}$-drift.** The entire VALAX codebase specifies SDEs under $\mathbb{Q}$. The `BlackScholesModel` stores `rate` (the risk-free rate) and `dividend`, not the real-world expected return. This is not a modeling choice — it is forced by no-arbitrage. The price of a derivative is $e^{-rT}\mathbb{E}^{\mathbb{Q}}[V_T]$, and $\mathbb{Q}$ is the measure under which the drift is $r - q$.

**2. Volatility is the same under both measures.** When a trader quotes "30% implied vol," that number is the same whether you're thinking under $\mathbb{P}$ or $\mathbb{Q}$. Girsanov only shifts drift — the diffusion coefficient (and hence the quadratic variation of the process) is invariant. This is why calibrating model volatilities to option prices (a $\mathbb{Q}$-world activity) gives parameters that are also meaningful for risk management (a $\mathbb{P}$-world activity, at least for the vol component).

**3. The market price of risk $\lambda$ is unobservable.** We never need to know $\mu$ (the real-world drift) to price derivatives. This is a feature, not a bug — estimating $\mu$ from historical data is extremely imprecise (you'd need decades of data), but option prices depend only on $\sigma$, $r$, $q$, and $T$, all of which are observable. Girsanov tells us that the entire unobservable risk premium is absorbed into the drift shift and cancels out of the pricing formula.

#### Change of numéraire: a second application of Girsanov

Girsanov's theorem is not only used for the $\mathbb{P} \to \mathbb{Q}$ switch. It also underlies **change of numéraire** — switching from one risk-neutral measure to another, each defined by a different numéraire (the asset used as the unit of account).

The general principle: if $N_t$ is a tradeable asset (the numéraire), then $V_t / N_t$ is a martingale under the measure $\mathbb{Q}^N$ associated with $N$. Switching numéraire from $N^{(1)}$ to $N^{(2)}$ is a Girsanov drift shift with:

$$
\lambda^{(1 \to 2)}_t = \frac{\sigma_{N^{(2)}}(t) - \sigma_{N^{(1)}}(t)}{\text{(volatility structure)}}
$$

Concrete examples in VALAX:

| Measure | Numéraire | Who uses it | Where in VALAX |
|---------|-----------|-------------|----------------|
| Risk-neutral $\mathbb{Q}$ | Money-market account $B_t = e^{\int_0^t r_s\,ds}$ | Black-Scholes, Heston | `GBMDrift(mu=r-q)` |
| $T$-forward measure $\mathbb{Q}^T$ | Zero-coupon bond $P(t,T)$ | Black-76, caplets | `SABRDrift` (zero drift — forward is a martingale) |
| $S_2$-measure | Asset $S_2$ | Margrabe exchange option | `margrabe_price` in `spread.py` |
| Spot LIBOR measure | Rolling bank account | LMM | `LMMDrift` with spot-measure correction |

In Black-76 (`valax/pricing/analytic/black76.py`), the forward price $F_t$ has **zero drift** under the $T$-forward measure — this is why the Black-76 SDE is $dF_t = \sigma F_t\,dW_t^T$ with no drift term. The drift didn't vanish — it was absorbed by the Girsanov shift from the money-market numéraire to the $T$-bond numéraire. The SABR model (`valax/models/sabr.py`) exploits this same fact: `SABRDrift.__call__` returns `jnp.array([0.0, 0.0])` because under the forward measure, the forward rate is driftless.

In the LMM (`valax/models/lmm.py`), each forward rate $F_i$ is a martingale under its own $T_{i+1}$-forward measure, but since we simulate all forwards under a single **spot LIBOR measure**, Girsanov introduces the no-arbitrage drift correction:

$$
\mu_i(t) = \sigma_i(t) \cdot \sum_{j=\eta(t)}^{i} \frac{\delta_j F_j(t)}{1 + \delta_j F_j(t)} \sigma_j(t)
$$

This drift is the price paid for simulating under a "wrong" measure — Girsanov tells us exactly what correction to apply, and `LMMDrift` computes it at every time step.

#### Importance sampling: Girsanov as a variance reduction technique

There is a computational application of Girsanov beyond pricing theory: **importance sampling**. The idea is to simulate under a *tilted* measure $\widetilde{\mathbb{Q}}$ that concentrates paths in the region where the payoff is large, then correct via the Radon-Nikodym derivative:

$$
\mathbb{E}^{\mathbb{Q}}[e^{-rT} V_T] = \mathbb{E}^{\widetilde{\mathbb{Q}}}\!\left[\frac{d\mathbb{Q}}{d\widetilde{\mathbb{Q}}}\,e^{-rT} V_T\right]
$$

Concretely, if you shift the drift by some $\tilde{\lambda}$, you simulate:

$$
S_{t+\Delta t} = S_t \exp\!\left[(r - q + \sigma\tilde{\lambda} - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]
$$

and multiply each payoff by the likelihood ratio:

$$
L = \exp\!\left(-\tilde{\lambda}\sum_i Z_i\sqrt{\Delta t} - \frac{1}{2}\tilde{\lambda}^2 T\right)
$$

For deep OTM options, choosing $\tilde{\lambda}$ to center the simulation around the strike can reduce MC variance by orders of magnitude. The Gaussian draws $Z_i$ are unchanged — Girsanov again operates purely through the drift. VALAX does not currently implement importance sampling, but the architecture supports it: one would only need to modify the drift in `GBMDrift` and apply the likelihood ratio weight in the MC engine's payoff averaging. The `diffrax.VirtualBrownianTree` and the payoff functions would require no changes.

#### Summary: the Girsanov hierarchy in VALAX

```
Physical measure P                 Risk-neutral measure Q
   drift = μ                          drift = r - q
   dW^P ~ N(0,dt)                     dW^Q ~ N(0,dt)
       │                                  │
       │  Girsanov: shift by λ=(μ-r)/σ    │
       └──────────────────────────────────→┘
                                           │
                    ┌──────────────────────┬┘
                    │                      │
            T-forward measure Q^T    Spot LIBOR measure
              drift = 0 (forwards     drift = Σ correction
              are martingales)         (LMM no-arb drift)
              Used by: Black-76,       Used by: LMM
              SABR, caplets            (valax/models/lmm.py)
```

Every arrow is a Girsanov drift shift. Every node uses the same Brownian noise — the same `jax.random` keys, the same `VirtualBrownianTree`. The **only** thing that changes is the coefficient in front of $dt$ in the SDE.

### 1.4 The Feynman-Kac Theorem

The **Feynman-Kac theorem** connects the risk-neutral expectation to a PDE:

If $V(t, S) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-r(T-t)} g(S_T) \mid S_t = S\right]$, then $V$ satisfies:

$$
\frac{\partial V}{\partial t} + (r - q)S\frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0
$$

with terminal condition $V(T, S) = g(S)$.

This is why Monte Carlo and PDE methods give the same answer: they are computing the same object (the risk-neutral expectation) via different routes. MC samples paths and averages the discounted payoff. PDE solves the differential equation backward from the terminal condition.

**In VALAX:** `valax/pricing/mc/engine.py` computes $\mathbb{E}^{\mathbb{Q}}[e^{-rT}g(S_T)]$ by simulation. `valax/pricing/pde/solvers.py` solves the Feynman-Kac PDE via Crank-Nicolson. For Black-Scholes models with European payoffs, both converge to the closed-form answer in `valax/pricing/analytic/black_scholes.py`. The QuantLib comparison tests (`tests/test_quantlib_comparison/`) verify this convergence.
