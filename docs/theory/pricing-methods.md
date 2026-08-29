[← Models & Theory index](index.md)

# 5. Pricing Methods

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

**Rannacher start-up:** Crank-Nicolson can produce spurious oscillations near a payoff discontinuity (digital) or kink (vanilla at the strike), which contaminate the second-order Greeks $\Gamma$ and vega. The remedy is **Rannacher start-up**: replace the first 2–4 CN steps with fully-implicit (backward-Euler, $\theta = 1$) steps. Backward Euler is strongly damping — it annihilates the high-frequency modes that CN merely propagates — after which CN resumes to recover second-order accuracy. This is essential for digitals (whose terminal data is a step) and barriers, and cheap insurance for vanillas.

#### Early exercise as a free-boundary problem

American and Bermudan options turn the PDE into a **free-boundary problem**: at each point the holder takes the larger of continuing and exercising, so the value satisfies the linear complementarity conditions

$$
V \ge g, \qquad \frac{\partial V}{\partial t} + \mathcal{L}V - rV \le 0, \qquad (V - g)\left(\frac{\partial V}{\partial t} + \mathcal{L}V - rV\right) = 0,
$$

where $g$ is the exercise (intrinsic) payoff. VALAX handles the two exercise styles with two projections, both chosen to keep the solve differentiable so Greeks flow from `jax.grad` (per `AGENTS.md`, never finite differences):

- **Penalty method (continuous American).** Add a forcing term $\rho\,\max(g - V, 0)$ to the discretised step that penalises any violation of $V \ge g$; solving the penalised system with a small **fixed** number of iterations per step pushes the solution onto the free boundary. A fixed iteration count of smooth operations differentiates cleanly.
- **Explicit projection (discrete-date Bermudan / callable / puttable).** Apply $V \leftarrow \max(V_{\text{cont}}, g)$ (holder-optimal) or $V \leftarrow \min(V_{\text{cont}}, \text{call price})$ (issuer-optimal) **only** at the exercise dates, which are snapped to specific time-steps via $\text{round}(t/\Delta t)$. This is exactly the projection the Hull-White trinomial tree already uses for callable/puttable bonds (§2.8).

The alternative, **PSOR** (projected successive over-relaxation), is rejected in VALAX: its data-dependent iteration count and non-smooth updates are awkward to differentiate under JAX.

#### Multi-dimensional PDEs and ADI

Models with two state variables — Heston $(\ln S, v)$, stochastic-local vol, or two correlated assets $(\ln S_1, \ln S_2)$ — give a 2-D backward PDE. For Heston, Itô on $V(t, \ln S, v)$ under the risk-neutral dynamics of §2.4 yields

$$
\frac{\partial V}{\partial t} + \tfrac{1}{2} v\,\frac{\partial^2 V}{\partial x^2} + \rho\xi v\,\frac{\partial^2 V}{\partial x\,\partial v} + \tfrac{1}{2}\xi^2 v\,\frac{\partial^2 V}{\partial v^2} + \left(r - q - \tfrac{1}{2}v\right)\frac{\partial V}{\partial x} + \kappa(\theta - v)\frac{\partial V}{\partial v} - rV = 0 ,
$$

with $x = \ln S$. The key difficulty is the **mixed derivative** $\partial_{x v}$ coming from the spot-vol correlation $\rho$ — it couples the two spatial directions, so a fully-implicit step would require inverting a large block-banded operator.

VALAX instead uses **Alternating-Direction Implicit (ADI)** operator splitting. Write the spatial operator as $\mathcal{L} = \mathcal{A}_0 + \mathcal{A}_1 + \mathcal{A}_2$, where $\mathcal{A}_1$ acts only along $x$, $\mathcal{A}_2$ only along $v$, and $\mathcal{A}_0$ is the mixed-derivative term. Each time step is then a sequence of **1-D tridiagonal solves per axis** with the cross term $\mathcal{A}_0$ treated **explicitly**. Three schemes trade cost for accuracy:

| Scheme | Cross-term order | Best for |
|--------|------------------|----------|
| **Douglas** | first | robust default, weak correlation |
| **Craig–Sneyd** | second | accuracy default (adds a correction stage) |
| **Hundsdorfer–Verwer** | second | strongly correlated / convection-dominated (large $\lvert\rho\rvert$) |

The Douglas scheme, for a step from $V^{n}$ to $V^{n+1}$ over $\Delta t$ with parameter $\theta$, reads schematically

$$
\begin{aligned}
Y_0 &= V^n + \Delta t\,\mathcal{L}V^n, \\
(I - \theta\Delta t\,\mathcal{A}_j)\,Y_j &= Y_{j-1} - \theta\Delta t\,\mathcal{A}_j V^n, \quad j = 1, 2, \\
V^{n+1} &= Y_2 ,
\end{aligned}
$$

so the only implicit solves are the two per-axis tridiagonal systems $(I - \theta\Delta t\,\mathcal{A}_j)$ — precisely the machinery already built for the 1-D solver. Craig–Sneyd and HV append one or two further explicit stages that reintroduce the mixed term to second order. ADI is unconditionally stable and is the industry standard for Heston and basket finite differences; VALAX chooses it over a sparse block solve specifically because it reuses the existing `lineax` tridiagonal solver per dimension.

#### Local-volatility PDE

Under Dupire local volatility (§4.4) the same 1-D Black-Scholes PDE holds but with a **state- and time-dependent** diffusion coefficient:

$$
\frac{\partial V}{\partial t} + \left(r - q - \tfrac{1}{2}\sigma_{\text{loc}}^2(x, t)\right)\frac{\partial V}{\partial x} + \tfrac{1}{2}\sigma_{\text{loc}}^2(x, t)\,\frac{\partial^2 V}{\partial x^2} - rV = 0 .
$$

The operator coefficients are no longer constant along the grid: at each node $(x_i, t_n)$ VALAX evaluates $\sigma_{\text{loc}} = \sigma_{\text{Dupire}}(k_i, t_n)$ via `dupire_local_vol` (with $k_i = x_i - \ln F(t_n)$), vectorised with `jax.vmap`. This is why the operator layer accepts callable coefficients rather than constant bands.

#### Boundary conditions

The finite domain requires boundary conditions on the truncated edges. VALAX uses three families: **Dirichlet** (a fixed value — barrier levels, and deep-ITM/OTM Black-Scholes asymptotics), **Neumann** (a fixed first derivative — value linear in $S$ at the far field), and a **linearity / PDE-at-boundary** condition ($\partial_{xx}V = 0$, the robust choice for the variance axis in Heston where no simple asymptotic is available). Continuously-monitored barriers are imposed as an **absorbing Dirichlet boundary** at the barrier level rather than only through the terminal payoff.

**When to use:** low-dimensional problems (one underlying, or one spot plus one variance factor) where you want the full solution surface, and especially early-exercise, barrier, and digital features where a backward grid with an absorbing boundary and Rannacher start-up beats MC and CRR. Faster than MC for $d \le 2$; the curse of dimensionality makes it impractical for $d > 3$, where Monte Carlo takes over.

**VALAX implementation:** the shipping solver is `valax/pricing/pde/solvers.py` (`pde_price`) — 1-D CN, `jax.lax.scan` backward loop, `lineax` tridiagonal solve, Black-Scholes asymptotic boundaries. The multivariate subsystem (2-D ADI, penalty/projection early exercise, model×instrument dispatch) is specified in the [PDE design doc](../architecture/pde-design.md) and delivered in phases; see the [PDE guide](../guide/pde.md) for the target coverage matrix.

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

VALAX uses diffrax's Euler-Maruyama for the linear-coefficient GBM and SABR SDEs (where Euler is unbiased), and a bespoke `jax.lax.scan` implementation of Andersen's (2008) Quadratic-Exponential scheme for the Heston variance process (where naïve Euler-with-reflection acquires bias at the absorbing boundary). See §2.4 for details on the QE algorithm.

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
