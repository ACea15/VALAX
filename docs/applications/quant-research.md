# Quant Research

*How VALAX serves front-office quants, structuring desks, model R&D teams, and academic quant-finance researchers — where "differentiable everything" stops being a talking point and starts being an unfair advantage in iteration speed.*

**Quant research may be VALAX's strongest *technical* fit of any audience.** Where [Market Risk & Model Validation](market-risk.md) is the strongest **commercial** case (regulator-mandated need, no incumbent lock-in), quant research is the strongest **technical** one: it is the audience where *every* JAX-native choice — autodiff, JIT, `vmap`, `equinox`, composability with the ML stack — compounds into a research-productivity moat that vendor C++ analytics libraries structurally cannot match.

---

## 1. The summary — differentiable pricing as a categorical research advantage

This section is the whole argument on one screen; the rest of the page is the worked detail. For quant research the JAX-native architecture is not a "nice implementation choice" — it is the primary source of research-productivity leverage, and it compounds:

| Design choice | Research payoff |
|---|---|
| **Autodiff Greeks** | Every new pricer ships with all Greeks. New model → new sensitivities → next model. No numerical noise, no bump-width tuning, no maintenance tax. |
| **Composable pure functions** | A new payoff is a Python function, not a class subtree. Iteration speed is Python speed. |
| **`vmap` everywhere** | Parameter sweeps, MC experiments, calibration-robustness studies — one line each, GPU-accelerated. |
| **`equinox.Module` pytrees** | Models, networks, and pricers share the same container. Pass a pricer to a network trainer, no serialisation. |
| **JAX ecosystem integration** | `optax`, `flax`, `blackjax`, `jaxopt` all work out of the box. Neural surrogate today, MCMC calibration tomorrow, deep hedging next week. |
| **JIT + XLA** | Prototype runs at C++ speed with no C++ code. Deployment is a `pip install`. |
| **Pure functions + explicit PRNG** | Bit-identical reproducibility across machines and years. Submissions and validators both trust the outputs. |

The compounding effect: **every model you have already implemented becomes a differentiable primitive for the next one.** A team of five researchers on VALAX outpaces a team of twenty on a C++ analytics library — not because JAX is magic, but because the tax of maintaining bump-and-reprice / serialisation-boundary / class-hierarchy infrastructure disappears entirely.

And the gap is **categorical, not incremental**:

- A bank MRM team can *do their job* on vendor risk tooling, however painfully. VALAX makes it better and cheaper, but the job is doable elsewhere.
- A quant research team **cannot** do neural surrogate pricing, deep hedging, or differentiable-SDE calibration on a C++ analytics library *at all* — those directions are structurally out of reach until the stack becomes differentiable. VALAX is not "better" here; it is *possible* here where the incumbent is not.

Strategically this means a bank should adopt VALAX in **Quant Research and Market Risk simultaneously** — different budgets, different buyers — and let research prototypes graduate into validated production without a rewrite, because it is the same library.

---

## 2. The problem quant research has today

Every quant researcher — sell-side derivatives R&D, buy-side systematic desk, hedge-fund model team, academic — hits the same wall inside three months of any project:

1. **The production pricer is a C++ black box.** You want to prototype a new payoff, a new model calibration, a new hedge scheme, but the incumbent pricer (Murex, Numerix, in-house C++) requires a full analytics-team ticket to modify. Two months of internal negotiation later you have a stub, not a prototype.
2. **Greeks come from bump-and-reprice.** Every new model needs custom finite-difference infrastructure for every Greek, tuned per-parameter to avoid numerical noise. Every re-parameterisation invalidates the bump-width tables.
3. **Calibration is a black box tied to the pricer.** You cannot try a new loss function, regulariser, or optimiser without rewriting the calibration inner loop. Researchers keep re-implementing Levenberg-Marquardt on the side.
4. **The pricer and the ML stack are two universes.** You want a neural surrogate, deep hedging on a novel payoff, or a neural SDE — but the pricer is C++, the ML stack is PyTorch/JAX, and the bindings are the graveyard where research projects die.
5. **Reproducibility is a wish, not a property.** Six months after a paper is written, the library has moved three versions and the seed is lost. Peer review and internal governance both suffer.

**Every one of these is a byproduct of the same architectural decision — building the pricer in a non-differentiable language with a rigid class hierarchy.** VALAX makes the opposite decision, and every benefit below flows from it.

---

## 3. What "differentiable everything" actually unlocks

The elevator pitch: **if your pricer is a JAX function, every Greek, every calibration, every sensitivity, every neural surrogate is one `jax.grad` call away — for any pricer you have ever written or ever will write, without additional code.**

### 3.1 A new pricer ships with all Greeks, immediately

```python
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

def my_new_exotic_payoff(
    paths: Float[Array, "n_paths n_steps"],
    strike: Float[Array, ""],
    barrier: Float[Array, ""],
) -> Float[Array, ""]:
    # ... whatever novel payoff structure you dreamed up
    return jnp.mean(discounted_payoff)
```

You now have — for free, with no additional code:

- **Delta:** `jax.grad(my_new_exotic_payoff, argnums=0)(paths, strike, barrier)`
- **Vega, rho, dividend sensitivity:** grad against whichever pricer inputs your pricer consumes.
- **Full Hessian for gamma, vanna, volga, cross-gamma:** `jax.hessian(...)`.
- **Batched delta across 10 000 strikes:** `jax.vmap(jax.grad(...))`.
- **JIT-compiled fast execution:** `@jax.jit` or `@eqx.filter_jit`.

In a C++ analytics library each of those five capabilities is a separate multi-week engineering ticket. Here they are one line each.

### 3.2 Calibration is an `optimistix` or `optax` call, not a subsystem

```python
import optimistix as optx

def calibration_residual(params, args):
    market_prices, market_data = args
    model_prices = jax.vmap(price_under_model)(instruments, params, market_data)
    return model_prices - market_prices  # residual vector

solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
solution = optx.least_squares(calibration_residual, solver, initial_params,
                              args=(market_prices, market_data))
```

No hand-written LM, no numerical Jacobian — `optimistix` gets the exact Jacobian by autodiff-differentiating the residual through your pricer. Want a different optimiser? Swap `LevenbergMarquardt` for `BFGS`, or `optax.adam` for a non-convex loss. The pricer does not change. Shipped calibrations demonstrate the pattern: [Heston](../guide/calibration.md), [SABR](../guide/vol-surfaces.md), [SLV](../guide/slv.md), [Hull-White tree](../api/models.md).

### 3.3 Rapid paper-to-calibrated-prototype pipeline

The loop from "I read an interesting paper this morning" to "I have a calibrated, GPU-accelerated implementation with Greeks" is measured in **hours, not months**:

1. **Write the SDE as a `diffrax` term** (~50 lines): drift, diffusion, correlation.
2. **Wrap the MC pricer as a pure function** (~30 lines): `V(instrument, params, market) → price`, `@eqx.filter_jit`.
3. **Calibrate with `optimistix` / `optax`** (~20 lines) — autodiff Jacobian.
4. **All Greeks from `jax.grad`; `vmap` for parameter sweeps** — no extra code.
5. **QuantLib comparison** for validation — pattern in `tests/test_quantlib_comparison/`.

Total: a calibrated, GPU-accelerated implementation of a novel model in **one to three days**. In a C++ analytics shop the same project is a Q3 planning item.

---

## 4. Composable payoffs — the design payoff for structuring

Structuring desks live and die by prototyping new payoffs on tight deal timelines. The traditional C++ path — subclass `Instrument`, override `pricingImpl`, register with the factory, rebuild, redeploy — is weeks per payoff. VALAX treats a payoff as **just a function**:

```python
def cliquet_payoff(
    paths: Float[Array, "n_paths n_steps"],
    local_cap: Float[Array, ""],
    local_floor: Float[Array, ""],
    global_floor: Float[Array, ""],
) -> Float[Array, ""]:
    """Locally-capped, globally-floored cliquet with monthly resets."""
    monthly_returns = jnp.diff(jnp.log(paths), axis=1)
    capped_floored = jnp.clip(monthly_returns, local_floor, local_cap)
    cumulative = jnp.sum(capped_floored, axis=1)
    return jnp.mean(jnp.maximum(cumulative, global_floor))
```

That is the pricer. Full Greeks are `jax.grad(cliquet_payoff, ...)`; calibration is `optimistix.least_squares` on the residual; `vmap` gives the price surface across strikes. The pattern generalises — barriers, Asians, quantos, autocallables, snowballs, target-redemption forwards are all Python functions in the same style. See [`valax/pricing/mc/payoffs.py`](../api/pricing.md) and the [Equity Exotics guide](../guide/equity-exotics.md). **Time from "trader asks for a new payoff" to "structured deal ready to hedge" collapses from weeks to hours.**

---

## 5. Neural surrogate pricers

A neural surrogate is a small network trained to approximate an expensive pricer, then dropped into inner loops that evaluate it repeatedly — XVA, real-time risk, portfolio optimisation, deep hedging. The traditional pipeline is painful: train in PyTorch, ship weights to C++, write ONNX bindings, hope precision matches. The VALAX pipeline is one file:

```python
import equinox as eqx
import optax

class SurrogateHestonPricer(eqx.Module):
    layers: list

    def __init__(self, key, in_dim=6, hidden=64, out_dim=1):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, hidden, key=keys[2]),
            eqx.nn.Linear(hidden, out_dim, key=keys[3]),
        ]

    def __call__(self, features):
        x = features
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)

# Ground truth: the existing VALAX Heston pricer
train_features = sample_feature_space(key, n_samples=100_000)
train_targets  = jax.vmap(lambda f: heston_price_cos(*unpack(f)))(train_features)

model = SurrogateHestonPricer(jax.random.PRNGKey(0))

@eqx.filter_jit
def loss(model, features, targets):
    preds = jax.vmap(model)(features)
    return jnp.mean((preds - targets) ** 2)
# ... standard optax training loop
```

The trained surrogate has all the Greeks (`jax.grad(model)`), `vmap`s for batch pricing, and lives in the same array framework as the teacher — so you can train it with a **pricer-gradient-matching loss** (reproduce the teacher's *delta and gamma*, not just its price), which is state-of-the-art surrogate training and essentially impossible across a C++/Python boundary. See [Vision § "Neural surrogates"](../vision.md).

---

## 6. Deep hedging — the flagship differentiable-finance application

Deep hedging (Buehler, Gonon, Teichmann, Wood 2018) trains a network to hedge a derivative under transaction costs and frictions where classical Δ-hedging breaks down. The training signal is the *gradient of hedged P&L through the SDE simulation*, which exists cleanly only if the path simulation, the payoff, and the network are all one differentiable graph — **all three natively met by VALAX**:

```python
paths = generate_heston_paths(key, model_params, path_config)   # (n_paths, n_steps)
hedging_policy = HedgingNetwork(key_net, in_dim=state_features, hidden=64)

def hedged_pnl(policy, paths, market):
    hedges = jax.vmap(policy)(state_features_from_paths(paths))
    pnl = accumulate_hedged_pnl(paths, hedges, transaction_cost, market)
    return -jnp.mean(cvar(pnl, alpha=0.05))   # loss = expected shortfall of P&L

grad_loss = eqx.filter_grad(hedged_pnl)      # gradient flows through paths, payoff, network
# ... optax training loop
```

Not a hypothetical: VALAX ships the exact primitives — [`diffrax`](../guide/monte-carlo.md) for SDE simulation, [`equinox`](../architecture/jax-patterns.md) for the network, [`optax`](../guide/calibration.md) for training — so a deep-hedging prototype needs essentially no infrastructure work. This is also the research root of the [trading-desk decision loop](trading-desk.md#2-the-actual-novelty-differentiability-that-crosses-into-the-decision-layer).

---

## 7. `vmap` parameter sweeps and Monte Carlo experiments

Quant research is an **experimental** discipline: sweep a parameter, plot the surface, look for anomalies. In a C++ library this is a driver script marshalling results into a DataFrame. In VALAX it is one `vmap`:

```python
# Price surface across 100 strikes × 50 expiries × 20 vols = 100 000 prices
surface = jax.vmap(jax.vmap(jax.vmap(price_fn,
                                     in_axes=(None, None, 0)),  # vary vol
                            in_axes=(None, 0, None)),           # vary expiry
                   in_axes=(0, None, None))(strikes, expiries, vols)
# surface.shape == (100, 50, 20)
```

One JIT-compiled kernel, GPU speed, no Python loop. The same pattern gives MC convergence studies (`vmap` over seed / path count), sensitivity landscapes (`vmap` `jax.grad` over a grid), calibration-robustness sweeps (`vmap` over perturbed market data), and rolling-window backtests. Experiments that were "a scheduler and 40 CPU hours" become "let me try it before lunch".

---

## 8. The JAX ecosystem is a research ecosystem

VALAX's dependencies are the state-of-the-art scientific-computing stack for JAX, and quant research uses each directly:

| Package | What it gives the researcher |
|---|---|
| [`diffrax`](https://github.com/patrick-kidger/diffrax) | Differentiable SDE / ODE solvers — any drift/diffusion, differentiable path simulation |
| [`equinox`](https://github.com/patrick-kidger/equinox) | Pytree dataclasses for models and networks; `filter_jit` / `filter_grad` |
| [`optimistix`](https://github.com/patrick-kidger/optimistix) | Root-finding, least-squares (LM), nonlinear solvers — the calibration backbone |
| [`optax`](https://github.com/google-deepmind/optax) | Gradient-based optimisers — the neural-surrogate and deep-hedging backbone |
| [`lineax`](https://github.com/patrick-kidger/lineax) | Structured linear solvers — PDE inner loop, covariance factorisations |
| [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) | Runtime shape/dtype checks — self-documenting research code |

The same stack powers cutting-edge scientific ML research: pull in `flax` for larger networks, `blackjax` for MCMC calibration, `jaxopt` for constrained optimisation — zero framework impedance. **One environment, one `pip install`, one mental model** — a pricing library that is also a neural-SDE library, a differentiable-optimisation library, and a bridge to the ML research world.

---

## 9. Publication and reproducibility

Peer-reviewed research and internal model-governance both live on reproducibility, which VALAX makes a *property*, not a wish:

- **Pure functions.** `V(instrument, market) → price` with no mutable state → bit-identical outputs across machines and years.
- **Integer-ordinal dates.** No `datetime` timezone / DST ambiguity.
- **Explicit PRNG keys.** Every stochastic op takes a `jax.random.PRNGKey` — no hidden global RNG. Seed once, replay forever.
- **Pinned dependency versions.** A git tag is a full reproducibility contract.
- **Golden tests.** `tests/golden/` ships reference outputs so a refactor cannot silently drift.

For academics, submissions ship with a working git tag rather than "code available on request". For internal governance, the audit trail from paper → model → production is a git log, not a folder of spreadsheets.

---

## 10. Coverage today vs. roadmap

Most of what quant research needs is already in the library — research productivity was a founding design goal:

| Research need | Status | Component(s) |
|---|---|---|
| Differentiable pricer signature | ✅ | Every `valax/pricing/*` function |
| Autodiff Greeks (any order) | ✅ | `jax.grad` / `jax.hessian` / `greeks` wrappers |
| SDE path simulation (differentiable) | ✅ | `diffrax` — GBM, Heston, LMM, SABR, local vol, SLV |
| Gradient-based calibration | ✅ | `optimistix` LM + `optax` — Heston, SABR, SLV shipped |
| Neural network container | ✅ | `equinox.Module` |
| `vmap` for parameter sweeps | ✅ | `jax.vmap` everywhere |
| GPU / TPU support | ✅ | JAX / XLA — same code, no changes |
| QuantLib comparison for validation | ✅ | `tests/test_quantlib_comparison/` (14 modules) |
| Golden tests for reproducibility | ✅ | `tests/golden/` with `golden_manifest.json` |
| Vol surface models (SABR, SVI, SLV, Dupire) | ✅ | `valax/surfaces/`, `valax/models/slv.py`, `valax/models/local_vol.py` |
| Neural surrogate example (worked end-to-end) | 📋 | Pattern documented; example notebook planned |
| Deep hedging example (worked end-to-end) | 📋 | All primitives shipped; canonical example planned |
| Differentiable portfolio construction | 📋 | Roadmap — the P5 frontier item in [Vision](../vision.md) |
| Rough volatility / neural SDE calibration | 📋 | `diffrax` supports the paths; canonical example planned |

The pattern: **the infrastructure is complete; the canonical worked examples are the roadmap.** For a research team this is close to ideal — you're not building tooling, you're doing research.

---

## 11. The pitch, tailored to the buyer

**To the Head of Quant Research / Chief Quant Officer:**
*Every model your team writes ships with autodiff Greeks, gradient-based calibration, GPU acceleration, and integration with the ML stack — for zero additional infrastructure. Time from paper to calibrated prototype collapses from months to days. Deep hedging and neural surrogates go from "future work" to "next sprint".*

**To the CTO / Head of Analytics:**
*One library serves the pricing kernel, the calibration engine, the risk engine, the surrogate trainer, and the SDE research bench. One environment, one dependency graph, one deployment story. Every research prototype is one refactor away from graduating to production because the language and the framework do not change.*

**To an academic partner / PhD student:**
*Publish with a `pip install` and a git tag. Every result is bit-reproducible. Every model composes with `diffrax`, `optax`, `flax`, `blackjax`. The paper-to-code delta is minutes, not weeks. Reviewers can run your notebook.*

**To a hedge-fund CIO / systematic PM:**
*The same differentiable pricer that computes your fair value gives you the gradient of P&L with respect to every parameter of the strategy — feed it into `optax` and the strategy tunes itself. Not a black-box RL loop; the actual gradient of the actual P&L.*

---

## 12. Where to read next

- **The applications overview and audience ranking** → [Applications](index.md).
- **The commercial-adoption companion** → [Market Risk & Model Validation](market-risk.md).
- **The front-office decision-loop companion** → [Front-Office Trading](trading-desk.md).
- **The mathematical foundations of every model in the library** → [Models & Theory](../theory/index.md).
- **Calibration patterns and shipped examples** → [Calibration](../guide/calibration.md), [Vol Surfaces](../guide/vol-surfaces.md), [SLV](../guide/slv.md).
- **SDE path simulation and Monte Carlo** → [Monte Carlo](../guide/monte-carlo.md).
- **The JAX design patterns behind the library** → [JAX Patterns](../architecture/jax-patterns.md).
- **The forward-looking vision — neural surrogates, deep hedging, differentiable portfolios** → [Vision](../vision.md).
