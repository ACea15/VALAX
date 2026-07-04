# Quant Research

*How VALAX serves front-office quants, structuring desks, model R&D teams, and academic quant-finance researchers — where "differentiable everything" stops being a talking point and starts being an unfair advantage in iteration speed.*

This page argues something the [Applications overview](index.md) hints at: **quant research may be VALAX's strongest technical fit of any audience** — stronger even than the middle-office adoption case, once you take the "who benefits how much per user" view rather than the "how many users" view. The reason is simple: quant research is the one audience where *every* JAX-native architectural choice — autodiff, JIT, `vmap`, `equinox`, composability with the ML stack — compounds into a genuine research-productivity moat that vendor C++ analytics libraries structurally cannot match.

Where [Market Risk & Model Validation](market-risk.md) is the strongest **commercial** adoption case (regulator-mandated need, no incumbent lock-in), quant research is the strongest **technical** case. Both matter. This page makes the technical one.

---

## 1. The problem quant research has today

Every quant researcher — sell-side derivatives R&D, buy-side systematic desk, hedge-fund model team, academic — hits the same wall inside three months of any project:

1. **The production pricer is a C++ black box.** You want to prototype a new payoff, a new model calibration, a new hedge scheme, but the incumbent pricer (Murex, Numerix, in-house C++) requires a full analytics-team ticket to modify. Two months of internal negotiation later you have a stub, not a prototype.

2. **Greeks come from bump-and-reprice.** Every new model needs custom finite-difference infrastructure for every Greek, tuned per-parameter to avoid numerical noise. Every re-parameterisation invalidates the bump-width tables. The research team spends more time fighting numerical noise than doing research.

3. **Calibration is a black box tied to the pricer.** You cannot try a new loss function, a new regulariser, or a new gradient-based optimiser without rewriting the calibration inner loop. Researchers keep re-implementing Levenberg-Marquardt on the side.

4. **The pricer and the ML stack are two universes.** You want to train a neural surrogate, or run deep hedging on a novel payoff, or fit a neural SDE. But the pricer is in C++, the ML stack is in PyTorch or JAX, and the bindings are the graveyard where research projects die.

5. **Reproducibility is a wish, not a property.** Six months after a paper is written, the pricer library has moved three versions, the calibration random seed is lost, and the numbers cannot be reproduced. Peer review suffers; internal governance suffers.

**Every one of these problems is a byproduct of the same architectural decision — building the pricer in a non-differentiable language with a rigid class hierarchy.** VALAX makes the opposite decision, and every research-productivity benefit downstream flows from it.

---

## 2. What "differentiable everything" actually unlocks

The elevator pitch: **if your pricer is a JAX function, every Greek, every calibration, every sensitivity, every neural surrogate is one `jax.grad` call away — for any pricer you have ever written or ever will write, without additional code.**

That is a big claim. Here is what it means concretely.

### 2.1 A new pricer ships with all Greeks, immediately

Write a new payoff:

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

That is what "differentiable everything" means. In a C++ analytics library each of those five capabilities is a separate multi-week engineering ticket. Here they are one line each.

### 2.2 Calibration is a `optimistix` or `optax` call, not a subsystem

If your pricer is differentiable, calibrating any model to any target is a gradient-based optimisation call:

```python
import optimistix as optx
import jax.numpy as jnp

def calibration_residual(params, args):
    market_prices, market_data = args
    model_prices = jax.vmap(price_under_model)(instruments, params, market_data)
    return model_prices - market_prices  # residual vector

# Levenberg-Marquardt via optimistix — three lines
solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
solution = optx.least_squares(calibration_residual, solver, initial_params,
                              args=(market_prices, market_data))
calibrated_params = solution.value
```

No hand-written LM. No numerical Jacobian. `optimistix` gets the exact Jacobian by autodiff-differentiating the residual through your pricer.

Want to try a different optimiser? Swap `LevenbergMarquardt` for `BFGS`, or move to `optax.adam` if the loss surface is non-convex. The pricer does not change.

Existing shipped calibrations demonstrate this pattern: [Heston](../guide/calibration.md), [SABR](../guide/vol-surfaces.md), [SLV](../guide/slv.md), [Hull-White tree](../api/models.md). Each is a few hundred lines and each ships with an autodiff Jacobian.

### 2.3 Rapid paper-to-prototype-to-calibrated pipeline

The concrete win for a researcher: **the loop from "I read an interesting paper this morning" to "I have a calibrated, GPU-accelerated implementation with Greeks" is now measured in hours, not months.**

The pattern:

1. **Read the paper.** SDE definition, characteristic function, calibration objective.
2. **Write the SDE as a `diffrax` term** (~50 lines): drift, diffusion, correlation structure.
3. **Wrap the Monte Carlo pricer as a pure function** (~30 lines): `V(instrument, model_params, market) → price`, decorated with `@eqx.filter_jit`.
4. **Calibrate with `optimistix.LevenbergMarquardt` or `optax.adam`** (~20 lines) — the Jacobian is autodiff-derived from the pricer.
5. **All Greeks come from `jax.grad`** — no extra code.
6. **`vmap` for parameter sweeps** across strike / expiry / spot — no loop.
7. **QuantLib comparison** for validation — pattern is in `tests/test_quantlib_comparison/`.

Total: a working, calibrated, GPU-accelerated implementation of a novel model in **one to three days**. In a C++ analytics shop the same project is a Q3 planning item.

---

## 3. Composable payoffs — the design payoff for structuring

Structuring desks live and die by the ability to prototype new payoffs on tight deal timelines. The traditional C++ approach — subclass `Instrument`, override `pricingImpl`, register with the pricer factory, rebuild the analytics library, redeploy — is measured in weeks per payoff.

VALAX's approach is to treat a payoff as **just a function**:

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
    payoff = jnp.maximum(cumulative, global_floor)
    return jnp.mean(payoff)
```

That is the pricer. Full Greeks are `jax.grad(cliquet_payoff, ...)`. Calibration is `optimistix.least_squares` on the residual against market quotes. `vmap` gives you the price surface across strikes.

The pattern generalises: barrier options, Asians, quantos, autocallables, snowballs, target-redemption forwards — all are just Python functions in the same style, all get autodiff Greeks, all `vmap` for portfolio pricing. See [`valax/pricing/mc/payoffs.py`](../api/pricing.md) and the [Equity Exotics guide](../guide/equity-exotics.md) for the shipped examples.

**Time from "trader asks for a new payoff" to "structured deal ready to hedge" collapses from weeks to hours.** For a structuring desk that ships a new payoff every quarter, this is the difference between winning and losing mandates.

---

## 4. Neural surrogate pricers

A neural surrogate is a small neural network trained to approximate an expensive pricer. Once trained it replaces the pricer in inner loops that repeatedly evaluate it — XVA, real-time risk, portfolio optimisation, deep hedging. Every bank with a serious quant R&D team is either running or investigating this.

The traditional pipeline is painful: train the network in PyTorch, ship the weights to C++, write ONNX bindings, hope the numerical precision matches. Every model change is a serialisation exercise.

The VALAX pipeline is one file:

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
def teacher(features):
    return heston_price_cos(*unpack(features))

# Training data — batched teacher evaluations
train_features = sample_feature_space(key, n_samples=100_000)
train_targets  = jax.vmap(teacher)(train_features)

# Train the surrogate with optax
model = SurrogateHestonPricer(jax.random.PRNGKey(0))
optim = optax.adam(1e-3)

@eqx.filter_jit
def loss(model, features, targets):
    preds = jax.vmap(model)(features)
    return jnp.mean((preds - targets) ** 2)

# ... standard optax training loop
```

The trained surrogate has all the Greeks (`jax.grad(model)`), can be `vmap`-ed for batch pricing, and lives in the same array framework as the teacher — so you can even train it with a **pricer-gradient-matching loss** (train the network to reproduce the teacher's *delta and gamma*, not just its price). That last trick is state-of-the-art surrogate training and is essentially impossible to do cleanly across a C++/Python boundary.

For the full context see [Vision § "Neural surrogates"](../vision.md) — this is one of the P5 frontier workstreams the library is explicitly architected to enable.

---

## 5. Deep hedging — the flagship differentiable-finance application

Deep hedging (Buehler, Gonon, Teichmann, Wood 2018) trains a neural network to hedge a derivative in the presence of transaction costs, market frictions, and constraints where classical Δ-hedging breaks down. The training signal is the *gradient of the hedged P&L through the SDE simulation*. That gradient only exists cleanly if:

- The underlying SDE simulation is differentiable.
- The payoff evaluation is differentiable.
- The neural network is in the same array framework as both.

**All three conditions are natively met by VALAX.** The pattern is:

```python
# diffrax generates the SDE paths, differentiably
paths = generate_heston_paths(key, model_params, path_config)   # (n_paths, n_steps)

# The hedging policy is an equinox neural network
hedging_policy = HedgingNetwork(key_net, in_dim=state_features, hidden=64)

# P&L accumulator walks the paths, calling the policy at each step,
# and computes end-of-horizon hedged P&L (all differentiable)
def hedged_pnl(policy, paths, market):
    hedges = jax.vmap(policy)(state_features_from_paths(paths))
    pnl = accumulate_hedged_pnl(paths, hedges, transaction_cost, market)
    return -jnp.mean(cvar(pnl, alpha=0.05))   # loss = expected shortfall of P&L

# Train the policy end-to-end
grad_loss = eqx.filter_grad(hedged_pnl)
policy_update = optax.adam(1e-3)
# ... training loop
```

Every gradient in that pipeline flows through the SDE simulation, the payoff, and the network — because they are all one JAX computation graph.

This is not a hypothetical. VALAX already ships the exact primitives this pattern requires: [`diffrax`](../guide/monte-carlo.md) for SDE simulation, [`equinox`](../architecture/jax-patterns.md) for the network container, [`optax`](../guide/calibration.md) for training. A researcher can build a deep-hedging prototype on top of the library today with essentially no infrastructure work — the entire stack is already differentiable.

---

## 6. `vmap` parameter sweeps and Monte Carlo experiments

Quant research is fundamentally an **experimental** discipline: sweep a parameter, plot the surface, look for anomalies. In a C++ analytics library this means writing a driver script that calls the pricer in a loop and marshals results into a DataFrame. In VALAX it is one `vmap`:

```python
# Price surface across 100 strikes × 50 expiries × 20 vols = 100 000 prices
surface = jax.vmap(jax.vmap(jax.vmap(price_fn,
                                     in_axes=(None, None, 0)),  # vary vol
                            in_axes=(None, 0, None)),           # vary expiry
                   in_axes=(0, None, None))(strikes, expiries, vols)
# surface.shape == (100, 50, 20)
```

That expression compiles into a single JIT-compiled kernel and runs at GPU speed. No Python loop. No intermediate DataFrame. Just an array.

The same pattern gives you:
- **Monte Carlo convergence studies** — `vmap` over seed, over path count, over payoff variant.
- **Sensitivity landscapes** — `vmap` `jax.grad` across a parameter grid.
- **Calibration robustness** — `vmap` calibration over 100 perturbations of the market data.
- **Historical rolling windows** — `vmap` a full risk pipeline over a rolling 250-day window in one call.

For research this is *quantitatively* faster (GPU) and *qualitatively* different: experiments that were "we'd need a scheduler and 40 CPU hours" become "let me try it before lunch".

---

## 7. The JAX ecosystem is a research ecosystem

The dependencies VALAX builds on are not incidental — they are the state-of-the-art scientific-computing ecosystem for JAX, and every one of them is used by quant research directly:

| Package | What it gives the researcher |
|---|---|
| [`diffrax`](https://github.com/patrick-kidger/diffrax) | Differentiable SDE / ODE solvers — plug in any drift/diffusion, get differentiable path simulation |
| [`equinox`](https://github.com/patrick-kidger/equinox) | Pytree dataclasses for models and networks; `filter_jit` / `filter_grad` handle static fields transparently |
| [`optimistix`](https://github.com/patrick-kidger/optimistix) | Root-finding, least-squares (LM), nonlinear solvers — the calibration backbone |
| [`optax`](https://github.com/google-deepmind/optax) | Gradient-based optimisers (Adam, RMSprop, SGD variants) — the neural surrogate and deep hedging backbone |
| [`lineax`](https://github.com/patrick-kidger/lineax) | Structured linear solvers — PDE inner loop, structured covariance factorisations |
| [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) | Runtime shape/dtype checks — makes research code self-documenting |

This same stack powers a lot of cutting-edge scientific ML research. When you write a VALAX prototype, you can pull in `flax` for larger neural architectures, `blackjax` for MCMC calibration, `jaxopt` for constrained optimisation — all with zero framework impedance. This is not the case for any C++ analytics library.

**The upshot for a research team:** the same infrastructure investment that gives you a pricing library also gives you a neural-SDE library, a differentiable-optimisation library, and a bridge to the broader ML research world. One environment. One `pip install`. One mental model.

---

## 8. Publication and reproducibility

Peer-reviewed quant-finance research — and internal model-governance documentation — both live and die on reproducibility. VALAX's design makes reproducibility a *property*, not a wish:

- **Pure functions.** Every pricer is `V(instrument, market) → price` with no mutable state. Given the same inputs, on the same version of the library, you get bit-identical outputs across machines and years. That is exactly what a journal reviewer and an internal validator both need.
- **Integer-ordinal dates.** No `datetime` timezone / DST ambiguity. Two runs on the same ordinal date produce the same discount factor.
- **Explicit PRNG keys.** Every stochastic operation (`historical_scenarios`, `parametric_scenarios`, MC path generation) takes a `jax.random.PRNGKey` explicitly. No hidden global RNG state. Seed once, replay forever.
- **Pinned dependency versions.** `pyproject.toml` pins `jax`, `equinox`, `diffrax`, `optimistix` at specific versions — a git tag is a full reproducibility contract.
- **Golden tests.** The `tests/golden/` package ships reference outputs for pricers and calibrations, so a future refactor cannot silently drift.

For an academic researcher this means submissions ship with a working git tag rather than a "code available on request" placeholder. For an internal model governance team this means the audit trail from paper → model → production is a git log — not a folder of spreadsheets.

---

## 9. Coverage today vs. roadmap

Most of what quant research needs is already in the library, because research productivity was one of the founding design goals:

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

The pattern: **the infrastructure is complete; the canonical worked examples are the roadmap**. For a research team this is close to ideal — you're not building tooling, you're doing research.

---

## 10. Why the JAX foundation matters for research specifically

Everything in this page reduces to a single thesis: **for quant research, the JAX-native architecture is not a "nice implementation choice" — it is the primary source of research-productivity leverage**, and it compounds over time because:

| Design choice | Research payoff |
|---|---|
| **Autodiff Greeks** | Every new pricer ships with all Greeks. New model → new sensitivities → next model. No numerical noise, no bump-width tuning, no maintenance tax. |
| **Composable pure functions** | A new payoff is a Python function, not a class subtree. Iteration speed is Python speed. |
| **`vmap` everywhere** | Parameter sweeps, MC experiments, calibration robustness studies — one line each, GPU-accelerated. |
| **`equinox.Module` pytrees** | Models, networks, and pricers share the same container. Pass a pricer to a network trainer, no serialisation. |
| **JAX ecosystem integration** | `optax`, `flax`, `blackjax`, `jaxopt` all work out of the box. Neural surrogate today, MCMC calibration tomorrow, deep hedging next week. |
| **JIT + XLA** | Prototype runs at C++ speed with no C++ code. Deployment is a `pip install`. |
| **Pure functions + explicit PRNG** | Bit-identical reproducibility across machines and years. Submissions and validators both trust the outputs. |

The compounding effect: **every model you have already implemented becomes a differentiable primitive for the next model you implement.** A team of five researchers on VALAX outpaces a team of twenty on a C++ analytics library — not because JAX is magic, but because the tax of maintaining bump-and-reprice / serialisation-boundary / class-hierarchy infrastructure disappears entirely.

---

## 11. So — the case for quant research as the strongest application

The [Applications overview](index.md) ranks Model Validation / MRM as the strongest **commercial** adoption case today because of the regulatory-mandated need and the absence of incumbent lock-in. That ranking is honest for a bank buyer.

But for a **technical** adoption case — measured in research output per researcher per unit time — quant research is the strongest fit, and the gap to any incumbent is not incremental. It is categorical. Consider:

- A bank MRM team can *do their job* with vendor risk tooling, however painfully. VALAX makes it better and cheaper, but the job is doable elsewhere.
- A bank quant research team **cannot** do neural surrogate pricing or deep hedging or differentiable-SDE calibration on a C++ analytics library. Those research directions are structurally out of reach until the analytics stack becomes differentiable. VALAX is not "better" for these — it is *possible* for these where the incumbent is not.

The strategic implication for a bank considering VALAX:

- **Adopt in Market Risk & Model Validation first** for the commercial and regulatory return ([market-risk.md](market-risk.md)).
- **Simultaneously adopt in Quant Research** for the innovation moat — different budget, different buyer, provides the frontier-capability story that keeps the analytics stack ahead of vendors long-term.

The two adoptions cross-pollinate: the Model Validation team benefits when Quant Research extends VALAX with new models; Quant Research benefits when their prototypes graduate into MRM's validated production stack without a rewrite. **That is the "one library, many audiences" thesis in practice** — and it is the thesis no incumbent C++ analytics library can offer, because their architecture forbids it.

---

## 12. The pitch, tailored to the buyer

**To the Head of Quant Research / Chief Quant Officer:**
*Every model your team writes ships with autodiff Greeks, gradient-based calibration, GPU acceleration, and integration with the ML stack — for zero additional infrastructure. Time from paper to calibrated prototype collapses from months to days. Deep hedging and neural surrogates go from "future work" to "next sprint".*

**To the CTO / Head of Analytics:**
*One library serves the pricing kernel, the calibration engine, the risk engine, the surrogate trainer, and the SDE research bench. One environment, one dependency graph, one deployment story. Every research prototype is one refactor away from graduating to production because the language and the framework do not change.*

**To an academic partner / PhD student:**
*Publish with a `pip install` and a git tag. Every result is bit-reproducible. Every model composes with `diffrax`, `optax`, `flax`, `blackjax`. The paper-to-code delta is minutes, not weeks. Reviewers can run your notebook.*

**To a hedge-fund CIO / systematic PM:**
*The same differentiable pricer that computes your fair value gives you the gradient of P&L with respect to every parameter of the strategy — feed it into `optax` and the strategy tunes itself. This is not a black-box RL loop; it is the actual gradient of the actual P&L.*

---

## 13. Where to read next

- **The applications overview and audience ranking** → [Applications](index.md).
- **The commercial-adoption companion** → [Market Risk & Model Validation](market-risk.md).
- **The mathematical foundations of every model in the library** → [Models & Theory](../theory.md).
- **Calibration patterns and shipped examples** → [Calibration](../guide/calibration.md), [Vol Surfaces](../guide/vol-surfaces.md), [SLV](../guide/slv.md).
- **SDE path simulation and Monte Carlo** → [Monte Carlo](../guide/monte-carlo.md).
- **PDE solvers and lattice methods** → [PDE Solvers](../guide/pde.md), [Binomial Trees](../guide/lattice.md).
- **The JAX design patterns behind the library** → [JAX Patterns](../architecture/jax-patterns.md).
- **The forward-looking vision — neural surrogates, deep hedging, differentiable portfolios** → [Vision](../vision.md).
- **Why the library makes the architectural choices it does** → [Design Rationale](../design-rationale.md).
