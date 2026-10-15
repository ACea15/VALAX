# Calibration

Model parameter fitting via gradient-based optimization. All routines
map constrained parameters to unconstrained space via
[transforms](#transforms), call `optimistix` / `optax` under the hood,
and return a [`CalibrationResult`](#valax.calibration.result.CalibrationResult):
the calibrated object is `result.value`, convergence is reported in
`result.diagnostics` (traceable, so it survives `jit` / `vmap`), and
`result.check()` raises `CalibrationError` on failure.

Every calibrator — the per-model ones below and the unified layer documented
first — is one calibration level `R(theta; quotes, upstream, fixed) = 0`, and
levels compose into chains. Solver choice is a single keyword-only
`settings=SolverSettings(method=..., rtol=..., atol=..., max_steps=...)`
argument; there are no string `solver=` / `max_steps=` keywords. Walkthrough:
[From Quotes to Risk](../guide/calibration-to-risk.md); design:
[Calibration, Pricing & Risk Architecture](../architecture/calibration-pricing-risk.md).

```python
from valax.calibration import SolverSettings, calibrate_sabr

result = calibrate_sabr(strikes, market_vols, forward, expiry,
                        fixed_beta=jnp.array(0.5),
                        settings=SolverSettings(method="levenberg_marquardt",
                                                rtol=1e-8, atol=1e-8))
result.check()                 # raises CalibrationError if not converged
fitted = result.value          # SABRModel
print(result.diagnostics.rmse, result.condition_number())
```

## Unified calibration layer

### Solver settings

::: valax.calibration.settings.SolverSettings

::: valax.calibration.settings.run_solver

::: valax.calibration.settings.make_solver

Defaults shipped with the library (each calibrator documents which one it
uses when `settings=None`):

| Constant | Method | Used by |
|---|---|---|
| `valax.calibration.settings.EXACT_SETTINGS` | Newton, `1e-10` | exact (root) levels: curve bootstraps, implied-vol inversions |
| `valax.calibration.settings.FIT_SETTINGS` | Levenberg–Marquardt, `1e-8`, 512 steps | least-squares smile / surface fits (SABR, SVI, Heston) |
| [`BASKET_SETTINGS`](#valax.calibration.basket.BASKET_SETTINGS) | BFGS, `1e-12` | Hull–White, G2++ basket fits |
| [`CHEYETTE_SETTINGS`](#valax.calibration.cheyette.CHEYETTE_SETTINGS) | BFGS, `1e-10` | Cheyette 1F / 2F PDE fits |
| [`STRIP_SETTINGS`](surfaces.md#valax.surfaces.caplet_stripping.STRIP_SETTINGS) | Newton, `1e-12` | caplet stripping (per bucket) |
| [`INVERSION_SETTINGS`](surfaces.md#swaption-premium-normal-vol) | Newton, `1e-12` | swaption premium → normal vol |
| [`SLV_SETTINGS`](#valax.calibration.slv.SLV_SETTINGS) | fixed point, `atol=5e-2` | SLV leverage iteration |

::: valax.calibration.settings.EXACT_SETTINGS

::: valax.calibration.settings.FIT_SETTINGS

### Levels and results

`CalibrationLevel`, `level_adjoint` and the conditioning diagnostics live in
`valax.calibration.level` and are documented with the risk they produce under
[Greeks › Generic calibration levels](greeks.md#generic-calibration-levels).

::: valax.calibration.level.LevelInputs

::: valax.calibration.result.CalibrationResult

::: valax.calibration.result.CalibrationDiagnostics

::: valax.calibration.result.CalibrationError

::: valax.calibration.result.CalibrationProvenance

::: valax.calibration.result.CLOSED_FORM_PROVENANCE

::: valax.calibration.result.engine_provenance

::: valax.calibration.result.check_engine

::: valax.calibration.result.fixed_point_diagnostics

::: valax.calibration.result.LEVEL_KINDS

::: valax.calibration.result.ENGINE_KINDS

::: valax.calibration.result.FAILURE_CLUSTER_FRACTION

### Solving a level

`valax.calibration.solve` is the one place a level is solved and packaged as
a `CalibrationResult`. Batched levels (`solve_batched_level`) solve each node
independently in one `vmap` and report per-node diagnostics aligned with
`result.labels`, so a failed smile / cube cell / strip bucket is named; a node
solved in stages (a cap-strip column) passes `sub_labels` and is labelled per
`(node, stage)`, e.g. `"K=3.00% cap 2Y"`.

::: valax.calibration.solve.solve_level

::: valax.calibration.solve.solve_batched_level

::: valax.calibration.solve.explicit_level

::: valax.calibration.solve.triangular_root_level

::: valax.calibration.solve.stack_slices

::: valax.calibration.solve.period_label

::: valax.calibration.solve.period_labels

### Chains

::: valax.calibration.chain.CalibrationNode

::: valax.calibration.chain.CalibrationChain

::: valax.calibration.chain.CalibratedMarket

::: valax.calibration.chain.calibrate_market

::: valax.calibration.chain.calibrate_model

::: valax.calibration.chain.install_curves

::: valax.calibration.chain.install_surface

::: valax.calibration.chain.install_market_fields

### Node factories

::: valax.calibration.nodes.market_inputs_node

::: valax.calibration.nodes.curve_graph_node

::: valax.calibration.nodes.instrument_quotes

::: valax.calibration.nodes.sabr_smile_node

::: valax.calibration.nodes.equity_forward

::: valax.calibration.nodes.price_fit_node

#### Rates volatility nodes

The rates vol levels as chain nodes: the swaption SABR cube read off the
curves, the caplet strip, and the per-expiry optionlet SABR surface fitted on
the strip.

::: valax.calibration.nodes.swaption_cube_node

::: valax.calibration.nodes.caplet_strip_node

::: valax.calibration.nodes.optionlet_sabr_node

#### Per-trade model calibration

Co-terminal swaption baskets and the per-trade short-rate fit on them (the
building blocks of [example 14](../examples.md#per-trade-model-calibration-for-bermudans)):
a least-squares fit of constant parameters, or the exact co-terminal
bootstrap of a piecewise-constant σ(t)
(`mode="coterminal_bootstrap"`, a `triangular_root_level`).

::: valax.calibration.nodes.coterminal_swaptions

::: valax.calibration.nodes.swaption_basket_node

::: valax.calibration.nodes.BASKET_MODES

## Short-rate basket fits

Hull–White, G2++ and Cheyette are all fitted by one weighted least-squares
level against a basket of instrument prices; the curves are its upstream
input, so the adjoint reports the curve risk of the fit. The default
[`BASKET_SETTINGS`](#valax.calibration.basket.BASKET_SETTINGS) is
line-searched BFGS rather than Levenberg–Marquardt: the closed-form swaption
engines (Jamshidian, Brigo–Mercurio) nest a root-find whose own implicit
derivative is non-finite far from the solution, where an undamped LM step can
land; BFGS's line search keeps the iterates where the pricer is defined. LM is
still accepted via `settings=`.

::: valax.calibration.basket.basket_fit_level

::: valax.calibration.basket.BASKET_SETTINGS

## SABR

Fit SABR to a volatility smile. Typical use: fix `beta` and calibrate
`(alpha, rho, nu)` from at-the-money forward vols.

::: valax.calibration.calibrate_sabr

The slice residual and its batched form are exposed so surface and cube
builders (and custom workflows) solve exactly the system `calibrate_sabr`
solves:

::: valax.calibration.sabr.sabr_slice_inputs

::: valax.calibration.sabr.sabr_slice_residual_fn

::: valax.calibration.sabr.sabr_batch_level

::: valax.calibration.sabr.stacked_sabr_inputs

::: valax.calibration.sabr.sabr_vol_fn

::: valax.calibration.sabr.SABR_FIXED_BETA_TRANSFORMS

## Heston

Fit Heston to option prices. The `pricing_fn` is dependency-injected
so the calibration machinery stays decoupled from the pricing engine
— pass either an analytic pricer or an MC pricer with signature
`(model, strike, spot, rate, dividend, expiry) -> price`.

::: valax.calibration.calibrate_heston

## Stochastic-Local Volatility (SLV)

Two-pass SLV calibration. Pass 1 fits Heston to vanillas; pass 2 fits
the leverage function \(L(k, t)\) via Markovian projection so that the
SLV model reproduces the input local-vol surface. See
[theory §4.5](../theory/volatility.md#45-stochastic-local-volatility) and the
[SLV guide](../guide/slv.md) for the mathematics and choice of
estimator (`particle` vs. `kernel`). Both return a `CalibrationResult`
(`.value` is the `LeverageGrid` / `SLVModel`); the leverage pass is a
fixed-point level whose diagnostics are the last outer-pass increment, and
`calibrate_slv` carries the Heston level as `result.children[0]` — fitted
(optionally multi-expiry, with `fixed_params`) or supplied via `heston=`.

::: valax.calibration.calibrate_slv_leverage

::: valax.calibration.calibrate_slv

::: valax.calibration.slv.SLV_SETTINGS

## Hull–White

Fits the two free short-rate parameters \((a, \sigma)\) to a European swaption
surface using the closed-form Jamshidian pricer, so the least-squares Jacobian
is exact autodiff rather than a bumped approximation. The fitted
`HullWhiteModel` (`result.value`) is **curve-free**: `valax.price` binds its
initial curve from the `MarketState` at pricing time.

The two parameters are strongly correlated along the ATM surface — \(a\) sets
the decay of forward-rate volatility with expiry, \(\sigma\) its level — so
fixing \(a\) and fitting only \(\sigma\) is usually the better-posed problem.
See [theory](../theory/hull-white-swaptions.md#5-why-this-makes-hullwhite-calibratable).

::: valax.calibration.calibrate_hull_white

::: valax.calibration.swaption_prices_from_vols

::: valax.calibration.hw_swaption_prices

::: valax.calibration.hull_white.JAMSHIDIAN_PROVENANCE

## G2++

Fits the five free G2++ parameters \((a, b, \sigma, \eta, \rho)\) jointly to a
European swaption surface using the semi-analytic Brigo–Mercurio pricer, so the
least-squares Jacobian is exact autodiff. Because the two mean reversions are
only weakly identified by an ATM surface — and the objective has degenerate
basins at \(\sigma \to 0\), \(\rho \to \pm 1\) — pinning \((a, b)\) via
`fixed_params` and starting the vols and \(\rho\) from a sensible prior is the
better-posed problem. See
[G2++ theory §6](../theory/g2pp.md#6-calibration).

::: valax.calibration.calibrate_g2pp

::: valax.calibration.g2pp_swaption_prices

## Cheyette (quasi-Gaussian)

Calibrated *through the finite-difference pricer* rather than an
approximate-analytic formula, so the full instrument set — caps, European and
Bermudan swaptions, callables — is fitted on one consistent engine. The PDE is
cleanly reverse-mode differentiable, so the least-squares Jacobian is exact
autodiff through the backward sweep (no finite-difference Greeks). One factor
$(\kappa, \sigma, \beta)$ fixes $\kappa$ and fits the level $\sigma$ and skew
$\beta$ against a **strike slice**; two factors add $\rho$ for decorrelation.
`calibrate_cheyette` selects the frozen-$y$ (fast) or exact-$y$ (bias-free)
engine; `calibrate_cheyette_2f` is always frozen-$y$. See the theory note
[Cheyette §9 Calibration](../theory/cheyette.md#9-calibration).

::: valax.calibration.calibrate_cheyette

::: valax.calibration.calibrate_cheyette_2f

::: valax.calibration.cheyette_pde_prices

::: valax.calibration.cheyette.cheyette_provenance

::: valax.calibration.cheyette.CHEYETTE_SETTINGS

## Transforms

Reparametrization utilities that map constrained parameters
(e.g. \(\alpha > 0\), \(-1 < \rho < 1\)) to unconstrained \(\mathbb{R}^n\)
so JAX optimizers can operate without explicit constraints.

::: valax.calibration.transforms.positive

::: valax.calibration.transforms.bounded

::: valax.calibration.transforms.unit_interval

::: valax.calibration.transforms.correlation

::: valax.calibration.transforms.model_to_unconstrained

::: valax.calibration.transforms.unconstrained_to_model
