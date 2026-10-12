# Calibration

Model parameter fitting via gradient-based optimization. All routines
map constrained parameters to unconstrained space via
[transforms](#transforms), call `optimistix` / `optax` under the hood,
and return the fitted model together with a solver `Solution` for
convergence diagnostics.

The per-model calibrators below predate the **unified calibration layer**
documented first, which expresses every calibration as a level returning a
`CalibrationResult` and composes levels into chains; they are being migrated
onto it. Walkthrough: [From Quotes to Risk](../guide/calibration-to-risk.md);
design: [Calibration, Pricing & Risk Architecture](../architecture/calibration-pricing-risk.md).

## Unified calibration layer

### Solver settings

::: valax.calibration.settings.SolverSettings

::: valax.calibration.settings.run_solver

::: valax.calibration.settings.make_solver

### Levels and results

`CalibrationLevel`, `level_adjoint` and the conditioning diagnostics live in
`valax.calibration.level` and are documented with the risk they produce under
[Greeks › Generic calibration levels](greeks.md#generic-calibration-levels).

::: valax.calibration.level.LevelInputs

::: valax.calibration.result.CalibrationResult

::: valax.calibration.result.CalibrationDiagnostics

::: valax.calibration.result.CalibrationError

### Chains

::: valax.calibration.chain.CalibrationNode

::: valax.calibration.chain.CalibrationChain

::: valax.calibration.chain.CalibratedMarket

::: valax.calibration.chain.calibrate_market

::: valax.calibration.chain.calibrate_model

::: valax.calibration.chain.solve_level

::: valax.calibration.chain.explicit_level

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

## SABR

Fit SABR to a volatility smile. Typical use: fix `beta` and calibrate
`(alpha, rho, nu)` from at-the-money forward vols.

::: valax.calibration.calibrate_sabr

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
estimator (`particle` vs. `kernel`).

::: valax.calibration.calibrate_slv_leverage

::: valax.calibration.calibrate_slv

## Hull–White

Fits the two free short-rate parameters \((a, \sigma)\) to a European swaption
surface using the closed-form Jamshidian pricer, so the least-squares Jacobian
is exact autodiff rather than a bumped approximation.

The two parameters are strongly correlated along the ATM surface — \(a\) sets
the decay of forward-rate volatility with expiry, \(\sigma\) its level — so
fixing \(a\) and fitting only \(\sigma\) is usually the better-posed problem.
See [theory](../theory/hull-white-swaptions.md#5-why-this-makes-hullwhite-calibratable).

::: valax.calibration.calibrate_hull_white

::: valax.calibration.swaption_prices_from_vols

::: valax.calibration.hw_swaption_prices

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

## Loss functions

Residual functions with the `optimistix` signature
`fn(y, args) -> residuals`. Used internally by the calibration
routines; exposed for custom calibration workflows.

::: valax.calibration.loss.vol_residuals

::: valax.calibration.loss.price_residuals

::: valax.calibration.loss.weighted_sse
