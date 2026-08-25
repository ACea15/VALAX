# Calibration

Model parameter fitting via gradient-based optimization. All routines
map constrained parameters to unconstrained space via
[transforms](#transforms), call `optimistix` / `optax` under the hood,
and return the fitted model together with a solver `Solution` for
convergence diagnostics.

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
[theory §4.5](../theory.md#45-stochastic-local-volatility) and the
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
