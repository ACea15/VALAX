# Greeks

Autodiff Greeks for any pure pricing function
`(instrument, *market_args) -> scalar` ([Theory § 6.2](../theory/greeks.md#62-automatic-differentiation)).
First- and second-order Greeks come from `jax.grad`. The kernel-layer
`greeks()` / `greek()` compute theta by a one-day backward bump of the
instrument's `expiry`, for any pricer; the unified [`theta`](#unified-market-model-contract-greeks)
below differentiates `expiry` exactly where the engine allows it.

!!! warning "Argument order matters"
    `greeks()` maps positional `market_args` to Greek names by index:
    arg 0 → `delta`, arg 1 → `vega`, arg 2 → `rho`, arg 3 → `dividend_rho`.
    This matches the Black–Scholes signature
    `(spot, vol, rate, dividend)`. For 3-argument pricers
    (Black-76, Bachelier) `dividend_rho` will error — call
    [`greek`][valax.greeks.greek] for the specific Greek you need instead.

## Compute all Greeks at once

::: valax.greeks.greeks

## Compute a single Greek

::: valax.greeks.greek

## Unified market / model / contract greeks

Sensitivities over the unified
[`price(instrument, model, market)`](core.md#the-unified-signature) call — see
[tutorial § 8](../guide/tutorial-unified-pricing.md#8-sensitivities-through-the-same-call)
for the walkthrough. One backward pass through the unified pricer yields a
pytree-shaped gradient for **market greeks** (`market_gradient`, `spot_delta`,
`curve_pillar_deltas`) and **model greeks** (`model_gradient`); **contract
greeks** (`dual_delta`, `theta`) differentiate the instrument's own leaves,
with theta following the autodiff-vs-bump policy (exact for analytic/lattice,
finite-difference for PDE/MC). `bucketed_curve_deltas` aggregates raw pillar
key-rate deltas onto FRTB tenor vertices.

::: valax.greeks.market_gradient

::: valax.greeks.model_gradient

::: valax.greeks.spot_delta

::: valax.greeks.curve_pillar_deltas

::: valax.greeks.bucketed_curve_deltas

::: valax.greeks.dual_delta

::: valax.greeks.theta

::: valax.greeks.market_greeks

### Every declared entry in one pass

`market_risk` is the risk counterpart of `price()`: one backward pass, reported
per declared market entry (every curve the trade or model reads, spots,
dividends, FX spots, inflation curves, surfaces), keyed by id. The declared set
is [`declared_entries`](core.md#valax.core.declared_entries); see
[tutorial § 8](../guide/tutorial-unified-pricing.md#every-entry-the-trade-touches-market_risk).

::: valax.greeks.market_risk

::: valax.greeks.MarketRisk

## Quote-space (hedge-space) risk

`curve_pillar_deltas` above answers "how does the price move if a *pillar
discount factor* moves". A desk hedges in instruments, not pillars, so the
sensitivity it needs is to the **market quotes the curves were bootstrapped
from**. That is a different number, not a rescaling: holding the other quotes
fixed forces the whole curve graph to re-solve, so an OIS quote picks up its
indirect effect on every projection curve that was calibrated against it.

[`curve_quote_deltas`](#valax.greeks.curve_quote_deltas) delivers it via the
implicit function theorem: one reverse sweep through the pricer, one transposed
linear solve against the bootstrap residual Jacobian, one reverse sweep through
the residual. The optimiser's Newton iterations are never differentiated, and
the quote Jacobian is never formed — so unlike
[`quote_jacobian`](curves.md#valax.curves.quote_jacobian) (which re-bootstraps
once per quote) the cost does not grow with the number of quotes.

Build a [`CurveAdjoint`](#valax.greeks.CurveAdjoint) once per market snapshot
and reuse it: it carries the factorised Jacobian, so each additional trade
costs one back-substitution.

::: valax.greeks.curve_adjoint

::: valax.greeks.CurveAdjoint

::: valax.greeks.curve_quote_deltas

::: valax.greeks.curve_log_df_cotangent

::: valax.greeks.quote_deltas_from_cotangent

### Generic calibration levels

The same theorem applies to every calibrated object, not just curves. A
[`CalibrationLevel`](#valax.greeks.CalibrationLevel) states a calibration as
`R(theta; inputs) = 0`; [`level_adjoint`](#valax.greeks.level_adjoint) then
differentiates it. This is the building block the volatility and model layers
plug into.

::: valax.greeks.CalibrationLevel

::: valax.greeks.level_adjoint

::: valax.greeks.LevelAdjoint

### Conditioning of a level

The theorem needs `dR/dtheta` invertible; *how* invertible decides how many
significant digits survive the transposed solve. A fit can converge to a tiny
residual and still produce quote deltas whose leading digits are noise, and the
residual norm will not say so. `level_condition_number` reports the condition
number for any level — curve, smile or model — on one scale, and
`condition_alert` tests it against the shared `ILL_CONDITIONED_THRESHOLD`.

See [Calibration § 8](../guide/calibration.md#8-is-the-fit-trustworthy-conditioning-at-every-level)
for the runnable example and for what the observed numbers mean.

::: valax.greeks.level_condition_number

::: valax.greeks.condition_alert

## Fixed inputs (`p`)

A calibration has two kinds of input. The **quotes** are solved against and
come back as hedge-space deltas above. The **fixed inputs** are everything
else the answer depends on but the solver holds still — Hull-White's mean
reversion, the SABR exponent and shift, futures convexity adjustments, turn
spreads, wing settings. They are not tradeable, so they are reported
*separately* from `q_bar` rather than folded into it: a desk cannot hedge mean
reversion, only size and govern the exposure to it.

A [`FixedInput`](#valax.greeks.FixedInput) stores **where** the value lives,
not the value. Those numbers already sit on the model and surface pytrees the
pricer consumes, so a container that copied them would be a second source of
truth — and the day it disagreed, the risk report would describe a model
nobody priced with. Differentiating through the re-injection makes the
reported sensitivity belong, by construction, to the leaf that was priced.

See [Calibration § 9](../guide/calibration.md#9-what-a-risk-report-has-to-contain)
for the runnable example, and in particular for why the *direct* and
*recalibrated* sensitivities to the same input differ by a factor of six.

::: valax.greeks.FixedInput

::: valax.greeks.FixedInputs

::: valax.greeks.fixed_inputs_from_paths

::: valax.greeks.fixed_input_adjoints

::: valax.greeks.FixedInputAdjoint

### Vega to volatility quotes

The same step one layer up. A smile fit is not a square root-find — there are
more quoted strikes than SABR parameters — so the system the theorem is applied
to is the fit's **first-order condition**, the stationarity of the weighted sum
of squares. That is square by construction, and differentiating it with
`jax.grad` of a gradient yields the **full** Hessian rather than the
Gauss-Newton approximation, so the adjoint stays correct when the fit is not
tight (which is when a sensitivity is most likely to be trusted wrongly).

Two input blocks come back, answering two different questions: the quoted vols
give **vega in quote space**, and the forward gives
$\partial V / \partial F_0$ — the term that chains the volatility layer back
into the curve layer (spec A-3). The second is non-zero only because the forward
is a genuine input to the fit, which is what
[`calibrate_swaption_cube_from_curves`](surfaces.md#swaption-cube-rates)
arranges (the cube is only ever calibrated from the curves); with a frozen
forward it would be structurally zero.

::: valax.greeks.vol_adjoint

## Quote risk through a calibration chain

The whole chain at once — curves, vol objects, a calibrated model — by two
independent paths: the explicit adjoint sweep, which walks the chain's levels
backwards with one transposed solve each and is linear in its seed (so a book
is swept once with its summed cotangent), and autodiff through the solves.
`reconcile_quote_risk` compares them. Walkthrough:
[From Quotes to Risk](../guide/calibration-to-risk.md); maths:
[Calibration Chains § I.4](../theory/calibration-chains.md#i4-the-adjoint-sweep).

::: valax.greeks.quote_risk.QuoteRisk

::: valax.greeks.quote_risk.chain_adjoint

::: valax.greeks.quote_risk.quote_risk_adjoint

::: valax.greeks.quote_risk.quote_risk_autodiff

::: valax.greeks.quote_risk.reconcile_quote_risk

::: valax.greeks.quote_risk.QuoteRiskReconciliation
