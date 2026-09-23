# Greeks

Automatic-differentiation Greeks for any pure pricing function
`(instrument, *market_args) -> scalar`. First- and second-order Greeks
come from `jax.grad`; theta is computed by bumping the instrument's
`expiry` field (autodiff w.r.t. calendar time would require making
`expiry` a float leaf, which is intentionally avoided).

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

Sensitivities over the ontology's `price(instrument, model, market)`
(pricing-ontology P4). One backward pass through the unified pricer yields a
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
