# Surfaces

Volatility surface construction and calibration. All surfaces are
callable `eqx.Module` pytrees with signature
`surface(strike, expiry) -> implied_vol`.

## Dupire / SLV protocol: `total_variance`

All three surface types additionally expose

```python
surface.total_variance(log_moneyness, expiry) -> Float[Array, ""]
```

returning total implied variance \(w(k, T) = \sigma_{\text{IV}}^2(k, T) \cdot T\)
as a function of log-moneyness \(k = \ln(K / F(T))\). This is the
duck-typed input expected by
[`dupire_local_vol`](pricing.md#dupire_local_vol) and by SLV's leverage
function calibration
([`calibrate_slv_leverage`][valax.calibration.calibrate_slv_leverage]).
The consistency identity

```python
surface.total_variance(jnp.log(K / F_T), T)  ==  surface(K, T) ** 2 * T
```

holds to machine precision for every surface implementation, where
`F_T = jnp.interp(T, surface.expiries, surface.forwards)` is the
query-expiry forward (or, for `GridVolSurface`, the convention is
documented per-surface below).

## Grid surface

Bilinear interpolation (via `valax.surfaces._interp.bilinear_2d`) with
flat extrapolation. The `vols` grid is differentiable.

For `GridVolSurface` the `strikes` axis is consumed directly as the
log-moneyness axis — i.e. a caller using a grid surface for Dupire
must build the grid in \((k, T)\) space rather than \((K, T)\). This
avoids embedding a forward-curve assumption inside the grid surface
itself. See the docstring on `GridVolSurface.total_variance` for the
full convention.

::: valax.surfaces.GridVolSurface

## SABR surface

Linearly interpolates SABR parameters to the query expiry, then
evaluates the Hagan formula. All parameters are differentiable.

`SABRVolSurface.total_variance(k, T)` converts back to an absolute
strike via the query-expiry forward
(\(K = F_T \cdot e^k\) with `F_T = jnp.interp(T, expiries, forwards)`)
and returns `surface(K, T) ** 2 * T`. Suitable as a Dupire input, but
note that Hagan's IV formula is itself an asymptotic expansion — at
extreme strikes the resulting "local vol" from Dupire becomes a
measurement of the expansion's residual rather than a true market local
vol.

::: valax.surfaces.SABRVolSurface

::: valax.surfaces.calibrate_sabr_surface

## SVI surface

SVI parametrizes total variance directly as
\(w(k) = a + b\bigl(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\bigr)\).
`SVIVolSurface.total_variance(k, T)` is the natural Dupire input —
total variance is the *intrinsic* SVI quantity, so no double
`sqrt → square` roundtrip is needed. The method evaluates each
calibrated slice's SVI total variance at the slice-relative
log-moneyness \(\ln(K / F_\text{slice})\), where \(K = F_T \cdot e^k\),
then linearly interpolates across slices in \(T\). Identity to
`__call__(K, T) ** 2 * T` holds exactly.

!!! note "Short-maturity extrapolation"
    Both `__call__` and `total_variance` extrapolate **linearly in
    T through the origin** below the first calibrated expiry (i.e.
    holding implied vol constant as \(T \to 0^+\)). This differs from
    `jnp.interp`'s flat-w default. The change makes Dupire's
    \(\partial w / \partial T\) positive at short expiries — without
    it, the \(1/w\) terms in the Dupire denominator diverge and any
    local-vol MC stepping through the boundary takes a zero-vol first
    step. See [theory §4.4](../theory.md#44-local-volatility-dupire).

### SVI slice primitives

::: valax.surfaces.SVISlice

::: valax.surfaces.svi_total_variance

::: valax.surfaces.svi_implied_vol

### SVI surface object

::: valax.surfaces.SVIVolSurface

### SVI calibration

::: valax.surfaces.calibrate_svi_slice

::: valax.surfaces.calibrate_svi_surface

## Leverage grid (SLV)

The leverage function \(L(k, t)\) tabulated on a fixed \((k, t)\)
grid. Stored with the same `(n_t, n_k)` (y outer, x inner) convention
as `GridVolSurface.vols`. The instance is callable
(`L(log_moneyness, time)`, bilinear, flat extrapolation).
Interpolation is delegated to the project-wide `bilinear_2d` helper.
Only `values` is differentiable; the grid axes are static
interpolation scaffolding. `jax.grad` through `values` flows
correctly into per-node leverage sensitivities, which is what the
calibration routine consumes internally.

Use `LeverageGrid.flat(...)` to build an \(L \equiv \text{value}\)
constant grid — the pure-Heston-limit warm start for
[`calibrate_slv_leverage`][valax.calibration.calibrate_slv_leverage]
and a useful Heston-limit reduction-test fixture. See the
[SLV guide](../guide/slv.md) for the role of the leverage function in
the Markovian-projection-based two-pass SLV calibration.

::: valax.surfaces.LeverageGrid
