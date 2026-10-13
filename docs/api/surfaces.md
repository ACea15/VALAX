# Surfaces

Volatility surface construction and calibration. All surfaces are
callable `eqx.Module` pytrees with signature
`surface(strike, expiry) -> implied_vol`.

Every surface calibrator returns a
[`CalibrationResult`](calibration.md#valax.calibration.result.CalibrationResult)
whose `.value` is the surface. Multi-slice builders
(`calibrate_sabr_surface`, `calibrate_svi_surface`,
`build_sabr_caplet_surface`, the swaption cube) are **batched levels**: each
slice / cell is solved independently in one `vmap`, per-node diagnostics are
aligned with `result.labels` (e.g. `"2Y"`, `"5Yx10Y"`), and a failure is
reported by name (`result.failed_labels()`, `result.check()`). Ragged strike
grids are padded with zero-weight points. Solver choice is the keyword-only
`settings=SolverSettings(...)`.

## Dupire / SLV protocol: `total_variance`

All three surface types additionally expose

```python
surface.total_variance(log_moneyness, expiry) -> Float[Array, ""]
```

returning total implied variance \(w(k, T) = \sigma_{\text{IV}}^2(k, T) \cdot T\)
as a function of log-moneyness \(k = \ln(K / F(T))\). This is the
duck-typed input expected by
[`dupire_local_vol`](pricing.md#valax.pricing.analytic.dupire_local_vol) and by SLV's leverage
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
    step. See [theory §4.4](../theory/volatility.md#44-local-volatility-dupire).

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

## Optionlet surface (rates)

The interest-rate counterpart of the SABR surface: one SABR model per
**optionlet expiry**, with no tenor axis (contrast the swaption cube,
whose underlying is a swap rate). `is_normal` selects the quoting
convention — lognormal Black-76 or normal Bachelier, the latter honouring
a displacement `shift` so zero and negative rates price finitely.

::: valax.surfaces.OptionletVolSurface

::: valax.surfaces.build_sabr_caplet_surface

## Cap → caplet (optionlet) stripping

Caps are quoted with **one flat vol per maturity**; nothing prices with a
flat vol. Stripping converts the quotes into a per-optionlet term
structure that reprices every quoted cap exactly, and it is the canonical
dual-curve operation — forwards are projected off the tenor curve,
payoffs discounted off OIS. The strip is solved bucket by bucket (a
sequential bootstrap, one Newton root-find per quoted maturity, labelled
`"cap 3Y"` etc.) and packaged as **one lower-triangular root level**, so the
whole strip is differentiable by a single implicit-function adjoint.
`strip_optionlet_surface` returns the optionlet SABR level with the strip
level as `result.children[0]`; its `settings=` drive the SABR fits and
`strip_settings=` the strip. Recipe and worked example:
[Volatility Surfaces § Cap → caplet stripping](../guide/vol-surfaces.md#cap-caplet-optionlet-stripping-rates).

::: valax.surfaces.CapletVolCurve

::: valax.surfaces.strip_caplet_vols

::: valax.surfaces.strip_optionlet_surface

::: valax.surfaces.cap_price_from_strip

::: valax.surfaces.optionlet_forwards

::: valax.surfaces.optionlet_expiries

### Strip level building blocks

::: valax.surfaces.caplet_stripping.caplet_strip_level

::: valax.surfaces.caplet_stripping.strip_buckets

::: valax.surfaces.caplet_stripping.strip_columns_level

::: valax.surfaces.caplet_stripping.caplet_strip_residual_fn

::: valax.surfaces.caplet_stripping.cap_strip_labels

::: valax.surfaces.caplet_stripping.STRIP_SETTINGS

## Swaption cube (rates)

One SABR model per **(expiry, tenor)** node, with the strike dimension supplied
in closed form rather than tabulated. `is_normal` selects lognormal Black-76 or
normal Bachelier, the latter honouring a displacement `shift`.

The cube is calibrated **from the curves**: give
`calibrate_swaption_cube_from_curves` (or `_from_graph`) the node *schedules*
and the curves and it derives the grid axes and each node's $(S_x, A_d)$
itself, so the fitted parameters are differentiable functions of the curve
pillars. Each (expiry, tenor) cell is one slice of a batched level, labelled
`"5Yx10Y"` etc. Recipe and worked example:
[Volatility Surfaces § Swaption cube](../guide/vol-surfaces.md#swaption-cube-curve-aware-calibration).

::: valax.surfaces.SwaptionCube

::: valax.surfaces.calibrate_swaption_cube_from_curves

::: valax.surfaces.calibrate_swaption_cube_from_graph

::: valax.surfaces.swaption_cube_level

::: valax.surfaces.swaption_node_labels

::: valax.surfaces.swaption_node_coordinates

::: valax.surfaces.swaption_node_forwards_and_annuities

### Smile dynamics

Which forward enters the Hagan evaluation when the curve has moved away from
the fitted one. A declared static choice on both rates vol objects, validated
against the quote form. The three conventions, the default and the caveats are
documented once in
[Volatility Surfaces § Smile dynamics](../guide/vol-surfaces.md#smile-dynamics-what-moves-when-the-curve-moves).

::: valax.surfaces.dynamics

## Swaption premium → normal vol

A premium quote is expressed *against a numeraire*, and the numeraire is a
function of the curves — so converting one to a volatility is a dual-curve
operation, not a unit change. The conversion is a scalar Newton root level
(`INVERSION_SETTINGS` by default) returning a `CalibrationResult`, and is
implicitly differentiated, so the implied vol carries genuine curve risk. Physical and cash settlement are distinct conventions; see
[Volatility Surfaces § Premium quotes](../guide/vol-surfaces.md#premium-quotes).

::: valax.surfaces.swaption_premium

## Shifted surface (scenario decorator)

`ShiftedVolSurface` adds an additive **parallel** vol shift to any
duck-typed surface — grid, SVI, SABR, cube, optionlet, or constant —
without touching its shape. It is the vol leg of a keyed market
scenario: see
[Scenarios & Shocks § Keyed scenarios](../risk-scenarios.md#keyed-scenarios-on-a-marketstate)
for why the parallel-shift representation was chosen over node-level
shocks, and [`valax.risk.shift_surface`](risk.md#keyed-shocks-on-a-marketstate)
for the constructor that applies one (it shifts a `ConstantVol` in place
rather than wrapping it).

::: valax.surfaces.ShiftedVolSurface
