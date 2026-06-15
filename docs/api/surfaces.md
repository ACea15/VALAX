# Surfaces

Volatility surface construction and calibration. All surfaces are callable `eqx.Module` pytrees with signature `surface(strike, expiry) -> implied_vol`.

## Dupire / SLV protocol: `total_variance`

All three surface types additionally expose

```python
surface.total_variance(log_moneyness, expiry) -> Float[Array, ""]
```

returning total implied variance $w(k, T) = \sigma_{\text{IV}}^2(k, T) \cdot T$ as a function of log-moneyness $k = \ln(K / F(T))$. This is the duck-typed input expected by [`dupire_local_vol`](pricing.md#dupire_local_vol) and (in due course) by SLV's leverage-function calibration. The consistency identity

```python
surface.total_variance(jnp.log(K / F_T), T)  ==  surface(K, T) ** 2 * T
```

holds to machine precision for every surface implementation, where `F_T = jnp.interp(T, surface.expiries, surface.forwards)` is the query-expiry forward (or, for `GridVolSurface`, the convention is documented per-surface below).

## Grid Surface

### `GridVolSurface`

```python
class GridVolSurface(eqx.Module):
    strikes: Float[Array, "n_strikes"]
    expiries: Float[Array, "n_expiries"]
    vols: Float[Array, "n_expiries n_strikes"]
```

Bilinear interpolation (via `valax.surfaces._interp.bilinear_2d`) with flat extrapolation. The `vols` grid is differentiable.

#### `total_variance` on a grid

For `GridVolSurface` the `strikes` axis is consumed directly as the log-moneyness axis — i.e. a caller using a grid surface for Dupire must build the grid in $(k, T)$ space rather than $(K, T)$. This avoids embedding a forward-curve assumption inside the grid surface itself. See the docstring on `GridVolSurface.total_variance` for the full convention.

---

## SABR Surface

### `SABRVolSurface`

```python
class SABRVolSurface(eqx.Module):
    expiries: Float[Array, "n_expiries"]
    forwards: Float[Array, "n_expiries"]
    alphas: Float[Array, "n_expiries"]
    betas: Float[Array, "n_expiries"]
    rhos: Float[Array, "n_expiries"]
    nus: Float[Array, "n_expiries"]
```

Linearly interpolates SABR parameters to the query expiry, then evaluates the Hagan formula. All parameters are differentiable.

#### `total_variance` on a SABR surface

`SABRVolSurface.total_variance(k, T)` converts back to an absolute strike via the query-expiry forward (`K = F_T · exp(k)` with `F_T = jnp.interp(T, expiries, forwards)`) and returns `surface(K, T) ** 2 * T`. Suitable as a Dupire input, but note that Hagan's IV formula is itself an asymptotic expansion — at extreme strikes the resulting "local vol" from Dupire becomes a measurement of the expansion's residual rather than a true market local vol.

### `calibrate_sabr_surface`

```python
calibrate_sabr_surface(
    strikes_per_expiry, market_vols_per_expiry,
    forwards, expiries, fixed_beta=None,
    solver="levenberg_marquardt", max_steps=256,
) -> SABRVolSurface
```

Fits each expiry slice independently via `calibrate_sabr`.

---

## SVI Surface

### `SVISlice`

```python
class SVISlice(eqx.Module):
    a: Float[Array, ""]      # variance level
    b: Float[Array, ""]      # wing slope (>= 0)
    rho: Float[Array, ""]    # asymmetry (-1, 1)
    m: Float[Array, ""]      # horizontal shift
    sigma: Float[Array, ""]  # vertex smoothing (> 0)
```

### `svi_total_variance`

```python
svi_total_variance(params: SVISlice, log_moneyness) -> Float[Array, ""]
```

$w(k) = a + b(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2})$

### `svi_implied_vol`

```python
svi_implied_vol(params, forward, strike, expiry) -> Float[Array, ""]
```

Returns $\sqrt{w(\log(K/F)) / T}$.

### `SVIVolSurface`

```python
class SVIVolSurface(eqx.Module):
    expiries: Float[Array, "n_expiries"]
    forwards: Float[Array, "n_expiries"]
    a_vec, b_vec, rho_vec, m_vec, sigma_vec: Float[Array, "n_expiries"]
```

Interpolates total variance (not vol) across expiries to preserve calendar spread no-arbitrage.

#### `total_variance` on an SVI surface

`SVIVolSurface.total_variance(k, T)` is the natural Dupire input — total variance is the *intrinsic* SVI quantity, so no double `sqrt → square` roundtrip is needed. The method evaluates each calibrated slice's SVI total variance at the slice-relative log-moneyness `log(K / F_slice)`, where `K = F_T · exp(k)`, then linearly interpolates across slices in `T`. Identity to `__call__(K, T) ** 2 * T` holds exactly.

#### Short-maturity extrapolation

Both `__call__` and `total_variance` extrapolate **linearly in T through the origin** below the first calibrated expiry (i.e. holding implied vol constant as $T \to 0^+$). This differs from `jnp.interp`'s flat-w default. The change makes Dupire's $\partial w/\partial T$ positive at short expiries — without it, the $1/w$ terms in the Dupire denominator diverge and any LV MC stepping through the boundary takes a zero-vol first step. See [theory §4.4](../theory.md#44-local-volatility-dupire).

### `calibrate_svi_slice`

```python
calibrate_svi_slice(strikes, market_vols, forward, expiry,
                    initial_guess=None, weights=None,
                    max_steps=256) -> (SVISlice, Solution)
```

Levenberg-Marquardt fit of SVI to a single smile.

### `calibrate_svi_surface`

```python
calibrate_svi_surface(strikes_per_expiry, market_vols_per_expiry,
                      forwards, expiries, max_steps=256) -> SVIVolSurface
```

Fits each expiry slice independently.
