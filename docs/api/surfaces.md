# Surfaces

Volatility surface construction and calibration. All surfaces are callable `eqx.Module` pytrees with signature `surface(strike, expiry) -> implied_vol`.

## Grid Surface

### `GridVolSurface`

```python
class GridVolSurface(eqx.Module):
    strikes: Float[Array, "n_strikes"]
    expiries: Float[Array, "n_expiries"]
    vols: Float[Array, "n_expiries n_strikes"]
```

Bilinear interpolation with flat extrapolation. The `vols` grid is differentiable.

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
