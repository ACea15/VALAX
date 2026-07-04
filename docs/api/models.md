# Models

Stochastic process definitions used by the Monte Carlo engine. Each
model is an `equinox.Module` carrying the process parameters and, in
most cases, companion `Drift` / `Diffusion` callables that plug into
the [`diffrax`](https://docs.kidger.site/diffrax/) SDE solvers.

## Black–Scholes (GBM)

Geometric Brownian motion — the workhorse spot dynamics for equity
and FX pricing:

\[
    dS = (r - q)\, S\, dt + \sigma\, S\, dW
\]

::: valax.models.BlackScholesModel

## Heston (stochastic vol)

Mean-reverting stochastic variance — the standard first move past
Black-Scholes for capturing volatility smile and skew:

\[
    dS = (r - q)\, S\, dt + \sqrt{V}\, S\, dW_1, \qquad
    dV = \kappa(\theta - V)\, dt + \xi\, \sqrt{V}\, dW_2,
\]

with \(\text{Corr}(dW_1, dW_2) = \rho\).

!!! note "Path scheme"
    Path generation works in log-spot space for numerical stability.
    Variance is floored at zero (full-truncation / absorption scheme).
    All parameters are differentiable; calibration lives in
    [`calibrate_heston`][valax.calibration.calibrate_heston].

::: valax.models.HestonModel

## SABR (stochastic vol for forwards)

Stochastic-Alpha-Beta-Rho model, standard for interest-rate smile
under the forward measure:

\[
    dF = \alpha\, F^{\beta}\, dW_1, \qquad
    d\alpha_t = \nu\, \alpha_t\, dW_2, \qquad
    \text{Corr}(dW_1, dW_2) = \rho.
\]

!!! note "Backbone"
    `beta` controls the backbone — \(\beta = 0\) gives normal
    dynamics, \(\beta = 1\) gives lognormal. Equity typically uses
    \(\beta = 0.5\); rates often uses \(\beta = 0\). Pricing goes via
    Hagan's asymptotic implied-vol formula fed into Black-76 (see
    [`sabr_price`](pricing.md#sabr_price)); calibration lives in
    [`calibrate_sabr`][valax.calibration.calibrate_sabr].

::: valax.models.SABRModel

## Local vol (Dupire)

State- and time-dependent diffusion calibrated to fit an implied-vol
surface by construction:

\[
    dS = (r - q)\, S\, dt + \sigma_{\text{loc}}(S, t)\, S\, dW,
\]

with \(\sigma_{\text{loc}}\) extracted from the surface on demand via
Gatheral's IV-space Dupire formula
([`dupire_local_vol`](pricing.md#dupire_local_vol)). Any object
exposing `total_variance(log_moneyness, expiry) -> Float[""]`
(`SVIVolSurface`, `SABRVolSurface`, `GridVolSurface`) satisfies the
duck-typed surface contract.

!!! warning "Precision"
    Requires `jax_enable_x64=True` (enforced at the Dupire layer —
    second derivatives of total variance are precision-sensitive).
    `valax/__init__.py` enables x64 globally on import.

!!! note "Forward curve"
    The implicit forward is \(F(t) = S_0\, e^{(r - q) t}\) — a
    deterministic-rate equity forward. Term-structured rates and
    dividends are a planned extension via a `forward_curve: Callable`
    field. Path generation uses `generate_local_vol_paths` (log-Euler
    with midpoint-in-time \(\sigma\)); see
    [theory §4.4](../theory.md#44-local-volatility-dupire) for the
    midpoint-vs-left-endpoint choice.

::: valax.models.LocalVolModel

## Stochastic-Local Vol (SLV)

Heston stochastic vol times a calibrated leverage function
\(L(k, t)\), with the leverage chosen so the SLV marginals reproduce a
calibrated implied-vol surface by Markovian projection:

\[
    \frac{dS_t}{S_t} = (r - q)\, dt + L(k_t, t)\, \sqrt{V_t}\, dW_1, \qquad
    dV_t = \kappa(\theta - V_t)\, dt + \xi\, \sqrt{V_t}\, dW_2,
\]

with \(k_t = \ln(S_t / F(t))\) and \(\langle dW_1, dW_2 \rangle = \rho\, dt\).
The leverage function comes from pass 2 of the SLV calibration
([`calibrate_slv_leverage`][valax.calibration.calibrate_slv_leverage]).
See [theory §4.5](../theory.md#45-stochastic-local-volatility) and
the [SLV guide](../guide/slv.md) for the underlying mathematics.

!!! note "Path generator never queries `surface`"
    `generate_slv_paths` reads only `leverage`; the `surface` field is
    kept on the model so leverage can be re-calibrated against the
    same target without external bookkeeping. Requires
    `jax_enable_x64=True` (enforced at construction).

::: valax.models.SLVModel

## Hull–White (short-rate)

Extended Vasicek one-factor short-rate — the workhorse of rates desks
for callables, puttables, Bermudan swaptions, and IR exotics:

\[
    dr(t) = [\theta(t) - a\, r(t)]\, dt + \sigma\, dW(t).
\]

!!! success "Exact fit"
    At \(t = 0\) with \(r = f^M(0, 0)\), `hw_bond_price` recovers the
    initial curve \(P^M(0, T)\) to machine precision by construction.
    Used by `callable_bond_price` and `puttable_bond_price` in
    [`valax.pricing.lattice`](pricing.md#callable--puttable-bonds) for
    trinomial-tree backward induction.

::: valax.models.HullWhiteModel

### Analytics

::: valax.models.hull_white.hw_bond_price

::: valax.models.hull_white.hw_B

::: valax.models.hull_white.hw_short_rate_variance

## LIBOR Market Model (LMM / BGM)

Brace–Gatarek–Musiela LIBOR Market Model for forward-rate simulation:

\[
    dF_i = \mu_i\, dt + \sigma_i(t)\, F_i \sum_j L_{ij}\, dW_j,
\]

where \(L\) is the Cholesky (or PCA) factor of the correlation matrix
and \(\mu_i\) is the spot-measure drift. Path simulation uses
log-Euler for guaranteed positivity; used by the Bermudan swaption
pricer (Longstaff–Schwartz on LMM paths).

::: valax.models.LMMModel

::: valax.models.build_lmm_model

### Volatility structures

::: valax.models.PiecewiseConstantVol

::: valax.models.RebonatoVol

### Correlation structures

::: valax.models.ExponentialCorrelation

::: valax.models.TwoParameterCorrelation

## Multi-asset GBM

\(N\)-asset correlated geometric Brownian motion under a single
risk-neutral measure:

\[
    dS_i(t) = (r - q_i)\, S_i(t)\, dt + \sigma_i\, S_i(t)\, dW_i(t),
    \qquad \langle dW_i, dW_j \rangle = \rho_{ij}\, dt.
\]

Powers the multi-asset MC recipes for
[`SpreadOption`](instruments.md#spreadoption) (validating Margrabe /
Kirk analytical) and
[`WorstOfBasketOption`](instruments.md#worstofbasketoption). The
Cholesky factor of `correlation` is computed inside
`generate_correlated_gbm_paths`; for repeated repricing at a fixed
correlation, wrap in `jax.jit` to amortize.

::: valax.models.MultiAssetGBMModel

### Correlation validator

::: valax.models.validate_correlation
