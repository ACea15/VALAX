# Models

Stochastic process definitions used by the Monte Carlo engine. Each
model is an `equinox.Module` carrying the process parameters and, in
most cases, companion `Drift` / `Diffusion` callables that plug into
the [`diffrax`](https://docs.kidger.site/diffrax/) SDE solvers.

!!! note "Constants or market-sourced"
    `BlackScholesModel`, `HestonModel` and `LocalVolModel` accept
    `rate` / `dividend` (and BS `vol`) either as explicit constants — the
    textbook form every closed form below assumes — or as `None`, in which
    case they are read from the `MarketState` at price time. The two modes,
    the projection conventions, and what it means for risk are documented
    once in
    [API → Core § Model binding](core.md#model-binding-constants-or-market-sourced).

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
    [`sabr_price`](pricing.md#valax.pricing.analytic.sabr_price)); calibration lives in
    [`calibrate_sabr`][valax.calibration.calibrate_sabr].

::: valax.models.SABRModel

## Market implied vol (rates options)

No dynamics: the market's quoting convention. `ImpliedVolModel(surface_id)`
prices caplets, caps and European swaptions through `valax.price` at the
implied vol read off the surface filed under `surface_id` — a
`SwaptionCube`, an `OptionletVolSurface`, or a `ConstantVol` quoted normal
(`is_normal=True`, Bachelier) or lognormal (Black-76) — at each instrument's
own coordinate and live forward, by the curve-aware kernels of
[`rates_smile`](pricing.md). It holds no numbers, so every sensitivity lands
on the curves and the surface. `swaption_basket_node` prices its targets
through it.

::: valax.models.ImpliedVolModel

## Local vol (Dupire)

State- and time-dependent diffusion calibrated to fit an implied-vol
surface by construction:

\[
    dS = (r - q)\, S\, dt + \sigma_{\text{loc}}(S, t)\, S\, dW,
\]

with \(\sigma_{\text{loc}}\) extracted from the surface on demand via
Gatheral's IV-space Dupire formula
([`dupire_local_vol`](pricing.md#valax.pricing.analytic.dupire_local_vol)). Any object
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
    [theory §4.4](../theory/volatility.md#44-local-volatility-dupire) for the
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
See [theory §4.5](../theory/volatility.md#45-stochastic-local-volatility) and
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
    [`valax.pricing.lattice`](pricing.md#callable-puttable-bonds) for
    trinomial-tree backward induction.

!!! note "Curve-free models, curve bound at pricing time"
    Hull–White, G2++ and Cheyette (1F/2F) carry
    `initial_curve: DiscountCurve | None = None` and a static
    `curve_id: str | None = None`. Construct them **without** a curve for
    `valax.price(...)`, which binds `initial_curve` from the `MarketState`
    (the curve `curve_id`, else the instrument's discount curve) and refuses
    a pre-bound model. Calibrators return curve-free models. Kernel-level
    functions (`hw_bond_price`, `hw_swaption_price`, PDE/MC recipes called
    directly, tree builders) need a bound model — bind with
    [`bind_curves`](core.md#valax.core.binding.bind_curves); an unbound
    model raises through `require_initial_curve`. See
    [API → Core § Curve fields](core.md#curve-fields-exact-fit-short-rate-models).

::: valax.models.HullWhiteModel

::: valax.models.short_rate.require_initial_curve

### Analytics

::: valax.models.hull_white.hw_bond_price

::: valax.models.hull_white.hw_B

::: valax.models.hull_white.hw_short_rate_variance

::: valax.models.hull_white.hw_state_variance

::: valax.models.hull_white.hw_mean_variance_rate

### Piecewise-constant volatility helpers

Shared by Hull–White, G2++ and Cheyette (`valax.models.short_rate`): bucket
construction and validation at the user boundary, and the closed-form
integrals of a piecewise-constant function that every Gaussian quantity of
these models is built from.

::: valax.models.short_rate.as_buckets

::: valax.models.short_rate.default_bucket_times

::: valax.models.short_rate.check_buckets

::: valax.models.short_rate.is_constant

::: valax.models.short_rate.bucket_index

::: valax.models.short_rate.volatility_bucket_fields

::: valax.models.short_rate.decayed_integral

::: valax.models.short_rate.piecewise_integral

## G2++ (two-factor Gaussian short-rate)

The two-additive-factor Gaussian model (equivalent to two-factor Hull–White).
A second stochastic factor lets rates of different maturities **decorrelate**,
unlocking CMS-spread options, steepeners/flatteners and decorrelation-sensitive
Bermudans that a one-factor model structurally cannot price:

\[
    r(t) = x(t) + y(t) + \varphi(t), \qquad
    dx = -a\,x\,dt + \sigma(t)\,dW_1, \quad
    dy = -b\,y\,dt + \eta(t)\,dW_2, \quad
    dW_1\,dW_2 = \rho\,dt.
\]

!!! note "Constructor: `volatility_x` and `volatility_ratio`"
    The model is parameterised by the first-factor volatility `volatility_x`
    (\(\sigma\), per bucket) and the **ratio** `volatility_ratio`
    (\(\eta / \sigma\), per bucket, sharing `volatility_times`; a one-bucket
    ratio is broadcast to the bucket count of `volatility_x`). The
    second-factor volatility \(\eta(t) = \text{ratio}(t)\,\sigma(t)\) is the
    read-only property `volatility_y` — it is **not** a constructor argument.

!!! success "Exact fit"
    At \(t = 0\) with \(x = y = 0\), `g2pp_bond_price` recovers the initial
    curve \(P^M(0, T)\) to machine precision by construction, via the shift
    \(\varphi\).

!!! note "\(\rho\) is an input, not the decorrelation"
    The tenor-rate decorrelation G2++ delivers is a model *output* bounded by
    \(\rho\): it vanishes as \(|\rho| \to 1\) (the factors collapse to a single
    shock) and is largest, typically with \(\rho\) strongly negative, when short
    and long rates should move most independently. See
    [G2++ theory §2](../theory/g2pp.md#2-decorrelation-and-to-what-extent).

::: valax.models.G2PPModel

### Analytics

::: valax.models.g2pp.g2pp_bond_price

::: valax.models.g2pp.g2pp_V

::: valax.models.g2pp.g2pp_phi

::: valax.models.g2pp.g2pp_B

::: valax.models.g2pp.g2pp_factor_covariance

::: valax.models.g2pp.g2pp_short_rate_variance

::: valax.models.g2pp.g2pp_phi_integral

::: valax.models.g2pp.g2pp_diffusion_rates

::: valax.models.g2pp.g2pp_state_covariance

::: valax.models.g2pp.g2pp_forward_measure_moments

## Cheyette (quasi-Gaussian short-rate)

The Markovian HJM model under a separable volatility. A **state-dependent**
local volatility $\sigma_r(t, x)$ gives rates a volatility **skew** — the
dynamic, low-dimensional smile the Gaussian short-rate models (Hull–White,
G2++) structurally cannot produce, and which a lattice or PDE can roll backward
for callable exotics:

\[
    r(t) = f^M(0, t) + x(t), \qquad
    dx = (y - \kappa x)\,dt + \sigma_r(t, x)\,dW, \quad
    dy = (\sigma_r^2 - 2\kappa y)\,dt,
\]

with the displaced-diffusion local vol $\sigma_r(t, x) = \sigma(t)(1 + \beta x)$,
whose level $\sigma(t)$ is piecewise constant (`volatility` per bucket plus
`volatility_times`, as for Hull–White). The two-factor `CheyetteModel2F` adds
decorrelation on top of the skew; it takes `volatility_1` (per bucket) and
`volatility_ratio` ($\sigma_2 = \text{ratio}\cdot\sigma_1$, read back as the
property `volatility_2`).

!!! success "Exact fit"
    With $r = f^M(0, t) + x$ and $x(0) = 0$, the convexity is carried
    dynamically by the $y$-drift, so `cheyette_bond_price` recovers the initial
    curve $P^M(0, T)$ at $t = 0$ to machine precision with no drift calibration.

!!! note "Affine bond for any local vol"
    $\sigma_r$ enters only the dynamics of $(x, y)$, never the bond formula, so
    the zero-coupon bond stays affine:
    $P(t, T) = \tfrac{P^M(0,T)}{P^M(0,t)} e^{-x G - \tfrac12 y G^2}$ with
    $G = (1 - e^{-\kappa(T-t)})/\kappa$. At $\beta = 0$ the model reduces exactly
    to Hull–White (2F to G2++) — the primary validation oracle. See
    [Cheyette theory](../theory/cheyette.md).

::: valax.models.CheyetteModel

::: valax.models.CheyetteModel2F

### Analytics

::: valax.models.cheyette.cheyette_bond_price

::: valax.models.cheyette.cheyette_bond_price_2f

::: valax.models.cheyette.cheyette_G

::: valax.models.cheyette.cheyette_local_vol

::: valax.models.cheyette.cheyette_y_gaussian

::: valax.models.cheyette.cheyette_gaussian_mean

::: valax.models.cheyette.cheyette_mean_variance_rate

::: valax.models.cheyette.cheyette_mean_variance_rates_2f

## LIBOR Market Model (LMM / BGM)

Brace–Gatarek–Musiela LIBOR Market Model for forward-rate simulation:

\[
    dF_i = \mu_i\, dt + \sigma_i(t)\, F_i \sum_j L_{ij}\, dW_j,
\]

where \(L\) is the Cholesky (or PCA) factor of the correlation matrix
and \(\mu_i\) is the spot-measure drift. Path simulation uses
log-Euler for guaranteed positivity; used by the Bermudan swaption
pricer (Longstaff–Schwartz on LMM paths).

The model is curve-free (`initial_curve=None`): `price()` binds the curve
from the `MarketState` and the initial forwards are derived from it, and
`build_lmm_model` returns a bound model for kernel-level calls
([Monte Carlo § LMM through `price()`](../guide/monte-carlo.md#lmm-through-price)).

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
[`SpreadOption`](instruments.md#valax.instruments.SpreadOption) (validating Margrabe /
Kirk analytical) and
[`WorstOfBasketOption`](instruments.md#valax.instruments.WorstOfBasketOption). The
Cholesky factor of `correlation` is computed inside
`generate_correlated_gbm_paths`; for repeated repricing at a fixed
correlation, wrap in `jax.jit` to amortize.

::: valax.models.MultiAssetGBMModel

### Correlation validator

::: valax.models.validate_correlation
