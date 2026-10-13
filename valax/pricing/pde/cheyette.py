r"""Cheyette (quasi-Gaussian) short-rate finite-difference pricing.

Prices interest-rate instruments with embedded optionality under the Cheyette
quasi-Gaussian model by solving its backward PDE on the existing 1-D theta-scheme
and 2-D ADI substrates built for Hull-White and G2++. It is the PDE companion to
the exact-``y`` Monte-Carlo engine (:mod:`valax.pricing.mc.cheyette_paths`), and
the route through which the model is *calibrated* (:mod:`valax.calibration.cheyette`).

The accumulator problem
-----------------------
The Cheyette state is the pair :math:`(x, y)` in one factor and the quintuple
:math:`(x_1, x_2, y_{11}, y_{22}, y_{12})` in two — so a naive PDE would be 2-D
(1F) or 5-D (2F). The auxiliary :math:`y` is **advection-only** (a pure-drift
accumulator, no Brownian term), which is what makes lower-dimensional treatments
possible. This module ships two tiers:

**Tier 1 — frozen-``y`` (the production engine).** Replace the stochastic
accumulator by its deterministic Gaussian-limit value
:math:`\bar y(t)` (:func:`~valax.models.cheyette.cheyette_y_gaussian`), so 1F
becomes a genuine **1-D** PDE in the factor and 2F a **2-D** ADI PDE — reusing
the Hull-White / G2++ substrate almost verbatim. Working in the *centred* factor
:math:`\tilde x_i = x_i - m_i(t)` with
:math:`m = ` :func:`~valax.models.cheyette.cheyette_gaussian_mean` recovers
exactly the Hull-White / G2++ centred dynamics
:math:`d\tilde x_i = -\kappa_i \tilde x_i\,dt + \sigma_{r,i}\,dW_i`, so the
centred grid, the linearity boundary condition and the exact-fit discount
factoring all apply directly. The **only** genuinely new ingredient is the
diffusion coefficient
:math:`\tfrac12\sigma_r(t, \tilde x + m)^2 = \tfrac12\sigma^2(1 + \beta(\tilde x + m))^2`,
which is *spatially varying* (Hull-White / G2++ have a constant
:math:`\tfrac12\sigma^2`); the operator builders already accept a per-node
diffusion field, so this needs only a new coefficient adapter.

**Tier 2 — exact-``y`` (one-factor validation oracle).** Carry :math:`y` as a
real, advection-only axis, giving a genuine **2-D** PDE in :math:`(x, y)`. This
is *not* an approximation: at :math:`\beta \neq 0` it reproduces the true-``y``
model, so the difference from Tier 1 is a clean, deterministic (zero-Monte-Carlo
noise) measurement of the frozen-``y`` bias. It is only offered for one factor
(the two-factor exact-``y`` PDE is 5-D — see the research note in
``docs/research-ideas.md``). The :math:`y`-axis is hyperbolic (advection only),
so it uses **first-order upwinding** for stability, unlike the parabolic
:math:`x`-axis.

Reduction oracle
----------------
At :math:`\beta = 0` frozen-``y`` is exact and the volatility is constant, so the
Cheyette PDE reduces to the Hull-White (1F) / G2++ (2F) PDE — the primary
validation, and through it to QuantLib's ``FdHullWhite*`` / ``FdG2`` engines.
There is no QuantLib Cheyette engine.

References:
    Andersen, L. & Piterbarg, V. (2010), *Interest Rate Modeling*, ch. 13.
    Cheyette, O. (1992), "Markov Representation of the HJM Model".
"""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from valax.dates.accrual import (
    bond_accrued_interest_at_dates,
    fixed_coupon_amounts,
)
from valax.dates.daycounts import year_fraction
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.instruments.rates import (
    BermudanSwaption,
    Cap,
    Caplet,
    CMSSpreadSwap,
    Swaption,
)
from valax.models.cheyette import (
    CheyetteModel,
    CheyetteModel2F,
    cheyette_G,
    cheyette_gaussian_mean,
    cheyette_gaussian_mean_2f,
    cheyette_market_df,
    cheyette_y_gaussian,
    cheyette_y_gaussian_2f,
)
from valax.models.g2pp import G2PPModel
from valax.models.hull_white import HullWhiteModel, hw_alpha_average
from valax.pricing.pde.boundary import (
    Boundary2D,
    apply_linearity_bc_1d,
    apply_linearity_bc_2d,
    zero_boundary,
)
from valax.pricing.pde.config import PDEConfig, PDEConfig2D, Scheme
from valax.pricing.pde.dispatch import PDEResult, register
from valax.pricing.pde.g2pp import _phi_step_discounts
from valax.pricing.pde.grids import (
    Grid1D,
    Grid2D,
    boundary_coords,
    centred_state_grid,
    read_off_1d,
    read_off_2d,
    uniform_linear_grid,
)
from valax.pricing.pde.operators import Operator1D, build_operator_1d
from valax.pricing.pde.operators2d import Operator2D, build_operator_2d
from valax.pricing.pde.payoff_smoothing import positive_part
from valax.pricing.pde.rates_payoffs import (
    bermudan_terminal_and_event,
    caplet_fixing_values,
    caplet_values_by_level,
    cms_spread_values_by_level,
    coupon_values_by_level,
    exercisable_terminal_and_event,
    exercise_obstacle_by_level,
    snap_to_levels,
    swap_tail_values,
)
from valax.pricing.pde.schemes import solve_backward_1d, theta_for_scheme
from valax.pricing.pde.schemes2d import solve_backward_2d

# Minimum fully-implicit start-up steps. Every instrument here has a kinked
# terminal payoff (swaptions) or a projection that reintroduces a kink
# (callables / Bermudan), both of which excite Crank-Nicolson / Craig-Sneyd
# oscillations.
_MIN_RANNACHER = 2

_Event1D = Callable[[Int[Array, ""], Float[Array, " n"]], Float[Array, " n"]]
_Event2D = Callable[
    [Int[Array, ""], Float[Array, "n_x n_y"]], Float[Array, "n_x n_y"]
]


# ─────────────────────────────────────────────────────────────────────
# Schedule helpers (local copies of the Hull-White / G2++ recipes', kept
# private so the short-rate PDE modules stay independent).
# ─────────────────────────────────────────────────────────────────────


def _ln_market_df(
    model: CheyetteModel | CheyetteModel2F,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    r"""Log market discount factor :math:`\ln P^M(0, t)` at year fraction(s)."""
    return jnp.log(cheyette_market_df(model, t))


# ═════════════════════════════════════════════════════════════════════
# Tier 1 — frozen-y, one factor (1-D PDE)
# ═════════════════════════════════════════════════════════════════════


def _hw_equivalent(model: CheyetteModel) -> HullWhiteModel:
    r"""Gaussian-limit Hull-White twin sharing :math:`\kappa`, :math:`\sigma`, curve.

    The deterministic discount shift of the frozen-``y`` Cheyette model,
    :math:`\alpha_{\mathrm{Ch}}(t) = f^M(0,t) + m(t)`, is *identically* the
    Hull-White :math:`\alpha(t)` of this twin (the Cheyette Gaussian-limit mean
    :math:`m(t)` equals the Hull-White convexity), so its exact per-step average
    can be reused verbatim — which is also what makes the :math:`\beta = 0`
    reduction exact at the operator level.

    Args:
        model: One-factor Cheyette model.

    Returns:
        The matching :class:`~valax.models.hull_white.HullWhiteModel`.
    """
    return HullWhiteModel(
        mean_reversion=model.mean_reversion,
        volatility=model.volatility,
        initial_curve=model.initial_curve,
    )


def cheyette_operator_stack(
    model: CheyetteModel,
    grid: Grid1D,
    *,
    expiry: Float[Array, ""],
    n_time: int,
) -> Operator1D:
    r"""Frozen-``y`` one-factor Cheyette operator stack in the centred factor.

    Works in the centred factor :math:`\tilde x = x - m(t)`, where the dynamics
    are the Hull-White centred OU dynamics
    :math:`d\tilde x = -\kappa \tilde x\,dt + \sigma_r(t, \tilde x + m)\,dW`. The
    backward pricing equation for :math:`V(\tilde x, \tau)` is

    .. math::

        V_\tau = \tfrac12 \sigma_r(t, \tilde x + m)^2\, V_{\tilde x\tilde x}
                 - \kappa \tilde x\, V_{\tilde x}
                 - \bigl(\tilde x + \alpha_{\mathrm{Ch}}(t)\bigr) V ,

    with :math:`\sigma_r(t, \tilde x + m) = \sigma(1 + \beta(\tilde x + m(t)))`
    and :math:`\alpha_{\mathrm{Ch}}(t) = f^M(0,t) + m(t)`.

    Only the discount and the (skew-driven) diffusion vary in time, so this
    returns a **stacked** operator of shape ``(n_time, n)``: row ``m`` is the
    operator used at backward step ``m`` (nearest expiry first). The discount's
    deterministic part :math:`\alpha_{\mathrm{Ch}}` is exactly step-averaged via
    the Hull-White twin's :func:`~valax.models.hull_white.hw_alpha_average`
    (preserving the exact fit); the diffusion samples the Gaussian-limit mean
    :math:`m` at each step's midpoint. At :math:`\beta = 0` the diffusion
    collapses to the constant :math:`\sigma^2` and this reduces bit-for-bit to
    :func:`~valax.pricing.pde.coefficients.hw_operator_stack`.

    Args:
        model: One-factor Cheyette model.
        grid: The centred-factor (:math:`\tilde x`) grid.
        expiry: Horizon ``T`` in year fractions (``dt = T / n_time``).
        n_time: Number of backward time steps (== number of stacked rows).

    Returns:
        An :class:`~valax.pricing.pde.operators.Operator1D` with bands of shape
        ``(n_time, n)``.
    """
    kappa = model.mean_reversion
    sigma = model.volatility
    beta = model.skew
    x = grid.nodes

    dt = expiry / n_time
    # Backward step m (m = 0 nearest expiry) spans forward levels k .. k+1.
    k = n_time - jnp.arange(n_time) - 1
    hw_twin = _hw_equivalent(model)
    alphas = hw_alpha_average(hw_twin, k * dt, (k + 1) * dt)  # (n_time,)
    m_mid = cheyette_gaussian_mean(model, (k + 0.5) * dt)     # (n_time,)

    drift = -kappa * x

    def _row_operator(
        alpha: Float[Array, ""], m: Float[Array, ""]
    ) -> Operator1D:
        sigma_r = jnp.maximum(sigma * (1.0 + beta * (x + m)), 0.0)
        return build_operator_1d(
            grid, drift=drift, diffusion=sigma_r**2, discount=x + alpha
        )

    return jax.vmap(_row_operator)(alphas, m_mid)


def _cheyette_coupon_bond_values(
    model: CheyetteModel,
    nodes: Float[Array, " n"],
    t: Float[Array, ""],
    payment_times: Float[Array, " n_cf"],
    cashflows: Float[Array, " n_cf"],
) -> Float[Array, " n"]:
    r"""Frozen-``y`` value of :math:`\sum_i c_i P(t, T_i)` at every centred node.

    Uses the quasi-Gaussian affine bond price with the accumulator frozen to its
    Gaussian-limit value :math:`\bar y(t)` and the actual factor recovered from
    the centred node as :math:`x = \tilde x + m(t)`:

    .. math::

        P(t, T \mid \tilde x)
            = \frac{P^M(0, T)}{P^M(0, t)}
              \exp\!\Bigl[-(\tilde x + m)\,G(t, T)
                          - \tfrac12 \bar y(t)\,G(t, T)^2\Bigr].

    Args:
        model: One-factor Cheyette model.
        nodes: Centred-factor (:math:`\tilde x`) grid coordinates.
        t: Valuation time in year fractions.
        payment_times: Cashflow times in year fractions.
        cashflows: Cashflow amounts.

    Returns:
        The portfolio value at each grid node.
    """
    kappa = model.mean_reversion
    m_t = cheyette_gaussian_mean(model, t)
    y_t = cheyette_y_gaussian(model, t)
    x = nodes + m_t                                            # (n,)

    G = cheyette_G(kappa, payment_times - t)                   # (n_cf,)
    ln_ratio = _ln_market_df(model, payment_times) - _ln_market_df(model, t)
    exponent = (
        ln_ratio[jnp.newaxis, :]
        - x[:, jnp.newaxis] * G[jnp.newaxis, :]
        - 0.5 * y_t * G[jnp.newaxis, :] ** 2
    )
    return jnp.exp(exponent) @ cashflows


def _cheyette_swap_tail_values(
    model: CheyetteModel,
    nodes: Float[Array, " n"],
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    day_count: str,
    is_payer: bool,
) -> Float[Array, " n"]:
    r"""Value of the underlying swap at ``exercise_date``, at every centred node.

    On unit notional a fixed-for-float swap entered at :math:`T_0` is worth
    :math:`1 - \sum_i c_i P(T_0, T_i)` to the payer, with :math:`c_i = K\tau_i`
    and the principal folded into the final flow; the receiver's value is the
    negative. Uses the frozen-``y`` affine bond price.

    Args:
        model: One-factor Cheyette model.
        nodes: Centred-factor grid coordinates.
        exercise_date: Swap start / option exercise date as an integer ordinal.
        payment_dates: Fixed-leg payment dates as integer ordinals.
        strike: Fixed rate.
        notional: Notional principal.
        day_count: Day-count convention.
        is_payer: ``True`` for a payer swap, ``False`` for a receiver.

    Returns:
        The swap value at each grid node.
    """
    reference_date = model.initial_curve.reference_date

    def bond_portfolio(t, payment_times, cashflows):
        return _cheyette_coupon_bond_values(
            model, nodes, t, payment_times, cashflows
        )

    return swap_tail_values(
        bond_portfolio,
        reference_date,
        exercise_date,
        payment_dates,
        strike,
        notional,
        day_count,
        is_payer,
    )


def _bond_horizon(bond, model) -> Float[Array, ""]:
    """Year fraction from the curve reference date to the bond's maturity."""
    return year_fraction(
        model.initial_curve.reference_date,
        bond.payment_dates[-1],
        bond.day_count,
    )


def _cheyette_coupon_values_by_level(
    bond,
    model: CheyetteModel,
    grid: Grid1D,
    config: PDEConfig,
    horizon: Float[Array, ""],
) -> Float[Array, "n_levels n"]:
    r"""Coupon value to inject at each time level, per centred-factor node.

    A coupon due at :math:`t_c` is attached to the nearest time level
    :math:`t_k` and scaled by the frozen-``y`` analytic discount factor
    :math:`P(t_k, t_c \mid \tilde x)`, removing the cashflow-timing error that
    plain snapping introduces.

    Args:
        bond: A fixed-coupon bond instrument.
        model: One-factor Cheyette model.
        grid: The centred-factor grid.
        config: 1-D grid configuration.
        horizon: Time to the bond's final payment, in year fractions.

    Returns:
        Coupon value indexed by ``(time level, node)``; zero where none is due.
    """
    reference_date = model.initial_curve.reference_date
    dt = horizon / config.n_time
    coupon = fixed_coupon_amounts(bond)

    def factor(level_time, payment_time):
        return _cheyette_coupon_bond_values(
            model,
            grid.nodes,
            level_time,
            payment_time[jnp.newaxis],
            jnp.ones((1,)),
        )

    return coupon_values_by_level(
        factor,
        reference_date,
        bond.payment_dates,
        bond.day_count,
        coupon,
        dt,
        config.n_time,
        (grid.n,),
    )


def _solve_cheyette_1f(
    model: CheyetteModel,
    config: PDEConfig,
    horizon: Float[Array, ""],
    build: Callable[[Grid1D], tuple[Float[Array, " n"], Optional[_Event1D]]],
) -> Float[Array, ""]:
    r"""Run the frozen-``y`` 1F backward sweep and read off at :math:`\tilde x = 0`.

    Sizes the centred grid from the Gaussian-limit factor variance
    :math:`\bar y(T)` (== the Hull-White short-rate variance at :math:`\beta=0`),
    folds the linearity boundary condition into the stacked operator, sweeps, and
    reads off at the origin (the factor starts at :math:`\tilde x = 0`).

    Args:
        model: One-factor Cheyette model.
        config: 1-D grid configuration.
        horizon: Time to the last modelled date, in year fractions.
        build: Given the grid, returns ``(terminal, event_fn)``.

    Returns:
        The price at the curve reference date.
    """
    std_dev = jnp.sqrt(cheyette_y_gaussian(model, horizon))
    grid = centred_state_grid(
        std_dev, n=config.n_spot, half_width=config.spot_range
    )
    operator = apply_linearity_bc_1d(
        cheyette_operator_stack(model, grid, expiry=horizon, n_time=config.n_time),
        grid,
    )
    terminal, event_fn = build(grid)
    values = solve_backward_1d(
        operator,
        zero_boundary(),
        terminal,
        expiry=horizon,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=max(config.rannacher_steps, _MIN_RANNACHER),
        event_fn=event_fn,
    )
    return read_off_1d(grid, values, jnp.zeros(()))


def _callable_style_price_1f(
    bond,
    model: CheyetteModel,
    config: PDEConfig,
    exercise_dates: Int[Array, " m"],
    exercise_prices: Float[Array, " m"],
    *,
    is_call: bool,
) -> Float[Array, ""]:
    """Shared frozen-``y`` sweep for callable / puttable fixed-rate bonds."""
    horizon = _bond_horizon(bond, model)
    dt = horizon / config.n_time
    strike_at_level = exercise_obstacle_by_level(
        exercise_dates,
        exercise_prices,
        bond.face_value,
        model.initial_curve.reference_date,
        bond.day_count,
        dt,
        config.n_time,
        is_call,
        accrued_at_exercise=bond_accrued_interest_at_dates(bond, exercise_dates),
    )

    def build(grid: Grid1D):
        coupons = _cheyette_coupon_values_by_level(
            bond, model, grid, config, horizon
        )
        return exercisable_terminal_and_event(
            coupons, bond.face_value, strike_at_level, is_call,
            smooth=config.smooth_payoff,
        )

    return _solve_cheyette_1f(model, config, horizon, build)


# ── Recipes (Tier 1, one factor) ──────────────────────────────────────


@register(FixedRateBond, CheyetteModel)
def _fixed_rate_bond_cheyette(*, instrument, model, config) -> PDEResult:
    """Option-free fixed-rate bond under one-factor Cheyette (frozen-``y``).

    The bond has no optionality, so the answer is fixed by the initial curve
    alone: the sharpest check on the scheme's numerics and its exact fit. At
    :math:`\\beta \\neq 0` a small frozen-``y`` repricing bias appears (the
    frozen accumulator no longer offsets the state convexity exactly); at
    :math:`\\beta = 0` it reprices to the scheme's ~1e-6 tolerance.
    """
    horizon = _bond_horizon(instrument, model)

    def build(grid: Grid1D):
        coupons = _cheyette_coupon_values_by_level(
            instrument, model, grid, config, horizon
        )
        terminal = (
            jnp.full(grid.n, instrument.face_value) + coupons[config.n_time]
        )
        return terminal, lambda level, values: values + coupons[level]

    return PDEResult(price=_solve_cheyette_1f(model, config, horizon, build))


@register(Swaption, CheyetteModel)
def _swaption_cheyette(*, instrument, model, config) -> PDEResult:
    """European swaption under one-factor Cheyette (frozen-``y``).

    Shares the entire discretisation with the Bermudan recipe below; a strike
    slice of these is where the skew :math:`\\beta` shows, which is what the
    calibrator fits.
    """
    horizon = year_fraction(
        model.initial_curve.reference_date,
        instrument.expiry_date,
        instrument.day_count,
    )

    def build(grid: Grid1D):
        swap = _cheyette_swap_tail_values(
            model,
            grid.nodes,
            instrument.expiry_date,
            instrument.fixed_dates,
            instrument.strike,
            instrument.notional,
            instrument.day_count,
            instrument.is_payer,
        )
        return positive_part(swap, config.smooth_payoff), None

    return PDEResult(price=_solve_cheyette_1f(model, config, horizon, build))


def _cheyette_discount_factor_1f(model: CheyetteModel, grid: Grid1D):
    """Node-level valuer for a single frozen-``y`` ``P(t, T)`` (one factor)."""
    def discount_factor(level_time, payment_time):
        return _cheyette_coupon_bond_values(
            model, grid.nodes, level_time, payment_time[jnp.newaxis], jnp.ones((1,))
        )

    return discount_factor


@register(Caplet, CheyetteModel)
def _caplet_cheyette(*, instrument, model, config) -> PDEResult:
    """Caplet / floorlet under one-factor Cheyette (frozen-``y``).

    A caplet is a European option on the accrual-end zero-coupon bond; a strike
    slice of caplets is the natural skew-``\\beta`` identification instrument for
    the calibrator. At :math:`\\beta = 0` this reduces to the Hull-White caplet.
    """
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    horizon = year_fraction(reference_date, instrument.fixing_date, day_count)
    end_time = year_fraction(reference_date, instrument.end_date, day_count)
    tau = year_fraction(instrument.start_date, instrument.end_date, day_count)

    def build(grid: Grid1D):
        bond = _cheyette_discount_factor_1f(model, grid)(horizon, end_time)
        terminal = caplet_fixing_values(
            bond, instrument.strike, tau, instrument.notional, instrument.is_cap,
            smooth=config.smooth_payoff,
        )
        return terminal, None

    return PDEResult(price=_solve_cheyette_1f(model, config, horizon, build))


@register(Cap, CheyetteModel)
def _cap_cheyette(*, instrument, model, config) -> PDEResult:
    """Cap / floor under one-factor Cheyette (frozen-``y``), single sweep."""
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    horizon = year_fraction(
        reference_date, instrument.fixing_dates[-1], day_count
    )
    dt = horizon / config.n_time

    def build(grid: Grid1D):
        caplet_vals = caplet_values_by_level(
            _cheyette_discount_factor_1f(model, grid),
            reference_date,
            instrument.fixing_dates,
            instrument.start_dates,
            instrument.end_dates,
            day_count,
            instrument.strike,
            instrument.notional,
            instrument.is_cap,
            dt,
            config.n_time,
            (grid.n,),
            smooth=config.smooth_payoff,
        )
        terminal = caplet_vals[config.n_time]
        return terminal, lambda level, values: values + caplet_vals[level]

    return PDEResult(price=_solve_cheyette_1f(model, config, horizon, build))


@register(BermudanSwaption, CheyetteModel)
def _bermudan_swaption_cheyette(*, instrument, model, config) -> PDEResult:
    """Bermudan swaption under one-factor Cheyette (frozen-``y``).

    The headline exotic: no closed form, priced natively because exercise is a
    pointwise projection on the value function. Tail-swap values are analytic
    from the frozen-``y`` affine bond, so exercise carries no discretisation
    error of its own.
    """
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    n_exercise = instrument.exercise_dates.shape[0]

    horizon = year_fraction(
        reference_date, instrument.exercise_dates[-1], day_count
    )
    dt = horizon / config.n_time
    levels = snap_to_levels(
        instrument.exercise_dates, reference_date, day_count, dt, config.n_time
    )

    def build(grid: Grid1D):
        tails = jnp.stack([
            _cheyette_swap_tail_values(
                model,
                grid.nodes,
                instrument.exercise_dates[e],
                instrument.fixed_dates[e:],
                instrument.strike,
                instrument.notional,
                day_count,
                instrument.is_payer,
            )
            for e in range(n_exercise)
        ])
        return bermudan_terminal_and_event(
            tails, levels, config.n_time, config.smooth_payoff
        )

    return PDEResult(price=_solve_cheyette_1f(model, config, horizon, build))


@register(CallableBond, CheyetteModel)
def _callable_bond_cheyette(*, instrument, model, config) -> PDEResult:
    """Callable fixed-rate bond under one-factor Cheyette (frozen-``y``)."""
    price = _callable_style_price_1f(
        instrument,
        model,
        config,
        instrument.call_dates,
        instrument.call_prices,
        is_call=True,
    )
    return PDEResult(price=price)


@register(PuttableBond, CheyetteModel)
def _puttable_bond_cheyette(*, instrument, model, config) -> PDEResult:
    """Puttable fixed-rate bond under one-factor Cheyette (frozen-``y``)."""
    price = _callable_style_price_1f(
        instrument,
        model,
        config,
        instrument.put_dates,
        instrument.put_prices,
        is_call=False,
    )
    return PDEResult(price=price)


# ═════════════════════════════════════════════════════════════════════
# Tier 1 — frozen-y, two factors (2-D ADI PDE)
# ═════════════════════════════════════════════════════════════════════


def _g2pp_equivalent(model: CheyetteModel2F) -> G2PPModel:
    r"""Gaussian-limit G2++ twin sharing :math:`\kappa_i, \sigma_i, \rho`, curve.

    The frozen-``y`` deterministic discount shift
    :math:`\varphi_{\mathrm{Ch}}(t) = f^M(0,t) + m_1(t) + m_2(t)` equals the
    G2++ :math:`\varphi(t)` of this twin, so its exact per-step discount can be
    reused verbatim — and the :math:`(\beta_1, \beta_2) = 0` reduction is exact.

    Args:
        model: Two-factor Cheyette model.

    Returns:
        The matching :class:`~valax.models.g2pp.G2PPModel`.
    """
    return G2PPModel(
        mean_reversion_x=model.mean_reversion_1,
        mean_reversion_y=model.mean_reversion_2,
        volatility_x=model.volatility_1,
        volatility_y=model.volatility_2,
        correlation=model.correlation,
        initial_curve=model.initial_curve,
    )


def cheyette_operator_2d(
    model: CheyetteModel2F,
    grid: Grid2D,
    *,
    ref_time: Float[Array, ""],
) -> Operator2D:
    r"""Frozen-``y`` two-factor Cheyette ADI operator in the centred factors.

    Works in the two centred factors :math:`\tilde x_i = x_i - m_i(t)`, whose
    dynamics are the G2++ centred dynamics
    :math:`d\tilde x_i = -\kappa_i \tilde x_i\,dt + \sigma_{r,i}\,dW_i`. The
    backward equation for :math:`V(t, \tilde x_1, \tilde x_2)` is

    .. math::

        V_\tau = \tfrac12 \sigma_{r,1}^2 V_{\tilde x_1\tilde x_1}
            + \tfrac12 \sigma_{r,2}^2 V_{\tilde x_2\tilde x_2}
            + \rho\,\sigma_{r,1}\sigma_{r,2}\, V_{\tilde x_1\tilde x_2}
            - \kappa_1 \tilde x_1 V_{\tilde x_1} - \kappa_2 \tilde x_2 V_{\tilde x_2}
            - (\tilde x_1 + \tilde x_2) V ,

    with :math:`\sigma_{r,i} = \sigma_i(1 + \beta_i(\tilde x_i + m_i))`. Unlike
    G2++ the diffusion **and** the cross term are spatially varying (through the
    skews), which :func:`~valax.pricing.pde.operators2d.build_operator_2d`
    supports directly via per-node coefficient fields. The deterministic,
    spatially-uniform discount :math:`\varphi_{\mathrm{Ch}}(t)` is factored out
    and applied per step (see :func:`_solve_cheyette_2f`); only the state part
    :math:`\tilde x_1 + \tilde x_2` enters here.

    The single-operator ADI substrate is time-independent, so the (mildly
    time-varying) Gaussian-limit means :math:`m_i` used inside the diffusion are
    frozen at ``ref_time``. This is a higher-order effect (the dominant skew term
    is :math:`\beta_i \tilde x_i`); at :math:`\beta = 0` the diffusion is exactly
    :math:`\sigma_i^2` and this reduces to
    :func:`~valax.pricing.pde.coefficients.g2pp_operator_2d`.

    Args:
        model: Two-factor Cheyette model.
        grid: The centred-factor :math:`(\tilde x_1, \tilde x_2)` grid.
        ref_time: Time at which the Gaussian-limit means are frozen (typically
            the horizon).

    Returns:
        The assembled :class:`~valax.pricing.pde.operators2d.Operator2D`.
    """
    k1 = model.mean_reversion_1
    k2 = model.mean_reversion_2
    s1 = model.volatility_1
    s2 = model.volatility_2
    b1 = model.skew_1
    b2 = model.skew_2
    rho = model.correlation

    m1, m2 = cheyette_gaussian_mean_2f(model, ref_time)

    x1 = grid.x.nodes[:, jnp.newaxis]  # (n_x, 1)
    x2 = grid.y.nodes[jnp.newaxis, :]  # (1, n_y)

    sig_r1 = jnp.maximum(s1 * (1.0 + b1 * (x1 + m1)), 0.0)  # (n_x, 1)
    sig_r2 = jnp.maximum(s2 * (1.0 + b2 * (x2 + m2)), 0.0)  # (1, n_y)

    diff_x = sig_r1**2                       # (n_x, 1)
    diff_v = sig_r2**2                       # (1, n_y)
    mixed = rho * sig_r1 * sig_r2            # (n_x, n_y)
    drift_x = -k1 * x1                       # (n_x, 1)
    drift_v = -k2 * x2                       # (1, n_y)
    discount = x1 + x2                       # (n_x, n_y) state part

    return build_operator_2d(
        grid,
        diff_x=diff_x,
        drift_x=drift_x,
        diff_v=diff_v,
        drift_v=drift_v,
        mixed=mixed,
        discount=discount,
    )


def _cheyette_coupon_bond_values_2f(
    model: CheyetteModel2F,
    grid: Grid2D,
    t: Float[Array, ""],
    payment_times: Float[Array, " n_cf"],
    cashflows: Float[Array, " n_cf"],
) -> Float[Array, "n_x n_y"]:
    r"""Frozen-``y`` value of :math:`\sum_i c_i P(t, T_i)` at every ``(x1, x2)`` node.

    Uses the two-factor quasi-Gaussian affine bond with the accumulator matrix
    frozen to its Gaussian-limit value :math:`\bar y_{ij}(t)` and the factors
    recovered as :math:`x_i = \tilde x_i + m_i(t)`.

    Args:
        model: Two-factor Cheyette model.
        grid: The centred-factor grid.
        t: Valuation time in year fractions.
        payment_times: Cashflow times in year fractions.
        cashflows: Cashflow amounts.

    Returns:
        The portfolio value at each node, shape ``(n_x, n_y)``.
    """
    m1, m2 = cheyette_gaussian_mean_2f(model, t)
    y_mat = cheyette_y_gaussian_2f(model, t)
    y11, y12, y22 = y_mat[0, 0], y_mat[0, 1], y_mat[1, 1]

    G1 = cheyette_G(model.mean_reversion_1, payment_times - t)  # (n_cf,)
    G2 = cheyette_G(model.mean_reversion_2, payment_times - t)  # (n_cf,)
    ln_ratio = _ln_market_df(model, payment_times) - _ln_market_df(model, t)
    quad = y11 * G1**2 + 2.0 * y12 * G1 * G2 + y22 * G2**2      # (n_cf,)

    x1 = (grid.x.nodes + m1)[:, jnp.newaxis, jnp.newaxis]  # (n_x, 1, 1)
    x2 = (grid.y.nodes + m2)[jnp.newaxis, :, jnp.newaxis]  # (1, n_y, 1)
    exponent = (
        ln_ratio - x1 * G1 - x2 * G2 - 0.5 * quad
    )  # (n_x, n_y, n_cf)
    return jnp.exp(exponent) @ cashflows


def _cheyette_discount_grid_2f(
    model: CheyetteModel2F,
    grid: Grid2D,
    t: Float[Array, ""],
    payment_time: Float[Array, ""],
) -> Float[Array, "n_x n_y"]:
    """Frozen-``y`` discount factor :math:`P(t, T \\mid x_1, x_2)` at every node."""
    return _cheyette_coupon_bond_values_2f(
        model, grid, t, payment_time[jnp.newaxis], jnp.ones((1,))
    )


def _cheyette_swap_tail_values_2f(
    model: CheyetteModel2F,
    grid: Grid2D,
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    day_count: str,
    is_payer: bool,
) -> Float[Array, "n_x n_y"]:
    """Value of the underlying swap at ``exercise_date``, at every node (2F)."""
    reference_date = model.initial_curve.reference_date

    def bond_portfolio(t, payment_times, cashflows):
        return _cheyette_coupon_bond_values_2f(
            model, grid, t, payment_times, cashflows
        )

    return swap_tail_values(
        bond_portfolio,
        reference_date,
        exercise_date,
        payment_dates,
        strike,
        notional,
        day_count,
        is_payer,
    )


def _cheyette_coupon_values_by_level_2f(
    bond,
    model: CheyetteModel2F,
    grid: Grid2D,
    config: PDEConfig2D,
    horizon: Float[Array, ""],
) -> Float[Array, "n_levels n_x n_y"]:
    """Coupon value to inject at each time level, per node (2F, frozen-``y``)."""
    reference_date = model.initial_curve.reference_date
    dt = horizon / config.n_time
    coupon = fixed_coupon_amounts(bond)

    def factor(level_time, payment_time):
        return _cheyette_discount_grid_2f(model, grid, level_time, payment_time)

    return coupon_values_by_level(
        factor,
        reference_date,
        bond.payment_dates,
        bond.day_count,
        coupon,
        dt,
        config.n_time,
        (grid.n_x, grid.n_y),
    )


def _zero_boundary_2d() -> Boundary2D:
    """Inert log-``x`` Dirichlet data (the linearity fold zeroes the edge bands)."""
    zero = lambda tau: jnp.zeros_like(tau)
    return Boundary2D(zero, zero)


def _solve_cheyette_2f(
    model: CheyetteModel2F,
    config: PDEConfig2D,
    horizon: Float[Array, ""],
    build: Callable[[Grid2D], tuple[Float[Array, "n_x n_y"], Optional[_Event2D]]],
) -> Float[Array, ""]:
    r"""Run the frozen-``y`` 2F ADI sweep and read off at :math:`(\tilde x_1, \tilde x_2) = 0`.

    Sizes the grid from the Gaussian-limit factor variances
    :math:`\bar y_{ii}(T)`, folds the linearity boundary condition into the
    operator, and applies the deterministic :math:`\varphi_{\mathrm{Ch}}(t)`
    discount per step through the ``event_fn`` seam (via the G2++ twin), exactly
    as :func:`~valax.pricing.pde.g2pp._solve_g2pp` does.

    Args:
        model: Two-factor Cheyette model.
        config: 2-D grid configuration.
        horizon: Time to the last modelled date, in year fractions.
        build: Given the grid, returns ``(terminal, instrument_event)``.

    Returns:
        The price at the curve reference date.
    """
    y_mat = cheyette_y_gaussian_2f(model, horizon)
    std_x = jnp.sqrt(y_mat[0, 0])
    std_y = jnp.sqrt(y_mat[1, 1])
    grid = Grid2D(
        x=centred_state_grid(std_x, n=config.n_x, half_width=config.x_range),
        y=centred_state_grid(std_y, n=config.n_y, half_width=config.x_range),
    )
    operator = apply_linearity_bc_2d(
        cheyette_operator_2d(model, grid, ref_time=horizon), grid
    )

    dt = horizon / config.n_time
    phi_disc = _phi_step_discounts(_g2pp_equivalent(model), dt, config.n_time)

    terminal, instrument_event = build(grid)

    if instrument_event is None:
        def event_fn(level, values):
            return values * phi_disc[level]
    else:
        def event_fn(level, values):
            return instrument_event(level, values * phi_disc[level])

    values = solve_backward_2d(
        operator,
        _zero_boundary_2d(),
        terminal,
        expiry=horizon,
        n_time=config.n_time,
        scheme=config.scheme,
        theta=config.theta,
        rannacher_steps=max(config.rannacher_steps, _MIN_RANNACHER),
        event_fn=event_fn,
    )
    return read_off_2d(grid, values, jnp.zeros(()), jnp.zeros(()))


# ── Recipes (Tier 1, two factors) ─────────────────────────────────────


@register(FixedRateBond, CheyetteModel2F)
def _fixed_rate_bond_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """Option-free fixed-rate bond under two-factor Cheyette (frozen-``y``)."""
    horizon = _bond_horizon(instrument, model)

    def build(grid: Grid2D):
        coupons = _cheyette_coupon_values_by_level_2f(
            instrument, model, grid, config, horizon
        )
        terminal = (
            jnp.full(grid.shape, instrument.face_value) + coupons[config.n_time]
        )
        return terminal, lambda level, values: values + coupons[level]

    return PDEResult(price=_solve_cheyette_2f(model, config, horizon, build))


@register(Swaption, CheyetteModel2F)
def _swaption_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """European swaption under two-factor Cheyette (frozen-``y``)."""
    horizon = year_fraction(
        model.initial_curve.reference_date,
        instrument.expiry_date,
        instrument.day_count,
    )

    def build(grid: Grid2D):
        swap = _cheyette_swap_tail_values_2f(
            model,
            grid,
            instrument.expiry_date,
            instrument.fixed_dates,
            instrument.strike,
            instrument.notional,
            instrument.day_count,
            instrument.is_payer,
        )
        return positive_part(swap, config.smooth_payoff), None

    return PDEResult(price=_solve_cheyette_2f(model, config, horizon, build))


def _cheyette_discount_factor_2f(model: CheyetteModel2F, grid: Grid2D):
    """Node-level valuer for a single frozen-``y`` ``P(t, T)`` (two factors)."""
    def discount_factor(level_time, payment_time):
        return _cheyette_discount_grid_2f(model, grid, level_time, payment_time)

    return discount_factor


@register(Caplet, CheyetteModel2F)
def _caplet_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """Caplet / floorlet under two-factor Cheyette (frozen-``y``)."""
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    horizon = year_fraction(reference_date, instrument.fixing_date, day_count)
    end_time = year_fraction(reference_date, instrument.end_date, day_count)
    tau = year_fraction(instrument.start_date, instrument.end_date, day_count)

    def build(grid: Grid2D):
        bond = _cheyette_discount_factor_2f(model, grid)(horizon, end_time)
        terminal = caplet_fixing_values(
            bond, instrument.strike, tau, instrument.notional, instrument.is_cap,
            smooth=config.smooth_payoff,
        )
        return terminal, None

    return PDEResult(price=_solve_cheyette_2f(model, config, horizon, build))


@register(Cap, CheyetteModel2F)
def _cap_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """Cap / floor under two-factor Cheyette (frozen-``y``), single sweep."""
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    horizon = year_fraction(
        reference_date, instrument.fixing_dates[-1], day_count
    )
    dt = horizon / config.n_time

    def build(grid: Grid2D):
        caplet_vals = caplet_values_by_level(
            _cheyette_discount_factor_2f(model, grid),
            reference_date,
            instrument.fixing_dates,
            instrument.start_dates,
            instrument.end_dates,
            day_count,
            instrument.strike,
            instrument.notional,
            instrument.is_cap,
            dt,
            config.n_time,
            (grid.n_x, grid.n_y),
            smooth=config.smooth_payoff,
        )
        terminal = caplet_vals[config.n_time]
        return terminal, lambda level, values: values + caplet_vals[level]

    return PDEResult(price=_solve_cheyette_2f(model, config, horizon, build))


@register(CMSSpreadSwap, CheyetteModel2F)
def _cms_spread_swap_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """CMS-spread swap (steepener / flattener) under two-factor Cheyette.

    The skew-capable analogue of the G2++ CMS-spread swap. Each period's coupon
    is a linear function of two par swap rates observed at the accrual start,
    built at every ``(x1, x2)`` node from the frozen-``y`` affine bond strip,
    scaled from fixing to payment by the node discount factor, injected at the
    (snapped) fixing level and discounted home in one backward pass. The
    frozen-``y`` bias is the same one carried by every other 2F recipe here.
    """
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    pay_times = year_fraction(
        reference_date, instrument.payment_dates, day_count
    )
    horizon = pay_times[-1]
    dt = horizon / config.n_time

    def build(grid: Grid2D):
        vals = cms_spread_values_by_level(
            _cheyette_discount_factor_2f(model, grid),
            reference_date,
            instrument.start_date,
            instrument.payment_dates,
            day_count,
            instrument.fixed_rate,
            instrument.notional,
            instrument.cms_tenor_long,
            instrument.cms_tenor_short,
            instrument.pay_fixed,
            dt,
            config.n_time,
            (grid.n_x, grid.n_y),
        )
        terminal = vals[config.n_time]
        return terminal, lambda level, values: values + vals[level]

    return PDEResult(price=_solve_cheyette_2f(model, config, horizon, build))


def _callable_style_price_2f(
    bond,
    model: CheyetteModel2F,
    config: PDEConfig2D,
    exercise_dates: Int[Array, " m"],
    exercise_prices: Float[Array, " m"],
    *,
    is_call: bool,
) -> Float[Array, ""]:
    """Shared frozen-``y`` 2-D ADI sweep for callable / puttable bonds (2F)."""
    horizon = _bond_horizon(bond, model)
    dt = horizon / config.n_time
    strike_at_level = exercise_obstacle_by_level(
        exercise_dates,
        exercise_prices,
        bond.face_value,
        model.initial_curve.reference_date,
        bond.day_count,
        dt,
        config.n_time,
        is_call,
        accrued_at_exercise=bond_accrued_interest_at_dates(bond, exercise_dates),
    )

    def build(grid: Grid2D):
        coupons = _cheyette_coupon_values_by_level_2f(
            bond, model, grid, config, horizon
        )
        return exercisable_terminal_and_event(
            coupons, bond.face_value, strike_at_level, is_call,
            smooth=config.smooth_payoff,
        )

    return _solve_cheyette_2f(model, config, horizon, build)


@register(CallableBond, CheyetteModel2F)
def _callable_bond_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """Callable fixed-rate bond under two-factor Cheyette (frozen-``y``).

    The skew-capable, decorrelation-sensitive analogue of the G2++ callable.
    """
    price = _callable_style_price_2f(
        instrument,
        model,
        config,
        instrument.call_dates,
        instrument.call_prices,
        is_call=True,
    )
    return PDEResult(price=price)


@register(PuttableBond, CheyetteModel2F)
def _puttable_bond_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """Puttable fixed-rate bond under two-factor Cheyette (frozen-``y``)."""
    price = _callable_style_price_2f(
        instrument,
        model,
        config,
        instrument.put_dates,
        instrument.put_prices,
        is_call=False,
    )
    return PDEResult(price=price)


@register(BermudanSwaption, CheyetteModel2F)
def _bermudan_swaption_cheyette_2f(*, instrument, model, config) -> PDEResult:
    """Bermudan swaption under two-factor Cheyette (frozen-``y``).

    The decorrelation-sensitive headline exotic — the skew-capable analogue of
    the G2++ Bermudan.
    """
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    n_exercise = instrument.exercise_dates.shape[0]

    horizon = year_fraction(
        reference_date, instrument.exercise_dates[-1], day_count
    )
    dt = horizon / config.n_time
    levels = snap_to_levels(
        instrument.exercise_dates, reference_date, day_count, dt, config.n_time
    )

    def build(grid: Grid2D):
        tails = jnp.stack([
            _cheyette_swap_tail_values_2f(
                model,
                grid,
                instrument.exercise_dates[e],
                instrument.fixed_dates[e:],
                instrument.strike,
                instrument.notional,
                day_count,
                instrument.is_payer,
            )
            for e in range(n_exercise)
        ])
        return bermudan_terminal_and_event(
            tails, levels, config.n_time, config.smooth_payoff
        )

    return PDEResult(price=_solve_cheyette_2f(model, config, horizon, build))


# ═════════════════════════════════════════════════════════════════════
# Tier 2 — exact-y, one factor (2-D PDE in (x, y))
# ═════════════════════════════════════════════════════════════════════
#
# Carries the accumulator ``y`` as a real advection-only axis, so the price is
# the true-``y`` price (no frozen-``y`` approximation). Used to measure the
# frozen-``y`` bias exactly (zero Monte-Carlo noise). The two-factor exact-``y``
# PDE is 5-D and is *not* implemented here — see docs/research-ideas.md.


def _apply_linearity_bc_x(operator: Operator2D, grid: Grid2D) -> Operator2D:
    r"""Fold the ``x``-axis linearity condition into ``A1`` only.

    The exact-``y`` operator diffuses only along ``x``; the ``y`` axis is
    hyperbolic and handled by upwinding, so — unlike
    :func:`~valax.pricing.pde.boundary.apply_linearity_bc_2d` — the ``V_yy = 0``
    fold must **not** be applied there. This imposes zero curvature on the two
    ``x`` edges (rows 0 and ``n_x - 1`` of ``A1``) exactly as the 1-D fold does,
    leaving ``A2`` untouched.

    Args:
        operator: The raw exact-``y`` :class:`Operator2D`.
        grid: The tensor-product ``(x, y)`` grid.

    Returns:
        A new :class:`Operator2D` with the two ``x`` edge rows folded.
    """
    x = grid.x.nodes
    x_lo, x_hi = boundary_coords(grid.x)
    ratio_lo = (x[0] - x_lo) / (x[1] - x[0])
    ratio_hi = (x_hi - x[-1]) / (x[-1] - x[-2])

    a1_lower, a1_diag, a1_upper = (
        operator.a1_lower,
        operator.a1_diag,
        operator.a1_upper,
    )
    al_first = a1_lower[0, :]
    a1_diag = a1_diag.at[0, :].add(al_first * (1.0 + ratio_lo))
    a1_upper = a1_upper.at[0, :].add(-al_first * ratio_lo)
    a1_lower = a1_lower.at[0, :].set(0.0)

    au_last = a1_upper[-1, :]
    a1_diag = a1_diag.at[-1, :].add(au_last * (1.0 + ratio_hi))
    a1_lower = a1_lower.at[-1, :].add(-au_last * ratio_hi)
    a1_upper = a1_upper.at[-1, :].set(0.0)

    return eqx.tree_at(
        lambda o: (o.a1_lower, o.a1_diag, o.a1_upper),
        operator,
        (a1_lower, a1_diag, a1_upper),
    )


def cheyette_exact_operator_2d(
    model: CheyetteModel,
    grid: Grid2D,
) -> Operator2D:
    r"""Exact-``y`` one-factor Cheyette operator on a :math:`(x, y)` grid.

    Carries the accumulator as a genuine axis. The backward equation for
    :math:`V(t, x, y)` is

    .. math::

        V_\tau = \tfrac12 \sigma_r(x)^2 V_{xx}
            + (y - \kappa x) V_x
            + \bigl(\sigma_r(x)^2 - 2\kappa y\bigr) V_y
            - x\,V ,

    with :math:`\sigma_r(x) = \sigma(1 + \beta x)` and the deterministic discount
    :math:`f^M(0,t)` factored out per step (see :func:`_solve_cheyette_exact`).
    The operator is **time-independent** (the local volatility has no explicit
    time dependence), so it is built once. The ``x`` axis is parabolic
    (central differences); the ``y`` axis is pure advection and uses **first-order
    upwinding** for stability, added to the ``A2`` bands produced with
    ``diff_v = drift_v = 0``.

    Args:
        model: One-factor Cheyette model.
        grid: The tensor-product ``(x, y)`` grid.

    Returns:
        The assembled exact-``y`` :class:`Operator2D` (``x`` diffusion + drift on
        ``A1``, upwinded ``y`` advection on ``A2``; no mixed term).
    """
    kappa = model.mean_reversion
    sigma = model.volatility
    beta = model.skew

    x = grid.x.nodes[:, jnp.newaxis]  # (n_x, 1)
    y = grid.y.nodes[jnp.newaxis, :]  # (1, n_y)

    sigma_r = jnp.maximum(sigma * (1.0 + beta * x), 0.0)  # (n_x, 1)
    sig2 = sigma_r**2                                     # (n_x, 1)

    diff_x = jnp.broadcast_to(sig2, grid.shape)                 # (n_x, n_y)
    drift_x = jnp.broadcast_to(y - kappa * x, grid.shape)       # (n_x, n_y)
    discount = jnp.broadcast_to(x, grid.shape)                  # (n_x, n_y) state

    # A1 (x diffusion + drift + half discount), A2 diag = -half discount.
    op = build_operator_2d(
        grid,
        diff_x=diff_x,
        drift_x=drift_x,
        diff_v=jnp.zeros(grid.shape),
        drift_v=jnp.zeros(grid.shape),
        mixed=jnp.zeros(grid.shape),
        discount=discount,
    )

    # --- Upwinded y-advection added to A2. ---
    # drift b(x, y) = sigma_r(x)^2 - 2 kappa y  (advection speed on the y axis).
    b = jnp.broadcast_to(sig2, grid.shape) - 2.0 * kappa * jnp.broadcast_to(
        y, grid.shape
    )
    yn = grid.y.nodes
    h_minus = jnp.concatenate([jnp.array([yn[1] - yn[0]]), yn[1:] - yn[:-1]])
    h_plus = jnp.concatenate([yn[1:] - yn[:-1], jnp.array([yn[-1] - yn[-2]])])
    h_minus = h_minus[jnp.newaxis, :]  # (1, n_y)
    h_plus = h_plus[jnp.newaxis, :]    # (1, n_y)

    pos = b > 0.0
    # Backward equation V_tau = b V_y: characteristics flow at dy/dtau = -b, so
    # the stable upwind differences the *downwind* neighbour — b > 0 uses the
    # forward difference (couples j+1), b < 0 the backward difference (j-1).
    # This keeps the advection contribution to the operator dissipative
    # (non-positive real eigenvalues), the condition for a stable ADI sweep.
    adv_upper = jnp.where(pos, b / h_plus, 0.0)
    adv_lower = jnp.where(pos, 0.0, -b / h_minus)
    adv_diag = jnp.where(pos, -b / h_plus, b / h_minus)

    a2_lower = op.a2_lower + adv_lower
    a2_diag = op.a2_diag + adv_diag
    a2_upper = op.a2_upper + adv_upper
    # Drop the out-of-domain couplings at the two y edges (first-order edge
    # treatment; harmless when the domain brackets the accumulator trajectory).
    a2_lower = a2_lower.at[:, 0].set(0.0)
    a2_upper = a2_upper.at[:, -1].set(0.0)

    op = eqx.tree_at(
        lambda o: (o.a2_lower, o.a2_diag, o.a2_upper),
        op,
        (a2_lower, a2_diag, a2_upper),
    )
    return _apply_linearity_bc_x(op, grid)


def _exact_y_grid(
    model: CheyetteModel,
    config: PDEConfig2D,
    horizon: Float[Array, ""],
) -> Grid2D:
    r"""Build the :math:`(x, y)` grid for the exact-``y`` solver.

    The ``x`` axis is the centred-state grid sized from the Gaussian-limit
    variance :math:`\bar y(T)`; the ``y`` axis is a linear grid bracketing the
    accumulator's trajectory ``[−0.25, config.y_max] × \bar y(T)`` so the
    read-off value ``y = 0`` sits inside the interior nodes.

    Args:
        model: One-factor Cheyette model.
        config: 2-D grid configuration (``n_x``, ``n_y``, ``x_range``, ``y_max``).
        horizon: Horizon in year fractions.

    Returns:
        The tensor-product :class:`Grid2D`.
    """
    ybar = cheyette_y_gaussian(model, horizon)
    std_x = jnp.sqrt(ybar)
    x = centred_state_grid(std_x, n=config.n_x, half_width=config.x_range)
    y = uniform_linear_grid(-0.25 * ybar, config.y_max * ybar, n=config.n_y)
    return Grid2D(x=x, y=y)


def _solve_cheyette_exact(
    model: CheyetteModel,
    config: PDEConfig2D,
    horizon: Float[Array, ""],
    build: Callable[[Grid2D], tuple[Float[Array, "n_x n_y"], Optional[_Event2D]]],
) -> Float[Array, ""]:
    r"""Run the exact-``y`` 1F backward sweep and read off at :math:`(x, y) = (0, 0)`.

    The deterministic discount :math:`f^M(0,t)` is spatially uniform, so it is
    factored out and applied as a per-step scalar (the telescoping market
    discount-factor ratio); the state part ``x`` stays in the operator, and the
    accumulator convexity is generated natively by the ``y`` dynamics (no shift
    to add). Uses the Douglas ADI scheme (the ``y`` axis is upwinded advection,
    for which the higher-order correctors are not needed).

    Args:
        model: One-factor Cheyette model.
        config: 2-D grid configuration.
        horizon: Horizon in year fractions.
        build: Given the grid, returns ``(terminal, instrument_event)``.

    Returns:
        The price at the curve reference date.
    """
    grid = _exact_y_grid(model, config, horizon)
    operator = cheyette_exact_operator_2d(model, grid)

    dt = horizon / config.n_time
    t_levels = jnp.arange(config.n_time + 1) * dt
    ln_pm = _ln_market_df(model, t_levels)
    # Per-step forward discount over [t_k, t_{k+1}]:
    #   exp(-int_{t_k}^{t_{k+1}} f^M ds) = P^M(0, t_{k+1}) / P^M(0, t_k) < 1,
    # indexed by forward level k = 0 .. n_time - 1 (the telescoping market
    # discount-factor ratio; same discipline as G2++'s phi factoring).
    phi_disc = jnp.exp(ln_pm[1:] - ln_pm[:-1])

    terminal, instrument_event = build(grid)

    if instrument_event is None:
        def event_fn(level, values):
            return values * phi_disc[level]
    else:
        def event_fn(level, values):
            return instrument_event(level, values * phi_disc[level])

    values = solve_backward_2d(
        operator,
        _zero_boundary_2d(),
        terminal,
        expiry=horizon,
        n_time=config.n_time,
        scheme=Scheme.DOUGLAS,
        theta=1.0,
        rannacher_steps=max(config.rannacher_steps, _MIN_RANNACHER),
        event_fn=event_fn,
    )
    return read_off_2d(grid, values, jnp.zeros(()), jnp.zeros(()))


def _cheyette_coupon_bond_values_exact(
    model: CheyetteModel,
    grid: Grid2D,
    t: Float[Array, ""],
    payment_times: Float[Array, " n_cf"],
    cashflows: Float[Array, " n_cf"],
) -> Float[Array, "n_x n_y"]:
    r"""Exact-``y`` value of :math:`\sum_i c_i P(t, T_i)` at every ``(x, y)`` node.

    Uses the affine bond price with the **actual** grid accumulator ``y`` (not a
    frozen value), which is what makes this the true-``y`` price.

    Args:
        model: One-factor Cheyette model.
        grid: The ``(x, y)`` grid.
        t: Valuation time in year fractions.
        payment_times: Cashflow times in year fractions.
        cashflows: Cashflow amounts.

    Returns:
        The portfolio value at each node, shape ``(n_x, n_y)``.
    """
    G = cheyette_G(model.mean_reversion, payment_times - t)  # (n_cf,)
    ln_ratio = _ln_market_df(model, payment_times) - _ln_market_df(model, t)

    x = grid.x.nodes[:, jnp.newaxis, jnp.newaxis]  # (n_x, 1, 1)
    y = grid.y.nodes[jnp.newaxis, :, jnp.newaxis]  # (1, n_y, 1)
    exponent = (
        ln_ratio - x * G - 0.5 * y * G**2
    )  # (n_x, n_y, n_cf)
    return jnp.exp(exponent) @ cashflows


def _cheyette_swap_tail_values_exact(
    model: CheyetteModel,
    grid: Grid2D,
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    day_count: str,
    is_payer: bool,
) -> Float[Array, "n_x n_y"]:
    """Value of the underlying swap at ``exercise_date``, every node (exact-``y``)."""
    reference_date = model.initial_curve.reference_date

    def bond_portfolio(t, payment_times, cashflows):
        return _cheyette_coupon_bond_values_exact(
            model, grid, t, payment_times, cashflows
        )

    return swap_tail_values(
        bond_portfolio,
        reference_date,
        exercise_date,
        payment_dates,
        strike,
        notional,
        day_count,
        is_payer,
    )


def cheyette_exact_price(
    instrument: FixedRateBond | Swaption | Caplet,
    model: CheyetteModel,
    config: PDEConfig2D,
) -> PDEResult:
    r"""Price a one-factor instrument with the exact-``y`` 2-D PDE.

    The exact-``y`` counterpart to the frozen-``y`` recipes registered on
    :func:`~valax.pricing.pde.pde_price_dispatch`. It is a **separate entry
    point** (not on the global registry) because it shares the
    ``(instrument, CheyetteModel)`` key with the frozen-``y`` recipe but takes a
    2-D config; the calibrator selects it via ``engine="exact"``.

    Supports :class:`~valax.instruments.bonds.FixedRateBond` (the option-free
    scheme calibration / exact-fit check),
    :class:`~valax.instruments.rates.Swaption` (European) and
    :class:`~valax.instruments.rates.Caplet` (the skew-``\beta`` slice
    instrument). The difference from the frozen-``y`` price is a deterministic
    measurement of the frozen-``y`` bias; at :math:`\beta = 0` the two agree.

    Args:
        instrument: A ``FixedRateBond``, European ``Swaption`` or ``Caplet``.
        model: One-factor Cheyette model.
        config: 2-D grid configuration for the ``(x, y)`` mesh.

    Returns:
        A :class:`~valax.pricing.pde.dispatch.PDEResult`.

    Raises:
        ValueError: If ``instrument`` is not a supported type.
    """
    if isinstance(instrument, FixedRateBond):
        horizon = _bond_horizon(instrument, model)
        reference_date = model.initial_curve.reference_date
        dt = horizon / config.n_time
        coupon = fixed_coupon_amounts(instrument)

        def build(grid: Grid2D):
            def factor(level_time, payment_time):
                return _cheyette_coupon_bond_values_exact(
                    model,
                    grid,
                    level_time,
                    payment_time[jnp.newaxis],
                    jnp.ones((1,)),
                )

            coupons = coupon_values_by_level(
                factor,
                reference_date,
                instrument.payment_dates,
                instrument.day_count,
                coupon,
                dt,
                config.n_time,
                (grid.n_x, grid.n_y),
            )
            terminal = (
                jnp.full(grid.shape, instrument.face_value)
                + coupons[config.n_time]
            )
            return terminal, lambda level, values: values + coupons[level]

        return PDEResult(price=_solve_cheyette_exact(model, config, horizon, build))

    if isinstance(instrument, Swaption):
        horizon = year_fraction(
            model.initial_curve.reference_date,
            instrument.expiry_date,
            instrument.day_count,
        )

        def build(grid: Grid2D):
            swap = _cheyette_swap_tail_values_exact(
                model,
                grid,
                instrument.expiry_date,
                instrument.fixed_dates,
                instrument.strike,
                instrument.notional,
                instrument.day_count,
                instrument.is_payer,
            )
            return positive_part(swap, config.smooth_payoff), None

        return PDEResult(price=_solve_cheyette_exact(model, config, horizon, build))

    if isinstance(instrument, Caplet):
        reference_date = model.initial_curve.reference_date
        day_count = instrument.day_count
        horizon = year_fraction(reference_date, instrument.fixing_date, day_count)
        end_time = year_fraction(reference_date, instrument.end_date, day_count)
        tau = year_fraction(instrument.start_date, instrument.end_date, day_count)

        def build(grid: Grid2D):
            bond = _cheyette_coupon_bond_values_exact(
                model, grid, horizon, end_time[jnp.newaxis], jnp.ones((1,))
            )
            terminal = caplet_fixing_values(
                bond,
                instrument.strike,
                tau,
                instrument.notional,
                instrument.is_cap,
                smooth=config.smooth_payoff,
            )
            return terminal, None

        return PDEResult(price=_solve_cheyette_exact(model, config, horizon, build))

    raise ValueError(
        f"cheyette_exact_price supports FixedRateBond, Swaption and Caplet, "
        f"got {type(instrument).__name__}."
    )
