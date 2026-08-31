r"""G2++ two-factor Gaussian short-rate finite-difference pricing.

Prices the headline **decorrelation-sensitive exotic** — the Bermudan swaption
— by solving the G2++ backward PDE on the existing 2-D ADI substrate that was
built for Heston. This is the fourth independent numerical route to the model,
joining the analytic Gauss-Hermite swaption formula
(:func:`~valax.pricing.analytic.g2pp_swaptions.g2pp_swaption_price`), the exact
Monte-Carlo engine (:mod:`valax.pricing.mc.g2pp_paths`) and the calibration
layer — and it is the only one that prices Bermudan exercise natively, because
the exercise decision is a pointwise projection on the value function.

State variables
---------------
The solver works in the two **centred** Gaussian factors ``x`` and ``y`` of the
decomposition :math:`r(t) = x(t) + y(t) + \varphi(t)` (see
:func:`~valax.models.g2pp.g2pp_phi`), where

.. math::

    dx = -a\,x\,dt + \sigma\,dW_1, \quad
    dy = -b\,y\,dt + \eta\,dW_2, \quad
    dW_1\,dW_2 = \rho\,dt , \quad x(0) = y(0) = 0 .

A claim value :math:`V(t, x, y)`, :math:`\tau = T - t`, then marches

.. math::

    V_\tau = \tfrac{1}{2}\sigma^2 V_{xx} + \tfrac{1}{2}\eta^2 V_{yy}
        + \rho\sigma\eta\, V_{xy} - a x\,V_x - b y\,V_y
        - \bigl(x + y + \varphi(t)\bigr) V

back from the terminal condition. As in the Hull-White 1-D recipe, the centred
factors make the drift/diffusion coefficients time-independent and anchor the
state at the origin, so the price is always read off at ``(x, y) = (0, 0)``.

The cross term ``rho sigma eta V_xy``
-------------------------------------
The cross derivative — the crux of two-factor decorrelation — is handled by the
existing ADI substrate exactly as Heston's ``rho xi v V_xv`` is: it enters as a
constant ``mixed`` coefficient into
:func:`~valax.pricing.pde.coefficients.g2pp_operator_2d`, is stored as the
explicit-only mixed operator ``A0``, and is zeroed on the four domain edges
(in't Hout & Foulon). The Craig-Sneyd / Hundsdorfer-Verwer correctors recover
second-order accuracy in it.

The exact-fit shift ``phi(t)``
------------------------------
:math:`\varphi(t)` is deterministic and spatially uniform, so a discount
``-varphi(t) V`` commutes with the spatial operator. Writing ``V = D(t) W`` with
:math:`D'/D = \varphi` **factors it out**: ``W`` solves the same PDE with the
state-only discount ``x + y`` (a *time-independent* operator, built once), and
the deterministic part re-enters as a per-step scalar discount
:math:`\exp(-\int_{t_k}^{t_{k+1}} \varphi\,ds)` applied along the backward
sweep. The step integral is evaluated **exactly** — the market-forward part
telescopes into a discount-factor ratio and the convexity/cross parts have
closed-form antiderivatives — which is what preserves the model's exact fit to
the initial curve (the same discipline as Hull-White's ``hw_alpha_average``; a
midpoint sample would inject a curved-forward bias). The per-step factor and the
Bermudan exercise projection both plug into the stepper's ``event_fn`` seam.

Boundaries
----------
Neither factor axis has a closed-form far-field value once the instrument
carries optionality, so both impose zero curvature via
:func:`~valax.pricing.pde.boundary.apply_linearity_bc_2d` (the 2-D analogue of
the Hull-White edge treatment). The domain is sized from the factors' terminal
covariance (:func:`~valax.models.g2pp.g2pp_factor_covariance`); the factors are
zero-mean so the grid is centred at the origin.

Registered recipes
------------------
================================  ==================================
Instrument                        Oracle used to validate it
================================  ==================================
``FixedRateBond``                 analytic curve price (no optionality)
``Swaption``                      analytic ``g2pp_swaption_price``
``BermudanSwaption``              co-terminal European bounds, QuantLib FD
================================  ==================================

The option-free bond is not a product of interest here — it is the *calibration*
of the scheme: if the PDE cannot reprice a plain coupon bond off its own input
curve, nothing built on top of it can be believed.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 4 (§4.2).
    K. J. in't Hout and S. Foulon, "ADI finite difference schemes for option
    pricing in the Heston model with correlation" (2010).
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from valax.dates.daycounts import year_fraction
from valax.instruments.bonds import FixedRateBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.g2pp import (
    G2PPModel,
    g2pp_B,
    g2pp_bond_price,
    g2pp_factor_covariance,
    g2pp_market_df,
)
from valax.pricing.pde.boundary import Boundary2D, apply_linearity_bc_2d
from valax.pricing.pde.coefficients import g2pp_operator_2d
from valax.pricing.pde.config import PDEConfig2D
from valax.pricing.pde.dispatch import PDEResult, register
from valax.pricing.pde.grids import Grid2D, centred_state_grid, read_off_2d
from valax.pricing.pde.schemes2d import solve_backward_2d

# Minimum fully-implicit (Rannacher) start-up steps. Every instrument here has a
# kinked terminal payoff (swaptions) or a projection that reintroduces a kink
# (Bermudan), both of which excite Craig-Sneyd oscillations.
_MIN_RANNACHER = 2

# An event returns a modified field given (forward level, field).
_Event = Callable[[Int[Array, ""], Float[Array, "n_x n_y"]], Float[Array, "n_x n_y"]]


# ─────────────────────────────────────────────────────────────────────
# Schedule helpers (local copies of the Hull-White recipe's, kept private
# so the two short-rate PDE modules stay independent).
# ─────────────────────────────────────────────────────────────────────


def _snap_to_levels(
    dates: Int[Array, " m"],
    reference_date: Int[Array, ""],
    day_count: str,
    dt: Float[Array, ""],
    n_time: int,
) -> Int[Array, " m"]:
    """Map ordinal dates to the nearest backward-sweep time level.

    Levels count forward time: ``0`` at the reference date, ``n_time`` at the
    horizon. Exercise decisions must land *at* a level; the snapping error is
    second order because the exercise boundary is smooth in time.

    Args:
        dates: Event dates as integer ordinals.
        reference_date: Curve reference date as an integer ordinal.
        day_count: Day-count convention used to convert to year fractions.
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.

    Returns:
        The clipped time-level index of each date.
    """
    times = year_fraction(reference_date, dates, day_count)
    return jnp.clip(jnp.round(times / dt).astype(jnp.int32), 0, n_time)


def _swap_accruals(
    start_date: Int[Array, ""],
    payment_dates: Int[Array, " n"],
    day_count: str,
) -> Float[Array, " n"]:
    """Year fractions for each fixed-leg accrual period.

    The first period accrues from ``start_date`` (the swap start / option
    exercise date); subsequent periods run between consecutive payment dates.

    Args:
        start_date: Swap effective date as an integer ordinal.
        payment_dates: Fixed-leg payment dates as integer ordinals.
        day_count: Day-count convention.

    Returns:
        The accrual factor for each fixed payment.
    """
    starts = jnp.concatenate([start_date[jnp.newaxis], payment_dates[:-1]])
    return year_fraction(starts, payment_dates, day_count)


# ─────────────────────────────────────────────────────────────────────
# Grid-node valuation via the G2++ affine bond price
# ─────────────────────────────────────────────────────────────────────


def _g2pp_coupon_bond_values(
    model: G2PPModel,
    grid: Grid2D,
    t: Float[Array, ""],
    payment_times: Float[Array, " n_cf"],
    cashflows: Float[Array, " n_cf"],
) -> Float[Array, "n_x n_y"]:
    r"""Value at ``t`` of :math:`\sum_i c_i P(t, T_i)` at every ``(x, y)`` node.

    Uses G2++'s affine structure
    :math:`P(t, T \mid x, y) = A(t, T)\,e^{-B_a x - B_b y}`, so the whole
    node :math:`\times` cashflow tensor is one broadcast product rather than a
    per-node solve. ``A`` is recovered by evaluating the validated analytic bond
    price at ``x = y = 0`` (one implementation of the coefficient in the
    codebase); ``B_a``, ``B_b`` are the two mean-reversion decay factors.

    Args:
        model: G2++ model.
        grid: The ``(x, y)`` factor grid.
        t: Valuation time in year fractions.
        payment_times: Cashflow times in year fractions.
        cashflows: Cashflow amounts.

    Returns:
        The portfolio value at each grid node, shape ``(n_x, n_y)``.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    zero = jnp.zeros(())

    A = g2pp_bond_price(model, zero, zero, t, payment_times)  # (n_cf,)
    B_a = g2pp_B(a, payment_times - t)                        # (n_cf,)
    B_b = g2pp_B(b, payment_times - t)                        # (n_cf,)

    x = grid.x.nodes[:, jnp.newaxis, jnp.newaxis]  # (n_x, 1, 1)
    y = grid.y.nodes[jnp.newaxis, :, jnp.newaxis]  # (1, n_y, 1)
    discounts = A * jnp.exp(-B_a * x - B_b * y)     # (n_x, n_y, n_cf)
    return discounts @ cashflows


def _g2pp_discount_grid(
    model: G2PPModel,
    grid: Grid2D,
    t: Float[Array, ""],
    payment_time: Float[Array, ""],
) -> Float[Array, "n_x n_y"]:
    """Discount factor :math:`P(t, T \\mid x, y)` at every grid node.

    Args:
        model: G2++ model.
        grid: The ``(x, y)`` factor grid.
        t: Valuation time in year fractions.
        payment_time: Bond maturity in year fractions.

    Returns:
        The discount factor at each grid node, shape ``(n_x, n_y)``.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    zero = jnp.zeros(())

    A = g2pp_bond_price(model, zero, zero, t, payment_time)
    B_a = g2pp_B(a, payment_time - t)
    B_b = g2pp_B(b, payment_time - t)
    x = grid.x.nodes[:, jnp.newaxis]
    y = grid.y.nodes[jnp.newaxis, :]
    return A * jnp.exp(-B_a * x - B_b * y)


def _swap_tail_values(
    model: G2PPModel,
    grid: Grid2D,
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    day_count: str,
    is_payer: bool,
) -> Float[Array, "n_x n_y"]:
    r"""Value of the underlying swap at ``exercise_date``, at every grid node.

    On unit notional a fixed-for-float swap entered at :math:`T_0` is worth
    :math:`1 - \sum_i c_i P(T_0, T_i)` to the payer, with :math:`c_i = K\tau_i`
    and the principal folded into the final flow — the standard replication of
    the floating leg. The receiver's value is the negative of that. Computed
    analytically from the affine bond price, so exercise decisions carry no
    discretisation error.

    Args:
        model: G2++ model.
        grid: The ``(x, y)`` factor grid.
        exercise_date: Swap start / option exercise date as an integer ordinal.
        payment_dates: Fixed-leg payment dates as integer ordinals.
        strike: Fixed rate.
        notional: Notional principal.
        day_count: Day-count convention.
        is_payer: ``True`` for a payer swap, ``False`` for a receiver.

    Returns:
        The swap value at each grid node, shape ``(n_x, n_y)``.
    """
    reference_date = model.initial_curve.reference_date
    t = year_fraction(reference_date, exercise_date, day_count)
    payment_times = year_fraction(reference_date, payment_dates, day_count)

    taus = _swap_accruals(exercise_date, payment_dates, day_count)
    cashflows = (strike * taus).at[-1].add(1.0)

    bond = _g2pp_coupon_bond_values(model, grid, t, payment_times, cashflows)
    payer = notional * (1.0 - bond)
    return payer if is_payer else -payer


# ─────────────────────────────────────────────────────────────────────
# Exact per-step phi(t) discount (the exact-fit discipline)
# ─────────────────────────────────────────────────────────────────────


def _phi_step_discounts(
    model: G2PPModel,
    dt: Float[Array, ""],
    n_time: int,
) -> Float[Array, " n_time"]:
    r"""Per-step scalar discount :math:`\exp(-\int_{t_k}^{t_{k+1}} \varphi\,ds)`.

    Returns one factor per backward step, indexed by the **forward** level
    ``k`` of the interval :math:`[k\,dt, (k+1)\,dt]` (``k = 0 .. n_time - 1``),
    so the recipe applies ``phi_disc[level]`` to the field just solved at
    forward ``level``.

    The integral is exact. The market-forward part telescopes,

    .. math::

        \int_{t_0}^{t_1} f^M(0, s)\,ds = \ln P^M(0, t_0) - \ln P^M(0, t_1),

    and the convexity/cross part has closed-form antiderivatives

    .. math::

        \int_0^t (1 - e^{-zs})^2\,ds
            = t - \tfrac{2}{z}(1 - e^{-zt}) + \tfrac{1}{2z}(1 - e^{-2zt}), \\
        \int_0^t (1 - e^{-as})(1 - e^{-bs})\,ds
            = t - \tfrac{1 - e^{-at}}{a} - \tfrac{1 - e^{-bt}}{b}
              + \tfrac{1 - e^{-(a+b)t}}{a + b},

    combined with the :math:`\varphi` convexity weights
    :math:`\sigma^2/2a^2`, :math:`\eta^2/2b^2`, :math:`\rho\sigma\eta/ab`.
    Doing this exactly (rather than sampling :math:`\varphi` at a midpoint) is
    what preserves the model's exact fit to the initial curve.

    Args:
        model: G2++ model carrying the initial curve and parameters.
        dt: Time-step size in year fractions.
        n_time: Number of backward time steps.

    Returns:
        The ``n_time`` per-step discount factors, indexed by forward level.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation

    t = jnp.arange(n_time + 1) * dt  # level boundaries t_0 .. t_{n_time}

    log_df = jnp.log(g2pp_market_df(model, t))  # F(t) = ln P^M(0, t)

    def _sq_integral(z: Float[Array, ""]) -> Float[Array, " n_time_plus_1"]:
        # int_0^t (1 - e^{-z s})^2 ds
        return (
            t
            - (2.0 / z) * (1.0 - jnp.exp(-z * t))
            + (1.0 / (2.0 * z)) * (1.0 - jnp.exp(-2.0 * z * t))
        )

    cross_integral = (
        t
        - (1.0 - jnp.exp(-a * t)) / a
        - (1.0 - jnp.exp(-b * t)) / b
        + (1.0 - jnp.exp(-(a + b) * t)) / (a + b)
    )
    convexity_integral = (
        (sigma**2 / (2.0 * a**2)) * _sq_integral(a)
        + (eta**2 / (2.0 * b**2)) * _sq_integral(b)
        + (rho * sigma * eta / (a * b)) * cross_integral
    )

    # int_{t_k}^{t_{k+1}} phi ds = [F(t_k) - F(t_{k+1})] + [C(t_{k+1}) - C(t_k)].
    int_forward = log_df[:-1] - log_df[1:]
    int_convexity = convexity_integral[1:] - convexity_integral[:-1]
    return jnp.exp(-(int_forward + int_convexity))


# ─────────────────────────────────────────────────────────────────────
# Backward-sweep driver
# ─────────────────────────────────────────────────────────────────────


def _zero_boundary_2d() -> Boundary2D:
    """Inert log-``x`` Dirichlet data (the linearity fold zeroes the edge bands).

    :func:`~valax.pricing.pde.boundary.apply_linearity_bc_2d` folds the exterior
    ghosts back into the interior and zeroes the first/last ``A1`` bands, so the
    stepper's Dirichlet term is multiplied by zero and never used. This is the
    correct inert placeholder to hand it.

    Returns:
        A :class:`~valax.pricing.pde.boundary.Boundary2D` returning zero.
    """
    zero = lambda tau: jnp.zeros_like(tau)
    return Boundary2D(zero, zero)


def _solve_g2pp(
    model: G2PPModel,
    config: PDEConfig2D,
    horizon: Float[Array, ""],
    build: Callable[[Grid2D], tuple[Float[Array, "n_x n_y"], Optional[_Event]]],
) -> Float[Array, ""]:
    """Run the G2++ backward ADI sweep and read the price off at ``(0, 0)``.

    Sizes the ``(x, y)`` grid from the factors' terminal covariance, builds the
    linearity-folded state-discount operator (``x + y``), obtains the terminal
    condition and an optional instrument event from ``build``, then sweeps with
    the deterministic :math:`\\varphi(t)` discount applied per step through the
    stepper's ``event_fn`` seam.

    The combined event applies ``phi_disc[level]`` to the just-solved field
    (completing the discount over the step) and then the instrument event (an
    additive coupon injection or a ``max`` exercise projection), so the
    instrument event always sees a fully-discounted continuation value.

    Args:
        model: G2++ model carrying the initial curve and parameters.
        config: 2-D grid configuration. ``n_x`` / ``n_y`` are the factor axis
            sizes and ``x_range`` the half-width (in std-dev units) applied to
            *both* factor axes.
        horizon: Time to the last modelled date, in year fractions.
        build: Given the grid, returns ``(terminal, instrument_event)``; the
            event is ``None`` for a European claim.

    Returns:
        The price at the curve reference date.
    """
    cov = g2pp_factor_covariance(model, horizon)
    std_x = jnp.sqrt(cov[0, 0])
    std_y = jnp.sqrt(cov[1, 1])
    grid = Grid2D(
        x=centred_state_grid(std_x, n=config.n_x, half_width=config.x_range),
        y=centred_state_grid(std_y, n=config.n_y, half_width=config.x_range),
    )
    operator = apply_linearity_bc_2d(g2pp_operator_2d(model, grid), grid)

    dt = horizon / config.n_time
    phi_disc = _phi_step_discounts(model, dt, config.n_time)  # (n_time,)

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
    # x(0) = y(0) = 0 by construction, so the price is read at the origin.
    return read_off_2d(grid, values, jnp.zeros(()), jnp.zeros(()))


# ─────────────────────────────────────────────────────────────────────
# Recipes
# ─────────────────────────────────────────────────────────────────────


def _bond_horizon(bond: FixedRateBond, model: G2PPModel) -> Float[Array, ""]:
    """Year fraction from the curve reference date to the bond's maturity."""
    return year_fraction(
        model.initial_curve.reference_date,
        bond.payment_dates[-1],
        bond.day_count,
    )


def _coupon_values_by_level(
    bond: FixedRateBond,
    model: G2PPModel,
    grid: Grid2D,
    config: PDEConfig2D,
    horizon: Float[Array, ""],
) -> Float[Array, "n_levels n_x n_y"]:
    r"""Coupon value to inject at each time level, per grid node.

    A coupon due at :math:`t_c` is attached to the nearest time level
    :math:`t_k` and scaled by the analytic G2++ discount factor
    :math:`P(t_k, t_c \mid x, y)`, which removes the cashflow-timing error that
    plain date-snapping would introduce (the factor is exact at every node for
    either sign of :math:`t_c - t_k`).

    Args:
        bond: A fixed-coupon bond instrument.
        model: G2++ model.
        grid: The factor grid.
        config: 2-D grid configuration.
        horizon: Time to the bond's final payment, in year fractions.

    Returns:
        Coupon value indexed by ``(time level, x node, y node)``; zero where no
        coupon is due.
    """
    reference_date = model.initial_curve.reference_date
    dt = horizon / config.n_time
    payment_times = year_fraction(
        reference_date, bond.payment_dates, bond.day_count
    )
    levels = jnp.clip(
        jnp.round(payment_times / dt).astype(jnp.int32), 0, config.n_time
    )
    coupon = bond.face_value * bond.coupon_rate / bond.frequency

    def factor(level_time, payment_time):
        return _g2pp_discount_grid(model, grid, level_time, payment_time)

    factors = jax.vmap(factor)(levels * dt, payment_times)  # (n_cpn, n_x, n_y)
    return (
        jnp.zeros((config.n_time + 1, grid.n_x, grid.n_y))
        .at[levels]
        .add(coupon * factors)
    )


@register(FixedRateBond, G2PPModel)
def _fixed_rate_bond_g2pp(*, instrument, model, config) -> PDEResult:
    """Option-free fixed-rate bond under G2++ — the scheme's calibration.

    The bond has no optionality, so the answer is fixed by the initial curve
    alone and is *independent of* the model parameters. That makes it the
    sharpest available check on the solver itself: any error here is pure
    numerics (mesh, ADI time-stepping, or the exact-fit :math:`\\varphi`
    treatment), with no model risk mixed in.
    """
    horizon = _bond_horizon(instrument, model)

    def build(grid: Grid2D):
        coupons = _coupon_values_by_level(
            instrument, model, grid, config, horizon
        )
        terminal = (
            jnp.full(grid.shape, instrument.face_value) + coupons[config.n_time]
        )
        return terminal, lambda level, values: values + coupons[level]

    return PDEResult(price=_solve_g2pp(model, config, horizon, build))


@register(Swaption, G2PPModel)
def _swaption_g2pp(*, instrument, model, config) -> PDEResult:
    """European swaption under G2++ via a 2-D finite-difference sweep.

    Redundant with the exact Gauss-Hermite decomposition in
    :func:`~valax.pricing.analytic.g2pp_swaptions.g2pp_swaption_price`, and
    deliberately so: it shares the entire discretisation with the Bermudan
    recipe below, which has no closed form, so agreeing with the analytic price
    to grid tolerance validates that machinery against an exact answer.
    """
    horizon = year_fraction(
        model.initial_curve.reference_date,
        instrument.expiry_date,
        instrument.day_count,
    )

    def build(grid: Grid2D):
        swap = _swap_tail_values(
            model,
            grid,
            instrument.expiry_date,
            instrument.fixed_dates,
            instrument.strike,
            instrument.notional,
            instrument.day_count,
            instrument.is_payer,
        )
        return jnp.maximum(swap, 0.0), None

    return PDEResult(price=_solve_g2pp(model, config, horizon, build))


@register(BermudanSwaption, G2PPModel)
def _bermudan_swaption_g2pp(*, instrument, model, config) -> PDEResult:
    """Bermudan swaption under G2++ via a 2-D finite-difference sweep.

    The headline instrument for this pricer, and the reason G2++ exists: a
    Bermudan swaption's value depends on the *joint* dynamics of short and long
    rates — the decorrelation a one-factor model structurally cannot express —
    and it has no closed form. The backward PDE handles it natively because the
    exercise decision is a pointwise projection on the value function.

    At exercise date ``exercise_dates[e]`` the holder enters the *tail* swap
    paying on ``fixed_dates[e:]``. That tail value is computed analytically at
    every node from G2++'s affine bond price rather than accrued through the
    mesh, so exercise decisions carry no discretisation error of their own —
    the only numerical error is in the continuation value.

    The horizon is the **last** exercise date; the option pays nothing before
    exercise, so between exercise dates the sweep is pure discounting.
    """
    reference_date = model.initial_curve.reference_date
    day_count = instrument.day_count
    n_exercise = instrument.exercise_dates.shape[0]

    horizon = year_fraction(
        reference_date, instrument.exercise_dates[-1], day_count
    )
    dt = horizon / config.n_time
    levels = _snap_to_levels(
        instrument.exercise_dates, reference_date, day_count, dt, config.n_time
    )

    def build(grid: Grid2D):
        # ``n_exercise`` is a static shape, so this Python loop unrolls at trace
        # time; each row uses a different (shorter) tail schedule.
        tails = jnp.stack([
            _swap_tail_values(
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
        ])  # (n_exercise, n_x, n_y)

        # The last exercise date *is* the horizon: exercise or expire worthless.
        terminal = jnp.maximum(tails[-1], 0.0)
        # -inf is the neutral obstacle: it never wins the maximum, so
        # non-exercise levels pass the continuation value straight through.
        exercise_at_level = (
            jnp.full((config.n_time + 1, grid.n_x, grid.n_y), -jnp.inf)
            .at[levels]
            .set(tails)
        )
        event_fn = lambda level, values: jnp.maximum(
            values, exercise_at_level[level]
        )
        return terminal, event_fn

    return PDEResult(price=_solve_g2pp(model, config, horizon, build))
