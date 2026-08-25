r"""Hull-White short-rate finite-difference pricing (roadmap PR-3).

Prices interest-rate instruments with embedded optionality by solving the
one-factor Hull-White backward PDE on the existing 1-D theta-scheme substrate.
This is the third independent numerical route to the same model, joining the
trinomial tree (:mod:`valax.pricing.lattice.hull_white_tree`) and the exact-OU
Monte-Carlo engine (:mod:`valax.pricing.mc.hull_white_paths`), which is exactly
the point: a callable-bond price that agrees across tree, MC and PDE is a price
worth trusting.

State variable
--------------
The solver works in the **centred** state ``x`` of the decomposition
:math:`r(t) = x(t) + \alpha(t)` (see :func:`~valax.models.hull_white.hw_alpha`),
where

.. math::

    dx(t) = -a\,x(t)\,dt + \sigma\,dW(t), \qquad x(0) = 0 .

Pricing a claim with value :math:`V(x, \tau)`, :math:`\tau = T - t`, then means
marching

.. math::

    V_\tau = \tfrac{1}{2}\sigma^2 V_{xx} - a x\,V_x
             - \bigl(x + \alpha(t)\bigr) V

back from the terminal condition. Two properties make this the right
coordinate: the drift and diffusion are time-independent (so the mesh can be
built once), and the state starts exactly at the origin (so the price is always
read off at ``x = 0`` — no moving read-off point, and no ``stop_gradient``
scaffolding of the kind the equity recipes need).

Boundaries
----------
There is no closed-form far-field value for a callable bond — whether the
issuer calls depends on the whole remaining schedule — so instead of Dirichlet
data the solver imposes zero curvature (:math:`V_{xx} = 0`) at both edges via
:func:`~valax.pricing.pde.boundary.apply_linearity_bc_1d`. Empirically the
resulting prices are insensitive to the domain width from ~4 standard
deviations outward.

Discrete events
---------------
Coupons and exercise decisions land through the stepper's ``event_fn`` seam.
The **ordering within a step matters and follows the tree**: exercise is
decided on the *ex-coupon* continuation value, and the coupon is added
afterwards, because call and put prices are quoted ex-coupon and a holder
called on a coupon date still receives that coupon. Getting this backwards
undervalued callable bonds by up to a full coupon — a real bug caught earlier
in the tree implementation by the QuantLib comparison harness.

Cashflow dates rarely land on a time level. Rather than simply snapping them —
which displaces every coupon by up to ``dt/2`` and injects an :math:`O(\Delta t)`
error that dominates everything else (it cost ~2e-3 on a 5-year bullet, three
orders of magnitude worse than the scheme's own ~1e-6) — a coupon due at
:math:`t_c` is injected at the nearest level :math:`t_k` **multiplied by the
analytic Hull-White bond price** :math:`P(t_k, t_c \mid x)`. That factor is
exact at every node and for either sign of :math:`t_c - t_k`, so cashflow
timing contributes no discretisation error at all and the scheme's second-order
convergence is visible rather than swamped. Exercise dates are still snapped —
an exercise decision has to happen *at* a time level — but that error is
second order, because the exercise boundary is smooth in time.

Registered recipes
------------------
================================  ==================================
Instrument                        Oracle used to validate it
================================  ==================================
``FixedRateBond``                 analytic curve price (no optionality)
``Swaption``                      Jamshidian ``hw_swaption_price``
``CallableBond`` / ``PuttableBond``  the HW trinomial tree
``BermudanSwaption``              co-terminal European bounds, QuantLib
================================  ==================================

The two option-free instruments are not interesting products in their own right
here — the tree and the analytic pricers already cover them — but they are the
*calibration* of the scheme: if the PDE cannot reprice a plain coupon bond off
its own input curve, nothing built on top of it can be believed.

References:
    Hull & White (1990), "Pricing Interest-Rate-Derivative Securities".
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 3.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from valax.dates.daycounts import year_fraction
from valax.instruments.bonds import CallableBond, FixedRateBond, PuttableBond
from valax.instruments.rates import BermudanSwaption, Swaption
from valax.models.hull_white import (
    HullWhiteModel,
    hw_B,
    hw_alpha,
    hw_bond_price,
    hw_short_rate_variance,
)
from valax.pricing.pde.boundary import apply_linearity_bc_1d, zero_boundary
from valax.pricing.pde.coefficients import hw_operator_stack
from valax.pricing.pde.config import PDEConfig
from valax.pricing.pde.dispatch import PDEResult, register
from valax.pricing.pde.grids import Grid1D, centred_state_grid, read_off_1d
from valax.pricing.pde.schemes import solve_backward_1d, theta_for_scheme

# Minimum fully-implicit start-up steps. Every instrument here has either a
# kinked terminal payoff (swaptions) or a projection that repeatedly reintroduces
# a kink (callables), both of which excite Crank-Nicolson oscillations.
_MIN_RANNACHER = 2


# ─────────────────────────────────────────────────────────────────────
# Shared machinery
# ─────────────────────────────────────────────────────────────────────


def _snap_to_levels(
    dates: Int[Array, " m"],
    reference_date: Int[Array, ""],
    day_count: str,
    dt: Float[Array, ""],
    n_time: int,
) -> Int[Array, " m"]:
    """Map ordinal dates to the nearest backward-sweep time level.

    Mirrors the trinomial tree's ``_snap_dates_to_steps`` so the two engines
    place contractual events on the same grid and can be compared like for
    like. Levels count forward time: ``0`` at the reference date, ``n_time`` at
    the horizon.

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


def _coupon_bond_values(
    model: HullWhiteModel,
    nodes: Float[Array, " n"],
    t: Float[Array, ""],
    payment_times: Float[Array, " n_cf"],
    cashflows: Float[Array, " n_cf"],
) -> Float[Array, " n"]:
    r"""Value at time ``t`` of :math:`\sum_i c_i P(t, T_i)` at every grid node.

    Uses Hull-White's affine structure, :math:`P(t,T\mid r) = A(t,T)e^{-B(t,T)r}`,
    so the whole node :math:`\times` cashflow matrix is one outer product rather
    than a per-node solve. ``A`` is recovered by evaluating the validated
    analytic bond price at :math:`r = 0`, which keeps a single implementation of
    the coefficient in the codebase.

    Args:
        model: Hull-White model.
        nodes: State-variable (``x``) grid coordinates.
        t: Valuation time in year fractions.
        payment_times: Cashflow times in year fractions.
        cashflows: Cashflow amounts.

    Returns:
        The portfolio value at each grid node.
    """
    rates = nodes + hw_alpha(model, t)                       # r = x + alpha(t)
    B = hw_B(model.mean_reversion, payment_times - t)        # (n_cf,)
    A = hw_bond_price(model, jnp.zeros(()), t, payment_times)  # (n_cf,)
    discounts = A * jnp.exp(-B * rates[:, jnp.newaxis])      # (n, n_cf)
    return discounts @ cashflows


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


def _solve_hull_white(
    model: HullWhiteModel,
    config: PDEConfig,
    horizon: Float[Array, ""],
    build: Callable[
        [Grid1D],
        tuple[
            Float[Array, " n"],
            Optional[Callable[[Int[Array, ""], Float[Array, " n"]], Float[Array, " n"]]],
        ],
    ],
) -> Float[Array, ""]:
    """Run the Hull-White backward sweep and read the price off at ``x = 0``.

    Args:
        model: Hull-White model carrying the initial curve and parameters.
        config: Grid configuration. ``n_spot`` is the number of state nodes and
            ``spot_range`` the domain half-width in standard deviations of the
            state at ``horizon``.
        horizon: Time to the last modelled date, in year fractions.
        build: Given the state grid, returns the terminal condition and an
            optional discrete-event hook. Both usually need node-level data
            (analytic bond prices at the grid coordinates), which is why they
            are built here rather than passed in ready-made.

    Returns:
        The price at the curve reference date.
    """
    std_dev = jnp.sqrt(hw_short_rate_variance(model, horizon))
    grid = centred_state_grid(
        std_dev, n=config.n_spot, half_width=config.spot_range
    )
    operator = apply_linearity_bc_1d(
        hw_operator_stack(model, grid, expiry=horizon, n_time=config.n_time),
        grid,
    )
    terminal, event_fn = build(grid)
    values = solve_backward_1d(
        operator,
        # The linearity fold zeroes the edge bands, so no Dirichlet data is
        # consumed; this is the inert placeholder the stepper still expects.
        zero_boundary(),
        terminal,
        expiry=horizon,
        n_time=config.n_time,
        theta=theta_for_scheme(config.scheme),
        rannacher_steps=max(config.rannacher_steps, _MIN_RANNACHER),
        event_fn=event_fn,
    )
    # x(0) = 0 by construction, so the price is always read at the origin.
    return read_off_1d(grid, values, jnp.zeros(()))


def _coupon_values_by_level(
    bond,
    model: HullWhiteModel,
    grid: Grid1D,
    config: PDEConfig,
    horizon: Float[Array, ""],
) -> Float[Array, "n_levels n"]:
    r"""Coupon value to inject at each time level, per grid node.

    A coupon due at :math:`t_c` is attached to the nearest time level
    :math:`t_k` and scaled by the analytic Hull-White discount factor
    :math:`P(t_k, t_c \mid x)`, which removes the cashflow-timing error that
    plain date-snapping would introduce. The factor is the correct analytic
    continuation for either sign of :math:`t_c - t_k`: when the true payment
    date falls *before* the level it becomes an accumulation factor greater
    than one.

    The scatter uses traced indices, so unlike the trinomial tree — which reads
    ``int(...)`` off the snapped dates and therefore needs concrete schedules —
    these recipes stay fully traceable under ``eqx.filter_jit``.

    Args:
        bond: A fixed-coupon bond instrument.
        model: Hull-White model (supplies the curve reference date).
        grid: The state-variable grid.
        config: Grid configuration.
        horizon: Time to the bond's final payment, in year fractions.

    Returns:
        Coupon value indexed by ``(time level, grid node)``; zero where no
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
        rates = grid.nodes + hw_alpha(model, level_time)
        return hw_bond_price(model, rates, level_time, payment_time)

    factors = jax.vmap(factor)(levels * dt, payment_times)  # (n_cpn, n)
    return (
        jnp.zeros((config.n_time + 1, grid.n))
        .at[levels]
        .add(coupon * factors)
    )


def _bond_horizon(bond, model: HullWhiteModel) -> Float[Array, ""]:
    """Year fraction from the curve reference date to the bond's maturity."""
    return year_fraction(
        model.initial_curve.reference_date,
        bond.payment_dates[-1],
        bond.day_count,
    )


def _callable_style_price(
    bond,
    model: HullWhiteModel,
    config: PDEConfig,
    exercise_dates: Int[Array, " m"],
    exercise_prices: Float[Array, " m"],
    *,
    is_call: bool,
) -> Float[Array, ""]:
    """Shared backward sweep for callable and puttable fixed-rate bonds.

    The two differ only in who owns the option: the issuer caps the value at
    the call price (``is_call``), the holder floors it at the put price.

    Args:
        bond: Callable or puttable bond instrument.
        model: Hull-White model.
        config: Grid configuration.
        exercise_dates: Call or put dates as integer ordinals.
        exercise_prices: Redemption price at each exercise date, as a fraction
            of face value.
        is_call: ``True`` for issuer-optimal (callable), ``False`` for
            holder-optimal (puttable).

    Returns:
        The dirty price at the curve reference date.
    """
    horizon = _bond_horizon(bond, model)
    dt = horizon / config.n_time

    levels = _snap_to_levels(
        exercise_dates,
        model.initial_curve.reference_date,
        bond.day_count,
        dt,
        config.n_time,
    )
    # Non-exercise levels get a neutral obstacle: +inf caps nothing, -inf
    # floors nothing.
    neutral = jnp.inf if is_call else -jnp.inf
    strike_at_level = (
        jnp.full(config.n_time + 1, neutral)
        .at[levels]
        .set(exercise_prices * bond.face_value)
    )

    def build(grid: Grid1D):
        coupons = _coupon_values_by_level(bond, model, grid, config, horizon)
        # Redemption of principal plus the final coupon.
        terminal = jnp.full(grid.n, bond.face_value) + coupons[config.n_time]

        def event_fn(level, values):
            # Exercise is decided *ex-coupon* — call/put prices are quoted
            # ex-coupon, so a holder redeemed on a coupon date still collects
            # that coupon. The coupon is added after the projection.
            obstacle = strike_at_level[level]
            exercised = (
                jnp.minimum(values, obstacle) if is_call
                else jnp.maximum(values, obstacle)
            )
            return exercised + coupons[level]

        return terminal, event_fn

    return _solve_hull_white(model, config, horizon, build)


def _swap_tail_values(
    model: HullWhiteModel,
    nodes: Float[Array, " n"],
    exercise_date: Int[Array, ""],
    payment_dates: Int[Array, " n_cf"],
    strike: Float[Array, ""],
    notional: Float[Array, ""],
    day_count: str,
    is_payer: bool,
) -> Float[Array, " n"]:
    r"""Value of the underlying swap at ``exercise_date``, at every grid node.

    On unit notional a fixed-for-float swap entered at :math:`T_0` is worth
    :math:`1 - \sum_i c_i P(T_0, T_i)` to the payer, with
    :math:`c_i = K\tau_i` and the principal folded into the final flow — the
    standard replication of the floating leg. The receiver's value is the
    negative of that.

    Args:
        model: Hull-White model.
        nodes: State-variable grid coordinates.
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
    t = year_fraction(reference_date, exercise_date, day_count)
    payment_times = year_fraction(reference_date, payment_dates, day_count)

    taus = _swap_accruals(exercise_date, payment_dates, day_count)
    cashflows = (strike * taus).at[-1].add(1.0)

    bond = _coupon_bond_values(model, nodes, t, payment_times, cashflows)
    payer = notional * (1.0 - bond)
    return payer if is_payer else -payer


# ─────────────────────────────────────────────────────────────────────
# Recipes
# ─────────────────────────────────────────────────────────────────────


@register(FixedRateBond, HullWhiteModel)
def _fixed_rate_bond_hw(*, instrument, model, config) -> PDEResult:
    """Option-free fixed-rate bond under Hull-White — the scheme's calibration.

    The bond has no optionality, so the answer is fixed by the initial curve
    alone and is *independent of* ``a`` and ``sigma``. That makes it the
    sharpest available check on the solver itself: any error here is pure
    numerics (mesh, time-stepping, or the exact-fit shift), with no model risk
    mixed in.
    """
    horizon = _bond_horizon(instrument, model)

    def build(grid: Grid1D):
        coupons = _coupon_values_by_level(
            instrument, model, grid, config, horizon
        )
        terminal = (
            jnp.full(grid.n, instrument.face_value) + coupons[config.n_time]
        )
        return terminal, lambda level, values: values + coupons[level]

    return PDEResult(price=_solve_hull_white(model, config, horizon, build))


@register(CallableBond, HullWhiteModel)
def _callable_bond_hw(*, instrument, model, config) -> PDEResult:
    """Callable fixed-rate bond under Hull-White.

    The issuer redeems at ``call_price`` whenever the ex-coupon continuation
    value exceeds it, which caps the bond's value to the holder below that of
    an otherwise identical bullet.
    """
    price = _callable_style_price(
        instrument,
        model,
        config,
        instrument.call_dates,
        instrument.call_prices,
        is_call=True,
    )
    return PDEResult(price=price)


@register(PuttableBond, HullWhiteModel)
def _puttable_bond_hw(*, instrument, model, config) -> PDEResult:
    """Puttable fixed-rate bond under Hull-White.

    The holder redeems at ``put_price`` whenever the ex-coupon continuation
    value falls below it, which floors the bond's value above that of an
    otherwise identical bullet.
    """
    price = _callable_style_price(
        instrument,
        model,
        config,
        instrument.put_dates,
        instrument.put_prices,
        is_call=False,
    )
    return PDEResult(price=price)


@register(Swaption, HullWhiteModel)
def _swaption_hw(*, instrument, model, config) -> PDEResult:
    """European swaption under Hull-White via finite differences.

    Redundant with the exact Jamshidian decomposition in
    :func:`~valax.pricing.analytic.hull_white_swaptions.hw_swaption_price`, and
    deliberately so: it shares the entire discretisation with the Bermudan
    recipe below, which has no closed form, so agreeing with Jamshidian to grid
    tolerance validates that machinery against an exact answer.
    """
    horizon = year_fraction(
        model.initial_curve.reference_date,
        instrument.expiry_date,
        instrument.day_count,
    )

    def build(grid: Grid1D):
        swap = _swap_tail_values(
            model,
            grid.nodes,
            instrument.expiry_date,
            instrument.fixed_dates,
            instrument.strike,
            instrument.notional,
            instrument.day_count,
            instrument.is_payer,
        )
        return jnp.maximum(swap, 0.0), None

    return PDEResult(price=_solve_hull_white(model, config, horizon, build))


@register(BermudanSwaption, HullWhiteModel)
def _bermudan_swaption_hw(*, instrument, model, config) -> PDEResult:
    """Bermudan swaption under Hull-White via finite differences.

    The headline instrument for this pricer: a Bermudan swaption has no closed
    form, and the backward PDE handles it natively because the exercise
    decision is a pointwise projection on the value function.

    At exercise date ``exercise_dates[e]`` the holder enters the *tail* swap
    paying on ``fixed_dates[e:]``. That tail value is computed analytically at
    every node from Hull-White's affine bond price rather than being accrued
    through the mesh, so exercise decisions carry no discretisation error of
    their own — the only numerical error is in the continuation value.

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

    def tail_values(grid: Grid1D) -> Float[Array, "n_exercise n"]:
        # ``n_exercise`` is a static shape, so this Python loop unrolls at
        # trace time; each row uses a different (shorter) tail schedule.
        return jnp.stack([
            _swap_tail_values(
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

    def build(grid: Grid1D):
        tails = tail_values(grid)
        # The last exercise date *is* the horizon: exercise or expire worthless.
        terminal = jnp.maximum(tails[-1], 0.0)
        # -inf is the neutral obstacle: it never wins the maximum, so
        # non-exercise levels pass the continuation value straight through.
        exercise_at_level = (
            jnp.full((config.n_time + 1, grid.n), -jnp.inf)
            .at[levels]
            .set(tails)
        )
        event_fn = lambda level, values: jnp.maximum(
            values, exercise_at_level[level]
        )
        return terminal, event_fn

    return PDEResult(price=_solve_hull_white(model, config, horizon, build))
