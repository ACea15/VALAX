r"""Hull-White European swaption pricing via Jamshidian decomposition.

Under a one-factor short-rate model the payoff of a European swaption is a
function of the single state variable :math:`r(T_0)`.  Jamshidian (1989)
exploits this: because every zero-coupon bond price :math:`P(T_0, T_i)` is
*monotonically decreasing* in :math:`r(T_0)`, so is the coupon bond

.. math::

    CB(r) = \sum_{i=1}^{n} c_i\,P(T_0, T_i, r),
    \qquad c_i = K\tau_i + \delta_{in},

and there is a unique critical rate :math:`r^\star` with
:math:`CB(r^\star) = V`, where :math:`V` is the value of the floating leg at
:math:`T_0` (:math:`V = 1` in the single-curve case, where the leg resets to
par).  The option on the *coupon* bond therefore decomposes exactly into a
portfolio of options on *zero-coupon* bonds, each struck at
:math:`X_i = P(T_0, T_i, r^\star)`:

.. math::

    \bigl(V - CB(r)\bigr)^+ = \sum_{i=1}^{n} c_i\,\bigl(X_i - P(T_0,T_i,r)\bigr)^+

With a distinct projection curve, a frozen deterministic tenor basis makes
:math:`V = b_1` and shifts each :math:`c_i` by :math:`1 - b_{i+1}`
(:mod:`valax.pricing.multicurve`), so the decomposition above carries over
unchanged — which is what keeps the dual-curve price closed-form.

Each zero-coupon bond option has a Black-like closed form under Hull-White,
so the swaption price is a finite sum of closed-form terms -- no lattice, no
simulation, and no numerical integration.

A payer swaption is a **put** on the coupon bond (you gain when rates rise and
the bond falls); a receiver swaption is a **call**.

The critical rate is found with an ``optimistix`` Newton solve.  Because
``optimistix`` root-finds are implicitly differentiable, ``jax.grad`` flows
through :math:`r^\star` correctly without differentiating the solver's
iterations.

This is the pricer that makes Hull-White *calibratable*: it turns a swaption
volatility surface into a set of model prices cheap enough to sit inside a
least-squares objective (see :mod:`valax.calibration.hull_white`).

References:
    Jamshidian (1989), "An Exact Bond Option Formula", *Journal of Finance*.
    Brigo & Mercurio (2006), *Interest Rate Models*, §3.3 (eqs. 3.40-3.41).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.instruments.rates import Swaption
from valax.models.hull_white import (
    HullWhiteModel,
    hw_B,
    hw_bond_price,
    hw_instantaneous_forward,
    hw_market_df,
    hw_short_rate_variance,
)
from valax.models.short_rate import is_constant, require_initial_curve
from valax.pricing.multicurve import (
    apply_swap_basis,
    model_swap_leg_basis,
)

# Floor for the zero-coupon-bond option volatility.  At zero time-to-expiry or
# zero model volatility the option is worth its intrinsic value; clamping keeps
# the division in `h` finite and the gradient well-defined rather than NaN.
_MIN_SIGMA_P = 1e-12


def _swaption_accruals(
    swaption: Swaption,
) -> Float[Array, " n_fixed"]:
    """Year fractions for each fixed-leg accrual period.

    Args:
        swaption: Swaption contract; the first period accrues from the
            expiry (swap start) date.

    Returns:
        Accrual factor for each fixed payment.
    """
    starts = jnp.concatenate(
        [swaption.expiry_date[None], swaption.fixed_dates[:-1]]
    )
    return year_fraction(starts, swaption.fixed_dates, swaption.day_count)


def hw_critical_rate(
    model: HullWhiteModel,
    expiry_time: Float[Array, ""],
    cashflow_times: Float[Array, " n_fixed"],
    cashflows: Float[Array, " n_fixed"],
    initial_guess: Float[Array, ""] | None = None,
    par_value: Float[Array, ""] | float = 1.0,
) -> Float[Array, ""]:
    r"""Short rate :math:`r^\star` at which the coupon bond is worth par.

    Solves :math:`\sum_i c_i P(T_0, T_i, r) = V` for :math:`r`, where
    :math:`V` is ``par_value``.  The left-hand side is strictly decreasing in
    :math:`r` (every :math:`B(T_0,T_i) > 0`, provided every :math:`c_i > 0`),
    so the root is unique and Newton's method converges from any sensible
    start.

    ``par_value`` is ``1`` in the single-curve case, where the floating leg
    resets to par.  With a deterministic tenor basis the floating leg is worth
    ``float_start`` instead (see
    :func:`valax.pricing.multicurve.apply_swap_basis`), and the exercise
    boundary moves accordingly.

    Args:
        model: Hull-White model.
        expiry_time: Option expiry :math:`T_0` in year fractions.
        cashflow_times: Fixed-leg payment times :math:`T_i` in year fractions.
        cashflows: Coupon-bond cash flows :math:`c_i` (unit notional, with the
            principal folded into the final flow).
        initial_guess: Starting short rate.  Defaults to the market
            instantaneous forward at ``expiry_time``.
        par_value: Value the coupon bond must equal at :math:`r^\star` — the
            swaption's effective strike on the bond.  Defaults to ``1.0``.

    Returns:
        Critical short rate :math:`r^\star`, or NaN if the Newton solve fails
        — a failed decomposition surfaces as a non-finite price (inside a
        calibration, a named non-finite residual) instead of a silently wrong
        one.

    Notes:
        **Guarded.** Differentiating through a *failed* root-find would make
        its implicit derivative solve a non-finite system, which raises at
        runtime inside ``jit`` / ``vmap`` (one absurd iterate of a
        calibration would abort the whole batch). So a gradient-free probe
        solve decides ``ok`` first, and the differentiable solve then runs on
        ``where(ok, real input, detached well-posed stand-in)``: on the
        healthy path every input is the real one, so the price *and* its
        gradient are exactly those of the unguarded solve; on failure the
        solve sees the stand-in and the result is NaN.
    """
    if initial_guess is None:
        initial_guess = hw_instantaneous_forward(model, expiry_time)

    def solve(m, c, pv, guess):
        def residual(r: Float[Array, ""], args) -> Float[Array, ""]:
            bond = jnp.sum(c * hw_bond_price(m, r, expiry_time, cashflow_times))
            return bond - pv

        return optx.root_find(
            residual,
            optx.Newton(rtol=1e-12, atol=1e-12),
            guess,
            max_steps=100,
            throw=False,
        )

    detach = jax.lax.stop_gradient
    par_value = jnp.asarray(par_value)
    probe = solve(
        jax.tree.map(detach, model), detach(cashflows), detach(par_value),
        detach(initial_guess),
    )
    ok = (probe.result == optx.RESULTS.successful) & jnp.isfinite(probe.value)

    # A benign problem with the same structure: moderate parameters, finite
    # flows. Only ever read on the failure path, and detached there.
    stand_in = eqx.tree_at(
        lambda m: (m.mean_reversion, m.volatility),
        model,
        (jnp.full_like(model.mean_reversion, 0.05),
         jnp.full_like(model.volatility, 0.01)),
    )
    safe_model = jax.tree.map(
        lambda x, s: jnp.where(ok, x, detach(s)), model, stand_in
    )
    safe_cashflows = jnp.where(
        ok, cashflows, detach(jnp.where(jnp.isfinite(cashflows), cashflows, 1.0))
    )
    safe_par = jnp.where(ok, par_value, 1.0)
    safe_guess = jnp.where(
        ok, initial_guess,
        detach(jnp.where(jnp.isfinite(initial_guess), initial_guess, 0.0)),
    )
    sol = solve(safe_model, safe_cashflows, safe_par, safe_guess)
    # A `where`, not a NaN multiplier: a NaN cotangent would reach the
    # implicit-derivative linear solve and raise there. The failed entry's own
    # gradient is therefore masked to zero — the failure is signalled by the
    # NaN *value*, which makes every price built on it (and its gradient) NaN.
    return jnp.where(
        ok & (sol.result == optx.RESULTS.successful), sol.value, jnp.nan
    )


def hw_zcb_option_price(
    model: HullWhiteModel,
    expiry_time: Float[Array, ""],
    maturity_times: Float[Array, " n"],
    strikes: Float[Array, " n"],
    is_call: bool,
) -> Float[Array, " n"]:
    r"""Closed-form Hull-White option on a zero-coupon bond.

    With :math:`T` the option expiry and :math:`S` the bond maturity,

    .. math::

        \sigma_p = \sigma\,\sqrt{\frac{1 - e^{-2aT}}{2a}}\;B(T, S),
        \qquad
        h = \frac{1}{\sigma_p}\ln\frac{P(0,S)}{P(0,T)X} + \frac{\sigma_p}{2}

    .. math::

        ZBC = P(0,S)\,\Phi(h) - X P(0,T)\,\Phi(h - \sigma_p)

    .. math::

        ZBP = X P(0,T)\,\Phi(-h + \sigma_p) - P(0,S)\,\Phi(-h)

    With a piecewise-constant :math:`\sigma(t)` only the bond-price volatility
    changes, to :math:`\sigma_p = B(T, S)\sqrt{\int_0^T \sigma(u)^2
    e^{-2a(T-u)}\,du}` (the variance of the state at the expiry), so the
    whole Jamshidian decomposition carries over unchanged.

    Args:
        model: Hull-White model.
        expiry_time: Option expiry :math:`T` in year fractions.
        maturity_times: Underlying bond maturities :math:`S` in year fractions.
        strikes: Strike price :math:`X` for each bond.
        is_call: ``True`` for calls (``ZBC``), ``False`` for puts (``ZBP``).

    Returns:
        Option price for each maturity, as of the reference date.
    """
    a = model.mean_reversion

    df_expiry = hw_market_df(model, expiry_time)
    df_maturity = hw_market_df(model, maturity_times)

    # Standard deviation of ln P(T, S) under the T-forward measure.
    if is_constant(model.volatility):
        sigma = model.volatility[0]
        variance_factor = jnp.sqrt(
            (1.0 - jnp.exp(-2.0 * a * expiry_time)) / (2.0 * a)
        )
        sigma_p = sigma * variance_factor * hw_B(a, maturity_times - expiry_time)
    else:
        sigma_p = jnp.sqrt(hw_short_rate_variance(model, expiry_time)) * hw_B(
            a, maturity_times - expiry_time
        )
    sigma_p = jnp.maximum(sigma_p, _MIN_SIGMA_P)

    h = (
        jnp.log(df_maturity / (df_expiry * strikes)) / sigma_p
        + 0.5 * sigma_p
    )

    Phi = jax.scipy.stats.norm.cdf
    if is_call:
        return df_maturity * Phi(h) - strikes * df_expiry * Phi(h - sigma_p)
    return strikes * df_expiry * Phi(-h + sigma_p) - df_maturity * Phi(-h)


def hw_swaption_price(
    swaption: Swaption,
    model: HullWhiteModel,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    r"""European swaption price under Hull-White (Jamshidian decomposition).

    Exact within the model -- no lattice or simulation.  The swaption is
    rewritten as a portfolio of :math:`n` zero-coupon bond options struck at
    the critical-rate bond prices, each of which has a closed form.

    Discounting uses ``model.initial_curve``, so the price is automatically
    consistent with the curve Hull-White was exact-fitted to.  Passing a
    distinct ``forward_curve`` projects the floating leg off that curve
    instead, through the frozen deterministic tenor basis of
    :mod:`valax.pricing.multicurve` — the same overlay (and the same
    :func:`~valax.pricing.multicurve.apply_swap_basis` algebra) the MC and PDE
    recipes use, so the analytic, lattice-free price stays consistent with
    them.  The floating leg is assumed to pay on the fixed schedule, matching
    the convention of :func:`valax.pricing.analytic.swaptions.swap_rate`.

    Args:
        swaption: Swaption contract (``is_payer`` selects payer vs receiver).
        model: Hull-White model carrying the initial discount curve.
        forward_curve: Optional projection curve for the underlying swap's
            floating leg.  ``None`` (or the discount curve itself) is the
            single-curve case, evaluated by the pre-existing code path
            bit-for-bit.

    Returns:
        Swaption price as of the curve reference date.

    Notes:
        Jamshidian's decomposition requires the coupon bond to be monotone in
        :math:`r`, i.e. every adjusted cash flow :math:`c_i > 0`.  The basis
        shifts each flow by :math:`1 - b_{i+1}`, which is small (a 50 bp
        one-year basis moves a flow by ~0.005) and therefore harmless at
        realistic strikes, but a near-zero strike combined with a very large
        basis could in principle flip a flow negative and invalidate the
        decomposition.  The dual-curve price is validated against the
        independent PDE recipe in ``tests/test_pricing/test_dual_curve_swaptions.py``.
    """
    curve: DiscountCurve = require_initial_curve(model)
    ref = curve.reference_date
    day_count = swaption.day_count

    expiry_time = year_fraction(ref, swaption.expiry_date, day_count)
    cashflow_times = year_fraction(ref, swaption.fixed_dates, day_count)

    # Coupon-bond cash flows on unit notional: K*tau_i, plus principal at T_n.
    taus = _swaption_accruals(swaption)
    cashflows = swaption.strike * taus
    cashflows = cashflows.at[-1].add(1.0)

    # Frozen tenor basis: single-curve returns None, a trace-time branch that
    # leaves `float_start = 1.0` and the cash flows untouched.
    basis = model_swap_leg_basis(
        model, forward_curve, swaption.expiry_date, swaption.fixed_dates
    )
    float_start, cashflows = apply_swap_basis(cashflows, basis)

    r_star = hw_critical_rate(
        model, expiry_time, cashflow_times, cashflows, par_value=float_start
    )
    strikes = hw_bond_price(model, r_star, expiry_time, cashflow_times)

    # Payer = put on the coupon bond, receiver = call.
    options = hw_zcb_option_price(
        model, expiry_time, cashflow_times, strikes,
        is_call=not swaption.is_payer,
    )
    return swaption.notional * jnp.sum(cashflows * options)
