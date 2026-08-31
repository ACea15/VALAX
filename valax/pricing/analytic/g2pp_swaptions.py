r"""G2++ European swaption pricing (Brigo-Mercurio semi-analytic).

Under the two-factor G2++ model the swaption payoff depends on the *pair*
of state variables :math:`(x(T), y(T))` at expiry, so Jamshidian's
one-dimensional decomposition no longer collapses the coupon-bond option to
a finite sum of closed forms.  Brigo & Mercurio (2006), §4.2.4, give the
exact price as a **one-dimensional integral** over the first factor with a
Jamshidian-style critical-boundary computed on the second factor:

.. math::

    \text{PS} = N\,P(0, T)\int_{-\infty}^{\infty}
        \frac{e^{-\frac12\left(\frac{x-\mu_x}{\sigma_x}\right)^2}}
             {\sigma_x\sqrt{2\pi}}
        \Bigl[\Phi(-h_1(x)) - \sum_{i} \lambda_i(x)\,
              e^{\kappa_i(x)}\,\Phi(-h_2^i(x))\Bigr]\,dx

for a payer swaption (a put on the coupon bond), and the sign-flipped
bracket for a receiver.  The outer integral is evaluated by Gauss-Hermite
quadrature; for each quadrature node the exercise boundary :math:`\bar y(x)`
solving :math:`\sum_i c_i A_i(x)\,e^{-B(b,T,t_i)\,y} = 1` is found with an
``optimistix`` Newton root-find, which is implicitly differentiable so
``jax.grad`` flows through the boundary cleanly.

The discount curve comes from ``model.initial_curve``, so the price is
automatically consistent with the curve G2++ was exact-fitted to.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, §4.2.4 (eq. 4.31).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.instruments.rates import Swaption
from valax.models.g2pp import (
    G2PPModel,
    g2pp_B,
    g2pp_bond_price,
    g2pp_market_df,
)

# Number of Gauss-Hermite quadrature nodes for the outer 1-D integral.  The
# G2++ swaption integrand is smooth and near-Gaussian, so 64 nodes reprice to
# well below 1e-8 relative error versus finer grids.
_N_QUAD = 64
_GH_NODES_NP, _GH_WEIGHTS_NP = np.polynomial.hermite.hermgauss(_N_QUAD)
_GH_NODES = jnp.asarray(_GH_NODES_NP)
_GH_WEIGHTS = jnp.asarray(_GH_WEIGHTS_NP)
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)

_Phi = jax.scipy.stats.norm.cdf


def _swaption_accruals(swaption: Swaption) -> Float[Array, " n_fixed"]:
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


def _forward_measure_moments(
    model: G2PPModel,
    expiry_time: Float[Array, ""],
) -> tuple[
    Float[Array, ""], Float[Array, ""],
    Float[Array, ""], Float[Array, ""],
    Float[Array, ""],
]:
    r"""Means, standard deviations and correlation of :math:`(x, y)` at expiry.

    Under the :math:`T`-forward measure the factors :math:`x(T), y(T)` are
    jointly Gaussian.  Returns :math:`(\mu_x, \mu_y, \sigma_x, \sigma_y,
    \rho_{xy})` following Brigo & Mercurio (2006), eq. (4.31).

    Args:
        model: G2++ model.
        expiry_time: Option expiry :math:`T` in year fractions.

    Returns:
        Tuple ``(mu_x, mu_y, sigma_x, sigma_y, rho_xy)``.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma = model.volatility_x
    eta = model.volatility_y
    rho = model.correlation
    T = expiry_time

    sigma_x = sigma * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T)) / (2.0 * a))
    sigma_y = eta * jnp.sqrt((1.0 - jnp.exp(-2.0 * b * T)) / (2.0 * b))
    rho_xy = (
        rho * sigma * eta / ((a + b) * sigma_x * sigma_y)
        * (1.0 - jnp.exp(-(a + b) * T))
    )

    M_x = (
        (sigma**2 / a**2 + rho * sigma * eta / (a * b)) * (1.0 - jnp.exp(-a * T))
        - (sigma**2 / (2.0 * a**2)) * (1.0 - jnp.exp(-2.0 * a * T))
        - (rho * sigma * eta / (b * (a + b))) * (1.0 - jnp.exp(-(a + b) * T))
    )
    M_y = (
        (eta**2 / b**2 + rho * sigma * eta / (a * b)) * (1.0 - jnp.exp(-b * T))
        - (eta**2 / (2.0 * b**2)) * (1.0 - jnp.exp(-2.0 * b * T))
        - (rho * sigma * eta / (a * (a + b))) * (1.0 - jnp.exp(-(a + b) * T))
    )
    return -M_x, -M_y, sigma_x, sigma_y, rho_xy


def _critical_y(
    lambdas0: Float[Array, " n_fixed"],
    B_b: Float[Array, " n_fixed"],
) -> Float[Array, ""]:
    r"""Exercise boundary :math:`\bar y` solving :math:`\sum_i \lambda_i^0
    e^{-B(b,T,t_i) y} = 1`.

    The left-hand side is strictly decreasing in :math:`y`, so the root is
    unique.  Solved with a Newton iteration (implicitly differentiable).

    Args:
        lambdas0: Coefficients :math:`\lambda_i^0 = c_i A_i e^{-B(a,T,t_i)x}`
            at the current outer-integration node.
        B_b: Second-factor decay factors :math:`B(b, T, t_i)`.

    Returns:
        Critical value :math:`\bar y`.
    """
    def residual(y: Float[Array, ""], args) -> Float[Array, ""]:
        return jnp.sum(lambdas0 * jnp.exp(-B_b * y)) - 1.0

    sol = optx.root_find(
        residual,
        optx.Newton(rtol=1e-12, atol=1e-12),
        jnp.zeros((), dtype=lambdas0.dtype),
        max_steps=100,
        throw=False,
    )
    return sol.value


def g2pp_swaption_price(
    swaption: Swaption,
    model: G2PPModel,
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, ""]:
    r"""European swaption price under G2++ (Brigo-Mercurio semi-analytic).

    Prices a physically-settled European swaption by the exact
    one-dimensional integral of eq. (4.31), evaluated with Gauss-Hermite
    quadrature.  A payer swaption is a put on the fixed-coupon bond, a
    receiver a call.

    Discounting uses ``model.initial_curve``.  The ``forward_curve`` argument
    is reserved for a future deterministic-basis dual-curve extension (mirror
    of the pattern in :mod:`valax.pricing.analytic.swaptions`); it must be
    ``None`` or the discount curve itself, since the single-curve G2++
    coupon-bond decomposition assumes the floating leg resets to par.

    Args:
        swaption: Swaption contract (``is_payer`` selects payer vs receiver).
        model: G2++ model carrying the initial discount curve.
        forward_curve: Reserved; must be ``None`` (single-curve).

    Returns:
        Swaption price as of the curve reference date.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2.4 (eq. 4.31).
    """
    if forward_curve is not None and forward_curve is not model.initial_curve:
        raise NotImplementedError(
            "Dual-curve (deterministic-basis) G2++ swaptions are not yet "
            "supported; pass forward_curve=None."
        )

    curve: DiscountCurve = model.initial_curve
    ref = curve.reference_date
    day_count = swaption.day_count

    expiry_time = year_fraction(ref, swaption.expiry_date, day_count)
    cashflow_times = year_fraction(ref, swaption.fixed_dates, day_count)

    # Coupon-bond cash flows on unit notional: K*tau_i, plus principal at T_n.
    taus = _swaption_accruals(swaption)
    cashflows = swaption.strike * taus
    cashflows = cashflows.at[-1].add(1.0)

    # Deterministic ZCB coefficients: P(T, t_i | x, y) = A_i e^{-B_a,i x - B_b,i y}.
    A_i = g2pp_bond_price(
        model,
        jnp.zeros((), dtype=cashflow_times.dtype),
        jnp.zeros((), dtype=cashflow_times.dtype),
        expiry_time,
        cashflow_times,
    )
    B_a = g2pp_B(model.mean_reversion_x, cashflow_times - expiry_time)
    B_b = g2pp_B(model.mean_reversion_y, cashflow_times - expiry_time)

    mu_x, mu_y, sigma_x, sigma_y, rho_xy = _forward_measure_moments(
        model, expiry_time
    )
    sqrt_1m = jnp.sqrt(1.0 - rho_xy**2)

    is_payer = swaption.is_payer

    def per_node(z: Float[Array, ""]) -> Float[Array, ""]:
        x = mu_x + jnp.sqrt(2.0) * sigma_x * z

        lambdas0 = cashflows * A_i * jnp.exp(-B_a * x)
        y_bar = _critical_y(lambdas0, B_b)

        h1 = (y_bar - mu_y) / (sigma_y * sqrt_1m) - (
            rho_xy * (x - mu_x) / (sigma_x * sqrt_1m)
        )
        h2 = h1 + B_b * sigma_y * sqrt_1m

        kappa = -B_b * (
            mu_y
            - 0.5 * (1.0 - rho_xy**2) * sigma_y**2 * B_b
            + rho_xy * sigma_y * (x - mu_x) / sigma_x
        )
        terms = lambdas0 * jnp.exp(kappa)

        if is_payer:
            bracket = _Phi(-h1) - jnp.sum(terms * _Phi(-h2))
        else:
            bracket = jnp.sum(terms * _Phi(h2)) - _Phi(h1)
        return bracket

    brackets = jax.vmap(per_node)(_GH_NODES)
    integral = jnp.sum(_GH_WEIGHTS * _INV_SQRT_PI * brackets)

    df_expiry = g2pp_market_df(model, expiry_time)
    return swaption.notional * df_expiry * integral
