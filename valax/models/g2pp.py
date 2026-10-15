"""G2++ two-additive-factor Gaussian short-rate model.

G2++ is the two-factor generalisation of Hull-White (equivalent to the
two-factor Hull-White / HW-2F model).  Adding a second stochastic factor
lets rates of different maturities **decorrelate** — something the
one-factor model structurally cannot do, since there every rate is driven
by a single shock and all tenor rates move in lock-step.  This unlocks
decorrelation-sensitive instruments: CMS-spread options, steepeners and
flatteners, spread range-accruals, and Bermudan swaptions whose value
depends on the joint dynamics of short and long rates.

The short rate is the sum of two mean-reverting factors plus a
deterministic shift that exact-fits the initial curve:

.. math::

    r(t) = x(t) + y(t) + \\varphi(t)

.. math::

    dx(t) = -a\\,x(t)\\,dt + \\sigma(t)\\,dW_1(t), \\quad x(0) = 0

    dy(t) = -b\\,y(t)\\,dt + \\eta(t)\\,dW_2(t), \\quad y(0) = 0

    dW_1(t)\\,dW_2(t) = \\rho\\,dt

Here :math:`a, b` are the mean-reversion speeds, :math:`\\sigma(t)` the
first factor's volatility, :math:`\\eta(t) = r(t)\\,\\sigma(t)` the second
factor's (parameterised by the **volatility ratio** :math:`r(t)`), and
:math:`\\rho \\in (-1, 1)` the instantaneous correlation between the two
driving Brownian motions.  Both volatilities are piecewise constant on the
same buckets (see :mod:`valax.models.short_rate`); a constant model is one
bucket and runs the original closed forms.  **The tenor-rate
decorrelation the model delivers is not the same as** :math:`\\rho`:
:math:`\\rho` is an input, whereas the correlation between rates of
different maturities is an output that also depends on the split between
:math:`a` and :math:`b`.  Genuine decorrelation requires :math:`|\\rho| <
1` — at :math:`\\rho = \\pm 1` the two Brownian motions collapse to a
single shock and the model degenerates to a one-factor world.

The model stays affine/Gaussian, so zero-coupon bond prices are
closed-form conditional on the two factors:

.. math::

    P(t, T \\mid x, y)
        = \\frac{P^M(0, T)}{P^M(0, t)}
          \\exp\\!\\Bigl[\\tfrac{1}{2}\\bigl(V(t, T) - V(0, T) + V(0, t)\\bigr)
                        - B(a, t, T)\\,x - B(b, t, T)\\,y\\Bigr]

which holds unchanged for deterministic time-dependent volatilities.

Piecewise-constant volatility
-----------------------------

With :math:`B_z(u, T) = (1 - e^{-z(T-u)})/z`, every Gaussian quantity is an
integral of a product of piecewise-constant volatilities against
exponentials :math:`e^{-k(T-u)}` with :math:`k \\in \\{0, a, 2a, b, 2b,
a+b\\}`. Writing, for :math:`v \\in \\{\\sigma^2, \\eta^2, \\sigma\\eta\\}`,

.. math::

    K^v_k(t, T) = \\int_t^T v(u)\\,e^{-k(T-u)}\\,du
               = I^v_k(T) - e^{-k(T-t)}\\,I^v_k(t),

with :math:`I^v_k` from :func:`~valax.models.short_rate.decayed_integral`
(and :func:`~valax.models.short_rate.piecewise_integral` for :math:`k = 0`),
all of them are finite sums of :math:`K` terms in closed form.

The model is an ``equinox.Module`` and hence a valid JAX pytree —
``jax.grad``, ``jax.jit``, and ``jax.vmap`` all work out of the box.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 4 (§4.2).
"""

import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import year_fraction
from valax.models.short_rate import (
    as_buckets,
    check_buckets,
    decayed_integral,
    default_bucket_times,
    is_constant,
    piecewise_integral,
    require_initial_curve,
    volatility_bucket_fields,
)


class G2PPModel(eqx.Module):
    """G2++ two-factor Gaussian short-rate model, piecewise-constant vols.

    The first factor's volatility :math:`\\sigma(t)` and the **volatility
    ratio** :math:`r(t)` are piecewise constant on shared buckets (see
    :mod:`valax.models.short_rate`): bucket ``j`` covers
    :math:`[\\tau_j, \\tau_{j+1})` with :math:`\\tau_0 = 0`, the interior
    breakpoints are ``volatility_times`` and the last value extends flat. The
    second factor's volatility is :math:`\\eta(t) = r(t)\\,\\sigma(t)`, exposed
    as the read-only property :attr:`volatility_y`. A constant model is one
    bucket; scalars are promoted to one, and every closed form then runs its
    original constant-parameter formula.

    Attributes:
        mean_reversion_x: Mean-reversion speed :math:`a` of the first
            factor (positive scalar).
        mean_reversion_y: Mean-reversion speed :math:`b` of the second
            factor (positive scalar).
        volatility_x: Volatility :math:`\\sigma` of the first factor per
            bucket (positive).
        volatility_ratio: Ratio :math:`r = \\eta / \\sigma` per bucket
            (positive).
        correlation: Instantaneous correlation :math:`\\rho \\in (-1, 1)`
            between the two driving Brownian motions.
        initial_curve: Initial discount curve :math:`P^M(0, t)` used for
            exact-fit :math:`\\varphi(t)` calibration.
            ``None`` (the default) leaves the model unbound:
            :func:`valax.core.pricing.price` binds it from the market (see
            :mod:`valax.models.short_rate`).
        curve_id: Market id of the curve to bind, or ``None`` to use the
            instrument's discount curve. Static.
        volatility_times: Interior bucket breakpoints in year fractions from
            the curve's reference date, increasing, shared by both
            volatilities; empty for one bucket.
    """

    mean_reversion_x: Float[Array, ""]
    mean_reversion_y: Float[Array, ""]
    volatility_x: Float[Array, " n_buckets"]
    volatility_ratio: Float[Array, " n_buckets"]
    correlation: Float[Array, ""]
    initial_curve: DiscountCurve | None
    curve_id: str | None = eqx.field(static=True)
    volatility_times: Float[Array, " n_buckets_minus_1"]

    def __init__(
        self,
        mean_reversion_x: Float[Array, ""],
        mean_reversion_y: Float[Array, ""],
        volatility_x: Float[Array, " n_buckets"] | Float[Array, ""],
        volatility_ratio: Float[Array, " n_buckets"] | Float[Array, ""],
        correlation: Float[Array, ""],
        initial_curve: DiscountCurve | None = None,
        curve_id: str | None = None,
        *,
        volatility_times: Float[Array, " n_buckets_minus_1"] | None = None,
    ):
        """Build a model; scalar volatilities become one bucket.

        A one-bucket ``volatility_ratio`` (or a scalar) is broadcast to the
        bucket count of ``volatility_x``.

        Args:
            mean_reversion_x: Mean-reversion speed :math:`a`.
            mean_reversion_y: Mean-reversion speed :math:`b`.
            volatility_x: :math:`\\sigma` per bucket, or a scalar.
            volatility_ratio: :math:`\\eta / \\sigma` per bucket, or a scalar.
            correlation: Correlation :math:`\\rho`.
            initial_curve: Bound initial curve, or ``None`` (bound by
                ``price``).
            curve_id: Market id of the curve to bind (static).
            volatility_times: Interior breakpoints (``n_buckets - 1``);
                required for more than one bucket.
        """
        self.mean_reversion_x = mean_reversion_x
        self.mean_reversion_y = mean_reversion_y
        self.volatility_x = as_buckets(volatility_x)
        ratio = as_buckets(volatility_ratio)
        v_shape = getattr(self.volatility_x, "shape", None)
        r_shape = getattr(ratio, "shape", None)
        if (
            v_shape is not None and r_shape is not None
            and len(v_shape) >= 1 and len(r_shape) >= 1
            and r_shape[-1] == 1 and v_shape[-1] != 1
        ):
            ratio = jnp.broadcast_to(ratio, r_shape[:-1] + (v_shape[-1],))
        self.volatility_ratio = ratio
        self.correlation = correlation
        self.initial_curve = initial_curve
        self.curve_id = curve_id
        self.volatility_times = (
            default_bucket_times(self.volatility_x)
            if volatility_times is None else volatility_times
        )

    def __check_init__(self) -> None:
        """Validate the bucket structure (shapes only)."""
        check_buckets("G2PPModel", self.volatility_x, self.volatility_times)
        check_buckets(
            "G2PPModel (volatility_ratio)",
            self.volatility_ratio,
            self.volatility_times,
        )

    @property
    def volatility_y(self) -> Float[Array, " n_buckets"]:
        """Second-factor volatility :math:`\\eta = r\\,\\sigma` per bucket."""
        return self.volatility_ratio * self.volatility_x


@volatility_bucket_fields.register(G2PPModel)
def _g2pp_bucket_fields(model: G2PPModel) -> tuple[str, ...]:
    return ("volatility_x", "volatility_ratio")


# ── Bucket helpers ────────────────────────────────────────────────────

def _is_constant(model: G2PPModel) -> bool:
    """Whether both volatilities are a single bucket (static, trace time)."""
    return is_constant(model.volatility_x) and is_constant(model.volatility_ratio)


def _constant_vols(
    model: G2PPModel,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """``(sigma, eta)`` of a one-bucket model, with ``eta = r * sigma``."""
    sigma = model.volatility_x[0]
    return sigma, model.volatility_ratio[0] * sigma


def _bucket_products(
    model: G2PPModel,
) -> tuple[
    Float[Array, " n_buckets"],
    Float[Array, " n_buckets"],
    Float[Array, " n_buckets"],
]:
    """Per-bucket :math:`(\\sigma^2, \\eta^2, \\sigma\\eta)` on shared buckets."""
    sigma = model.volatility_x
    eta = model.volatility_ratio * sigma
    sigma = jnp.broadcast_to(sigma, eta.shape)
    return sigma**2, eta**2, sigma * eta


def _decayed(
    model: G2PPModel,
    values: Float[Array, " n_buckets"],
    rate: Float[Array, ""],
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """:math:`I_k(t) = \\int_0^t v(u) e^{-k(t-u)} du` (piecewise ``v``)."""
    return decayed_integral(values, model.volatility_times, rate, t)[0]


def _window(
    model: G2PPModel,
    values: Float[Array, " n_buckets"],
    rate: Float[Array, ""],
    t: Float[Array, "..."],
    T: Float[Array, "..."],
) -> Float[Array, "..."]:
    """:math:`K_k(t, T) = \\int_t^T v(u) e^{-k(T-u)} du = I_k(T) - e^{-k(T-t)} I_k(t)`."""
    return (
        _decayed(model, values, rate, T)
        - jnp.exp(-rate * (T - t)) * _decayed(model, values, rate, t)
    )


def _window0(
    model: G2PPModel,
    values: Float[Array, " n_buckets"],
    t: Float[Array, "..."],
    T: Float[Array, "..."],
) -> Float[Array, "..."]:
    """:math:`K_0(t, T) = \\int_t^T v(u)\\,du` (no decay)."""
    times = model.volatility_times
    return piecewise_integral(values, times, T) - piecewise_integral(values, times, t)


#: Terms of the small-argument series of :func:`_b_product_integral`; at the
#: switch point ``(p + q) s = 1`` the truncation is below 1e-20 relative.
_N_SERIES = 24


def _b_product_integral(
    p: Float[Array, ""],
    q: Float[Array, ""],
    s: Float[Array, "..."],
) -> Float[Array, "..."]:
    """:math:`H(s) = \\int_0^s (1 - e^{-pu})(1 - e^{-qu})\\,du`, accurately.

    Closed form :math:`s - \\frac{1 - e^{-ps}}{p} - \\frac{1 - e^{-qs}}{q}
    + \\frac{1 - e^{-(p+q)s}}{p+q}` for :math:`(p+q)s \\ge 1`; below, where
    that loses ~eps / (ps)^2 to cancellation, the alternating series

    .. math::

        H(s) = s\\sum_{n \\ge 3} \\frac{(-1)^{n+1}}{n!}
            \\sum_{j=1}^{n-2}\\binom{n-1}{j}(ps)^j (qs)^{n-1-j},

    whose terms are all positive inside the inner sum (no cancellation).
    The inner sum :math:`A_k = (x+y)^k - x^k - y^k` (with :math:`x = ps`,
    :math:`y = qs`, :math:`k = n - 1`) is built by the recurrence
    :math:`A_{k+1} = (x+y)A_k + xy\\,(x^{k-1} + y^{k-1})`, :math:`A_1 = 0`,
    whose terms are all non-negative too — as accurate as the binomial sum,
    at a handful of operations per term instead of :math:`k` (the expanded
    double sum made the traced graph, and a reverse-mode compile through a
    G2++ PDE, an order of magnitude larger).

    Args:
        p: First decay rate (positive).
        q: Second decay rate (positive).
        s: Window lengths (non-negative).

    Returns:
        :math:`H(s)`, shaped like ``s``.
    """
    s = jnp.asarray(s)
    small = (p + q) * s < 1.0
    ss = jnp.where(small, s, 0.0)
    x, y = p * ss, q * ss
    x_plus_y, xy = x + y, x * y
    a_k = jnp.zeros_like(ss)          # A_1 = 0
    x_pow = y_pow = jnp.ones_like(ss)  # x^0, y^0
    series = jnp.zeros_like(ss)
    for n in range(3, _N_SERIES + 1):
        # A_{n-1} from A_{n-2} (and x^{n-3}, y^{n-3}).
        a_k = x_plus_y * a_k + xy * (x_pow + y_pow)
        x_pow, y_pow = x_pow * x, y_pow * y
        sign = 1.0 if n % 2 == 1 else -1.0
        series = series + (sign / math.factorial(n)) * a_k
    series = ss * series
    closed = (
        s
        + jnp.expm1(-p * s) / p
        + jnp.expm1(-q * s) / q
        - jnp.expm1(-(p + q) * s) / (p + q)
    )
    return jnp.where(small, series, closed)


# ── Helpers ───────────────────────────────────────────────────────────

def g2pp_B(
    z: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Mean-reversion decay factor :math:`B(z, \\tau) = (1 - e^{-z\\tau})/z`.

    The same shape function :func:`~valax.models.hull_white.hw_B` carries
    per factor; G2++ evaluates it once with :math:`z = a` and once with
    :math:`z = b`.  For :math:`z \\to 0` it reduces to :math:`\\tau`.

    Args:
        z: Mean-reversion speed of the factor.
        tau: Time to maturity :math:`T - t` (year fractions).

    Returns:
        The decay factor :math:`B(z, \\tau)`.
    """
    return (1.0 - jnp.exp(-z * tau)) / z


def _pillar_times(curve: DiscountCurve) -> Float[Array, " n"]:
    """Year fractions from the reference date to each pillar."""
    return year_fraction(
        curve.reference_date, curve.pillar_dates, curve.day_count
    )


def _log_df_grid(
    curve: DiscountCurve,
) -> tuple[Float[Array, " n_nodes"], Float[Array, " n_nodes"]]:
    """Log-DF interpolation nodes, anchored at the reference date.

    Mirrors the Hull-White accessor: prepends the no-arbitrage anchor
    :math:`\\ln P^M(0, 0) = 0` so a curve whose first pillar sits strictly
    after the reference date does not silently drive the short-end
    instantaneous forward to zero.

    Args:
        curve: Initial market discount curve.

    Returns:
        Pair of ``(times, log_discount_factors)`` interpolation nodes.
    """
    pillar_t = _pillar_times(curve)
    log_dfs = jnp.log(curve.discount_factors)
    return (
        jnp.concatenate([jnp.zeros((1,), dtype=pillar_t.dtype), pillar_t]),
        jnp.concatenate([jnp.zeros((1,), dtype=log_dfs.dtype), log_dfs]),
    )


def _log_df_at_time(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Log discount factor :math:`\\ln P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(require_initial_curve(model))
    return jnp.interp(t, times, log_dfs)


def _market_df(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at year-fraction *t*."""
    times, log_dfs = _log_df_grid(require_initial_curve(model))
    return jnp.exp(jnp.interp(t, times, log_dfs))


def _instantaneous_forward(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Instantaneous forward rate :math:`f^M(0, t) = -d\\ln P^M(0,t)/dt`."""
    return -jax.grad(lambda s: _log_df_at_time(model, s))(t)


# ── Public curve accessors (continuous time) ──────────────────────────

def g2pp_market_df(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Market discount factor :math:`P^M(0, t)` at a year fraction.

    Args:
        model: G2++ model carrying the initial curve.
        t: Year fraction(s) from the curve reference date.

    Returns:
        Market discount factor at each ``t``.
    """
    return _market_df(model, t)


def g2pp_instantaneous_forward(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Market instantaneous forward rate :math:`f^M(0, t)`.

    Args:
        model: G2++ model carrying the initial curve.
        t: Year fraction from the curve reference date.

    Returns:
        Instantaneous forward rate at ``t``.
    """
    return _instantaneous_forward(model, t)


# ── Gaussian variance term ────────────────────────────────────────────

def _factor_variance_term(
    z: Float[Array, ""],
    vol: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Single-factor contribution to :math:`V(t, T)` with :math:`\\tau = T - t`.

    .. math::

        \\frac{\\text{vol}^2}{z^2}\\Bigl[\\tau + \\frac{2}{z}e^{-z\\tau}
            - \\frac{1}{2z}e^{-2z\\tau} - \\frac{3}{2z}\\Bigr]
    """
    return (vol**2 / z**2) * (
        tau
        + (2.0 / z) * jnp.exp(-z * tau)
        - (1.0 / (2.0 * z)) * jnp.exp(-2.0 * z * tau)
        - 3.0 / (2.0 * z)
    )


def _cross_variance_term(
    a: Float[Array, ""],
    b: Float[Array, ""],
    sigma: Float[Array, ""],
    eta: Float[Array, ""],
    rho: Float[Array, ""],
    tau: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Cross-factor contribution to :math:`V(t, T)` with :math:`\\tau = T - t`.

    .. math::

        2\\rho\\frac{\\sigma\\eta}{ab}\\Bigl[\\tau
            + \\frac{e^{-a\\tau} - 1}{a}
            + \\frac{e^{-b\\tau} - 1}{b}
            - \\frac{e^{-(a+b)\\tau} - 1}{a + b}\\Bigr]
    """
    return 2.0 * rho * sigma * eta / (a * b) * (
        tau
        + (jnp.exp(-a * tau) - 1.0) / a
        + (jnp.exp(-b * tau) - 1.0) / b
        - (jnp.exp(-(a + b) * tau) - 1.0) / (a + b)
    )


def g2pp_V(
    model: G2PPModel,
    t: Float[Array, "..."],
    T: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Gaussian variance term :math:`V(t, T)` of the integrated short rate.

    This is the variance of :math:`\\int_t^T (x(u) + y(u))\\,du` conditional
    on the factors at time :math:`t`; it drives both the exact-fit ZCB price
    and the Gaussian swaption formula. In general

    .. math::

        V(t, T) = \\int_t^T \\Bigl[\\sigma^2 B_a(u,T)^2 + \\eta^2 B_b(u,T)^2
            + 2\\rho\\sigma\\eta\\,B_a(u,T) B_b(u,T)\\Bigr] du
        = \\frac{K^{\\sigma^2}_0 - 2K^{\\sigma^2}_a + K^{\\sigma^2}_{2a}}{a^2}
        + \\frac{K^{\\eta^2}_0 - 2K^{\\eta^2}_b + K^{\\eta^2}_{2b}}{b^2}
        + \\frac{2\\rho}{ab}\\bigl(K^{\\sigma\\eta}_0 - K^{\\sigma\\eta}_a
            - K^{\\sigma\\eta}_b + K^{\\sigma\\eta}_{a+b}\\bigr),

    with :math:`K^v_k = K^v_k(t, T)` the decayed window integrals of the
    module docstring. For constant volatilities it depends on :math:`t, T`
    only through :math:`\\tau = T - t`:

    .. math::

        V(t, T)
        = \\frac{\\sigma^2}{a^2}\\Bigl[(T-t) + \\tfrac{2}{a}e^{-a(T-t)}
            - \\tfrac{1}{2a}e^{-2a(T-t)} - \\tfrac{3}{2a}\\Bigr]
        + \\frac{\\eta^2}{b^2}\\Bigl[(T-t) + \\tfrac{2}{b}e^{-b(T-t)}
            - \\tfrac{1}{2b}e^{-2b(T-t)} - \\tfrac{3}{2b}\\Bigr]
        + 2\\rho\\frac{\\sigma\\eta}{ab}\\Bigl[(T-t)
            + \\tfrac{e^{-a(T-t)} - 1}{a}
            + \\tfrac{e^{-b(T-t)} - 1}{b}
            - \\tfrac{e^{-(a+b)(T-t)} - 1}{a+b}\\Bigr]

    Args:
        model: G2++ model carrying the factor parameters.
        t: Start time in year fractions.
        T: End time in year fractions.

    Returns:
        The variance term :math:`V(t, T)`.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2 (eq. 4.10).
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    rho = model.correlation
    if _is_constant(model):
        sigma, eta = _constant_vols(model)
        tau = T - t
        return (
            _factor_variance_term(a, sigma, tau)
            + _factor_variance_term(b, eta, tau)
            + _cross_variance_term(a, b, sigma, eta, rho, tau)
        )
    # Evaluated bucket by bucket: on the part [u0, u1] of bucket j inside
    # [t, T], int (1 - e^{-p(T-u)})(1 - e^{-q(T-u)}) du = H(s0) - H(s1) with
    # s = T - u. Equal to the K-combination above, without its cancellation
    # (relative error ~eps / (k tau)^2 on a short window).
    s2, e2, se = _bucket_products(model)
    times = model.volatility_times
    t = jnp.asarray(t)[..., jnp.newaxis]
    T = jnp.asarray(T)[..., jnp.newaxis]
    starts = jnp.concatenate([jnp.zeros((1,), dtype=times.dtype), times])
    ends = jnp.concatenate([times, jnp.full((1,), jnp.inf, dtype=times.dtype)])
    s0 = T - jnp.clip(starts, t, T)  # (..., n_buckets)
    s1 = T - jnp.clip(ends, t, T)

    # The three rate pairs (a, a), (b, b), (a, b) at both window ends in one
    # broadcast evaluation, (pair, end, ..., n_buckets): elementwise, so each
    # entry is computed exactly as six separate calls would, but the series
    # is traced once rather than six times.
    lead = (1,) * (s0.ndim + 1)
    p = jnp.reshape(jnp.stack([a, b, a]), (3,) + lead)
    q = jnp.reshape(jnp.stack([a, b, b]), (3,) + lead)
    h = _b_product_integral(p, q, jnp.stack([s0, s1])[jnp.newaxis])
    window = h[:, 0] - h[:, 1]  # (pair, ..., n_buckets)

    terms = (
        s2 * window[0] / a**2
        + e2 * window[1] / b**2
        + 2.0 * rho * se * window[2] / (a * b)
    )
    return jnp.sum(terms, axis=-1)


# ── Deterministic exact-fit shift ─────────────────────────────────────

def g2pp_phi(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Deterministic shift :math:`\\varphi(t)` in :math:`r = x + y + \\varphi`.

    The shift carries the whole dependence on the initial curve, so the
    factors :math:`x, y` are zero-mean OU processes — the parameterisation
    every numerical scheme wants. It is
    :math:`\\varphi(t) = f^M(0, t) + \\tfrac12 \\partial_t V(0, t)`, i.e.

    .. math::

        \\varphi(t) = f^M(0, t)
            + \\frac{I^{\\sigma^2}_a - I^{\\sigma^2}_{2a}}{a}
            + \\frac{I^{\\eta^2}_b - I^{\\eta^2}_{2b}}{b}
            + \\rho\\Bigl[\\frac{I^{\\sigma\\eta}_b - I^{\\sigma\\eta}_{a+b}}{a}
                + \\frac{I^{\\sigma\\eta}_a - I^{\\sigma\\eta}_{a+b}}{b}\\Bigr]

    with :math:`I^v_k = I^v_k(t)` (:func:`~valax.models.short_rate.decayed_integral`),
    which for constant volatilities is

    .. math::

        \\varphi(t) = f^M(0, t)
            + \\frac{\\sigma^2}{2a^2}\\bigl(1 - e^{-at}\\bigr)^2
            + \\frac{\\eta^2}{2b^2}\\bigl(1 - e^{-bt}\\bigr)^2
            + \\rho\\frac{\\sigma\\eta}{ab}\\bigl(1 - e^{-at}\\bigr)
              \\bigl(1 - e^{-bt}\\bigr)

    At :math:`t = 0` every convexity term vanishes and
    :math:`\\varphi(0) = f^M(0, 0) = r(0)`.

    Args:
        model: G2++ model carrying the initial curve and parameters.
        t: Year fraction(s) from the curve reference date.

    Returns:
        The shift :math:`\\varphi(t)` at each ``t``.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2 (eq. 4.5).
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    rho = model.correlation
    t_arr = jnp.asarray(t)
    forward = jnp.vectorize(lambda s: _instantaneous_forward(model, s))(t_arr)
    if _is_constant(model):
        sigma, eta = _constant_vols(model)
        ea = 1.0 - jnp.exp(-a * t_arr)
        eb = 1.0 - jnp.exp(-b * t_arr)
        convexity = (
            (sigma**2 / (2.0 * a**2)) * ea**2
            + (eta**2 / (2.0 * b**2)) * eb**2
            + rho * sigma * eta / (a * b) * ea * eb
        )
        return forward + convexity
    s2, e2, se = _bucket_products(model)
    i_se_ab = _decayed(model, se, a + b, t_arr)
    convexity = (
        (_decayed(model, s2, a, t_arr) - _decayed(model, s2, 2.0 * a, t_arr)) / a
        + (_decayed(model, e2, b, t_arr) - _decayed(model, e2, 2.0 * b, t_arr)) / b
        + rho * (
            (_decayed(model, se, b, t_arr) - i_se_ab) / a
            + (_decayed(model, se, a, t_arr) - i_se_ab) / b
        )
    )
    return forward + convexity


def g2pp_phi_integral(
    model: G2PPModel,
    t: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Integrated shift :math:`\\int_0^t \\varphi(s)\\,ds` in closed form.

    The exact-fit condition :math:`P^M(0, t) = \\exp(-\\int_0^t \\varphi
    + \\tfrac12 V(0, t))` gives

    .. math::

        \\int_0^t \\varphi(s)\\,ds = -\\ln P^M(0, t) + \\tfrac{1}{2} V(0, t),

    so the exact step average of :math:`\\varphi` that a discretised scheme
    needs (PDE discount, Monte-Carlo numeraire) follows from :math:`V(0,\\cdot)`
    and the curve alone, for any piecewise-constant volatilities.

    Args:
        model: G2++ model carrying the initial curve and parameters.
        t: Year fraction(s) from the curve reference date.

    Returns:
        :math:`\\int_0^t \\varphi` at each ``t``.
    """
    t_arr = jnp.asarray(t)
    log_df = jnp.log(_market_df(model, t_arr))
    return -log_df + 0.5 * g2pp_V(model, jnp.zeros_like(t_arr), t_arr)


def g2pp_diffusion_rates(
    model: G2PPModel,
    t0: Float[Array, "..."],
    t1: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Time averages of the factor diffusion coefficients over ``[t0, t1]``.

    The coefficients a discretised scheme should use for a step that may
    straddle a bucket breakpoint: the exact step averages of
    :math:`\\sigma^2`, :math:`\\eta^2` and :math:`\\rho\\sigma\\eta`.

    Args:
        model: G2++ model.
        t0: Start of each interval in year fractions.
        t1: End of each interval in year fractions (``t1 > t0``).

    Returns:
        ``(sigma2, eta2, rho_sigma_eta)``: the averages of :math:`\\sigma^2`,
        :math:`\\eta^2` and :math:`\\rho\\sigma\\eta` over each interval.
    """
    shape = jnp.broadcast_shapes(jnp.shape(t0), jnp.shape(t1))
    rho = model.correlation
    if _is_constant(model):
        sigma, eta = _constant_vols(model)
        return (
            jnp.broadcast_to(sigma**2, shape),
            jnp.broadcast_to(eta**2, shape),
            jnp.broadcast_to(rho * sigma * eta, shape),
        )
    s2, e2, se = _bucket_products(model)
    t0 = jnp.asarray(t0)
    t1 = jnp.asarray(t1)
    dt = t1 - t0
    return (
        _window0(model, s2, t0, t1) / dt,
        _window0(model, e2, t0, t1) / dt,
        rho * _window0(model, se, t0, t1) / dt,
    )


# ── Analytic zero-coupon bond price ───────────────────────────────────

def g2pp_bond_price(
    model: G2PPModel,
    x: Float[Array, ""],
    y: Float[Array, ""],
    t: Float[Array, ""],
    T: Float[Array, ""],
) -> Float[Array, ""]:
    """Zero-coupon bond price under G2++ given factors *x*, *y* at time *t*.

    .. math::

        P(t, T \\mid x, y)
            = \\frac{P^M(0, T)}{P^M(0, t)}
              \\exp\\!\\Bigl[\\tfrac{1}{2}\\bigl(V(t, T) - V(0, T) + V(0, t)\\bigr)
                            - B(a, t, T)\\,x - B(b, t, T)\\,y\\Bigr]

    (**Exact-fit property**: at :math:`t = 0` with :math:`x = y = 0` the
    bracket vanishes and this recovers the initial curve discount factor
    :math:`P^M(0, T)`.) The formula holds unchanged for piecewise-constant
    volatilities, through :func:`g2pp_V`.

    Args:
        model: G2++ model (carries initial curve and parameters).
        x: Current value of the first factor (scalar).
        y: Current value of the second factor (scalar).
        t: Current time in year fractions.
        T: Bond maturity time in year fractions.

    Returns:
        Zero-coupon bond price :math:`P(t, T)`.

    References:
        Brigo & Mercurio (2006), *Interest Rate Models*, §4.2 (eq. 4.14).
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y

    zero = jnp.zeros_like(t)
    v_tT = g2pp_V(model, t, T)
    v_0T = g2pp_V(model, zero, T)
    v_0t = g2pp_V(model, zero, t)

    B_a = g2pp_B(a, T - t)
    B_b = g2pp_B(b, T - t)

    ln_PM_T = _log_df_at_time(model, T)
    ln_PM_t = _log_df_at_time(model, t)

    ln_A = (
        ln_PM_T - ln_PM_t
        + 0.5 * (v_tT - v_0T + v_0t)
        - B_a * x - B_b * y
    )
    return jnp.exp(ln_A)


# ── Factor distribution ──────────────────────────────────────────────

def g2pp_state_covariance(
    model: G2PPModel,
    s: Float[Array, "..."],
    t: Float[Array, "..."],
) -> Float[Array, "... 2 2"]:
    """Conditional covariance of :math:`(x(t), y(t))` given the factors at *s*.

    .. math::

        \\text{Var}[x] = \\int_s^t \\sigma(u)^2 e^{-2a(t-u)}\\,du, \\quad
        \\text{Var}[y] = \\int_s^t \\eta(u)^2 e^{-2b(t-u)}\\,du, \\quad
        \\text{Cov}[x, y] = \\rho\\int_s^t \\sigma(u)\\eta(u)e^{-(a+b)(t-u)}\\,du,

    each evaluated as :math:`I_k(t) - e^{-k(t-s)} I_k(s)`. The conditional
    means are :math:`x(s)e^{-a(t-s)}` and :math:`y(s)e^{-b(t-s)}`. This is the
    exact Monte-Carlo step covariance, and from :math:`s = 0` the
    unconditional covariance of the factors.

    Args:
        model: G2++ model.
        s: Start time(s) in year fractions.
        t: End time(s) in year fractions (``t >= s``).

    Returns:
        The ``2x2`` covariance matrix of :math:`(x, y)` per ``(s, t)`` pair.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    rho = model.correlation
    s = jnp.asarray(s)
    t = jnp.asarray(t)
    if _is_constant(model):
        sigma, eta = _constant_vols(model)
        dt = t - s
        var_x = (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * dt))
        var_y = (eta**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * dt))
        cov_xy = rho * sigma * eta / (a + b) * (1.0 - jnp.exp(-(a + b) * dt))
    else:
        s2, e2, se = _bucket_products(model)
        var_x = _window(model, s2, 2.0 * a, s, t)
        var_y = _window(model, e2, 2.0 * b, s, t)
        cov_xy = rho * _window(model, se, a + b, s, t)
    row_x = jnp.stack([var_x, cov_xy], axis=-1)
    row_y = jnp.stack([cov_xy, var_y], axis=-1)
    return jnp.stack([row_x, row_y], axis=-2)


def g2pp_factor_covariance(
    model: G2PPModel,
    dt: Float[Array, ""],
) -> Float[Array, "2 2"]:
    """Covariance of :math:`(x, y)` over a step of length *dt* from time 0.

    For constant volatilities the factors are time-homogeneous: given
    :math:`(x(s), y(s))`, the increment to :math:`(x(s+dt), y(s+dt))` is
    Gaussian with mean :math:`(x(s)e^{-a\\,dt}, y(s)e^{-b\\,dt})` and the
    covariance returned here, independent of :math:`s`:

    .. math::

        \\text{Var}[x] = \\frac{\\sigma^2}{2a}\\bigl(1 - e^{-2a\\,dt}\\bigr),
        \\quad
        \\text{Var}[y] = \\frac{\\eta^2}{2b}\\bigl(1 - e^{-2b\\,dt}\\bigr)

    .. math::

        \\text{Cov}[x, y]
            = \\rho\\frac{\\sigma\\eta}{a + b}\\bigl(1 - e^{-(a+b)\\,dt}\\bigr)

    With piecewise-constant volatilities the step covariance depends on
    where the step starts; this returns the one over :math:`[0, dt]` — the
    unconditional covariance of :math:`(x(dt), y(dt))` — and
    :func:`g2pp_state_covariance` gives any other step.

    Args:
        model: G2++ model carrying the factor parameters.
        dt: Step length in year fractions.

    Returns:
        The ``2x2`` covariance matrix of :math:`(x, y)`.
    """
    if not _is_constant(model):
        return g2pp_state_covariance(model, jnp.zeros_like(dt), dt)
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma, eta = _constant_vols(model)
    rho = model.correlation

    var_x = (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * dt))
    var_y = (eta**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * dt))
    cov_xy = rho * sigma * eta / (a + b) * (1.0 - jnp.exp(-(a + b) * dt))

    return jnp.array([[var_x, cov_xy], [cov_xy, var_y]])


def g2pp_forward_measure_moments(
    model: G2PPModel,
    expiry_time: Float[Array, ""],
) -> tuple[
    Float[Array, ""], Float[Array, ""],
    Float[Array, ""], Float[Array, ""],
    Float[Array, ""],
]:
    r"""Means, standard deviations and correlation of :math:`(x, y)` at expiry.

    Under the :math:`T`-forward measure the factors :math:`x(T), y(T)` are
    jointly Gaussian, with the covariance of :func:`g2pp_state_covariance`
    from 0 and the drift-adjusted means

    .. math::

        \mu_x(T) = -\int_0^T e^{-a(T-u)}\sigma(u)\bigl[\sigma(u) B_a(u,T)
            + \rho\,\eta(u) B_b(u,T)\bigr] du
            = -\Bigl[\frac{I^{\sigma^2}_a - I^{\sigma^2}_{2a}}{a}
                + \rho\,\frac{I^{\sigma\eta}_a - I^{\sigma\eta}_{a+b}}{b}\Bigr],

    .. math::

        \mu_y(T) = -\int_0^T e^{-b(T-u)}\eta(u)\bigl[\eta(u) B_b(u,T)
            + \rho\,\sigma(u) B_a(u,T)\bigr] du
            = -\Bigl[\frac{I^{\eta^2}_b - I^{\eta^2}_{2b}}{b}
                + \rho\,\frac{I^{\sigma\eta}_b - I^{\sigma\eta}_{a+b}}{a}\Bigr],

    with :math:`I^v_k = I^v_k(T)`; for constant volatilities these are
    Brigo & Mercurio (2006), eq. (4.31).

    Args:
        model: G2++ model.
        expiry_time: Option expiry :math:`T` in year fractions.

    Returns:
        Tuple ``(mu_x, mu_y, sigma_x, sigma_y, rho_xy)``.
    """
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    rho = model.correlation
    T = expiry_time

    if _is_constant(model):
        sigma, eta = _constant_vols(model)
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

    s2, e2, se = _bucket_products(model)
    var_x = _decayed(model, s2, 2.0 * a, T)
    var_y = _decayed(model, e2, 2.0 * b, T)
    i_se_ab = _decayed(model, se, a + b, T)
    sigma_x = jnp.sqrt(var_x)
    sigma_y = jnp.sqrt(var_y)
    rho_xy = rho * i_se_ab / (sigma_x * sigma_y)
    mu_x = -(
        (_decayed(model, s2, a, T) - var_x) / a
        + rho * (_decayed(model, se, a, T) - i_se_ab) / b
    )
    mu_y = -(
        (_decayed(model, e2, b, T) - var_y) / b
        + rho * (_decayed(model, se, b, T) - i_se_ab) / a
    )
    return mu_x, mu_y, sigma_x, sigma_y, rho_xy


def g2pp_short_rate_variance(
    model: G2PPModel,
    t: Float[Array, ""],
) -> Float[Array, ""]:
    """Variance of the short rate at time *t* (unconditional).

    .. math::

        \\text{Var}[r(t)]
            = \\frac{\\sigma^2}{2a}(1 - e^{-2at})
            + \\frac{\\eta^2}{2b}(1 - e^{-2bt})
            + 2\\rho\\frac{\\sigma\\eta}{a + b}(1 - e^{-(a+b)t})

    for constant volatilities, and
    :math:`\\text{Var}[x(t)] + \\text{Var}[y(t)] + 2\\,\\text{Cov}[x(t), y(t)]`
    from :func:`g2pp_state_covariance` in general.

    Args:
        model: G2++ model carrying the factor parameters.
        t: Year fraction from the curve reference date.

    Returns:
        The variance of :math:`r(t) = x(t) + y(t) + \\varphi(t)`.
    """
    if not _is_constant(model):
        cov = g2pp_state_covariance(model, jnp.zeros_like(t), t)
        return cov[..., 0, 0] + cov[..., 1, 1] + 2.0 * cov[..., 0, 1]
    a = model.mean_reversion_x
    b = model.mean_reversion_y
    sigma, eta = _constant_vols(model)
    rho = model.correlation

    var_x = (sigma**2 / (2.0 * a)) * (1.0 - jnp.exp(-2.0 * a * t))
    var_y = (eta**2 / (2.0 * b)) * (1.0 - jnp.exp(-2.0 * b * t))
    cov_xy = rho * sigma * eta / (a + b) * (1.0 - jnp.exp(-(a + b) * t))
    return var_x + var_y + 2.0 * cov_xy
