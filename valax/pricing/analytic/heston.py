"""Heston semi-analytic European option pricing via the Fang-Oosterlee COS method.

The pricing engine combines:

* **Lord-Kahl "Little Trap" characteristic function** for the log-spot under
  the Heston dynamics. The Little Trap form uses the
  ``(d - g) / (d + g)`` ratio convention which keeps the complex logarithm
  on its principal branch for all expiries and all ``|rho| < 1``, avoiding
  the branch-cut discontinuity in the original Heston (1993) formulation
  that breaks autodiff for long-dated or high-correlation parameter regions.

* **Fang & Oosterlee (2008) COS expansion** of the risk-neutral density of
  log-moneyness ``Y_T = log(S_T / K)``. Closed-form payoff cosine
  coefficients ``U_k`` use the ``chi_k`` / ``psi_k`` integrals (FO eq. 22-24)
  with the integration interval ``[0, b]`` for calls and ``[a, 0]`` for puts
  — strictly more accurate than computing puts via parity for deep OTM
  strikes (no catastrophic cancellation).

* **Cumulant-based truncation** ``[a, b] = c1 +/- L * sqrt(c2)`` with the
  closed-form Heston cumulants from FO eq. 33 (``c4`` set to zero as in the
  reference paper). Default ``L = 12`` and ``N = 160`` give truncation
  error well below ``1e-10`` over the standard equity Heston parameter
  range.

The pricer is JIT-able, vmap-able and autodifferentiable through every
Heston parameter via ``jax.grad`` — no internal ``@jit`` is applied, in
line with the rest of ``valax/pricing/analytic`` (left to the caller).

References:
    Fang, F. & Oosterlee, C. W. (2008). "A novel pricing method for
    European options based on Fourier-cosine series expansions."
    SIAM J. Sci. Comput., 31(2), 826-848.

    Lord, R. & Kahl, C. (2010). "Complex logarithms in Heston-like
    models." Mathematical Finance, 20(4), 671-694.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption
from valax.models.heston import HestonModel


# ── Heston cumulants of log-spot increment ─────────────────────────────

def _heston_cumulants(
    T: Float[Array, ""],
    mu: Float[Array, ""],
    model: HestonModel,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """First and second cumulants of ``X_T - X_0 = log(S_T / S_0)`` under Heston.

    Closed forms from Fang-Oosterlee (2008), eq. 33. Used to build the
    truncation interval ``[a, b]`` for the COS expansion. ``c4`` is set
    to zero following the same reference.

    Args:
        T: Expiry in year fractions.
        mu: Risk-neutral drift ``rate - dividend``.
        model: Heston parameters.

    Returns:
        ``(c1, c2)`` — first cumulant (mean) and second cumulant (variance).
    """
    v0 = model.v0
    kappa = model.kappa
    theta = model.theta
    xi = model.xi
    rho = model.rho

    ekt = jnp.exp(-kappa * T)
    e2kt = jnp.exp(-2.0 * kappa * T)

    c1 = (
        mu * T
        + (1.0 - ekt) * (theta - v0) / (2.0 * kappa)
        - 0.5 * theta * T
    )

    c2 = (1.0 / (8.0 * kappa**3)) * (
        xi * T * kappa * ekt * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
        + kappa * rho * xi * (1.0 - ekt) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * xi + xi**2 + 4.0 * kappa**2)
        + xi**2 * ((theta - 2.0 * v0) * e2kt + theta * (6.0 * ekt - 7.0) + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - ekt)
    )

    return c1, c2


# ── Heston characteristic function (Lord-Kahl "Little Trap" form) ──────

def _heston_char_fn(
    u: Float[Array, ""],
    T: Float[Array, ""],
    x0: Float[Array, ""],
    mu: Float[Array, ""],
    model: HestonModel,
) -> Array:
    """Characteristic function of ``Y_T = log(S_T / K)`` under Heston.

    Uses the Lord-Kahl (2010) "Little Trap" formulation with the
    ``(d_root - d) / (d_root + d)`` (minus) sign convention. This keeps
    the principal branch of the complex logarithm intact for all
    ``T > 0`` and all admissible ``rho``, avoiding the spurious
    discontinuity in the original Heston-1993 form that breaks autodiff.

    Args:
        u: Real-valued Fourier frequency.
        T: Expiry in year fractions.
        x0: Initial log-moneyness ``log(spot / strike)``.
        mu: Risk-neutral drift ``rate - dividend``.
        model: Heston parameters.

    Returns:
        Complex-valued ``phi_Y(u) = E[exp(i u Y_T)]``.
    """
    kappa = model.kappa
    theta = model.theta
    xi = model.xi
    rho = model.rho
    v0 = model.v0

    iu = 1j * u

    # Lord-Kahl "Little Trap": d_root = kappa - rho * xi * i * u, then
    # d = sqrt(d_root^2 + xi^2 * (i u + u^2)), g = (d_root - d) / (d_root + d).
    d_root = kappa - rho * xi * iu
    d = jnp.sqrt(d_root**2 + xi**2 * (iu + u**2))
    g = (d_root - d) / (d_root + d)
    e_minus_dT = jnp.exp(-d * T)

    # Heston A(u, T) and B(u, T) coefficients of the affine cf:
    # phi(u) = exp(A + B * v0 + i u (x0 + mu T))
    A = (kappa * theta / xi**2) * (
        (d_root - d) * T
        - 2.0 * jnp.log((1.0 - g * e_minus_dT) / (1.0 - g))
    )
    B = ((d_root - d) / xi**2) * (1.0 - e_minus_dT) / (1.0 - g * e_minus_dT)

    return jnp.exp(iu * (x0 + mu * T) + A + B * v0)


# ── COS payoff coefficients (call: [0, b]; put: [a, 0]) ────────────────

def _chi_psi(
    k: Float[Array, " N"],
    a: Float[Array, ""],
    b: Float[Array, ""],
    c: Float[Array, ""],
    d: Float[Array, ""],
) -> tuple[Float[Array, " N"], Float[Array, " N"]]:
    """Closed-form ``chi_k`` and ``psi_k`` integrals on ``[c, d] \\subset [a, b]``.

    From Fang-Oosterlee (2008), eq. 22-24:

    * ``chi_k = integral_c^d cos(k pi (y - a) / (b - a)) * exp(y) dy``
    * ``psi_k = integral_c^d cos(k pi (y - a) / (b - a)) dy``

    Uses an autodiff-safe "double where" for ``psi_k`` at ``k = 0`` so
    ``jax.grad`` does not propagate NaNs through the unused branch.
    """
    kpi_ba = k * jnp.pi / (b - a)

    cos_d = jnp.cos(kpi_ba * (d - a))
    cos_c = jnp.cos(kpi_ba * (c - a))
    sin_d = jnp.sin(kpi_ba * (d - a))
    sin_c = jnp.sin(kpi_ba * (c - a))

    exp_d = jnp.exp(d)
    exp_c = jnp.exp(c)

    chi = (1.0 / (1.0 + kpi_ba**2)) * (
        cos_d * exp_d
        - cos_c * exp_c
        + kpi_ba * sin_d * exp_d
        - kpi_ba * sin_c * exp_c
    )

    # psi: k = 0 is a removable singularity (the integrand is constant 1
    # so the integral is (d - c)). Use a "safe k" inside the branch and
    # mask the result with `where` to keep gradients NaN-free.
    safe_k = jnp.where(k == 0, 1.0, k)
    psi_nonzero = (b - a) / (safe_k * jnp.pi) * (sin_d - sin_c)
    psi = jnp.where(k == 0, d - c, psi_nonzero)

    return chi, psi


def _cos_payoff_coeffs(
    k: Float[Array, " N"],
    a: Float[Array, ""],
    b: Float[Array, ""],
    strike: Float[Array, ""],
    is_call: bool,
) -> Float[Array, " N"]:
    """Payoff cosine coefficients ``U_k`` for a European call or put.

    The payoff is expressed in log-moneyness ``Y = log(S_T / K)``:

    * call: ``K * max(exp(Y) - 1, 0)`` is non-zero only on ``Y in [0, b]``
    * put:  ``K * max(1 - exp(Y), 0)`` is non-zero only on ``Y in [a, 0]``

    ``is_call`` is a static Python bool (it is a static field of
    ``EuropeanOption``), so this branch resolves at trace time and
    JIT-compiles to a single specialized graph.
    """
    if is_call:
        chi, psi = _chi_psi(k, a, b, jnp.zeros_like(a), b)
        return (2.0 / (b - a)) * strike * (chi - psi)
    else:
        chi, psi = _chi_psi(k, a, b, a, jnp.zeros_like(a))
        return (2.0 / (b - a)) * strike * (-chi + psi)


# ── Public pricer ──────────────────────────────────────────────────────

def heston_cos_price(
    option: EuropeanOption,
    spot: Float[Array, ""],
    rate: Float[Array, ""] | None = None,
    dividend: Float[Array, ""] | None = None,
    model: HestonModel | None = None,
    *,
    N: int = 160,
    L: float = 12.0,
) -> Float[Array, ""]:
    """Heston European option price via the Fang-Oosterlee COS method.

    Args:
        option: European option contract (``strike``, ``expiry``,
            static ``is_call``).
        spot: Current underlying price.
        rate: Risk-free rate (continuously compounded). If ``None``,
            falls back to ``model.rate``. Pass an explicit value to
            override the model's curve (e.g. for repricing the same
            fitted model under a stressed discount curve).
        dividend: Continuous dividend yield. If ``None``, falls back
            to ``model.dividend``.
        model: Heston parameters (``v0``, ``kappa``, ``theta``, ``xi``,
            ``rho``, plus ``rate`` / ``dividend`` carried for default
            fall-back). Required.
        N: Number of cosine basis terms. Default 160 gives ``< 1e-10``
            truncation error on standard equity Heston parameters and
            costs ~160 complex exp evaluations per call.
        L: Truncation width in cumulant-standard-deviations. Default 12.

    Returns:
        Option price as a scalar JAX array.

    Notes:
        - The function is scalar-in / scalar-out. Use ``jax.vmap`` for
          batching over strikes / expiries (see
          ``valax/calibration/heston.py`` for the calibration call site).
        - Greeks are obtained via ``jax.grad`` on the appropriate argument
          (or on the ``model`` pytree for parameter sensitivities).
        - For a put the COS expansion uses the put-payoff coefficients
          directly rather than put-call parity, which is strictly more
          accurate for deep OTM puts.
        - Backward-compat note: prior versions required ``rate`` and
          ``dividend`` as positional non-default args. Existing call
          sites that pass them positionally continue to work; new call
          sites can omit them and the model's own values are used.
    """
    if model is None:
        raise TypeError("heston_cos_price: `model` is required")
    if rate is None:
        rate = model.rate
    if dividend is None:
        dividend = model.dividend

    strike = option.strike
    T = option.expiry
    mu = rate - dividend

    x0 = jnp.log(spot / strike)

    # Truncation interval from Heston cumulants of log-moneyness Y_T:
    # c1_Y = x0 + c1_X (X = log-spot increment), c2_Y = c2_X.
    c1_x, c2_x = _heston_cumulants(T, mu, model)
    c1 = x0 + c1_x
    # Guard against tiny negative c2 from floating-point cancellation in
    # extreme parameter regions; downstream sqrt would NaN otherwise.
    c2_safe = jnp.maximum(c2_x, 1e-30)
    width = L * jnp.sqrt(c2_safe)
    a = c1 - width
    b = c1 + width

    # Cosine grid u_k = k pi / (b - a), k = 0 .. N-1.
    # Inherit floating-point precision from T (respects jax_enable_x64).
    k = jnp.arange(N).astype(T.dtype)
    u = k * jnp.pi / (b - a)

    # Characteristic function evaluated at every u_k.
    phi = jax.vmap(lambda uk: _heston_char_fn(uk, T, x0, mu, model))(u)

    # Payoff coefficients U_k.
    U = _cos_payoff_coeffs(k, a, b, strike, option.is_call)

    # COS sum: w_0 = 1/2, w_k = 1 for k > 0.
    weights = jnp.ones_like(k).at[0].set(0.5)
    terms = jnp.real(phi * jnp.exp(-1j * u * a)) * U

    return jnp.exp(-rate * T) * jnp.sum(weights * terms)
