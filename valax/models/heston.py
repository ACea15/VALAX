"""Heston stochastic volatility model.

dS = (r - q) S dt + sqrt(V) S dW_1
dV = kappa (theta - V) dt + xi sqrt(V) dW_2
Corr(dW_1, dW_2) = rho
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array


class HestonModel(eqx.Module):
    r"""Heston stochastic volatility model parameters.

    The Heston risk-neutral dynamics are

    .. math::

        dS_t &= (r - q) S_t\, dt + \sqrt{V_t}\, S_t\, dW_1 \\
        dV_t &= \kappa (\theta - V_t)\, dt + \xi \sqrt{V_t}\, dW_2 \\
        d\langle W_1, W_2\rangle_t &= \rho\, dt

    Attributes:
        v0: Initial variance :math:`V_0` (variance, not volatility).
        kappa: Mean-reversion speed :math:`\kappa` of the variance
            process.
        theta: Long-run variance level :math:`\theta`.
        xi: Volatility of volatility :math:`\xi` (sometimes written
            :math:`\sigma_V`).
        rho: Instantaneous correlation :math:`\rho \in [-1, 1]` between
            the spot and variance Brownian motions.
        rate: Continuously-compounded risk-free rate :math:`r`.
        dividend: Continuous dividend yield :math:`q`.
    """

    v0: Float[Array, ""]       # initial variance
    kappa: Float[Array, ""]    # mean reversion speed
    theta: Float[Array, ""]    # long-run variance
    xi: Float[Array, ""]       # vol of vol
    rho: Float[Array, ""]      # correlation between spot and vol Brownians
    rate: Float[Array, ""]     # risk-free rate
    dividend: Float[Array, ""] # continuous dividend yield


class HestonDrift(eqx.Module):
    r"""Drift term for the :math:`(\ln S, V)` Heston system.

    Callable that returns the drift vector expected by :mod:`diffrax`
    SDE solvers. See :class:`HestonModel` for the underlying SDE.

    Attributes:
        rate: Risk-free rate :math:`r`.
        dividend: Continuous dividend yield :math:`q`.
        kappa: Variance mean-reversion speed.
        theta: Long-run variance level.
    """

    rate: Float[Array, ""]
    dividend: Float[Array, ""]
    kappa: Float[Array, ""]
    theta: Float[Array, ""]

    def __call__(self, t, y, args):
        r"""Drift of :math:`(\ln S, V)` at state ``y``.

        Args:
            t: Time (unused; the drift is time-homogeneous).
            y: State vector ``[log_S, V]``.
            args: Unused solver args (kept for the diffrax signature).

        Returns:
            Drift vector ``[d(log_S)/dt, dV/dt]`` as a length-2 JAX array,
            with the variance floored at zero (full-truncation scheme)
            to guard against negative-variance excursions.
        """
        log_s, v = y
        v_pos = jnp.maximum(v, 0.0)
        d_log_s = (self.rate - self.dividend) - 0.5 * v_pos
        d_v = self.kappa * (self.theta - v_pos)
        return jnp.array([d_log_s, d_v])


class HestonDiffusion(eqx.Module):
    r"""Diffusion term for the :math:`(\ln S, V)` Heston system.

    Returns a :math:`(2, 2)` diffusion matrix that reproduces the
    :math:`\rho`-correlated Brownian structure of :class:`HestonModel`
    when driven by two independent Brownian motions.

    Attributes:
        xi: Volatility of volatility :math:`\xi`.
        rho: Instantaneous correlation :math:`\rho` between spot and
            variance Brownians.
    """

    xi: Float[Array, ""]
    rho: Float[Array, ""]

    def __call__(self, t, y, args):
        r"""Diffusion matrix at state ``y``.

        Args:
            t: Time (unused; the diffusion is time-homogeneous).
            y: State vector ``[log_S, V]``.
            args: Unused solver args (kept for the diffrax signature).

        Returns:
            :math:`(2, 2)` diffusion matrix driving independent
            Brownians :math:`(dB_1, dB_2)` such that the resulting
            :math:`(d\ln S, dV)` has the correlation structure of
            :class:`HestonModel`. Variance is floored at zero (full
            truncation) before taking :math:`\sqrt{V}`.
        """
        _, v = y
        sqrt_v = jnp.sqrt(jnp.maximum(v, 0.0))
        return jnp.array([
            [sqrt_v, 0.0],
            [self.xi * sqrt_v * self.rho, self.xi * sqrt_v * jnp.sqrt(1.0 - self.rho**2)],
        ])
