"""Stochastic-Local Volatility model (Heston × leverage).

SDE under Q:

.. math::

    dS_t / S_t &= (r - q)\\,dt + L(k_t, t)\\,\\sqrt{V_t}\\,dW_1 \\\\
    dV_t &= \\kappa(\\theta - V_t)\\,dt + \\xi\\,\\sqrt{V_t}\\,dW_2 \\\\
    \\langle dW_1, dW_2 \\rangle &= \\rho\\,dt,

with :math:`k_t = \\log(S_t / F(t))`. The leverage function
:math:`L(k, t)` is **calibrated** so that the marginal of ``S_t``
matches a given implied-vol surface (Markovian projection):

.. math::

    L^2(k, t) = \\sigma_{\\mathrm{Dupire}}^2(k, t)
                \\,/\\,\\mathbb{E}[V_t \\mid k_t = k].

The model pytree carries:

- the Heston block ``(v0, kappa, theta, xi, rho)``,
- the rate / dividend pair,
- the implied-vol ``surface`` (duck-typed via ``total_variance``) —
  kept on the model so the leverage can be re-calibrated against the
  same target without external bookkeeping, and so consumers can
  introspect what the model was calibrated against,
- the ``leverage`` :class:`LeverageGrid` produced by pass 2 of the
  two-pass calibration.

The path generator (``generate_slv_paths``) never queries the surface;
it only consumes the leverage grid + Heston params. The surface is a
calibration artifact, not a pricing-time input.

x64 is enforced at construction time — both Dupire (calibration
target) and Andersen-QE (path generation) need it.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jax import Array
from jaxtyping import Float

# ``LeverageGrid`` is imported lazily inside ``from_heston_and_leverage``
# to avoid a latent circular-import chain at package-init time. The
# ``leverage`` field is annotated as ``eqx.Module`` (matching the
# ``surface: eqx.Module`` pattern in ``LocalVolModel``) so pytree
# registration does not need the concrete class.


def _check_x64() -> None:
    """Match the Dupire-layer x64 guard."""
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "SLVModel requires jax_enable_x64=True. "
            "Add `jax.config.update('jax_enable_x64', True)` at the top "
            "of your script/notebook before importing valax."
        )


class SLVModel(eqx.Module):
    """Heston × leverage stochastic-local volatility model.

    Field order is fixed for ``eqx.tree_at`` and pytree
    flatten/unflatten stability.

    Attributes:
        v0, kappa, theta, xi, rho: Heston parameters (pass-1
            calibration output).
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        surface: Implied-vol surface (duck-typed; must expose
            ``total_variance(k, T) -> Float[""]``). SVI / SABR / Grid
            all comply.
        leverage: Calibrated leverage grid (pass-2 calibration output).
    """

    v0: Float[Array, ""]
    kappa: Float[Array, ""]
    theta: Float[Array, ""]
    xi: Float[Array, ""]
    rho: Float[Array, ""]
    rate: Float[Array, ""]
    dividend: Float[Array, ""]
    surface: eqx.Module
    leverage: eqx.Module  # in practice a ``valax.surfaces.LeverageGrid``

    @classmethod
    def from_heston_and_leverage(
        cls,
        heston: "HestonModel",
        surface: eqx.Module,
        leverage: eqx.Module,
    ) -> "SLVModel":
        """Construct an SLV model from a calibrated Heston + surface + leverage.

        Args:
            heston: A ``HestonModel`` (typically the output of
                ``calibrate_heston``).
            surface: The implied-vol surface used as the Dupire target
                for the leverage calibration.
            leverage: The ``LeverageGrid`` from pass-2 calibration.

        Returns:
            A fully-built ``SLVModel``.
        """
        _check_x64()
        return cls(
            v0=heston.v0,
            kappa=heston.kappa,
            theta=heston.theta,
            xi=heston.xi,
            rho=heston.rho,
            rate=heston.rate,
            dividend=heston.dividend,
            surface=surface,
            leverage=leverage,
        )
