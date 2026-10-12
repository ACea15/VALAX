r"""Layer 1b of the adjoint chain: vega to *volatility quotes*.

:mod:`valax.greeks.adjoint` pushes a price's curve-node gradient back to the
curve **quotes**. This module does the matching step one layer up: it pushes a
price's gradient with respect to a surface's *fitted SABR parameters* back to
the **volatility quotes** the smile was fitted from — and, in the same sweep,
back to the forward the smile was anchored on, which is what couples the
volatility layer to the curve layer.

The least-squares twist
-----------------------
A curve bootstrap is a square root-find, so its residual is ready-made for the
implicit function theorem. A smile fit is not: there are more quoted strikes
than SABR parameters. The system that *is* square is the fit's **first-order
condition** — the stationarity of the weighted sum of squares,

.. math::

    R_v(\theta_v;\, \sigma^{\text{mkt}}, F)
      = \nabla_{\theta_v}\,
        \tfrac12 \sum_k w_k
        \bigl(\sigma_{\text{SABR}}(K_k;\, \theta_v, F) - \sigma^{\text{mkt}}_k\bigr)^2
      = 0,

which has exactly as many equations as free parameters. Differentiating it is
the standard route from a least-squares calibration to its sensitivities, and
it is the *only* place the choice of Hessian enters: ``jax.grad`` of the
gradient gives the **full** Hessian
:math:`J^T W J + \sum_k r_k \nabla^2 r_k`, not the Gauss-Newton approximation
:math:`J^T W J`. On a well-fitting smile the two coincide to the size of the
residual; keeping the full Hessian means the adjoint stays correct when the fit
is *not* tight, which is when a sensitivity is most likely to be trusted
wrongly. This is the documented resolution of the "Gauss-Newton vs full
Hessian" choice in the
[audit conformance map](../architecture/aad-audit-conformance.md).

:func:`~valax.greeks.adjoint.level_adjoint` then does the rest, unchanged:
solve :math:`(\partial R_v/\partial\theta_v)^T \lambda_v = \bar\theta_v` and
contract against :math:`\partial R_v/\partial\text{inputs}`. The two input
blocks are the two answers wanted:

* ``market_vols`` → **vega to the quoted vols**, one per strike;
* ``forward`` → :math:`\partial V/\partial F_0`, the term that chains into the
  curve level's cotangent. It is non-zero **only** because the forward is a
  genuine input to the fit, which is what
  :func:`valax.surfaces.swaption_cube.calibrate_swaption_cube_from_curves`
  arranged; with a caller-supplied forward it is structurally zero and the
  volatility layer carries no curve risk at all.

Parametrisation
---------------
``theta_v`` is a flat vector in the SABR parameters' **natural** space,
``(alpha, rho, nu)`` with beta fixed or ``(alpha, beta, rho, nu)`` with beta
free. The solver itself works in an unconstrained reparametrisation (see
:mod:`valax.calibration.transforms`), but stationarity is invariant under an
invertible change of variables at an interior optimum, so the theorem may be
applied in whichever parametrisation the answers are wanted in — and node
parameters are what a cube actually stores.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Float
from jax import Array

from valax.calibration.level import CalibrationLevel, level_adjoint
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.surfaces.optionlet_surface import OptionletVolSurface
from valax.surfaces.swaption_cube import SwaptionCube


#: Order of ``theta_v`` with beta held fixed.
SABR_FIXED_BETA_PARAMS = ("alpha", "rho", "nu")

#: Order of ``theta_v`` with beta free.
SABR_FREE_BETA_PARAMS = ("alpha", "beta", "rho", "nu")


def sabr_theta_names(free_beta: bool = False) -> tuple[str, ...]:
    """Names of the entries of ``theta_v``, in order.

    Args:
        free_beta: Whether beta is a free parameter of the fit.

    Returns:
        :data:`SABR_FREE_BETA_PARAMS` when ``free_beta``, else
        :data:`SABR_FIXED_BETA_PARAMS`.
    """
    return SABR_FREE_BETA_PARAMS if free_beta else SABR_FIXED_BETA_PARAMS


def sabr_theta(
    model: SABRModel,
    *,
    free_beta: bool = False,
) -> Float[Array, " n_params"]:
    """Pack a :class:`SABRModel` into a flat ``theta_v`` vector.

    Args:
        model: Fitted SABR parameters.
        free_beta: Include beta in the vector.

    Returns:
        The parameter vector in :func:`sabr_theta_names` order.
    """
    if free_beta:
        return jnp.stack([model.alpha, model.beta, model.rho, model.nu])
    return jnp.stack([model.alpha, model.rho, model.nu])


def sabr_model_from_theta(
    theta: Float[Array, " n_params"],
    beta: Float[Array, ""],
    *,
    free_beta: bool = False,
) -> SABRModel:
    """Unpack a flat ``theta_v`` vector back into a :class:`SABRModel`.

    Args:
        theta: Parameter vector in :func:`sabr_theta_names` order.
        beta: The held-fixed beta. Ignored when ``free_beta`` is True.
        free_beta: Whether ``theta`` carries beta.

    Returns:
        The reconstructed SABR model.
    """
    if free_beta:
        return SABRModel(
            alpha=theta[0], beta=theta[1], rho=theta[2], nu=theta[3]
        )
    return SABRModel(alpha=theta[0], beta=beta, rho=theta[1], nu=theta[2])


class SABRSliceInputs(eqx.Module):
    """The curve- and market-dependent inputs to one smile fit.

    Both blocks are differentiated, and each answers a different question:
    ``market_vols`` gives vega in quote space, ``forward`` gives the coupling
    back to the curve layer.

    Attributes:
        market_vols: Quoted volatilities, one per strike, in the slice's
            quoting convention.
        forward: The forward the smile is anchored on, :math:`F_0`.
    """

    market_vols: Float[Array, " n_k"]
    forward: Float[Array, ""]


class SliceVolAdjoint(eqx.Module):
    """Result of differentiating one smile fit.

    Attributes:
        vol_quote_vegas: :math:`\\partial V / \\partial \\sigma^{mkt}_k`, one
            per quoted strike — vega in the space the desk is quoted in.
        forward_bar: :math:`\\partial V / \\partial F_0` through the fit: the
            contribution this volatility level adds to the curve level's
            cotangent (spec A-3).
        lam: The adjoint multiplier :math:`\\lambda_v`, one entry per free
            parameter.
        stationarity_norm: ``max|R_v|`` at the fitted parameters — the
            first-order condition's violation. The implicit function theorem
            assumes this is zero, so a large value invalidates everything
            above: the fit did not actually reach a stationary point.
    """

    vol_quote_vegas: Float[Array, " n_k"]
    forward_bar: Float[Array, ""]
    lam: Float[Array, " n_params"]
    stationarity_norm: Float[Array, ""]


def sabr_slice_level(
    model: SABRModel,
    strikes: Float[Array, " n_k"],
    market_vols: Float[Array, " n_k"],
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    *,
    weights: Float[Array, " n_k"] | None = None,
    is_normal: bool = False,
    shift: Float[Array, ""] | float = 0.0,
    free_beta: bool = False,
    quote_form: str = "absolute_strike",
) -> CalibrationLevel:
    r"""Express one fitted SABR smile as ``R_v(theta_v; inputs) = 0``.

    The residual is the stationarity condition of the weighted sum of squares
    (see the module docstring), so the level is square and the implicit
    function theorem applies.

    Under ``quote_form="atm_offset"`` the strike grid is
    :math:`K_k = F_0 + \delta_k` and therefore moves with the forward; the
    residual rebuilds it from ``inputs.forward`` so
    :math:`\partial R_v / \partial F_0` picks up both the smile-anchor effect
    and the strike-grid effect. Under ``"absolute_strike"`` the grid is fixed
    and only the anchor moves.

    Args:
        model: The *fitted* SABR parameters at this node.
        strikes: Absolute strikes, or offsets to the forward when
            ``quote_form="atm_offset"``.
        market_vols: Quoted volatilities at those strikes.
        forward: The forward the slice was fitted at.
        expiry: Time to expiry (year fraction).
        weights: Per-strike least-squares weights. Defaults to uniform, which
            is what :func:`valax.calibration.sabr.calibrate_sabr` uses.
        is_normal: Quote and expansion convention — normal (Bachelier) if
            True, lognormal (Black-76) otherwise. Must match the fit.
        shift: Displacement for the normal expansion; ignored when
            ``is_normal`` is False. Must match the fit.
        free_beta: Whether beta was a free parameter of the fit. Determines
            the length and order of ``theta_v``.
        quote_form: ``"absolute_strike"`` or ``"atm_offset"``.

    Returns:
        A :class:`~valax.greeks.adjoint.CalibrationLevel` whose ``theta`` is
        the flat parameter vector and whose ``inputs`` is a
        :class:`SABRSliceInputs`.

    Raises:
        ValueError: If ``quote_form`` is not a recognised form.
    """
    if quote_form not in ("absolute_strike", "atm_offset"):
        raise ValueError(
            "quote_form must be 'absolute_strike' or 'atm_offset', got "
            f"{quote_form!r}."
        )

    if is_normal:
        vol_fn = functools.partial(
            sabr_normal_implied_vol, shift=jnp.asarray(shift)
        )
    else:
        vol_fn = sabr_implied_vol

    w = jnp.ones_like(strikes) if weights is None else jnp.asarray(weights)
    beta = model.beta
    offsets = jnp.asarray(strikes)
    is_offset = quote_form == "atm_offset"

    def residual_fn(
        theta: Float[Array, " n_params"],
        inputs: SABRSliceInputs,
    ) -> Float[Array, " n_params"]:
        """Stationarity of the weighted SSE at ``theta``."""

        def sse(th: Float[Array, " n_params"]) -> Float[Array, ""]:
            m = sabr_model_from_theta(th, beta, free_beta=free_beta)
            k = inputs.forward + offsets if is_offset else offsets
            fitted = jax.vmap(
                lambda kk: vol_fn(m, inputs.forward, kk, expiry)
            )(k)
            r = w * (fitted - inputs.market_vols)
            return 0.5 * jnp.sum(r * r)

        return jax.grad(sse)(theta)

    return CalibrationLevel(
        residual_fn=residual_fn,
        theta=sabr_theta(model, free_beta=free_beta),
        inputs=SABRSliceInputs(
            market_vols=jnp.asarray(market_vols),
            forward=jnp.asarray(forward),
        ),
    )


def vol_quote_vegas(
    level: CalibrationLevel,
    theta_bar: Float[Array, " n_params"],
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> SliceVolAdjoint:
    r"""Push a parameter cotangent back to the quoted vols and the forward.

    One implicit-function-theorem step on a smile fit built by
    :func:`sabr_slice_level`. Costs one transposed solve of a tiny
    (3x3 or 4x4) system plus two reverse sweeps through the residual,
    regardless of how many strikes are quoted.

    Args:
        level: A level from :func:`sabr_slice_level` (or any level whose
            ``inputs`` is a :class:`SABRSliceInputs`).
        theta_bar: :math:`\partial V / \partial \theta_v`, in
            :func:`sabr_theta_names` order.
        solver: Linear solver for the transposed system. Defaults to
            ``lineax.AutoLinearSolver(well_posed=True)``.

    Returns:
        A :class:`SliceVolAdjoint`.
    """
    adj = level_adjoint(level, theta_bar, solver=solver)
    return SliceVolAdjoint(
        vol_quote_vegas=adj.inputs_bar.market_vols,
        forward_bar=adj.inputs_bar.forward,
        lam=adj.lam,
        stationarity_norm=adj.residual_norm,
    )


# ── Cube / surface convenience ───────────────────────────────────────


def cube_node_level(
    cube: SwaptionCube,
    expiry_index: int,
    tenor_index: int,
    strikes: Float[Array, " n_k"],
    market_vols: Float[Array, " n_k"],
    *,
    weights: Float[Array, " n_k"] | None = None,
    free_beta: bool = False,
) -> CalibrationLevel:
    """Build a :func:`sabr_slice_level` from one node of a swaption cube.

    Reads the node's fitted parameters, its fitted forward, its expiry and the
    cube's quoting conventions (``is_normal``, ``shift``, ``quote_form``) off
    the object, so the level cannot drift from the cube it describes.

    Args:
        cube: The calibrated cube.
        expiry_index: Row index of the node.
        tenor_index: Column index of the node.
        strikes: The strikes this node was fitted at, in the cube's
            ``quote_form`` (absolute rates, or offsets to the node forward).
        market_vols: The quoted vols at those strikes.
        weights: Per-strike least-squares weights; defaults to uniform.
        free_beta: Whether beta was free in the fit.

    Returns:
        The node's calibration level.
    """
    i, j = expiry_index, tenor_index
    model = SABRModel(
        alpha=cube.alphas[i, j],
        beta=cube.betas[i, j],
        rho=cube.rhos[i, j],
        nu=cube.nus[i, j],
    )
    return sabr_slice_level(
        model,
        strikes,
        market_vols,
        cube.forwards[i, j],
        cube.expiries[i],
        weights=weights,
        is_normal=cube.is_normal,
        shift=cube.shift,
        free_beta=free_beta,
        quote_form=cube.quote_form,
    )


def optionlet_slice_level(
    surface: OptionletVolSurface,
    expiry_index: int,
    strikes: Float[Array, " n_k"],
    market_vols: Float[Array, " n_k"],
    *,
    weights: Float[Array, " n_k"] | None = None,
    free_beta: bool = False,
) -> CalibrationLevel:
    """Build a :func:`sabr_slice_level` from one expiry of an optionlet surface.

    The caplet-side counterpart of :func:`cube_node_level`; the optionlet
    surface has no tenor axis, so a slice is indexed by expiry alone.

    Args:
        surface: The calibrated optionlet surface.
        expiry_index: Index of the expiry slice.
        strikes: The strikes this slice was fitted at, in the surface's
            ``quote_form``.
        market_vols: The quoted vols at those strikes.
        weights: Per-strike least-squares weights; defaults to uniform.
        free_beta: Whether beta was free in the fit.

    Returns:
        The slice's calibration level.
    """
    i = expiry_index
    model = SABRModel(
        alpha=surface.alphas[i],
        beta=surface.betas[i],
        rho=surface.rhos[i],
        nu=surface.nus[i],
    )
    return sabr_slice_level(
        model,
        strikes,
        market_vols,
        surface.forwards[i],
        surface.expiries[i],
        weights=weights,
        is_normal=surface.is_normal,
        shift=surface.shift,
        free_beta=free_beta,
        quote_form=surface.quote_form,
    )


def cube_node_cotangent(
    cube_bar: SwaptionCube,
    expiry_index: int,
    tenor_index: int,
    *,
    free_beta: bool = False,
) -> Float[Array, " n_params"]:
    """Extract one node's ``theta_bar`` from a cube-shaped cotangent.

    ``jax.grad`` of a price over a :class:`SwaptionCube` returns a pytree with
    the cube's structure; this picks the node's parameter cotangents out of it
    in :func:`sabr_theta_names` order, ready for :func:`vol_quote_vegas`.

    Args:
        cube_bar: A cube-shaped cotangent (the output of ``jax.grad`` /
            ``eqx.filter_grad`` over a cube).
        expiry_index: Row index of the node.
        tenor_index: Column index of the node.
        free_beta: Whether beta was free in the fit, and so appears in
            ``theta_bar``.

    Returns:
        The node's parameter cotangent vector.
    """
    i, j = expiry_index, tenor_index
    if free_beta:
        return jnp.stack([
            cube_bar.alphas[i, j],
            cube_bar.betas[i, j],
            cube_bar.rhos[i, j],
            cube_bar.nus[i, j],
        ])
    return jnp.stack([
        cube_bar.alphas[i, j], cube_bar.rhos[i, j], cube_bar.nus[i, j]
    ])


__all__ = [
    "SABRSliceInputs",
    "SABR_FIXED_BETA_PARAMS",
    "SABR_FREE_BETA_PARAMS",
    "SliceVolAdjoint",
    "cube_node_cotangent",
    "cube_node_level",
    "optionlet_slice_level",
    "sabr_model_from_theta",
    "sabr_slice_level",
    "sabr_theta",
    "sabr_theta_names",
    "vol_quote_vegas",
]
