"""Spatial differential operators for 2-D (ADI) finite-difference solvers.

An :class:`Operator2D` holds the three-way operator splitting used by the
Douglas / Craig-Sneyd / Hundsdorfer-Verwer ADI schemes (in't Hout & Foulon,
2010) for a two-factor pricing PDE such as Heston. In log-spot ``x = ln S`` and
variance ``v`` the spatial operator ``L`` is split as ``L = A0 + A1 + A2`` with

.. math::

    A_1 V &= \\tfrac{1}{2} c_{xx}\\, V_{xx} + c_x\\, V_x - \\tfrac{1}{2} r V, \\\\
    A_2 V &= \\tfrac{1}{2} c_{vv}\\, V_{vv} + c_v\\, V_v - \\tfrac{1}{2} r V, \\\\
    A_0 V &= c_{xv}\\, V_{xv},

where ``A1`` is tridiagonal along the log-spot axis, ``A2`` is tridiagonal along
the variance axis, and ``A0`` is the mixed second derivative (a 3x3 cross
stencil). The discount ``-rV`` is split evenly between ``A1`` and ``A2``, the
standard ADI convention that keeps each implicit per-axis solve a well-posed
tridiagonal system. ``A0`` is applied **only explicitly** by the ADI stepper --
it is never inverted -- so it is stored as a per-node coefficient together with
the separable central first-derivative weights whose composition realises
``V_xv = D_x(D_v V)``.

All finite differences are the three-point non-uniform central stencils (exact
for quadratics), so concentrated grids (:func:`~valax.pricing.pde.grids.sinh_concentrated_grid`
along the variance axis) are supported. Coefficient arrays are broadcast to the
full ``(n_x, n_y)`` field shape; edge rows use a zero exterior (ghost) value in
the operator action and are corrected by the boundary layer / ADI stepper.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from valax.pricing.pde.grids import Grid1D, Grid2D, boundary_coords


def _axis_stencil(
    grid: Grid1D,
) -> tuple[
    Float[Array, " n"],
    Float[Array, " n"],
    Float[Array, " n"],
    Float[Array, " n"],
    Float[Array, " n"],
    Float[Array, " n"],
]:
    """Per-node three-point stencil weights for one axis.

    Returns the second-derivative weights ``(dxx_lower, dxx_diag, dxx_upper)``
    scaled for the ``1/2 c V_xx`` term (the ``1/2`` is absorbed) and the
    first-derivative weights ``(dx_lower, dx_diag, dx_upper)`` for the ``c V_x``
    term, both on the (possibly non-uniform) node spacing. Exterior spacings at
    the two ends come from :func:`~valax.pricing.pde.grids.boundary_coords`, so
    the operator shares its boundary placement with the Dirichlet values.

    Args:
        grid: The 1-D axis grid.

    Returns:
        ``(dxx_lower, dxx_diag, dxx_upper, dx_lower, dx_diag, dx_upper)``, each
        of length ``grid.n``.
    """
    x_lo, x_hi = boundary_coords(grid)
    full = jnp.concatenate([x_lo[jnp.newaxis], grid.nodes, x_hi[jnp.newaxis]])
    h_minus = grid.nodes - full[:-2]
    h_plus = full[2:] - grid.nodes
    h_sum = h_minus + h_plus

    # Second-derivative weights for 1/2 * V_xx (the 1/2 absorbed):
    #   1/2 * [2/(h- h_s), -2/(h- h+), 2/(h+ h_s)].
    dxx_lower = 1.0 / (h_minus * h_sum)
    dxx_diag = -1.0 / (h_minus * h_plus)
    dxx_upper = 1.0 / (h_plus * h_sum)

    # First-derivative weights (second-order non-uniform central).
    dx_lower = -h_plus / (h_minus * h_sum)
    dx_diag = (h_plus - h_minus) / (h_minus * h_plus)
    dx_upper = h_minus / (h_plus * h_sum)

    return dxx_lower, dxx_diag, dxx_upper, dx_lower, dx_diag, dx_upper


class Operator2D(eqx.Module):
    """Three-way ADI operator split ``L = A0 + A1 + A2`` on a :class:`Grid2D`.

    Fields with a leading ``a1_`` are the tridiagonal bands of the log-spot
    operator ``A1`` (coupling ``V[i-1, j], V[i, j], V[i+1, j]``); ``a2_`` are the
    tridiagonal bands of the variance operator ``A2`` (coupling
    ``V[i, j-1], V[i, j], V[i, j+1]``). All are shape ``(n_x, n_y)``. The mixed
    operator ``A0`` is stored as its per-node coefficient ``c0`` plus the
    first-derivative weights along each axis (``sx_*`` length ``n_x``, ``sv_*``
    length ``n_y``); its action is ``c0 * D_x(D_v V)``.

    Attributes:
        a1_lower, a1_diag, a1_upper: Log-spot tridiagonal bands, ``(n_x, n_y)``.
        a2_lower, a2_diag, a2_upper: Variance tridiagonal bands, ``(n_x, n_y)``.
        c0: Mixed-derivative coefficient ``rho sigma v``, ``(n_x, n_y)``.
        sx_lower, sx_diag, sx_upper: Log-spot first-derivative weights, ``(n_x,)``.
        sv_lower, sv_diag, sv_upper: Variance first-derivative weights, ``(n_y,)``.
    """

    a1_lower: Float[Array, "n_x n_y"]
    a1_diag: Float[Array, "n_x n_y"]
    a1_upper: Float[Array, "n_x n_y"]
    a2_lower: Float[Array, "n_x n_y"]
    a2_diag: Float[Array, "n_x n_y"]
    a2_upper: Float[Array, "n_x n_y"]
    c0: Float[Array, "n_x n_y"]
    sx_lower: Float[Array, " n_x"]
    sx_diag: Float[Array, " n_x"]
    sx_upper: Float[Array, " n_x"]
    sv_lower: Float[Array, " n_y"]
    sv_diag: Float[Array, " n_y"]
    sv_upper: Float[Array, " n_y"]

    def apply_a1(self, v: Float[Array, "n_x n_y"]) -> Float[Array, "n_x n_y"]:
        """Apply the log-spot operator ``A1`` (tridiagonal along axis 0)."""
        v_m = jnp.pad(v, ((1, 0), (0, 0)))[:-1]  # V[i-1, j], zero ghost at i=0
        v_p = jnp.pad(v, ((0, 1), (0, 0)))[1:]   # V[i+1, j], zero ghost at i=n_x-1
        return self.a1_lower * v_m + self.a1_diag * v + self.a1_upper * v_p

    def apply_a2(self, v: Float[Array, "n_x n_y"]) -> Float[Array, "n_x n_y"]:
        """Apply the variance operator ``A2`` (tridiagonal along axis 1)."""
        v_m = jnp.pad(v, ((0, 0), (1, 0)))[:, :-1]  # V[i, j-1], zero ghost at j=0
        v_p = jnp.pad(v, ((0, 0), (0, 1)))[:, 1:]   # V[i, j+1], zero ghost at j=n_y-1
        return self.a2_lower * v_m + self.a2_diag * v + self.a2_upper * v_p

    def _d_v(self, v: Float[Array, "n_x n_y"]) -> Float[Array, "n_x n_y"]:
        """Central first derivative along the variance axis (axis 1)."""
        v_m = jnp.pad(v, ((0, 0), (1, 0)))[:, :-1]
        v_p = jnp.pad(v, ((0, 0), (0, 1)))[:, 1:]
        return self.sv_lower * v_m + self.sv_diag * v + self.sv_upper * v_p

    def _d_x(self, w: Float[Array, "n_x n_y"]) -> Float[Array, "n_x n_y"]:
        """Central first derivative along the log-spot axis (axis 0)."""
        w_m = jnp.pad(w, ((1, 0), (0, 0)))[:-1]
        w_p = jnp.pad(w, ((0, 1), (0, 0)))[1:]
        return (
            self.sx_lower[:, jnp.newaxis] * w_m
            + self.sx_diag[:, jnp.newaxis] * w
            + self.sx_upper[:, jnp.newaxis] * w_p
        )

    def apply_a0(self, v: Float[Array, "n_x n_y"]) -> Float[Array, "n_x n_y"]:
        """Apply the mixed operator ``A0 V = c0 * V_xv`` (explicit only).

        Realises the cross derivative as the composition of central first
        derivatives ``D_x(D_v V)`` (a 3x3 stencil), scaled by the per-node mixed
        coefficient ``c0``.
        """
        return self.c0 * self._d_x(self._d_v(v))

    def apply(self, v: Float[Array, "n_x n_y"]) -> Float[Array, "n_x n_y"]:
        """Apply the full spatial operator ``L V = (A0 + A1 + A2) V``."""
        return self.apply_a0(v) + self.apply_a1(v) + self.apply_a2(v)


def build_operator_2d(
    grid: Grid2D,
    *,
    diff_x: Float[Array, "n_x n_y"] | Float[Array, ""],
    drift_x: Float[Array, "n_x n_y"] | Float[Array, ""],
    diff_v: Float[Array, "n_x n_y"] | Float[Array, ""],
    drift_v: Float[Array, "n_x n_y"] | Float[Array, ""],
    mixed: Float[Array, "n_x n_y"] | Float[Array, ""],
    discount: Float[Array, ""],
) -> Operator2D:
    """Assemble an :class:`Operator2D` from per-node coefficient fields.

    The coefficients follow the same convention as
    :func:`~valax.pricing.pde.operators.build_operator_1d`: ``diff_*`` is the
    coefficient of the second derivative (the ``1/2`` is applied by the stencil),
    ``drift_*`` the coefficient of the first derivative, and ``mixed`` the
    coefficient of the mixed second derivative ``V_xv`` (no ``1/2``). The scalar
    ``discount`` is split evenly between ``A1`` and ``A2``.

    Args:
        grid: The tensor-product grid.
        diff_x: Coefficient ``c_xx`` of ``V_xx`` (e.g. ``v`` for Heston).
        drift_x: Coefficient ``c_x`` of ``V_x`` (e.g. ``r - q - v/2``).
        diff_v: Coefficient ``c_vv`` of ``V_vv`` (e.g. ``sigma^2 v``).
        drift_v: Coefficient ``c_v`` of ``V_v`` (e.g. ``kappa (theta - v)``).
        mixed: Coefficient ``c_xv`` of ``V_xv`` (e.g. ``rho sigma v``).
        discount: Discount rate ``r`` (split ``r/2`` onto each axis).

    Returns:
        The assembled :class:`Operator2D`.
    """
    shape = grid.shape
    half_r = 0.5 * discount

    xxl, xxd, xxu, xl, xd, xu = _axis_stencil(grid.x)  # each (n_x,)
    vvl, vvd, vvu, vl, vd, vu = _axis_stencil(grid.y)  # each (n_y,)

    # A1: tridiagonal along log-spot (broadcast x-weights across the v axis).
    a1_lower = jnp.broadcast_to(diff_x * xxl[:, None] + drift_x * xl[:, None], shape)
    a1_diag = jnp.broadcast_to(
        diff_x * xxd[:, None] + drift_x * xd[:, None] - half_r, shape
    )
    a1_upper = jnp.broadcast_to(diff_x * xxu[:, None] + drift_x * xu[:, None], shape)

    # A2: tridiagonal along variance (broadcast v-weights across the x axis).
    a2_lower = jnp.broadcast_to(diff_v * vvl[None, :] + drift_v * vl[None, :], shape)
    a2_diag = jnp.broadcast_to(
        diff_v * vvd[None, :] + drift_v * vd[None, :] - half_r, shape
    )
    a2_upper = jnp.broadcast_to(diff_v * vvu[None, :] + drift_v * vu[None, :], shape)

    c0 = jnp.broadcast_to(jnp.asarray(mixed) * jnp.ones(shape), shape)

    return Operator2D(
        a1_lower=a1_lower,
        a1_diag=a1_diag,
        a1_upper=a1_upper,
        a2_lower=a2_lower,
        a2_diag=a2_diag,
        a2_upper=a2_upper,
        c0=c0,
        sx_lower=xl,
        sx_diag=xd,
        sx_upper=xu,
        sv_lower=vl,
        sv_diag=vd,
        sv_upper=vu,
    )
