"""Vol-layer adjoint: vega to volatility quotes, and the curve coupling.

Spec A-3 / section 7.3 step 3. Two claims are gated here:

1. ``vol_quote_vegas`` agrees with a **full-rebuild** central finite
   difference — recalibrate the smile with one quote bumped, reprice, difference.
2. ``forward_bar`` (the ``dR_v/dtheta_c`` contribution) is non-zero and agrees
   with the same full-rebuild finite difference in the forward. This is the
   term that is structurally zero when a cube is handed a precomputed forward,
   so it is the whole reason the curve-aware calibration route exists.

Tolerances are stated on each test. The reference is a rebuild through the
production ``optimistix`` Levenberg-Marquardt fit (``rtol=atol=1e-8``), so the
finite differences carry solver noise of order ``1e-8 / h``; the bump sizes
below are chosen to keep that below the asserted tolerance.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from valax.calibration.sabr import calibrate_sabr
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.dates.schedule import generate_schedule
from valax.greeks.adjoint import level_adjoint
from valax.greeks.vol_adjoint import (
    SABR_FIXED_BETA_PARAMS,
    SABR_FREE_BETA_PARAMS,
    SABRSliceInputs,
    cube_node_cotangent,
    cube_node_level,
    optionlet_slice_level,
    sabr_model_from_theta,
    sabr_slice_level,
    sabr_theta,
    sabr_theta_names,
    vol_quote_vegas,
)
from valax.instruments.rates import Swaption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.surfaces import (
    OptionletVolSurface,
    SwaptionCube,
    calibrate_swaption_cube_from_curves,
    swaption_node_coordinates,
    swaption_node_forwards_and_annuities,
)


REF = ymd_to_ordinal(2025, 1, 1)
BETA = 0.5

#: Offsets to the forward defining the quoted strike grid.
OFFSETS = jnp.array([-0.010, -0.005, 0.0, 0.005, 0.010])

#: Bump sizes for the full-rebuild finite differences, in vol units.
VOL_BUMPS = (1e-4, 1e-3)

#: Bump sizes for the forward finite difference, in rate units (0.1 bp, 1 bp).
FWD_BUMPS = (1e-5, 1e-4)

#: Agreement tolerance against the full-rebuild reference, relative to the
#: largest component of the finite-difference vector.
FD_RTOL = 1e-4


def _curve(zero=0.03):
    pillars = jnp.array([
        int(ymd_to_ordinal(y, 1, 1))
        for y in (2025, 2026, 2027, 2028, 2030, 2032, 2035, 2040, 2045)
    ], dtype=jnp.int32)
    times = (pillars - int(REF)).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-zero * times),
        reference_date=REF,
    )


def _nodes():
    """A 2x2 (expiry x tenor) grid of annual-pay swaptions."""
    grid = []
    for e in (1, 2):
        row = []
        for n in (5, 10):
            y = 2025 + e
            row.append(Swaption(
                expiry_date=ymd_to_ordinal(y, 1, 1),
                fixed_dates=generate_schedule(
                    y, 1, 1, y + n, 1, 1, frequency=1,
                ).payment_dates,
                strike=jnp.array(0.03),
                notional=jnp.array(1_000_000.0),
                is_payer=True,
            ))
        grid.append(row)
    return grid


#: The true SABR model the synthetic quotes are generated from. Lognormal with
#: a 3% forward gives vols around 130%, which is arithmetically fine and keeps
#: the fit exact, so the first-order condition really is satisfied.
TRUTH = SABRModel(
    alpha=jnp.array(0.22), beta=jnp.array(BETA),
    rho=jnp.array(-0.25), nu=jnp.array(0.40),
)


def _slice_quotes(forward, expiry):
    """Self-consistent (strikes, vols) for one smile slice."""
    strikes = forward + OFFSETS
    vols = jax.vmap(
        lambda k: sabr_implied_vol(TRUTH, forward, k, expiry)
    )(strikes)
    return strikes, vols


def _fit(strikes, vols, forward, expiry):
    """The production fit, beta held fixed."""
    model = calibrate_sabr(
        strikes=strikes, market_vols=vols, forward=forward, expiry=expiry,
        fixed_beta=jnp.array(BETA),
    ).value
    return model


def _assert_matches_fd(aad, fd, rtol=FD_RTOL):
    """Compare against a finite-difference vector, scaled by its own size."""
    aad = jnp.atleast_1d(jnp.asarray(aad))
    fd = jnp.atleast_1d(jnp.asarray(fd))
    scale = float(jnp.max(jnp.abs(fd)))
    assert scale > 1e-12, "the finite-difference reference is identically zero"
    err = float(jnp.max(jnp.abs(aad - fd))) / scale
    assert err < rtol, f"relative error {err:.3e} (aad={aad}, fd={fd})"


# ── Packing ───────────────────────────────────────────────────────────

class TestParameterPacking:
    def test_names_match_the_vector_length(self):
        assert sabr_theta_names(False) == SABR_FIXED_BETA_PARAMS
        assert sabr_theta_names(True) == SABR_FREE_BETA_PARAMS
        assert sabr_theta(TRUTH).shape == (3,)
        assert sabr_theta(TRUTH, free_beta=True).shape == (4,)

    @pytest.mark.parametrize("free_beta", [False, True])
    def test_pack_unpack_round_trip(self, free_beta):
        th = sabr_theta(TRUTH, free_beta=free_beta)
        back = sabr_model_from_theta(th, TRUTH.beta, free_beta=free_beta)
        for name in ("alpha", "beta", "rho", "nu"):
            assert float(getattr(back, name)) == float(getattr(TRUTH, name))


# ── The level is stationary at the fitted parameters ──────────────────

class TestLevelIsStationary:
    def test_residual_vanishes_at_the_fit(self):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        r = level.residual_fn(level.theta, level.inputs)
        assert r.shape == (3,)
        # The fit is exact, so the SSE gradient is at machine-noise level.
        assert float(jnp.max(jnp.abs(r))) < 1e-12

    def test_hessian_is_the_full_one_not_gauss_newton(self):
        """A deliberately imperfect fit: the two Hessians must differ."""
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        # Perturb the quotes away from any SABR smile so r != 0 at the optimum.
        vols = vols + jnp.array([0.05, -0.03, 0.04, -0.02, 0.06])
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)

        hess = jax.jacobian(
            lambda th: level.residual_fn(th, level.inputs)
        )(level.theta)

        def resid_vec(th):
            m = sabr_model_from_theta(th, model.beta)
            fitted = jax.vmap(
                lambda k: sabr_implied_vol(m, level.inputs.forward, k, expiry)
            )(strikes)
            return fitted - level.inputs.market_vols

        J = jax.jacobian(resid_vec)(level.theta)
        gauss_newton = J.T @ J
        assert float(jnp.max(jnp.abs(hess - gauss_newton))) > 1e-6
        # ...and the full Hessian is what the level reports.
        assert hess.shape == (3, 3)

    def test_atm_offset_quote_form_moves_the_strike_grid(self):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)

        abs_level = sabr_slice_level(
            model, strikes, vols, forward, expiry,
            quote_form="absolute_strike",
        )
        off_level = sabr_slice_level(
            model, OFFSETS, vols, forward, expiry, quote_form="atm_offset",
        )
        # At the base point the two agree (K = F0 + delta there)...
        assert float(jnp.max(jnp.abs(
            abs_level.residual_fn(abs_level.theta, abs_level.inputs)
            - off_level.residual_fn(off_level.theta, off_level.inputs)
        ))) < 1e-12
        # ...but they respond differently to a forward move.
        def r_of_f(level, f):
            return level.residual_fn(
                level.theta, SABRSliceInputs(level.inputs.market_vols, f)
            )
        bumped = jnp.array(0.031)
        assert float(jnp.max(jnp.abs(
            r_of_f(abs_level, bumped) - r_of_f(off_level, bumped)
        ))) > 1e-9

    def test_unknown_quote_form_is_rejected(self):
        with pytest.raises(ValueError, match="quote_form must be"):
            sabr_slice_level(
                TRUTH, OFFSETS, OFFSETS, jnp.array(0.03), jnp.array(1.0),
                quote_form="log_moneyness",
            )


# ── The headline gates ────────────────────────────────────────────────

@pytest.mark.audit
class TestVolQuoteVegasAgainstFiniteDifferences:
    """``vol_quote_vegas`` vs a full rebuild of the smile fit."""

    @pytest.mark.parametrize("h", VOL_BUMPS)
    def test_vol_sensitivity_of_a_queried_vol(self, h):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        query = strikes[3]

        def vol_at_query(quoted):
            m = _fit(strikes, quoted, forward, expiry)
            return sabr_implied_vol(m, forward, query, expiry)

        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        theta_bar = jax.grad(
            lambda th: sabr_implied_vol(
                sabr_model_from_theta(th, model.beta), forward, query, expiry
            )
        )(level.theta)

        aad = vol_quote_vegas(level, theta_bar).vol_quote_vegas

        fd = []
        for k in range(strikes.shape[0]):
            e = jnp.zeros_like(vols).at[k].set(h)
            fd.append(
                (float(vol_at_query(vols + e)) - float(vol_at_query(vols - e)))
                / (2.0 * h)
            )
        _assert_matches_fd(aad, jnp.asarray(fd))

    @pytest.mark.parametrize("h", VOL_BUMPS)
    def test_vol_sensitivity_of_a_swaption_price(self, h):
        """End to end: the vega a desk would hedge with, in quote space."""
        from valax.pricing.analytic.rates_smile import swaption_price

        curve = _curve()
        nodes = _nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        f00, t00 = forwards[0, 0], expiries[0]
        strikes, vols = _slice_quotes(f00, t00)

        trade = eqx.tree_at(
            lambda s: s.strike, nodes[0][0], jnp.asarray(strikes[3])
        )

        def price_from_quotes(quoted):
            m = _fit(strikes, quoted, f00, t00)
            cube = SwaptionCube(
                expiries=expiries[:1], tenors=jnp.asarray([5.0]),
                forwards=f00.reshape(1, 1),
                alphas=m.alpha.reshape(1, 1),
                betas=m.beta.reshape(1, 1),
                rhos=m.rho.reshape(1, 1),
                nus=m.nu.reshape(1, 1),
            )
            return swaption_price(trade, curve, cube)

        model = _fit(strikes, vols, f00, t00)
        level = sabr_slice_level(model, strikes, vols, f00, t00)

        def price_from_theta(th):
            m = sabr_model_from_theta(th, model.beta)
            cube = SwaptionCube(
                expiries=expiries[:1], tenors=jnp.asarray([5.0]),
                forwards=f00.reshape(1, 1),
                alphas=m.alpha.reshape(1, 1),
                betas=m.beta.reshape(1, 1),
                rhos=m.rho.reshape(1, 1),
                nus=m.nu.reshape(1, 1),
            )
            return swaption_price(trade, curve, cube)

        theta_bar = jax.grad(price_from_theta)(level.theta)
        aad = vol_quote_vegas(level, theta_bar).vol_quote_vegas

        fd = []
        for k in range(strikes.shape[0]):
            e = jnp.zeros_like(vols).at[k].set(h)
            fd.append(
                (float(price_from_quotes(vols + e))
                 - float(price_from_quotes(vols - e))) / (2.0 * h)
            )
        _assert_matches_fd(aad, jnp.asarray(fd))

    def test_vegas_are_not_all_zero(self):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        out = vol_quote_vegas(level, jnp.array([1.0, 0.0, 0.0]))
        assert float(jnp.max(jnp.abs(out.vol_quote_vegas))) > 1e-6
        assert float(out.stationarity_norm) < 1e-12


@pytest.mark.audit
class TestCurveContributionThroughTheFit:
    r"""``forward_bar`` is the ``dR_v/dtheta_c`` coupling (spec A-3).

    Gated against a **full rebuild** of the smile fit under both quote forms:
    recalibrate with the forward bumped, reprice, difference. Under
    ``atm_offset`` the strike grid moves with the forward
    (:math:`K_k = F_0 + \delta_k`); under ``absolute_strike`` it does not, and
    the residual rebuilds the grid accordingly, so both paths are exercised.

    .. note::

        An earlier version of this suite gated the ``absolute_strike`` case
        against ``jax.grad`` through :func:`calibrate_sabr` -- optimistix's own
        ``ImplicitAdjoint`` -- on the grounds that the finite difference was
        unreliable because the least-squares Hessian is ~1e6-conditioned here.
        That was wrong twice over. The finite difference was *correct*, and the
        two "independent" adjoints agreed only because they **share the same
        primitive**: ``sabr_implied_vol`` pinned Hagan's ``z / x(z)`` to a
        constant at the money, which has the right value and zero derivative,
        so both routes inherited a corrupted
        :math:`\partial\sigma/\partial F`. Two implementations agreeing is
        worth nothing when they share the thing that is broken -- the lesson
        already on record as entry 4 of ``docs/architecture/numerical-pitfalls.md``.

        The primitive is fixed (regression:
        ``tests/test_models/test_sabr.py::TestATMDerivatives``), the finite
        difference now agrees with the adjoint to ~1e-6, and the conditioning
        was a red herring: it is still ~1e6 and the FD is still fine. The
        optimistix comparison is retained below, honestly labelled as a
        consistency check rather than an independent one.
    """

    @pytest.mark.parametrize("h", FWD_BUMPS)
    @pytest.mark.parametrize("quote_form", ["absolute_strike", "atm_offset"])
    def test_forward_sensitivity_matches_a_full_rebuild(self, h, quote_form):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        query = strikes[1]

        def vol_at_query(f):
            k = f + OFFSETS if quote_form == "atm_offset" else strikes
            m = _fit(k, vols, f, expiry)
            return sabr_implied_vol(m, forward, query, expiry)

        model = _fit(strikes, vols, forward, expiry)
        grid = OFFSETS if quote_form == "atm_offset" else strikes
        level = sabr_slice_level(
            model, grid, vols, forward, expiry, quote_form=quote_form
        )
        theta_bar = jax.grad(
            lambda th: sabr_implied_vol(
                sabr_model_from_theta(th, model.beta), forward, query, expiry
            )
        )(level.theta)

        aad = vol_quote_vegas(level, theta_bar).forward_bar
        fd = (float(vol_at_query(forward + h))
              - float(vol_at_query(forward - h))) / (2.0 * h)
        _assert_matches_fd(aad, jnp.asarray([fd]))

    @pytest.mark.parametrize("quote_form", ["absolute_strike", "atm_offset"])
    def test_forward_jacobian_agrees_with_optimistix_implicit_adjoint(
        self, quote_form
    ):
        """Consistency check -- *not* independent; see the class note."""
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        grid = OFFSETS if quote_form == "atm_offset" else strikes
        level = sabr_slice_level(
            model, grid, vols, forward, expiry, quote_form=quote_form
        )

        def theta_of_forward(f):
            k = f + OFFSETS if quote_form == "atm_offset" else strikes
            m = _fit(k, vols, f, expiry)
            return jnp.stack([m.alpha, m.rho, m.nu])

        reference = jax.jacobian(theta_of_forward)(forward)
        ours = jnp.stack([
            vol_quote_vegas(level, jnp.zeros(3).at[i].set(1.0)).forward_bar
            for i in range(3)
        ])
        _assert_matches_fd(ours, reference, rtol=1e-9)

    def test_vol_quote_jacobian_agrees_with_optimistix_implicit_adjoint(self):
        """The same consistency check on the quote block of the inputs."""
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)

        reference = jax.jacobian(
            lambda v: jnp.stack(
                [_fit(strikes, v, forward, expiry).alpha,
                 _fit(strikes, v, forward, expiry).rho,
                 _fit(strikes, v, forward, expiry).nu]
            )
        )(vols)
        ours = jnp.stack([
            vol_quote_vegas(level, jnp.zeros(3).at[i].set(1.0)).vol_quote_vegas
            for i in range(3)
        ])
        _assert_matches_fd(ours, reference, rtol=1e-9)

    def test_ill_conditioning_does_not_invalidate_the_finite_difference(self):
        """The conditioning is real; it was never the reason for a mismatch.

        Recorded so the earlier misdiagnosis is not re-derived: the smile
        Hessian is ~1e6-conditioned *and* the full-rebuild FD above agrees to
        ~1e-6. High condition number is a reason to be careful, not a licence
        to dismiss a disagreeing reference.
        """
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        hess = jax.jacobian(
            lambda th: level.residual_fn(th, level.inputs)
        )(level.theta)
        assert float(jnp.linalg.cond(hess)) > 1e5

    def test_forward_contribution_is_non_zero(self):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        out = vol_quote_vegas(level, jnp.array([1.0, 0.0, 0.0]))
        assert abs(float(out.forward_bar)) > 1e-6

    def test_the_coupling_is_absent_from_a_frozen_forward_fit(self):
        """Why the curve-aware calibration route had to exist.

        A fit whose forward is a constant cannot produce this term: the
        cotangent w.r.t. the forward is identically zero because the forward is
        not an input to the residual at all. Simulated here by differentiating
        a level whose ``inputs.forward`` is detached from the curve.
        """
        curve = _curve()
        nodes = _nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        f00, t00 = forwards[0, 0], expiries[0]
        strikes, vols = _slice_quotes(f00, t00)

        def cube_alpha(dfs, curve_aware):
            c = eqx.tree_at(lambda x: x.discount_factors, curve, dfs)
            if curve_aware:
                cube = calibrate_swaption_cube_from_curves(
                    nodes,
                    [[strikes, strikes], [strikes, strikes]],
                    [[vols, vols], [vols, vols]],
                    c, fixed_beta=jnp.array(BETA),
                ).value
            else:
                # What the retired frozen-forward cube did: the forward is
                # computed off the curve but enters the fit as a constant.
                f, _ = swaption_node_forwards_and_annuities(nodes, c)
                return calibrate_sabr(
                    strikes, vols, jax.lax.stop_gradient(f[0, 0]), t00,
                    fixed_beta=jnp.array(BETA),
                ).value.alpha
            return cube.alphas[0, 0]

        aware = jax.grad(lambda d: cube_alpha(d, True))(curve.discount_factors)
        frozen = jax.grad(lambda d: cube_alpha(d, False))(curve.discount_factors)
        assert float(jnp.max(jnp.abs(aware))) > 1e-6
        assert float(jnp.max(jnp.abs(frozen))) == 0.0


# ── Cube / surface wiring ─────────────────────────────────────────────

class TestCubeWiring:
    def test_cube_node_level_reads_the_cubes_own_conventions(self):
        curve = _curve()
        nodes = _nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        quotes = [
            [_slice_quotes(forwards[i, j], expiries[i]) for j in range(2)]
            for i in range(2)
        ]
        cube = calibrate_swaption_cube_from_curves(
            nodes,
            [[OFFSETS, OFFSETS], [OFFSETS, OFFSETS]],
            [[quotes[i][j][1] for j in range(2)] for i in range(2)],
            curve, quote_form="atm_offset", fixed_beta=jnp.array(BETA),
        ).value
        assert cube.quote_form == "atm_offset"

        level = cube_node_level(cube, 1, 0, OFFSETS, quotes[1][0][1])
        assert float(level.inputs.forward) == float(cube.forwards[1, 0])
        assert float(level.theta[0]) == float(cube.alphas[1, 0])
        # The fit reached a stationary point, so the IFT applies.
        r = level.residual_fn(level.theta, level.inputs)
        assert float(jnp.max(jnp.abs(r))) < 1e-10

    def test_cube_node_cotangent_picks_the_right_entries(self):
        curve = _curve()
        nodes = _nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        vols = [
            [_slice_quotes(forwards[i, j], expiries[i])[1] for j in range(2)]
            for i in range(2)
        ]
        cube = calibrate_swaption_cube_from_curves(
            nodes,
            [[OFFSETS, OFFSETS], [OFFSETS, OFFSETS]],
            vols, curve, quote_form="atm_offset", fixed_beta=jnp.array(BETA),
        ).value

        cube_bar = jax.grad(
            lambda c: c(
                jnp.array(0.03), c.expiries[1], c.tenors[0],
                forward=c.forwards[1, 0],
            )
        )(cube)
        theta_bar = cube_node_cotangent(cube_bar, 1, 0)
        assert theta_bar.shape == (3,)
        assert float(theta_bar[0]) == float(cube_bar.alphas[1, 0])
        assert float(theta_bar[1]) == float(cube_bar.rhos[1, 0])
        assert float(theta_bar[2]) == float(cube_bar.nus[1, 0])
        # Querying node (1, 0) must not touch any other node.
        assert float(cube_bar.alphas[0, 0]) == 0.0

    def test_optionlet_slice_level_mirrors_the_cube_one(self):
        expiry = jnp.array(1.0)
        forward = jnp.array(0.03)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        surf = OptionletVolSurface(
            expiries=jnp.array([0.5, 1.0, 2.0]),
            forwards=jnp.full(3, float(forward)),
            alphas=jnp.full(3, float(model.alpha)),
            betas=jnp.full(3, BETA),
            rhos=jnp.full(3, float(model.rho)),
            nus=jnp.full(3, float(model.nu)),
        )
        level = optionlet_slice_level(surf, 1, strikes, vols)
        assert float(level.inputs.forward) == float(forward)
        assert float(jnp.max(jnp.abs(
            level.residual_fn(level.theta, level.inputs)
        ))) < 1e-10


class TestJaxContract:
    def test_vol_quote_vegas_is_jittable(self):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        out = eqx.filter_jit(vol_quote_vegas)(level, jnp.array([1.0, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(out.vol_quote_vegas))
        assert jnp.isfinite(out.forward_bar)

    def test_generic_level_adjoint_agrees_with_the_named_wrapper(self):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(model, strikes, vols, forward, expiry)
        theta_bar = jnp.array([1.0, -0.5, 0.25])

        named = vol_quote_vegas(level, theta_bar)
        generic = level_adjoint(level, theta_bar)
        assert float(jnp.max(jnp.abs(
            named.vol_quote_vegas - generic.inputs_bar.market_vols
        ))) == 0.0
        assert float(named.forward_bar) == float(generic.inputs_bar.forward)

    @pytest.mark.parametrize("free_beta", [False, True])
    def test_free_beta_level_is_square(self, free_beta):
        forward, expiry = jnp.array(0.03), jnp.array(1.0)
        strikes, vols = _slice_quotes(forward, expiry)
        model = _fit(strikes, vols, forward, expiry)
        level = sabr_slice_level(
            model, strikes, vols, forward, expiry, free_beta=free_beta
        )
        n = 4 if free_beta else 3
        assert level.theta.shape == (n,)
        assert level.residual_fn(level.theta, level.inputs).shape == (n,)
