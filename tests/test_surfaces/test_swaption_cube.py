"""Tests for the SABR SwaptionCube (expiry x tenor x strike).

Covers parameter interpolation, lognormal and normal calibration round-trips,
the normal-vs-lognormal quoting flag, negative-strike capability, and a jit
smoke test.
"""

import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph
from valax.dates.daycounts import ymd_to_ordinal, year_fraction
from valax.dates.schedule import generate_schedule
from valax.instruments.rates import Swaption
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol
from valax.pricing.analytic.swaptions import swaption_forward_and_annuity
from valax.surfaces import (
    SwaptionCube,
    calibrate_swaption_cube,
    calibrate_swaption_cube_from_curves,
    calibrate_swaption_cube_from_graph,
    swaption_node_coordinates,
    swaption_node_forwards_and_annuities,
)


EXPIRIES = jnp.array([1.0, 2.0, 5.0])
TENORS = jnp.array([2.0, 10.0])


def _const_grids(alpha_grid, nu_grid, forward=0.03, beta=0.5, rho=-0.3):
    """Build a SwaptionCube with given alpha/nu grids and constant beta/rho/fwd."""
    shape = alpha_grid.shape
    return SwaptionCube(
        expiries=EXPIRIES,
        tenors=TENORS,
        forwards=jnp.full(shape, forward),
        alphas=alpha_grid,
        betas=jnp.full(shape, beta),
        rhos=jnp.full(shape, rho),
        nus=nu_grid,
    )


class TestInterpolation:
    def test_node_values_exact(self):
        alphas = jnp.array([[0.30, 0.31], [0.32, 0.33], [0.35, 0.36]])
        nus = jnp.array([[0.40, 0.41], [0.42, 0.43], [0.45, 0.46]])
        cube = _const_grids(alphas, nus)
        # At an exact grid node the interpolation returns the node parameters.
        m = cube.model_at(EXPIRIES[1], TENORS[1])
        assert abs(float(m.alpha) - 0.33) < 1e-12
        assert abs(float(m.nu) - 0.43) < 1e-12
        assert abs(float(cube.forward_at(EXPIRIES[2], TENORS[0])) - 0.03) < 1e-12

    def test_midpoint_is_bilinear_average(self):
        alphas = jnp.array([[0.30, 0.40], [0.50, 0.60], [0.90, 1.00]])
        nus = jnp.full((3, 2), 0.4)
        cube = _const_grids(alphas, nus)
        # Halfway in expiry (1 <-> 2 => 1.5) and tenor (2 <-> 10 => 6.0):
        # average of the four surrounding nodes 0.30,0.40,0.50,0.60 = 0.45.
        m = cube.model_at(jnp.array(1.5), jnp.array(6.0))
        assert abs(float(m.alpha) - 0.45) < 1e-12

    def test_flat_extrapolation(self):
        alphas = jnp.array([[0.30, 0.31], [0.32, 0.33], [0.35, 0.36]])
        cube = _const_grids(alphas, jnp.full((3, 2), 0.4))
        # Query below the grid clamps to the first node.
        m = cube.model_at(jnp.array(0.1), jnp.array(0.5))
        assert abs(float(m.alpha) - 0.30) < 1e-12


class TestLognormalCalibration:
    def test_node_round_trip(self):
        """Fit per-node SABR smiles generated from known models; recover vols."""
        forward = 100.0
        strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])

        def truth(i, j):
            return SABRModel(
                alpha=jnp.array(0.28 + 0.01 * i + 0.02 * j),
                beta=jnp.array(0.5),
                rho=jnp.array(-0.3),
                nu=jnp.array(0.35 + 0.05 * j),
            )

        strikes_per_node = [[strikes for _ in range(2)] for _ in range(3)]
        market = [
            [
                jax.vmap(lambda K: sabr_implied_vol(truth(i, j), jnp.array(forward), K, EXPIRIES[i]))(strikes)
                for j in range(2)
            ]
            for i in range(3)
        ]
        forwards = jnp.full((3, 2), forward)

        cube = calibrate_swaption_cube(
            strikes_per_node, market, forwards, EXPIRIES, TENORS,
            fixed_beta=jnp.array(0.5),
        )

        max_err = 0.0
        for i in range(3):
            for j in range(2):
                for k in range(strikes.shape[0]):
                    got = cube(strikes[k], EXPIRIES[i], TENORS[j])
                    max_err = max(max_err, abs(float(got - market[i][j][k])))
        assert max_err < 1e-3, f"max node vol error {max_err}"


class TestNormalCalibration:
    def test_node_round_trip(self):
        """Fit normal-quoted (beta=0, shifted) smiles and recover them.

        Guards against the historical vol_fn plumbing bug where normal quotes
        were silently fit with the lognormal formula (see
        examples/pitfalls/01_sabr_vol_convention_mismatch.py).
        """
        forward = 0.025
        shift = 0.03
        strikes = jnp.array([0.015, 0.02, 0.025, 0.03, 0.035])
        vol_fn = functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(shift))

        def truth(i, j):
            return SABRModel(
                alpha=jnp.array(0.009 + 0.001 * i + 0.001 * j),
                beta=jnp.array(0.0),
                rho=jnp.array(-0.2),
                nu=jnp.array(0.3 + 0.05 * j),
            )

        strikes_per_node = [[strikes for _ in range(2)] for _ in range(3)]
        market = [
            [
                jax.vmap(lambda K: vol_fn(truth(i, j), jnp.array(forward), K, EXPIRIES[i]))(strikes)
                for j in range(2)
            ]
            for i in range(3)
        ]
        forwards = jnp.full((3, 2), forward)

        cube = calibrate_swaption_cube(
            strikes_per_node, market, forwards, EXPIRIES, TENORS,
            is_normal=True, shift=shift, fixed_beta=jnp.array(0.0),
        )
        assert cube.is_normal is True

        max_err = 0.0
        for i in range(3):
            for j in range(2):
                for k in range(strikes.shape[0]):
                    got = cube(strikes[k], EXPIRIES[i], TENORS[j])
                    max_err = max(max_err, abs(float(got - market[i][j][k])))
        assert max_err < 1e-6, f"max node vol error {max_err}"


class TestQuotingConvention:
    def test_lognormal_cube_nan_at_negative_strike(self):
        cube = _const_grids(jnp.full((3, 2), 0.3), jnp.full((3, 2), 0.4))
        assert cube.is_normal is False
        vol = cube(jnp.array(-5.0), EXPIRIES[1], TENORS[0])
        assert jnp.isnan(vol)

    def test_normal_cube_matches_direct_hagan_at_node(self):
        model = SABRModel(alpha=jnp.array(0.01), beta=jnp.array(0.0),
                          rho=jnp.array(-0.2), nu=jnp.array(0.3))
        cube = SwaptionCube(
            expiries=EXPIRIES, tenors=TENORS,
            forwards=jnp.full((3, 2), 0.025),
            alphas=jnp.full((3, 2), 0.01),
            betas=jnp.full((3, 2), 0.0),
            rhos=jnp.full((3, 2), -0.2),
            nus=jnp.full((3, 2), 0.3),
            shift=jnp.array(0.03),
            is_normal=True,
        )
        K = jnp.array(0.02)
        got = cube(K, EXPIRIES[1], TENORS[0])
        want = sabr_normal_implied_vol(model, jnp.array(0.025), K, EXPIRIES[1], jnp.array(0.03))
        assert abs(float(got - want)) < 1e-12

    def test_normal_cube_finite_at_negative_strike(self):
        cube = SwaptionCube(
            expiries=EXPIRIES, tenors=TENORS,
            forwards=jnp.full((3, 2), 0.01),
            alphas=jnp.full((3, 2), 0.01),
            betas=jnp.full((3, 2), 0.0),
            rhos=jnp.full((3, 2), -0.2),
            nus=jnp.full((3, 2), 0.3),
            shift=jnp.array(0.03),
            is_normal=True,
        )
        vol = cube(jnp.array(-0.005), EXPIRIES[1], TENORS[1])
        assert jnp.isfinite(vol)
        assert float(vol) > 0.0


class TestJit:
    def test_filter_jit_smoke(self):
        cube = _const_grids(jnp.full((3, 2), 0.3), jnp.full((3, 2), 0.4))
        vol = eqx.filter_jit(cube.__call__)(jnp.array(100.0), EXPIRIES[1], TENORS[0])
        assert jnp.isfinite(vol)


# ── Curve-aware calibration ───────────────────────────────────────────

class TestCurveAwareCalibration:
    """The forwards come from the curves, so curve risk reaches the vols.

    T-1 gate for the cube layer: every fitted quote must be reproduced by the
    calibrated object at its own node coordinate. Tolerance: 1e-8 absolute in
    vol units on self-consistent synthetic data (the SABR fit converges to
    machine precision there; the slack is for the solver's 1e-8 rtol/atol).
    """

    #: T-1 repricing tolerance, absolute, in the quotes' vol units.
    T1_ATOL = 1e-8

    @staticmethod
    def _curve(ref_date, zero=0.03):
        pillars = jnp.array([
            int(ymd_to_ordinal(y, 1, 1))
            for y in (2025, 2026, 2027, 2028, 2030, 2032, 2035, 2040, 2045)
        ], dtype=jnp.int32)
        times = (pillars - int(ref_date)).astype(jnp.float64) / 365.0
        return DiscountCurve(
            pillar_dates=pillars,
            discount_factors=jnp.exp(-zero * times),
            reference_date=ref_date,
        )

    @staticmethod
    def _nodes(expiry_years=(1, 2), tenor_years=(5, 10)):
        """A rectangular (expiry x tenor) grid of annual-pay swaptions."""
        grid = []
        for e in expiry_years:
            row = []
            for n in tenor_years:
                start_y = 2025 + e
                row.append(Swaption(
                    expiry_date=ymd_to_ordinal(start_y, 1, 1),
                    fixed_dates=generate_schedule(
                        start_y, 1, 1, start_y + n, 1, 1, frequency=1,
                    ).payment_dates,
                    strike=jnp.array(0.03),
                    notional=jnp.array(1_000_000.0),
                    is_payer=True,
                ))
            grid.append(row)
        return grid

    @staticmethod
    def _synthetic_quotes(forwards, expiries, offsets):
        """Self-consistent lognormal SABR vols at ``F0 + offsets`` per node."""
        truth = SABRModel(
            alpha=jnp.array(0.22), beta=jnp.array(0.5),
            rho=jnp.array(-0.25), nu=jnp.array(0.40),
        )
        strikes, vols = [], []
        for i in range(forwards.shape[0]):
            k_row, v_row = [], []
            for j in range(forwards.shape[1]):
                k = forwards[i, j] + offsets
                k_row.append(k)
                v_row.append(jax.vmap(
                    lambda kk: sabr_implied_vol(
                        truth, forwards[i, j], kk, expiries[i]
                    )
                )(k))
            strikes.append(k_row)
            vols.append(v_row)
        return strikes, vols

    def test_axes_match_the_pricers_own_coordinate(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        expiries, tenors = swaption_node_coordinates(nodes, curve)

        # Exactly what rates_smile.swaption_price computes for node (0, 0).
        s = nodes[0][0]
        assert float(expiries[0]) == float(
            year_fraction(curve.reference_date, s.expiry_date, s.day_count)
        )
        assert float(tenors[0]) == float(
            year_fraction(s.expiry_date, s.fixed_dates[-1], s.day_count)
        )
        assert expiries.shape == (2,)
        assert tenors.shape == (2,)

    def test_forwards_match_the_pricers_own_forward(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        forwards, annuities = swaption_node_forwards_and_annuities(nodes, curve)
        assert forwards.shape == (2, 2)
        assert annuities.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                S, A = swaption_forward_and_annuity(nodes[i][j], curve)
                assert float(forwards[i, j]) == float(S)
                assert float(annuities[i, j]) == float(A)
        assert float(jnp.min(annuities)) > 0.0

    @pytest.mark.audit
    @pytest.mark.parametrize("quote_form", ["absolute_strike", "atm_offset"])
    def test_t1_every_fitted_quote_reprices(self, quote_form):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        offsets = jnp.array([-0.01, -0.005, 0.0, 0.005, 0.01])
        abs_strikes, vols = self._synthetic_quotes(forwards, expiries, offsets)

        if quote_form == "atm_offset":
            quoted = [[offsets for _ in row] for row in abs_strikes]
        else:
            quoted = abs_strikes

        cube = calibrate_swaption_cube_from_curves(
            nodes, quoted, vols, curve,
            quote_form=quote_form, fixed_beta=jnp.array(0.5),
        )

        # The cube recorded the curve-derived anchors verbatim.
        assert float(jnp.max(jnp.abs(cube.forwards - forwards))) == 0.0

        for i in range(len(nodes)):
            for j in range(len(nodes[0])):
                got = jax.vmap(lambda k: cube(
                    k, cube.expiries[i], cube.tenors[j],
                    forward=cube.forwards[i, j],
                ))(abs_strikes[i][j])
                err = float(jnp.max(jnp.abs(got - vols[i][j])))
                assert err < self.T1_ATOL, (i, j, err)

    def test_atm_offset_quotes_imply_sticky_moneyness(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        offsets = jnp.array([-0.01, 0.0, 0.01])
        _, vols = self._synthetic_quotes(forwards, expiries, offsets)
        quoted = [[offsets, offsets], [offsets, offsets]]

        cube = calibrate_swaption_cube_from_curves(
            nodes, quoted, vols, curve,
            quote_form="atm_offset", fixed_beta=jnp.array(0.5),
        )
        assert cube.quote_form == "atm_offset"
        assert cube.dynamics == "sticky_moneyness"

        # Absolute-strike quotes imply the other one, and the default is never
        # the backbone (which has an unverified prerequisite).
        abs_cube = calibrate_swaption_cube_from_curves(
            nodes,
            [[forwards[i, j] + offsets for j in range(2)] for i in range(2)],
            vols, curve, fixed_beta=jnp.array(0.5),
        )
        assert abs_cube.dynamics == "sticky_strike"

    def test_backbone_must_be_asked_for_explicitly(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        offsets = jnp.array([-0.01, 0.0, 0.01])
        abs_strikes, vols = self._synthetic_quotes(forwards, expiries, offsets)

        cube = calibrate_swaption_cube_from_curves(
            nodes, abs_strikes, vols, curve,
            dynamics="sabr_backbone", fixed_beta=jnp.array(0.5),
        )
        assert cube.dynamics == "sabr_backbone"
        # Self-consistency: at F == F0 the backbone reproduces the quote, which
        # is exactly what the curve-aware fit buys.
        got = cube(
            abs_strikes[0][0][1], cube.expiries[0], cube.tenors[0],
            forward=cube.forwards[0, 0],
        )
        assert abs(float(got) - float(vols[0][0][1])) < self.T1_ATOL

    def test_inconsistent_dynamics_is_rejected(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        offsets = jnp.array([-0.01, 0.0, 0.01])
        quoted = [[offsets, offsets], [offsets, offsets]]
        with pytest.raises(ValueError, match="contradict each other"):
            calibrate_swaption_cube_from_curves(
                nodes, quoted, quoted, curve,
                quote_form="atm_offset", dynamics="sticky_strike",
            )

    def test_cube_vol_has_curve_sensitivity(self):
        """The point of the exercise: d(fitted vol) / d(curve DF) != 0."""
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes(expiry_years=(1,), tenor_years=(5,))
        expiries, _ = swaption_node_coordinates(nodes, curve)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, curve)
        offsets = jnp.array([-0.01, -0.005, 0.0, 0.005, 0.01])
        abs_strikes, vols = self._synthetic_quotes(forwards, expiries, offsets)
        query = abs_strikes[0][0][3]

        def fitted_vol(dfs):
            c = eqx.tree_at(lambda x: x.discount_factors, curve, dfs)
            cube = calibrate_swaption_cube_from_curves(
                nodes, abs_strikes, vols, c, fixed_beta=jnp.array(0.5),
                dynamics="sabr_backbone",
            )
            return cube(
                query, cube.expiries[0], cube.tenors[0],
                forward=cube.forwards[0, 0],
            )

        g = jax.grad(fitted_vol)(curve.discount_factors)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 1e-6

    def test_dual_curve_basis_moves_the_fit(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        disc = self._curve(ref, zero=0.03)
        proj = self._curve(ref, zero=0.035)
        nodes = self._nodes()

        f_single, _ = swaption_node_forwards_and_annuities(nodes, disc)
        f_dual, _ = swaption_node_forwards_and_annuities(nodes, disc, proj)
        # A 50 bp tenor basis must move the at-the-money anchors.
        assert float(jnp.min(jnp.abs(f_dual - f_single))) > 1e-4

        expiries, _ = swaption_node_coordinates(nodes, disc)
        offsets = jnp.array([-0.01, 0.0, 0.01])
        abs_strikes, vols = self._synthetic_quotes(f_dual, expiries, offsets)
        cube = calibrate_swaption_cube_from_curves(
            nodes, abs_strikes, vols, disc, proj, fixed_beta=jnp.array(0.5),
        )
        assert float(jnp.max(jnp.abs(cube.forwards - f_dual))) == 0.0

    def test_graph_entry_point_agrees_with_the_curve_one(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        disc = self._curve(ref, zero=0.03)
        graph = CurveGraph(curves={"USD.SOFR.OIS": disc})
        nodes = self._nodes()
        expiries, _ = swaption_node_coordinates(nodes, disc)
        forwards, _ = swaption_node_forwards_and_annuities(nodes, disc)
        offsets = jnp.array([-0.01, 0.0, 0.01])
        abs_strikes, vols = self._synthetic_quotes(forwards, expiries, offsets)

        a = calibrate_swaption_cube_from_curves(
            nodes, abs_strikes, vols, disc, fixed_beta=jnp.array(0.5)
        )
        b = calibrate_swaption_cube_from_graph(
            nodes, abs_strikes, vols, graph, "USD.SOFR.OIS",
            fixed_beta=jnp.array(0.5),
        )
        assert float(jnp.max(jnp.abs(a.alphas - b.alphas))) == 0.0
        assert float(jnp.max(jnp.abs(a.forwards - b.forwards))) == 0.0

    def test_ragged_grid_is_rejected(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        ragged = [nodes[0], nodes[1][:1]]
        with pytest.raises(ValueError, match="not rectangular"):
            swaption_node_coordinates(ragged, curve)

    def test_mixed_expiry_row_is_rejected(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        # Put a 2y-expiry node in the 1y-expiry row.
        broken = [[nodes[0][0], nodes[1][1]], nodes[1]]
        with pytest.raises(ValueError, match="spans several expiries"):
            swaption_node_coordinates(broken, curve)

    def test_mixed_tenor_column_is_rejected(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        # Row 1 has its columns swapped, so column 0 mixes 5y and 10y.
        broken = [nodes[0], [nodes[1][1], nodes[1][0]]]
        with pytest.raises(ValueError, match="Columns must share one"):
            swaption_node_coordinates(broken, curve)

    def test_quote_grid_shape_is_checked(self):
        ref = ymd_to_ordinal(2025, 1, 1)
        curve = self._curve(ref)
        nodes = self._nodes()
        k = jnp.array([0.02, 0.03, 0.04])
        with pytest.raises(ValueError, match="must match the swaption node"):
            calibrate_swaption_cube_from_curves(
                nodes, [[k, k]], [[k, k]], curve
            )


class TestSingletonAxes:
    """A one-node cube is flat, not NaN.

    ``bilinear_2d`` used to divide by ``grid[0] - grid[0]`` on a length-1 axis,
    so every query against a single-expiry or single-tenor cube returned NaN.
    Calibrating one node is a legitimate thing to do (it is the smallest
    curve-aware fit), so the degenerate axis is now flat along that direction.
    """

    @staticmethod
    def _cube(shape):
        n_e, n_t = shape
        return SwaptionCube(
            expiries=jnp.linspace(1.0, 5.0, n_e),
            tenors=jnp.linspace(2.0, 10.0, n_t),
            forwards=jnp.full(shape, 0.03),
            alphas=jnp.full(shape, 0.22),
            betas=jnp.full(shape, 0.5),
            rhos=jnp.full(shape, -0.25),
            nus=jnp.full(shape, 0.4),
        )

    @pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1)])
    def test_query_is_finite_on_and_off_node(self, shape):
        cube = self._cube(shape)
        on_node = cube(jnp.array(0.03), cube.expiries[0], cube.tenors[0])
        off_node = cube(jnp.array(0.035), jnp.array(2.5), jnp.array(7.0))
        assert jnp.isfinite(on_node) and float(on_node) > 0.0
        assert jnp.isfinite(off_node) and float(off_node) > 0.0

    @pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1)])
    def test_degenerate_axis_is_flat(self, shape):
        cube = self._cube(shape)
        # Moving along a length-1 axis cannot change the answer.
        a = cube(jnp.array(0.03), jnp.array(1.0), jnp.array(2.0))
        b = cube(
            jnp.array(0.03),
            jnp.array(1.0) if shape[0] == 1 else jnp.array(1.0),
            jnp.array(50.0) if shape[1] == 1 else jnp.array(2.0),
        )
        if shape[1] == 1:
            assert float(a) == float(b)

    def test_gradient_through_a_singleton_cube_is_finite(self):
        cube = self._cube((1, 1))

        def vol(alphas):
            c = eqx.tree_at(lambda x: x.alphas, cube, alphas)
            return c(jnp.array(0.035), jnp.array(1.0), jnp.array(5.0))

        g = jax.grad(vol)(cube.alphas)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.abs(g[0, 0])) > 0.0
