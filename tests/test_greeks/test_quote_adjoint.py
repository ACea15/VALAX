"""Conformance gates for quote-space (hedge-space) AAD risk.

Verifies :mod:`valax.greeks.adjoint` against the multi-curve calibration /
AAD risk audit specification.  The ``audit`` marker collects these with the
rest of the T-suite (``pytest -m audit``).

Gates implemented here, with the tolerances this implementation commits to:

* **T-2** — AAD quote sensitivities vs **full-rebuild central finite
  differences**, every quote, at two bump sizes (0.1 bp and 1 bp).
  Tolerance: ``rtol = 1e-5`` on entries whose FD magnitude exceeds 1.0,
  ``atol = 1e-4`` elsewhere (near-zero entries are FD-noise-dominated — the
  adjoint returns a much cleaner zero than the finite difference does).
* **T-3** — parallel-shift identity: the sum of quote deltas equals a rebuild
  with every quote bumped together.  Tolerance ``rtol = 1e-6``.
* **T-15** — the discount-quote delta computed with the other *quotes* held
  fixed differs from one computed with projection *nodes* held fixed, and the
  API reports the former.

Plus unit coverage of the level-generic machinery: that ``lam`` really solves
the transposed system, that the generic and curve-specialised paths agree, and
that a reused factorisation is numerically identical to a fresh one (A-7).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from valax.core.market_state import MarketState
from valax.core.pricing import price
from valax.curves import (
    CurveSpec,
    DepositRate,
    IborSwapRate,
    OISSwapRate,
    bootstrap_curve_graph,
)
from valax.dates.daycounts import ymd_to_ordinal
from valax.greeks.adjoint import (
    curve_adjoint,
    curve_log_df_cotangent,
    curve_quote_deltas,
    level_adjoint,
    quote_deltas_from_cotangent,
)
from valax.greeks.market_greeks import curve_pillar_deltas, market_gradient
from valax.instruments.bonds import FixedRateBond
from valax.instruments.rates import InterestRateSwap


REF = ymd_to_ordinal(2025, 1, 1)
OIS = "USD.SOFR.OIS"
P3M = "USD.TERM.3M"

# Bump sizes the T-2 gate is validated at: 0.1 bp and 1 bp.
BUMPS = (1e-5, 1e-4)


def _d(y, m, d):
    return int(ymd_to_ordinal(y, m, d))


# ── Single-curve fixture ─────────────────────────────────────────────

SC_PILLARS = jnp.array(
    [_d(2026, 1, 1), _d(2027, 1, 1), _d(2030, 1, 1)], dtype=jnp.int32
)
SC_SPEC = CurveSpec(
    curve_id=OIS, currency="USD", pillar_dates=SC_PILLARS, day_count="act_365"
)
SC_BASE = (0.045, 0.047, 0.050)


def _sc_quotes(rates):
    """Deposit + two OIS swaps on one discount curve."""
    return [
        DepositRate(
            start_date=REF,
            end_date=int(SC_PILLARS[0]),
            rate=jnp.asarray(rates[0]),
            day_count="act_365",
            curves_touched=(OIS,),
        ),
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [_d(2026, 1, 1), _d(2027, 1, 1)], jnp.int32
            ),
            rate=jnp.asarray(rates[1]),
            day_count="act_365",
            curves_touched=(OIS,),
        ),
        OISSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [
                    _d(2026, 1, 1),
                    _d(2027, 1, 1),
                    _d(2028, 1, 1),
                    _d(2029, 1, 1),
                    _d(2030, 1, 1),
                ],
                jnp.int32,
            ),
            rate=jnp.asarray(rates[2]),
            day_count="act_365",
            curves_touched=(OIS,),
        ),
    ]


def _sc_bond():
    return FixedRateBond(
        payment_dates=jnp.array(
            [_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1)], jnp.int32
        ),
        settlement_date=REF,
        coupon_rate=jnp.array(0.04),
        face_value=jnp.array(100.0),
        frequency=1,
        discount_id=OIS,
    )


# ── Dual-curve fixture (coupled blocks) ──────────────────────────────

DC_PILLARS = jnp.array([_d(2026, 1, 1), _d(2028, 1, 1)], dtype=jnp.int32)
DC_SPECS = [
    CurveSpec(
        curve_id=OIS, currency="USD",
        pillar_dates=DC_PILLARS, day_count="act_365",
    ),
    CurveSpec(
        curve_id=P3M, currency="USD",
        pillar_dates=DC_PILLARS, day_count="act_365",
    ),
]
ANN_1Y = jnp.array([_d(2026, 1, 1)], jnp.int32)
ANN_3Y = jnp.array([_d(2026, 1, 1), _d(2027, 1, 1), _d(2028, 1, 1)], jnp.int32)
FIX_1Y = jnp.array([REF], jnp.int32)
FIX_3Y = jnp.array([REF, _d(2026, 1, 1), _d(2027, 1, 1)], jnp.int32)
DC_BASE = (0.045, 0.048, 0.0475, 0.0505)


def _dc_quotes(q):
    """Two OIS quotes (discount block) + two IBOR swap quotes (projection)."""
    return [
        DepositRate(
            start_date=REF, end_date=int(DC_PILLARS[0]),
            rate=jnp.asarray(q[0]), day_count="act_365",
            curves_touched=(OIS,),
        ),
        OISSwapRate(
            start_date=REF, fixed_dates=ANN_3Y,
            rate=jnp.asarray(q[1]), day_count="act_365",
            curves_touched=(OIS,),
        ),
        IborSwapRate(
            start_date=REF, fixed_dates=ANN_1Y, float_dates=ANN_1Y,
            fixing_dates=FIX_1Y, rate=jnp.asarray(q[2]),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=(OIS, P3M), index_id=P3M,
        ),
        IborSwapRate(
            start_date=REF, fixed_dates=ANN_3Y, float_dates=ANN_3Y,
            fixing_dates=FIX_3Y, rate=jnp.asarray(q[3]),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=(OIS, P3M), index_id=P3M,
        ),
    ]


def _dc_swap():
    return InterestRateSwap(
        start_date=REF,
        fixed_dates=ANN_3Y,
        fixed_rate=jnp.array(0.04),
        notional=jnp.array(1_000_000.0),
        discount_id=OIS,
        forward_id=P3M,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _build(specs, quote_fn, rates):
    res = bootstrap_curve_graph(REF, specs, quote_fn(rates))
    assert bool(res.converged)
    graph = res.value
    # R-4: residuals must sit far below the smallest validation bump.
    assert float(res.diagnostics.max_abs_residual) < min(BUMPS) * 1e-2
    return graph


def _pv(instrument, specs, quote_fn, rates):
    graph = _build(specs, quote_fn, rates)
    return float(price(instrument, None, MarketState(curves=graph)).price)


def _fd_quote_deltas(instrument, specs, quote_fn, base, eps):
    """Full-rebuild central differences, one quote at a time."""
    out = []
    for k in range(len(base)):
        up, dn = list(base), list(base)
        up[k] += eps
        dn[k] -= eps
        out.append(
            (
                _pv(instrument, specs, quote_fn, up)
                - _pv(instrument, specs, quote_fn, dn)
            )
            / (2 * eps)
        )
    return np.array(out)


def _aad_quote_deltas(instrument, specs, quote_fn, base):
    graph = _build(specs, quote_fn, base)
    quotes = quote_fn(base)
    market = MarketState(curves=graph)
    adj = curve_adjoint(graph, REF, specs, quotes)
    return np.asarray(
        curve_quote_deltas(instrument, None, market, adjoint=adj)
    ), adj


def _assert_matches_fd(aad, fd, *, rtol=1e-5, atol=1e-4):
    """Relative where the FD has signal, absolute where it is noise."""
    big = np.abs(fd) > 1.0
    if big.any():
        rel = np.abs(aad[big] - fd[big]) / np.abs(fd[big])
        assert rel.max() < rtol, f"rel err {rel.max():.3e}\nAAD={aad}\nFD={fd}"
    if (~big).any():
        err = np.abs(aad[~big] - fd[~big])
        assert err.max() < atol, f"abs err {err.max():.3e}\nAAD={aad}\nFD={fd}"


# ── T-2: AAD vs full-rebuild central finite differences ──────────────


@pytest.mark.audit
class TestT2AgainstFiniteDifferences:
    """Spec T-2 / A-1..A-5: every quote, two bump sizes, full rebuild."""

    @pytest.mark.parametrize("eps", BUMPS)
    def test_single_curve_bond(self, eps):
        aad, _ = _aad_quote_deltas(_sc_bond(), [SC_SPEC], _sc_quotes, SC_BASE)
        fd = _fd_quote_deltas(_sc_bond(), [SC_SPEC], _sc_quotes, SC_BASE, eps)
        _assert_matches_fd(aad, fd)

    @pytest.mark.parametrize("eps", BUMPS)
    def test_dual_curve_swap(self, eps):
        """Coupled blocks: the IBOR quotes touch both curves."""
        aad, _ = _aad_quote_deltas(_dc_swap(), DC_SPECS, _dc_quotes, DC_BASE)
        fd = _fd_quote_deltas(_dc_swap(), DC_SPECS, _dc_quotes, DC_BASE, eps)
        _assert_matches_fd(aad, fd)

    def test_deltas_are_not_all_zero(self):
        """Guard against a vacuously-passing chain."""
        aad, _ = _aad_quote_deltas(_dc_swap(), DC_SPECS, _dc_quotes, DC_BASE)
        assert np.abs(aad).max() > 1.0


# ── T-3: parallel-shift identity ─────────────────────────────────────


@pytest.mark.audit
class TestT3ParallelShiftIdentity:
    """Spec T-3 / A-3, C-6: summed quote deltas == all-quotes-bumped rebuild."""

    @pytest.mark.parametrize(
        "specs,quote_fn,base,instrument",
        [
            ([SC_SPEC], _sc_quotes, SC_BASE, _sc_bond()),
            (DC_SPECS, _dc_quotes, DC_BASE, _dc_swap()),
        ],
        ids=["single_curve", "dual_curve"],
    )
    def test_sum_matches_parallel_rebuild(
        self, specs, quote_fn, base, instrument
    ):
        aad, _ = _aad_quote_deltas(instrument, specs, quote_fn, base)
        eps = 1e-6
        up = [r + eps for r in base]
        dn = [r - eps for r in base]
        parallel = (
            _pv(instrument, specs, quote_fn, up)
            - _pv(instrument, specs, quote_fn, dn)
        ) / (2 * eps)
        rel = abs(aad.sum() - parallel) / max(abs(parallel), 1.0)
        assert rel < 1e-6, f"sum={aad.sum()} parallel={parallel}"


# ── T-15: quote-space vs node-space ──────────────────────────────────


@pytest.mark.audit
class TestT15QuotesFixedVsNodesFixed:
    """Spec T-15 / 7.4: the two conventions differ; we report quotes-fixed."""

    def test_quote_space_differs_from_node_space(self):
        graph = _build(DC_SPECS, _dc_quotes, DC_BASE)
        market = MarketState(curves=graph)
        quotes = _dc_quotes(DC_BASE)
        adj = curve_adjoint(graph, REF, DC_SPECS, quotes)

        quote_space = np.asarray(
            curve_quote_deltas(_dc_swap(), None, market, adjoint=adj)
        )
        # Node-space: bump discount pillars with every other node frozen.
        node_space = np.asarray(
            curve_pillar_deltas(_dc_swap(), None, market, curve_id=OIS)
        )

        # Discount-curve quotes are the first two columns.
        assert not np.allclose(
            quote_space[:2], node_space[1:], rtol=1e-2, atol=1e-2
        ), "quote-space and node-space deltas must not coincide"

    def test_quote_space_is_the_reported_number(self):
        """The reported figure is the one validated against a full rebuild."""
        aad, _ = _aad_quote_deltas(_dc_swap(), DC_SPECS, _dc_quotes, DC_BASE)
        fd = _fd_quote_deltas(
            _dc_swap(), DC_SPECS, _dc_quotes, DC_BASE, BUMPS[0]
        )
        _assert_matches_fd(aad, fd)

    def test_discount_quote_delta_carries_the_indirect_part(self):
        """Moving an OIS quote re-solves the projection block too.

        With the projection *quotes* held fixed, the projection curve must
        move to keep repricing them off a shifted discount curve.  That
        indirect path is what makes the number differ from a pure
        discount-node bump, so it must be non-zero.
        """
        graph = _build(DC_SPECS, _dc_quotes, DC_BASE)
        quotes = _dc_quotes(DC_BASE)
        adj = curve_adjoint(graph, REF, DC_SPECS, quotes)

        # Jacobian rows for the projection block w.r.t. the OIS quotes.
        n_ois = int(DC_SPECS[0].pillar_dates.shape[0])
        theta_bar = jnp.zeros_like(adj.log_dfs).at[n_ois:].set(1.0)
        projection_only = np.asarray(
            quote_deltas_from_cotangent(adj, theta_bar)
        )
        assert np.abs(projection_only[:2]).max() > 1e-8, (
            "projection nodes show no dependence on the discount quotes; "
            "the coupled blocks are not being solved jointly"
        )


# ── Level-generic machinery ──────────────────────────────────────────


class TestLevelAdjoint:
    def test_lambda_solves_the_transposed_system(self):
        graph = _build(DC_SPECS, _dc_quotes, DC_BASE)
        market = MarketState(curves=graph)
        quotes = _dc_quotes(DC_BASE)
        adj = curve_adjoint(graph, REF, DC_SPECS, quotes)

        grad = market_gradient(_dc_swap(), None, market)
        theta_bar = curve_log_df_cotangent(grad, market, DC_SPECS)
        result = level_adjoint(adj.level(), theta_bar)

        # (dR/dtheta)^T lam == theta_bar
        lhs = adj.residual_jacobian.T @ result.lam
        assert np.allclose(
            np.asarray(lhs), np.asarray(theta_bar), rtol=1e-8, atol=1e-10
        )

    def test_generic_path_agrees_with_specialised(self):
        graph = _build(DC_SPECS, _dc_quotes, DC_BASE)
        market = MarketState(curves=graph)
        quotes = _dc_quotes(DC_BASE)
        adj = curve_adjoint(graph, REF, DC_SPECS, quotes)

        grad = market_gradient(_dc_swap(), None, market)
        theta_bar = curve_log_df_cotangent(grad, market, DC_SPECS)

        generic = np.asarray(level_adjoint(adj.level(), theta_bar).inputs_bar)
        special = np.asarray(quote_deltas_from_cotangent(adj, theta_bar))
        assert np.allclose(generic, special, rtol=1e-10, atol=1e-12)

    def test_residual_norm_is_tiny_at_the_solution(self):
        """R-4: the theorem only applies at a converged solution."""
        graph = _build(DC_SPECS, _dc_quotes, DC_BASE)
        market = MarketState(curves=graph)
        adj = curve_adjoint(graph, REF, DC_SPECS, _dc_quotes(DC_BASE))
        grad = market_gradient(_dc_swap(), None, market)
        theta_bar = curve_log_df_cotangent(grad, market, DC_SPECS)
        result = level_adjoint(adj.level(), theta_bar)
        assert float(result.residual_norm) < 1e-10


class TestFactorisationReuse:
    """Spec A-7: one factorisation per snapshot, reused across trades."""

    def test_reused_handle_matches_fresh_handles(self):
        graph = _build([SC_SPEC], _sc_quotes, SC_BASE)
        market = MarketState(curves=graph)
        quotes = _sc_quotes(SC_BASE)
        shared = curve_adjoint(graph, REF, [SC_SPEC], quotes)

        bond = _sc_bond()
        swap = InterestRateSwap(
            start_date=REF,
            fixed_dates=jnp.array(
                [_d(2026, 1, 1), _d(2027, 1, 1)], jnp.int32
            ),
            fixed_rate=jnp.array(0.045),
            notional=jnp.array(1_000_000.0),
            discount_id=OIS,
            forward_id=OIS,
        )

        for inst in (bond, swap):
            reused = np.asarray(
                curve_quote_deltas(inst, None, market, adjoint=shared)
            )
            fresh = np.asarray(
                curve_quote_deltas(
                    inst,
                    None,
                    market,
                    adjoint=curve_adjoint(graph, REF, [SC_SPEC], quotes),
                )
            )
            assert np.allclose(reused, fresh, rtol=1e-12, atol=1e-14)

    def test_condition_number_is_reported(self):
        """A-6 / T-13: well-posedness is observable, not implicit."""
        graph = _build([SC_SPEC], _sc_quotes, SC_BASE)
        adj = curve_adjoint(graph, REF, [SC_SPEC], _sc_quotes(SC_BASE))
        cond = float(adj.jacobian_condition_number)
        assert np.isfinite(cond) and cond > 1.0
        assert cond < 1e10, "benign strip should be well-conditioned"


class TestCotangentConversion:
    def test_reference_pillar_is_dropped_and_log_scaled(self):
        """theta is log(DF) excluding the reference pillar."""
        graph = _build([SC_SPEC], _sc_quotes, SC_BASE)
        market = MarketState(curves=graph)
        grad = market_gradient(_sc_bond(), None, market)
        theta_bar = curve_log_df_cotangent(grad, market, [SC_SPEC])

        n_pillars = int(SC_SPEC.pillar_dates.shape[0])
        assert theta_bar.shape == (n_pillars,)

        dfs = graph[OIS].discount_factors
        expected = grad.curves.curves[OIS].discount_factors[1:] * dfs[1:]
        assert np.allclose(np.asarray(theta_bar), np.asarray(expected))

    def test_unknown_curve_raises(self):
        graph = _build([SC_SPEC], _sc_quotes, SC_BASE)
        market = MarketState(curves=graph)
        grad = market_gradient(_sc_bond(), None, market)
        # Well-formed identifier, but absent from this market.
        absent = CurveSpec(
            curve_id="GBP.SONIA.OIS", currency="GBP",
            pillar_dates=SC_PILLARS, day_count="act_365",
        )
        with pytest.raises(KeyError, match="GBP.SONIA.OIS"):
            curve_log_df_cotangent(grad, market, [absent])


class TestJaxContract:
    def test_quote_deltas_from_cotangent_is_jittable(self):
        """The per-trade leg must be JIT-able even though the build is not."""
        graph = _build([SC_SPEC], _sc_quotes, SC_BASE)
        market = MarketState(curves=graph)
        adj = curve_adjoint(graph, REF, [SC_SPEC], _sc_quotes(SC_BASE))
        grad = market_gradient(_sc_bond(), None, market)
        theta_bar = curve_log_df_cotangent(grad, market, [SC_SPEC])

        eager = np.asarray(quote_deltas_from_cotangent(adj, theta_bar))
        jitted = np.asarray(
            jax.jit(lambda tb: quote_deltas_from_cotangent(adj, tb))(theta_bar)
        )
        assert np.allclose(eager, jitted, rtol=1e-10, atol=1e-12)
