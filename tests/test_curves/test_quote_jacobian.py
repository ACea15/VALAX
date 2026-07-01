"""Tests for :func:`valax.curves.bootstrap_graph.quote_jacobian`.

Verifies:

* Output shape matches ``(n_pillars_total, n_quotes)``.
* Reverse-mode implicit-adjoint agrees with a central-difference
  Jacobian to ``rtol ≈ 1e-4``.
* The ``by="log_df" / "df" / "zero_rate"`` switch produces
  consistent gradients (each is a smooth transform of the others).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves import CurveSpec, DepositRate, SwapRate, quote_jacobian
from valax.curves.bootstrap_graph import bootstrap_curve_graph
from valax.dates.daycounts import ymd_to_ordinal


REF = ymd_to_ordinal(2025, 1, 1)


def _date(y, m, d):
    return ymd_to_ordinal(y, m, d)


def _deposit_strip():
    """4-pillar deposit strip on a single OIS curve."""
    pillars = jnp.array(
        [
            int(_date(2025, 4, 1)),
            int(_date(2025, 7, 1)),
            int(_date(2026, 1, 1)),
            int(_date(2027, 1, 1)),
        ],
        dtype=jnp.int32,
    )
    spec = CurveSpec(
        curve_id="USD.SOFR.OIS", currency="USD",
        pillar_dates=pillars, day_count="act_365",
    )
    rates = [0.045, 0.048, 0.050, 0.052]
    deps = [
        DepositRate(
            start_date=REF, end_date=int(pillars[i]),
            rate=jnp.array(r), day_count="act_365",
            curves_touched=("USD.SOFR.OIS",),
        )
        for i, r in enumerate(rates)
    ]
    return spec, deps, rates


# ── Shape ────────────────────────────────────────────────────────────


class TestShape:
    def test_shape_matches_pillar_x_quote_count(self):
        spec, deps, _ = _deposit_strip()
        J = quote_jacobian(REF, [spec], deps, by="df")
        assert J.shape == (4, 4)


# ── Autodiff vs finite differences ───────────────────────────────────


class TestFiniteDifferenceAgreement:
    """Recalibrate the graph after bumping each quote by ±ε and diff the
    output DFs.  The autodiff Jacobian must agree with the resulting
    central-difference matrix.
    """

    def _fd_jacobian(self, spec, deps, base_rates, eps=1e-6):
        n = len(base_rates)
        fd = np.zeros((n, n))
        for k in range(n):
            for sgn, out_col in [(+1, None), (-1, None)]:
                bumped_rates = list(base_rates)
                bumped_rates[k] = base_rates[k] + sgn * eps
                bumped_deps = [
                    DepositRate(
                        start_date=REF, end_date=int(spec.pillar_dates[i]),
                        rate=jnp.array(bumped_rates[i]), day_count="act_365",
                        curves_touched=("USD.SOFR.OIS",),
                    )
                    for i in range(n)
                ]
                graph, _ = bootstrap_curve_graph(
                    REF, [spec], bumped_deps,
                )
                dfs = np.asarray(graph["USD.SOFR.OIS"].discount_factors[1:])
                if sgn == +1:
                    plus = dfs
                else:
                    minus = dfs
            fd[:, k] = (plus - minus) / (2 * eps)
        return fd

    def test_df_jacobian_matches_fd(self):
        spec, deps, rates = _deposit_strip()
        ad = np.asarray(quote_jacobian(REF, [spec], deps, by="df"))
        fd = self._fd_jacobian(spec, deps, rates)
        assert ad.shape == fd.shape
        # Absolute tolerance is looser than relative because the
        # off-diagonal entries are exactly zero for independent deposits.
        assert np.allclose(ad, fd, rtol=1e-4, atol=1e-8), (
            f"autodiff vs FD mismatch:\nAD=\n{ad}\nFD=\n{fd}"
        )


# ── by-switch consistency ────────────────────────────────────────────


class TestBySwitch:
    """The three ``by`` variants must be mutually consistent under the
    chain rule for a deposit strip.
    """

    def test_by_switch_shapes_agree(self):
        spec, deps, _ = _deposit_strip()
        J_log = quote_jacobian(REF, [spec], deps, by="log_df")
        J_df = quote_jacobian(REF, [spec], deps, by="df")
        J_zr = quote_jacobian(REF, [spec], deps, by="zero_rate")
        assert J_log.shape == J_df.shape == J_zr.shape == (4, 4)

    def test_df_and_log_df_agree_via_chain_rule(self):
        """d(DF)/dr = DF * d(log DF)/dr"""
        spec, deps, _ = _deposit_strip()
        J_log = np.asarray(quote_jacobian(REF, [spec], deps, by="log_df"))
        J_df = np.asarray(quote_jacobian(REF, [spec], deps, by="df"))
        graph, _ = bootstrap_curve_graph(REF, [spec], deps)
        pillar_dfs = np.asarray(graph["USD.SOFR.OIS"].discount_factors[1:])
        assert np.allclose(
            J_df, pillar_dfs[:, None] * J_log, rtol=1e-8, atol=1e-12,
        )


# ── Rejects invalid by= ──────────────────────────────────────────────


class TestValidation:
    def test_invalid_by_raises(self):
        spec, deps, _ = _deposit_strip()
        with pytest.raises(ValueError, match="by="):
            quote_jacobian(REF, [spec], deps, by="not_supported")
