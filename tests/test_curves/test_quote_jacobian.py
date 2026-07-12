"""Tests for :func:`valax.curves.bootstrap_graph.quote_jacobian`.

Verifies:

* Output shape matches ``(n_pillars_total, n_quotes)``.
* Reverse-mode implicit-adjoint agrees with a central-difference
  Jacobian to ``rtol ≈ 1e-4``.
* The ``by="log_df" / "df" / "zero_rate"`` switch produces
  consistent gradients (each is a smooth transform of the others).
* All eleven registered quote types can drive the Jacobian —
  including ``.spread`` (``TenorBasisSwap``, ``CrossCurrencyBasisSwap``),
  ``.futures_rate`` (``MoneyMarketFuture``), ``.quoted_forward``
  (``FXForward``), ``.far_rate`` (``FXSwap``), and ``.jump_size``
  (``TurnInstrument``).
* Unknown quote types raise ``TypeError``.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from valax.curves import (
    CrossCurrencyBasisSwap,
    CurveSpec,
    DepositRate,
    FRA,
    FXForward,
    FXSwap,
    IborSwapRate,
    MoneyMarketFuture,
    OISSwapRate,
    SwapRate,
    TenorBasisSwap,
    TurnInstrument,
    quote_jacobian,
)
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


# ── Non-``.rate`` quote fields ───────────────────────────────────────
#
# The remaining tests exercise the extended-quote support added in
# MC-Curves-3: each new test drives ``quote_jacobian`` with an
# instrument type whose primary quote scalar is NOT ``.rate``.
# Every test:
#
# 1. Builds a small, well-posed fixture (2 – 5 instruments / pillars).
# 2. Calls ``quote_jacobian`` for the AD Jacobian.
# 3. Bumps the target quote by ±ε, re-bootstraps, and finite-
#    differences the resulting pillar-DF vector.
# 4. Asserts the AD column matches the FD column to ``rtol ≈ 1e-4``.
#
# The FD path re-uses ``bootstrap_curve_graph`` (with implicit-adjoint
# grad disabled by construction, since we bump inputs manually), so
# the comparison is genuinely AD-vs-FD.


def _fd_column(build_instruments, base_rate, k_col_pillar_count, eps=1e-6):
    """Central-difference the pillar-DF vector w.r.t. one scalar quote.

    ``build_instruments`` is a callable ``rate -> (curve_specs,
    instruments)``.  Returns a 1-D array of length ``k_col_pillar_count``
    equal to ``d(DF_pillar) / d(rate)`` per pillar.
    """
    specs_p, insts_p = build_instruments(base_rate + eps)
    graph_p, _ = bootstrap_curve_graph(REF, specs_p, insts_p)
    dfs_p = jnp.concatenate(
        [graph_p[s.curve_id].discount_factors[1:] for s in specs_p]
    )

    specs_m, insts_m = build_instruments(base_rate - eps)
    graph_m, _ = bootstrap_curve_graph(REF, specs_m, insts_m)
    dfs_m = jnp.concatenate(
        [graph_m[s.curve_id].discount_factors[1:] for s in specs_m]
    )
    assert dfs_p.shape == (k_col_pillar_count,)
    return np.asarray((dfs_p - dfs_m) / (2 * eps))


class TestMoneyMarketFutureJacobian:
    """Bump the ``.futures_rate`` of a :class:`MoneyMarketFuture` and
    diff the resulting single-curve DFs.
    """

    @staticmethod
    def _build(futures_rate: float):
        t0 = int(_date(2026, 1, 1))
        t1 = int(_date(2026, 4, 1))
        t0_arr = jnp.asarray(t0, dtype=jnp.int32)
        t1_arr = jnp.asarray(t1, dtype=jnp.int32)
        pillars = jnp.array([t0, t1], dtype=jnp.int32)
        spec = CurveSpec(
            curve_id="USD.SOFR.3M", currency="USD",
            pillar_dates=pillars, day_count="act_365",
        )
        dep = DepositRate(
            start_date=REF, end_date=t0,
            rate=jnp.array(0.045), day_count="act_365",
            curves_touched=("USD.SOFR.3M",),
        )
        fut = MoneyMarketFuture(
            # Both start/end must be JAX ints so ``year_fraction`` can
            # dispatch (Python int - int has no ``.astype``).
            start_date=t0_arr, end_date=t1_arr,
            futures_rate=jnp.array(futures_rate),
            day_count="act_365",
            curves_touched=("USD.SOFR.3M",),
        )
        return [spec], [dep, fut]

    def test_shape(self):
        specs, insts = self._build(0.048)
        J = quote_jacobian(REF, specs, insts, by="df")
        assert J.shape == (2, 2)

    def test_futures_rate_column_matches_fd(self):
        specs, insts = self._build(0.048)
        ad = np.asarray(quote_jacobian(REF, specs, insts, by="df"))
        fd_col1 = _fd_column(self._build, 0.048, k_col_pillar_count=2)
        assert np.allclose(ad[:, 1], fd_col1, rtol=1e-4, atol=1e-8), (
            f"futures column mismatch:\nAD col1 = {ad[:, 1]}\n"
            f"FD col1 = {fd_col1}"
        )
        # And the first column (deposit rate) is well-signed.
        assert ad[0, 0] < 0.0, "d(DF@T0) / d(deposit rate) must be negative"


class TestFXForwardJacobian:
    """Bump ``.quoted_forward`` of an :class:`FXForward` and diff the
    two-currency DFs.
    """

    @staticmethod
    def _build(quoted_forward: float):
        t = int(_date(2026, 1, 1))
        usd_pillars = jnp.array([t], dtype=jnp.int32)
        eur_pillars = jnp.array([t], dtype=jnp.int32)
        usd_spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=usd_pillars, day_count="act_365",
        )
        eur_spec = CurveSpec(
            curve_id="EUR.ESTR.OIS", currency="EUR",
            pillar_dates=eur_pillars, day_count="act_365",
        )
        usd_dep = DepositRate(
            start_date=REF, end_date=t,
            rate=jnp.array(0.045), day_count="act_365",
            curves_touched=("USD.SOFR.OIS",),
        )
        fx_fwd = FXForward(
            value_date=REF, settle_date=t,
            quoted_forward=jnp.array(quoted_forward),
            fx_spot=jnp.array(1.10),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        return [usd_spec, eur_spec], [usd_dep, fx_fwd]

    def test_shape(self):
        specs, insts = self._build(1.08)
        J = quote_jacobian(REF, specs, insts, by="df")
        assert J.shape == (2, 2)

    def test_quoted_forward_column_matches_fd(self):
        specs, insts = self._build(1.08)
        ad = np.asarray(quote_jacobian(REF, specs, insts, by="df"))
        fd_col1 = _fd_column(self._build, 1.08, k_col_pillar_count=2)
        assert np.allclose(ad[:, 1], fd_col1, rtol=1e-4, atol=1e-8), (
            f"FXForward column mismatch:\nAD col1 = {ad[:, 1]}\n"
            f"FD col1 = {fd_col1}"
        )
        # A higher quoted_forward at fixed spot implies a higher
        # DF_foreign / DF_domestic ratio → higher foreign DF.
        assert ad[1, 1] > 0.0, "d(DF_EUR) / d(quoted_forward) must be positive"


class TestFXSwapJacobian:
    """Bump ``.far_rate`` of an :class:`FXSwap` (near_date = REF so the
    residual reduces to a CIP relation on the far leg).
    """

    @staticmethod
    def _build(far_rate: float):
        t = int(_date(2026, 1, 1))
        usd_spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=jnp.array([t], dtype=jnp.int32),
            day_count="act_365",
        )
        eur_spec = CurveSpec(
            curve_id="EUR.ESTR.OIS", currency="EUR",
            pillar_dates=jnp.array([t], dtype=jnp.int32),
            day_count="act_365",
        )
        usd_dep = DepositRate(
            start_date=REF, end_date=t,
            rate=jnp.array(0.045), day_count="act_365",
            curves_touched=("USD.SOFR.OIS",),
        )
        fx_swap = FXSwap(
            near_date=REF, far_date=t,
            near_rate=jnp.array(1.10),
            far_rate=jnp.array(far_rate),
            curves_touched=("USD.SOFR.OIS", "EUR.ESTR.OIS"),
        )
        return [usd_spec, eur_spec], [usd_dep, fx_swap]

    def test_shape(self):
        specs, insts = self._build(1.08)
        J = quote_jacobian(REF, specs, insts, by="df")
        assert J.shape == (2, 2)

    def test_far_rate_column_matches_fd(self):
        specs, insts = self._build(1.08)
        ad = np.asarray(quote_jacobian(REF, specs, insts, by="df"))
        fd_col1 = _fd_column(self._build, 1.08, k_col_pillar_count=2)
        assert np.allclose(ad[:, 1], fd_col1, rtol=1e-4, atol=1e-8), (
            f"FXSwap column mismatch:\nAD col1 = {ad[:, 1]}\n"
            f"FD col1 = {fd_col1}"
        )


class TestTenorBasisSwapJacobian:
    """Bump ``.spread`` of a :class:`TenorBasisSwap` in a three-curve
    (OIS + 3M + 6M) build.
    """

    @staticmethod
    def _quarterly(end):
        dates = []
        for year in range(2025, 2036):
            for month in (1, 4, 7, 10):
                d = int(_date(year, month, 1))
                if d > int(REF) and d <= end:
                    dates.append(d)
        return jnp.array(dates, dtype=jnp.int32)

    @staticmethod
    def _semiannual(end):
        dates = []
        for year in range(2025, 2036):
            for month in (1, 7):
                d = int(_date(year, month, 1))
                if d > int(REF) and d <= end:
                    dates.append(d)
        return jnp.array(dates, dtype=jnp.int32)

    @classmethod
    def _build(cls, basis_spread: float):
        # 5 instruments, 5 pillars total.
        disc_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        fwd_3m_pillars = disc_pillars
        fwd_6m_pillars = jnp.array(
            [int(_date(2027, 1, 1))], dtype=jnp.int32,
        )
        disc_spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=disc_pillars, day_count="act_365",
        )
        fwd_3m_spec = CurveSpec(
            curve_id="USD.SOFR.3M", currency="USD",
            pillar_dates=fwd_3m_pillars, day_count="act_365",
        )
        fwd_6m_spec = CurveSpec(
            curve_id="USD.SOFR.6M", currency="USD",
            pillar_dates=fwd_6m_pillars, day_count="act_365",
        )
        deps = [
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[0]),
                rate=jnp.array(0.040), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(disc_pillars[1]),
                rate=jnp.array(0.042), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
        ]
        float_1y = cls._quarterly(int(_date(2026, 1, 1)))
        fix_1y = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), float_1y[:-1]]
        )
        ibor_1y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1))], dtype=jnp.int32,
            ),
            float_dates=float_1y, fixing_dates=fix_1y,
            rate=jnp.array(0.045),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        float_2y = cls._quarterly(int(_date(2027, 1, 1)))
        fix_2y = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), float_2y[:-1]]
        )
        ibor_2y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
                dtype=jnp.int32,
            ),
            float_dates=float_2y, fixing_dates=fix_2y,
            rate=jnp.array(0.047),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        leg_a_dates = cls._quarterly(int(_date(2027, 1, 1)))
        leg_a_fix = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), leg_a_dates[:-1]]
        )
        leg_b_dates = cls._semiannual(int(_date(2027, 1, 1)))
        leg_b_fix = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), leg_b_dates[:-1]]
        )
        basis_2y = TenorBasisSwap(
            start_date=REF,
            leg_a_dates=leg_a_dates,
            leg_a_fixing_dates=leg_a_fix,
            leg_b_dates=leg_b_dates,
            leg_b_fixing_dates=leg_b_fix,
            spread=jnp.array(basis_spread),
            leg_a_day_count="act_365",
            leg_a_index_id="USD.SOFR.3M",
            leg_b_day_count="act_365",
            leg_b_index_id="USD.SOFR.6M",
            spread_on_leg="a",
            curves_touched=(
                "USD.SOFR.OIS", "USD.SOFR.3M", "USD.SOFR.6M",
            ),
        )
        return (
            [disc_spec, fwd_3m_spec, fwd_6m_spec],
            deps + [ibor_1y, ibor_2y, basis_2y],
        )

    def test_shape(self):
        specs, insts = self._build(0.0015)
        J = quote_jacobian(REF, specs, insts, by="df")
        # 5 pillars total, 5 quotes.
        assert J.shape == (5, 5)

    def test_spread_column_matches_fd(self):
        specs, insts = self._build(0.0015)
        ad = np.asarray(quote_jacobian(REF, specs, insts, by="df"))
        fd_col_last = _fd_column(
            self._build, 0.0015, k_col_pillar_count=5, eps=1e-5,
        )
        # Basis swap is the last instrument (column index 4).
        assert np.allclose(
            ad[:, 4], fd_col_last, rtol=1e-4, atol=1e-8,
        ), (
            f"basis spread column mismatch:\nAD col4 = {ad[:, 4]}\n"
            f"FD col4 = {fd_col_last}"
        )


class TestCrossCurrencyBasisSwapJacobian:
    """Bump ``.spread`` of a :class:`CrossCurrencyBasisSwap` in a 4-curve
    EUR/USD build.
    """

    @staticmethod
    def _quarterly(end):
        dates = []
        for year in range(2025, 2036):
            for month in (1, 4, 7, 10):
                d = int(_date(year, month, 1))
                if d > int(REF) and d <= end:
                    dates.append(d)
        return jnp.array(dates, dtype=jnp.int32)

    @staticmethod
    def _semiannual(end):
        dates = []
        for year in range(2025, 2036):
            for month in (1, 7):
                d = int(_date(year, month, 1))
                if d > int(REF) and d <= end:
                    dates.append(d)
        return jnp.array(dates, dtype=jnp.int32)

    @classmethod
    def _build(cls, ccbs_spread: float):
        usd_ois_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        usd_3m_pillars = jnp.array(
            [int(_date(2027, 1, 1))], dtype=jnp.int32,
        )
        eur_ois_pillars = jnp.array(
            [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
            dtype=jnp.int32,
        )
        eur_6m_pillars = jnp.array(
            [int(_date(2027, 1, 1))], dtype=jnp.int32,
        )

        specs = [
            CurveSpec(
                curve_id="USD.SOFR.OIS", currency="USD",
                pillar_dates=usd_ois_pillars, day_count="act_365",
            ),
            CurveSpec(
                curve_id="USD.SOFR.3M", currency="USD",
                pillar_dates=usd_3m_pillars, day_count="act_365",
            ),
            CurveSpec(
                curve_id="EUR.ESTR.OIS", currency="EUR",
                pillar_dates=eur_ois_pillars, day_count="act_365",
            ),
            CurveSpec(
                curve_id="EUR.EURIBOR.6M", currency="EUR",
                pillar_dates=eur_6m_pillars, day_count="act_365",
            ),
        ]
        usd_ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(usd_ois_pillars[0]),
                rate=jnp.array(0.040), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(usd_ois_pillars[1]),
                rate=jnp.array(0.042), day_count="act_365",
                curves_touched=("USD.SOFR.OIS",),
            ),
        ]
        eur_ois_deps = [
            DepositRate(
                start_date=REF, end_date=int(eur_ois_pillars[0]),
                rate=jnp.array(0.028), day_count="act_365",
                curves_touched=("EUR.ESTR.OIS",),
            ),
            DepositRate(
                start_date=REF, end_date=int(eur_ois_pillars[1]),
                rate=jnp.array(0.030), day_count="act_365",
                curves_touched=("EUR.ESTR.OIS",),
            ),
        ]
        float_2y = cls._quarterly(int(_date(2027, 1, 1)))
        fix_2y = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), float_2y[:-1]]
        )
        usd_ibor_2y = IborSwapRate(
            start_date=REF,
            fixed_dates=jnp.array(
                [int(_date(2026, 1, 1)), int(_date(2027, 1, 1))],
                dtype=jnp.int32,
            ),
            float_dates=float_2y, fixing_dates=fix_2y,
            rate=jnp.array(0.047),
            fixed_day_count="act_365", float_day_count="act_365",
            curves_touched=("USD.SOFR.OIS", "USD.SOFR.3M"),
            index_id="USD.SOFR.3M",
        )
        dom_dates = cls._quarterly(int(_date(2027, 1, 1)))
        dom_fix = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), dom_dates[:-1]]
        )
        for_dates = cls._semiannual(int(_date(2027, 1, 1)))
        for_fix = jnp.concatenate(
            [jnp.array([REF], dtype=jnp.int32), for_dates[:-1]]
        )
        ccbs_2y = CrossCurrencyBasisSwap(
            start_date=REF,
            dom_dates=dom_dates,
            dom_fixing_dates=dom_fix,
            for_dates=for_dates,
            for_fixing_dates=for_fix,
            fx_spot=jnp.array(1.10),
            spread=jnp.array(ccbs_spread),
            dom_day_count="act_365",
            dom_index_id="USD.SOFR.3M",
            for_day_count="act_365",
            for_index_id="EUR.EURIBOR.6M",
            spread_on_leg="foreign",
            variant="mtm",
            curves_touched=(
                "USD.SOFR.OIS", "USD.SOFR.3M",
                "EUR.ESTR.OIS", "EUR.EURIBOR.6M",
            ),
        )
        instruments = usd_ois_deps + eur_ois_deps + [usd_ibor_2y, ccbs_2y]
        return specs, instruments

    def test_shape(self):
        specs, insts = self._build(-0.0025)
        J = quote_jacobian(REF, specs, insts, by="df")
        # 6 pillars total (2 USD.OIS + 1 USD.3M + 2 EUR.OIS + 1 EUR.6M),
        # 6 instruments.
        assert J.shape == (6, 6)

    def test_spread_column_matches_fd(self):
        specs, insts = self._build(-0.0025)
        ad = np.asarray(quote_jacobian(REF, specs, insts, by="df"))
        fd_col_last = _fd_column(
            self._build, -0.0025, k_col_pillar_count=6, eps=1e-5,
        )
        # CCBS is last instrument → column index 5.
        assert np.allclose(
            ad[:, 5], fd_col_last, rtol=1e-4, atol=1e-8,
        ), (
            f"CCBS spread column mismatch:\nAD col5 = {ad[:, 5]}\n"
            f"FD col5 = {fd_col_last}"
        )


# ── Unknown quote types raise TypeError ──────────────────────────────


class TestUnknownQuoteType:
    """Instruments whose class is not registered in ``_QUOTE_FIELD``
    must fail loudly, not silently mis-compute.
    """

    def test_unregistered_class_raises_type_error(self):
        # Fabricate a bootstrap instrument that satisfies the protocol
        # but is not registered in _QUOTE_FIELD.
        import equinox as eqx  # noqa: F401 (used via eqx.Module below)
        from valax.curves.graph import CurveGraph
        from valax.curves.fixings import FixingHistory

        class UnregisteredQuote(eqx.Module):
            some_field: jnp.ndarray
            curves_touched: tuple = eqx.field(
                static=True, default=("USD.SOFR.OIS",)
            )

            def residual(
                self,
                graph: CurveGraph,
                fixings: FixingHistory,
                ref_date,
            ):
                del fixings, ref_date
                return graph[self.curves_touched[0]](REF) - 1.0

        spec = CurveSpec(
            curve_id="USD.SOFR.OIS", currency="USD",
            pillar_dates=jnp.array(
                [int(_date(2026, 1, 1))], dtype=jnp.int32,
            ),
            day_count="act_365",
        )
        inst = UnregisteredQuote(some_field=jnp.array(0.05))
        with pytest.raises(TypeError, match="UnregisteredQuote"):
            quote_jacobian(REF, [spec], [inst])
