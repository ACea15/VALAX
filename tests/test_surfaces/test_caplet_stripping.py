"""Tests for cap → caplet (optionlet) stripping.

Mirrors ``valax/surfaces/caplet_stripping.py``.  The load-bearing property
is **exact repricing**: a stripped term structure must return every quoted
cap's premium, single- and dual-curve alike.  These are also the first
multi-curve tests under ``tests/test_surfaces/``.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from valax.calibration import CalibrationError, LevelInputs
from valax.curves.discount import DiscountCurve
from valax.dates.daycounts import ymd_to_ordinal
from valax.instruments.rates import Cap
from valax.pricing.analytic.caplets import (
    cap_price_bachelier,
    cap_price_black76,
)
from valax.surfaces.caplet_stripping import (
    CapletVolCurve,
    cap_price_from_strip,
    optionlet_expiries,
    optionlet_forwards,
    strip_caplet_vols,
    strip_columns_level,
    strip_optionlet_surface,
)


# ── Fixtures ─────────────────────────────────────────────────────────

REF = int(ymd_to_ordinal(2025, 1, 1))


def _quarterly_dates(n_quarters: int, first_month: int = 4):
    """Quarterly start/end ordinals for ``n_quarters`` accrual periods.

    Starts one quarter after the reference date by default, so no
    optionlet has already fixed (the market convention for a quoted
    strip).
    """
    months = [(first_month + 3 * i) for i in range(n_quarters + 1)]
    dates = []
    for m in months:
        year = 2025 + (m - 1) // 12
        month = (m - 1) % 12 + 1
        dates.append(int(ymd_to_ordinal(year, month, 1)))
    grid = jnp.array(dates, dtype=jnp.int32)
    return grid[:-1], grid[1:]


def _flat_curve(rate: float, years: float = 12.0) -> DiscountCurve:
    """Flat continuously-compounded curve on an annual pillar grid."""
    n = int(years) + 1
    pillars = jnp.array(
        [int(ymd_to_ordinal(2025 + i, 1, 1)) for i in range(n)],
        dtype=jnp.int32,
    )
    times = (pillars - REF).astype(jnp.float64) / 365.0
    return DiscountCurve(
        pillar_dates=pillars,
        discount_factors=jnp.exp(-rate * times),
        reference_date=jnp.asarray(REF),
    )


def _cap(
    n_quarters: int,
    strike: float,
    is_cap: bool = True,
    first_month: int = 4,
) -> Cap:
    """Cap with ``n_quarters`` quarterly optionlets fixing in advance."""
    starts, ends = _quarterly_dates(n_quarters, first_month)
    return Cap(
        fixing_dates=starts,
        start_dates=starts,
        end_dates=ends,
        strike=jnp.asarray(strike),
        notional=jnp.asarray(1_000_000.0),
        is_cap=is_cap,
    )


@pytest.fixture
def curve():
    """OIS / discount curve: flat 3%."""
    return _flat_curve(0.03)


@pytest.fixture
def forward_curve():
    """Projection curve: flat 3.4%, i.e. a +40 bp tenor basis over OIS."""
    return _flat_curve(0.034)


@pytest.fixture
def caps():
    """Nested 1Y / 2Y / 3Y / 4Y quarterly cap strip at 3.5%."""
    return [_cap(n, 0.035) for n in (4, 8, 12, 16)]


#: Rising flat-vol quotes, one per cap in ``caps``.
QUOTES = jnp.array([0.30, 0.34, 0.36, 0.37])


# ── Repricing: the defining property ─────────────────────────────────


@pytest.mark.audit
class TestExactRepricing:
    """T-1 gate for the cap layer: every quoted cap reprices from the strip.

    Stripping is a per-maturity root-find, so repricing is exact up to the
    solver's tolerance by construction rather than by approximation —
    hence the 1e-10 relative bar. Marked ``audit`` so ``pytest -m audit``
    collects the cap half of T-1 along with the curve, swaption-cube and
    basket halves.
    """

    def test_reprices_every_quoted_cap_single_curve(self, caps, curve):
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        for i, cap in enumerate(caps):
            target = cap_price_black76(cap, curve, QUOTES[i])
            got = cap_price_from_strip(cap, strip, curve)
            assert float(jnp.abs(got - target)) / float(target) < 1e-10

    def test_reprices_every_quoted_cap_dual_curve(
        self, caps, curve, forward_curve,
    ):
        strip = strip_caplet_vols(
            caps, curve, cap_vols=QUOTES, forward_curve=forward_curve,
        ).value
        for i, cap in enumerate(caps):
            target = cap_price_black76(cap, curve, QUOTES[i], forward_curve)
            got = cap_price_from_strip(cap, strip, curve, forward_curve)
            assert float(jnp.abs(got - target)) / float(target) < 1e-10

    def test_premium_quotes_match_vol_quotes(self, caps, curve):
        premiums = jnp.stack([
            cap_price_black76(cap, curve, QUOTES[i])
            for i, cap in enumerate(caps)
        ])
        from_vols = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        from_premiums = strip_caplet_vols(
            caps, curve, cap_premiums=premiums,
        ).value
        assert jnp.allclose(from_vols.vols, from_premiums.vols, atol=1e-10)

    def test_floor_strip_reprices(self, curve):
        floors = [_cap(n, 0.035, is_cap=False) for n in (4, 8, 12)]
        quotes = jnp.array([0.30, 0.33, 0.35])
        strip = strip_caplet_vols(floors, curve, cap_vols=quotes).value
        for i, floor in enumerate(floors):
            target = cap_price_black76(floor, curve, quotes[i])
            got = cap_price_from_strip(floor, strip, curve)
            assert float(jnp.abs(got - target)) / float(jnp.abs(target)) < 1e-10


# ── Shape of the stripped term structure ─────────────────────────────


class TestStrippedShape:
    def test_one_vol_per_optionlet(self, caps, curve):
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        assert strip.vols.shape == (16,)
        assert strip.expiries.shape == (16,)
        assert strip.forwards.shape == (16,)
        assert strip.bucket_ends == (4, 8, 12, 16)

    def test_flat_quotes_give_flat_strip(self, caps, curve):
        flat = jnp.full((4,), 0.32)
        strip = strip_caplet_vols(caps, curve, cap_vols=flat).value
        assert jnp.allclose(strip.vols, 0.32, atol=1e-10)

    def test_constant_within_bucket(self, caps, curve):
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        for lo, hi in zip((0,) + strip.bucket_ends[:-1], strip.bucket_ends):
            bucket = strip.vols[lo:hi]
            assert jnp.allclose(bucket, bucket[0], atol=1e-12)

    def test_rising_quotes_give_forward_vols_above_the_quote(
        self, caps, curve,
    ):
        # A rising flat-vol term structure implies forward (optionlet)
        # vols above the flat quote at the long end — the same convexity
        # relation as forward rates versus zero rates.
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        assert float(strip.vols[-1]) > float(QUOTES[-1])
        assert float(strip.vols[0]) == pytest.approx(float(QUOTES[0]), abs=1e-9)

    def test_strike_is_carried(self, caps, curve):
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        assert float(strip.strike) == pytest.approx(0.035)


# ── The dual-curve split ─────────────────────────────────────────────


class TestDualCurve:
    def test_forwards_come_from_the_projection_curve(
        self, caps, curve, forward_curve,
    ):
        cap = caps[-1]
        single = optionlet_forwards(cap, curve)
        dual = optionlet_forwards(cap, curve, forward_curve)
        # +40 bp of continuously-compounded basis lifts every forward.
        assert jnp.all(dual > single)
        assert float(jnp.mean(dual - single)) == pytest.approx(0.004, abs=5e-4)

    def test_basis_moves_the_stripped_vols(self, caps, curve, forward_curve):
        single = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        dual = strip_caplet_vols(
            caps, curve, cap_vols=QUOTES, forward_curve=forward_curve,
        ).value
        # The quotes are identical; only the basis changed. A stripper that
        # ignored the projection curve would return the same numbers.
        assert not jnp.allclose(single.vols, dual.vols, atol=1e-6)

    def test_projection_curve_is_a_differentiable_input(
        self, caps, curve, forward_curve,
    ):
        def long_vol(dfs):
            fwd = eqx.tree_at(
                lambda c: c.discount_factors, forward_curve, dfs,
            )
            strip = strip_caplet_vols(
                caps, curve, cap_vols=QUOTES, forward_curve=fwd,
            ).value
            return strip.vols[-1]

        g = jax.grad(long_vol)(forward_curve.discount_factors)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0.0


# ── Normal (Bachelier) branch ────────────────────────────────────────


class TestNormalBranch:
    def test_strips_at_negative_rates(self):
        curve = _flat_curve(-0.004)
        caps = [_cap(n, -0.002) for n in (4, 8, 12)]
        quotes = jnp.array([0.0060, 0.0070, 0.0075])
        strip = strip_caplet_vols(
            caps, curve, cap_vols=quotes, is_normal=True,
        ).value
        assert strip.is_normal
        assert jnp.all(strip.vols > 0.0)
        for i, cap in enumerate(caps):
            target = cap_price_bachelier(cap, curve, quotes[i])
            got = cap_price_from_strip(cap, strip, curve)
            assert float(jnp.abs(got - target)) / float(jnp.abs(target)) < 1e-10

    def test_flat_normal_quotes_give_flat_strip(self):
        curve = _flat_curve(0.002)
        caps = [_cap(n, 0.0) for n in (4, 8)]
        strip = strip_caplet_vols(
            caps, curve, cap_vols=jnp.full((2,), 0.0055), is_normal=True,
        ).value
        assert jnp.allclose(strip.vols, 0.0055, atol=1e-10)


# ── Term-structure lookup ────────────────────────────────────────────


class TestCapletVolCurveLookup:
    def test_piecewise_constant_picks_the_bucket(self, caps, curve):
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        # Read at the quoted expiries themselves.
        assert jnp.allclose(strip(strip.expiries), strip.vols, atol=1e-12)
        # Just inside a bucket resolves to that bucket's vol.
        mid = 0.5 * (strip.expiries[4] + strip.expiries[5])
        assert float(strip(mid)) == pytest.approx(float(strip.vols[5]))

    def test_flat_extrapolation_beyond_the_strip(self, caps, curve):
        strip = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        assert float(strip(jnp.asarray(50.0))) == pytest.approx(
            float(strip.vols[-1])
        )

    def test_linear_mode_interpolates(self, caps, curve):
        strip = strip_caplet_vols(
            caps, curve, cap_vols=QUOTES, interp="linear",
        ).value
        mid = 0.5 * (strip.expiries[4] + strip.expiries[5])
        lo, hi = float(strip.vols[4]), float(strip.vols[5])
        assert float(strip(mid)) == pytest.approx(0.5 * (lo + hi), rel=1e-9)

    def test_rejects_unknown_interp(self):
        with pytest.raises(ValueError, match="interp"):
            CapletVolCurve(
                expiries=jnp.array([1.0]),
                vols=jnp.array([0.2]),
                strike=jnp.asarray(0.03),
                forwards=jnp.array([0.03]),
                interp="cubic",
            )


# ── Differentiability and JIT ────────────────────────────────────────


class TestJaxContract:
    def test_filter_jit(self, caps, curve):
        @eqx.filter_jit
        def run(quotes):
            return strip_caplet_vols(caps, curve, cap_vols=quotes).value.vols

        vols = run(QUOTES)
        assert vols.shape == (16,)
        assert jnp.all(jnp.isfinite(vols))
        assert jnp.allclose(
            vols, strip_caplet_vols(caps, curve, cap_vols=QUOTES).value.vols,
            atol=1e-10,
        )

    def test_grad_through_the_bootstrap(self, caps, curve):
        # Implicit differentiation through the sequential root-finds: the
        # long-end optionlet vol rises when the long cap quote rises.
        def long_vol(bump):
            quotes = QUOTES.at[-1].add(bump)
            return strip_caplet_vols(caps, curve, cap_vols=quotes).value.vols[-1]

        g = jax.grad(long_vol)(jnp.asarray(0.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0

    def test_earlier_buckets_are_independent_of_later_quotes(
        self, caps, curve,
    ):
        def first_vol(bump):
            quotes = QUOTES.at[-1].add(bump)
            return strip_caplet_vols(caps, curve, cap_vols=quotes).value.vols[0]

        g = jax.grad(first_vol)(jnp.asarray(0.0))
        assert float(jnp.abs(g)) < 1e-8


# ── Validation at the user boundary ──────────────────────────────────


class TestValidation:
    def test_empty_strip(self, curve):
        with pytest.raises(ValueError, match="at least one quoted cap"):
            strip_caplet_vols([], curve, cap_vols=jnp.array([0.2]))

    def test_both_quote_kinds(self, caps, curve):
        with pytest.raises(ValueError, match="exactly one of"):
            strip_caplet_vols(
                caps, curve, cap_vols=QUOTES, cap_premiums=QUOTES,
            )

    def test_neither_quote_kind(self, caps, curve):
        with pytest.raises(ValueError, match="exactly one of"):
            strip_caplet_vols(caps, curve)

    def test_quote_count_mismatch(self, caps, curve):
        with pytest.raises(ValueError, match="one quote per cap"):
            strip_caplet_vols(caps, curve, cap_vols=jnp.array([0.3, 0.3]))

    def test_non_nested_schedules(self, curve):
        caps = [_cap(4, 0.035), _cap(8, 0.035)]
        shifted = eqx.tree_at(
            lambda c: c.start_dates,
            caps[1],
            caps[1].start_dates.at[0].add(7),
        )
        with pytest.raises(ValueError, match="nested"):
            strip_caplet_vols(
                [caps[0], shifted], curve, cap_vols=jnp.array([0.3, 0.32]),
            )

    def test_non_increasing_maturities(self, curve):
        caps = [_cap(8, 0.035), _cap(4, 0.035)]
        with pytest.raises(ValueError, match="ascending in maturity"):
            strip_caplet_vols(caps, curve, cap_vols=jnp.array([0.3, 0.32]))

    def test_strike_mismatch(self, curve):
        caps = [_cap(4, 0.035), _cap(8, 0.04)]
        with pytest.raises(ValueError, match="different strike"):
            strip_caplet_vols(caps, curve, cap_vols=jnp.array([0.3, 0.32]))

    def test_already_fixed_optionlet(self, curve):
        # A spot-starting cap whose first period has already fixed: zero
        # vega, nothing to strip.
        caps = [
            _cap(4, 0.035, first_month=1),
            _cap(8, 0.035, first_month=1),
        ]
        with pytest.raises(ValueError, match="no volatility to strip"):
            strip_caplet_vols(caps, curve, cap_vols=jnp.array([0.3, 0.32]))

    def test_mixed_cap_and_floor(self, curve):
        caps = [_cap(4, 0.035), _cap(8, 0.035, is_cap=False)]
        with pytest.raises(ValueError, match="all caps or all floors"):
            strip_caplet_vols(caps, curve, cap_vols=jnp.array([0.3, 0.32]))


# ── Strip → SABR surface ─────────────────────────────────────────────


class TestOptionletSurface:
    @pytest.fixture
    def quote_matrix(self):
        """Smiled flat-vol quotes: 5 strikes × 3 cap maturities."""
        skew = jnp.array([0.06, 0.02, 0.0, 0.01, 0.04])[:, None]
        term = jnp.array([0.30, 0.33, 0.35])[None, :]
        return term + skew

    @pytest.fixture
    def strikes(self):
        return jnp.array([0.020, 0.028, 0.035, 0.045, 0.060])

    def test_builds_a_surface_from_quotes(
        self, curve, forward_curve, strikes, quote_matrix,
    ):
        caps = [_cap(n, 0.035) for n in (4, 8, 12)]
        surface = strip_optionlet_surface(
            caps, strikes, quote_matrix, curve,
            forward_curve=forward_curve,
        ).value
        assert surface.expiries.shape == (12,)
        assert surface.alphas.shape == (12,)
        vol = surface(jnp.asarray(0.035), surface.expiries[6])
        assert jnp.isfinite(vol) and float(vol) > 0.0

    def test_forwards_are_read_off_the_projection_curve(
        self, curve, forward_curve, strikes, quote_matrix,
    ):
        caps = [_cap(n, 0.035) for n in (4, 8, 12)]
        surface = strip_optionlet_surface(
            caps, strikes, quote_matrix, curve,
            forward_curve=forward_curve,
        ).value
        expected = optionlet_forwards(caps[-1], curve, forward_curve)
        assert jnp.allclose(surface.forwards, expected, atol=1e-12)

    def test_expiry_grid_matches_the_optionlet_grid(
        self, curve, strikes, quote_matrix,
    ):
        caps = [_cap(n, 0.035) for n in (4, 8, 12)]
        surface = strip_optionlet_surface(caps, strikes, quote_matrix, curve).value
        assert jnp.allclose(
            surface.expiries, optionlet_expiries(caps[-1], curve), atol=1e-12,
        )

    def test_smile_is_recovered_at_the_quoted_strikes(
        self, curve, strikes, quote_matrix,
    ):
        caps = [_cap(n, 0.035) for n in (4, 8, 12)]
        surface = strip_optionlet_surface(caps, strikes, quote_matrix, curve).value
        columns = [
            strip_caplet_vols(
                [eqx.tree_at(lambda c: c.strike, cap, strikes[k])
                 for cap in caps],
                curve,
                cap_vols=quote_matrix[k],
            ).value.vols
            for k in range(strikes.shape[0])
        ]
        stripped = jnp.stack(columns)  # (n_strikes, n_optionlets)
        expiry = surface.expiries[6]
        fitted = jnp.stack([surface(k, expiry) for k in strikes])
        # A 4-parameter SABR fit to 5 points: within a few vol points.
        assert float(jnp.max(jnp.abs(fitted - stripped[:, 6]))) < 0.02

    def test_rejects_mis_shaped_quote_matrix(self, curve, strikes):
        caps = [_cap(n, 0.035) for n in (4, 8, 12)]
        with pytest.raises(ValueError, match="cap_vol_matrix has shape"):
            strip_optionlet_surface(
                caps, strikes, jnp.zeros((3, 5)) + 0.3, curve,
            )


# ── The strip as a calibration level ─────────────────────────────────


class TestStripLevel:
    """The strip reports like every other level: labelled, checkable."""

    def test_result_contract(self, caps, curve, forward_curve):
        res = strip_caplet_vols(
            caps, curve, cap_vols=QUOTES, forward_curve=forward_curve
        ).check(max_residual=1e-12)
        assert res.problem == "root"
        assert res.labels == ("cap 1Y", "cap 2Y", "cap 3Y", "cap 4Y")
        assert bool(res.converged)
        # The level residual is the per-cap price error per unit notional.
        assert res.diagnostics.residuals.shape == (4, 1)
        assert float(res.condition_number()) < 1e3

    def test_impossible_bucket_is_named(self, caps, curve):
        """Anti-vacuity: a premium below the shorter cap's has no root."""
        good = strip_caplet_vols(caps, curve, cap_vols=QUOTES).value
        prem = jnp.stack([cap_price_from_strip(c, good, curve) for c in caps])
        prem = prem.at[2].set(0.9 * prem[1])
        res = strip_caplet_vols(caps, curve, cap_premiums=prem)
        assert not bool(res.converged)
        assert "cap 3Y" in res.failed_labels()
        assert "cap 1Y" not in res.failed_labels()
        with pytest.raises(CalibrationError, match="cap 3Y"):
            res.check()

    def test_strike_columns_equal_per_strike_strips(self, curve, forward_curve):
        caps3 = [_cap(n, 0.035) for n in (4, 8, 12)]
        strikes = jnp.array([0.025, 0.035, 0.045])
        quotes = jnp.array([[0.33, 0.35, 0.36], [0.30, 0.32, 0.33],
                            [0.29, 0.31, 0.32]])
        cols = strip_columns_level(
            "cols", caps3,
            LevelInputs(quotes=quotes, upstream=(curve, forward_curve)),
            lambda up: up, strikes=strikes,
        ).check()
        assert cols.labels == ("K=2.50%", "K=3.50%", "K=4.50%")
        for k in range(3):
            one = strip_caplet_vols(
                [eqx.tree_at(lambda c: c.strike, cap, strikes[k]) for cap in caps3],
                curve, cap_vols=quotes[k], forward_curve=forward_curve,
            ).value
            assert float(jnp.max(jnp.abs(cols.value.vols[k] - one.vols))) < 1e-12

    def test_surface_carries_the_strip_as_a_child(self, curve):
        caps3 = [_cap(n, 0.035) for n in (4, 8, 12)]
        strikes = jnp.array([0.02, 0.03, 0.04, 0.05])
        quotes = jnp.array([[0.36, 0.37, 0.38], [0.32, 0.33, 0.34],
                            [0.30, 0.31, 0.32], [0.31, 0.32, 0.33]])
        res = strip_optionlet_surface(caps3, strikes, quotes, curve)
        assert len(res.children) == 1
        assert res.children[0].name == "optionlet_surface.strip"
        assert res.children[0].diagnostics.converged.shape == (4, 3)
        res.check()
