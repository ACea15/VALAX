"""Stage 3 — chain validation: caplet vols → SABR per-expiry → cap strip.

Stage 3 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

**Currently a placeholder.** Two non-blockers:

1. The per-expiry SABR-on-caplet-vols pipeline is not yet wrapped in
   a public VALAX helper. The ingredients exist
   (``valax.calibration.calibrate_sabr``, ``valax.surfaces.SABRVolSurface``,
   ``valax.pricing.analytic.caplets.cap_price_black76``); they have not
   yet been composed into a single "build SABR cube → price cap"
   convenience.
2. QL's ``ql.CapHelper`` / ``ql.PiecewiseATMOptionletVolatility`` plus
   ``ql.OptionletStripper1`` calibration path requires fixture work
   (cap-floor parity instruments, schedule construction).

Once the above are in place, this file mirrors
``test_exotics_on_sabr_surface_ql.py``:

  1. Synthesise per-expiry caplet vol smiles via the synthetic generator.
  2. SABR-calibrate per expiry on both sides (or QL calibrates and
     VALAX adopts).
  3. Build a cap as a strip of caplets.
  4. Both engines price each caplet via Black-76, reading the
     per-expiry SABR vol at the cap strike.
  5. Compare cap NPVs at ``rel < 1e-6``.

Tracked in the session log of the validation-pyramid plan doc.
"""

import pytest


pytestmark = pytest.mark.skip(
    reason=(
        "Needs a `build_sabr_caplet_surface` convenience on the VALAX "
        "side and the QL OptionletStripper fixture. Tracked in the "
        "QuantLib Validation Pyramid plan doc."
    ),
)


def test_cap_strip_chain_placeholder():
    pass
