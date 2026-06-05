"""Stage 3 — chain validation: calibrated Heston surface → Asian option.

Stage 3 of the [QuantLib Validation Pyramid](
../../docs/architecture/quantlib-validation-pyramid.md).

**Currently a placeholder.** Blocked by roadmap item **HE-1**
(Andersen QE or full-truncation Heston discretisation): the existing
Euler-with-reflection scheme in
``valax/pricing/mc/paths.py::generate_heston_paths`` exhibits
``O(1/sqrt(n_steps))`` bias when QL's single-expiry Heston calibration
lands at Feller-violating parameters, and adding a path-dependent
payoff on top compounds the issue.

The chain pattern (calibration → exotic) is identical to
``test_exotics_on_sabr_surface_ql.py``; once HE-1 ships, this file
becomes:

  1. Synthesise a SABR smile from a ground truth.
  2. QL calibrates a Heston model to the smile.
  3. Adopt the fitted parameters into VALAX's ``HestonModel``.
  4. Both engines price an arithmetic Asian call via MC.
  5. Compare means within ``3 * combined_stderr``.

Tracked in the session log of the validation-pyramid plan doc.
"""

import pytest


pytestmark = pytest.mark.skip(
    reason=(
        "Blocked by HE-1 (Heston Euler bias under violated Feller). "
        "See docs/roadmap.md HE-series backlog."
    ),
)


def test_heston_asian_chain_placeholder():
    pass
