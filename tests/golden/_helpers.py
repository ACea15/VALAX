"""Versioned golden-dataset comparison helpers.

A *golden* is a numerical reference output checked into the
repository under ``tests/golden/v{version}/{name}.npz`` and indexed
by a single manifest ``tests/golden/golden_manifest.json``.

Lifecycle
---------
- The first time a test references a golden, run with
  ``REGEN_GOLDEN=1`` to create the artifact and add a manifest entry.
- On subsequent runs (without ``REGEN_GOLDEN``), the helper loads the
  manifest, verifies the artifact's sha256, and compares the freshly
  computed value against the stored one with the supplied tolerances.
- A bytes drift without a ``version`` bump fails loudly.  To
  intentionally update a golden, bump its ``version=`` keyword in the
  call site, then regenerate.

Schema
------
``golden_manifest.json`` is a mapping ``{name: entry}`` where each
entry is::

    {
      "version": int,
      "sha256": str,
      "library_version": str,
      "jax_version": str,
      "shape": list[int],
      "dtype": str,
      "master_seed": int
    }
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

import valax


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GOLDEN_ROOT = _REPO_ROOT / "tests" / "golden"
_MANIFEST_PATH = _GOLDEN_ROOT / "golden_manifest.json"


def _regen_enabled() -> bool:
    return os.environ.get("REGEN_GOLDEN", "").lower() in ("1", "true", "yes")


def _load_manifest() -> dict[str, dict[str, Any]]:
    if not _MANIFEST_PATH.exists():
        return {}
    with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_manifest(manifest: dict[str, dict[str, Any]]) -> None:
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            dict(sorted(manifest.items())),
            fh,
            indent=2,
            sort_keys=True,
        )
        fh.write("\n")


def _artifact_path(name: str, version: int) -> Path:
    return _GOLDEN_ROOT / f"v{version}" / f"{name}.npz"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion to a numpy array."""
    arr = np.asarray(value)
    if arr.dtype == object:
        raise TypeError(
            f"Cannot golden-compare object array (dtype={arr.dtype}); "
            "convert to a numeric array first."
        )
    return arr


def assert_matches_golden(
    name: str,
    value: Any,
    *,
    version: int = 1,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    master_seed: int | None = None,
) -> None:
    """Assert ``value`` matches the stored golden artifact.

    Args:
        name: Stable identifier; becomes the manifest key and the
            artifact filename stem.
        value: Numerical value to compare (array-like, JAX or numpy).
        version: Golden version.  Bump when the numerical contract
            changes intentionally; pair with a ``REGEN_GOLDEN=1`` run.
        rtol: ``numpy.allclose`` relative tolerance.
        atol: ``numpy.allclose`` absolute tolerance.
        master_seed: Master seed embedded in the manifest (for audit).
            Pass ``None`` to leave the manifest's value unchanged.

    Behaviour:
        - ``REGEN_GOLDEN=1``: write artifact + manifest entry, return.
        - Otherwise: load + compare; fail if missing, drifted, or
          version mismatch.
    """
    arr = _to_numpy(value)
    manifest = _load_manifest()
    art_path = _artifact_path(name, version)

    if _regen_enabled():
        art_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(art_path, value=arr)
        manifest[name] = {
            "version": int(version),
            "sha256": _sha256_of_file(art_path),
            "library_version": valax.__version__,
            "jax_version": jax.__version__,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "master_seed": (
                int(master_seed) if master_seed is not None else -1
            ),
        }
        _save_manifest(manifest)
        return

    if name not in manifest:
        raise AssertionError(
            f"No golden registered for {name!r}.  Run "
            f"REGEN_GOLDEN=1 pytest -k <test> to create it."
        )
    entry = manifest[name]
    if entry["version"] != int(version):
        raise AssertionError(
            f"Golden {name!r} version mismatch: manifest has "
            f"{entry['version']}, test requested {version}.  Either "
            "align the version or regenerate."
        )
    if not art_path.exists():
        raise AssertionError(
            f"Manifest references missing artifact {art_path}.  "
            "Regenerate with REGEN_GOLDEN=1."
        )
    actual_sha = _sha256_of_file(art_path)
    if actual_sha != entry["sha256"]:
        raise AssertionError(
            f"Golden artifact {art_path} drifted (sha256 mismatch) "
            "but version was not bumped.  Bump version= and regenerate."
        )

    with np.load(art_path) as data:
        stored = data["value"]
    if not np.allclose(arr, stored, rtol=rtol, atol=atol):
        max_abs = float(np.max(np.abs(arr - stored)))
        raise AssertionError(
            f"Value for {name!r} does not match golden v{version}: "
            f"max abs diff = {max_abs:g} (rtol={rtol}, atol={atol})."
        )


__all__ = ["assert_matches_golden"]
