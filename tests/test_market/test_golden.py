"""Tests for the golden-dataset harness itself.

These tests don't reference any production golden file — they create
a temporary manifest inside the test, exercise the helper, and tear
it down.  Production goldens live in ``tests/golden/v*/`` and are
referenced from the data-shape tests in this directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.golden import _helpers as helpers


@pytest.fixture
def temp_golden(tmp_path, monkeypatch):
    """Redirect the helper's manifest/artifact paths to a tmp dir."""
    fake_root = tmp_path / "golden"
    fake_root.mkdir()
    monkeypatch.setattr(helpers, "_GOLDEN_ROOT", fake_root, raising=True)
    monkeypatch.setattr(
        helpers,
        "_MANIFEST_PATH",
        fake_root / "golden_manifest.json",
        raising=True,
    )
    return fake_root


class TestGoldenHarness:
    def test_regen_writes_artifact_and_manifest(
        self, temp_golden, monkeypatch
    ):
        monkeypatch.setenv("REGEN_GOLDEN", "1")
        value = np.array([1.0, 2.0, 3.0])
        helpers.assert_matches_golden(
            "test.array", value, version=1, master_seed=42
        )
        manifest = json.loads(
            (temp_golden / "golden_manifest.json").read_text()
        )
        assert "test.array" in manifest
        assert manifest["test.array"]["version"] == 1
        assert manifest["test.array"]["master_seed"] == 42
        assert (temp_golden / "v1" / "test.array.npz").exists()

    def test_replay_matches_golden(self, temp_golden, monkeypatch):
        monkeypatch.setenv("REGEN_GOLDEN", "1")
        value = np.array([1.0, 2.0, 3.0])
        helpers.assert_matches_golden("test.array", value, version=1)

        monkeypatch.delenv("REGEN_GOLDEN")
        # Re-asserting with the same value passes.
        helpers.assert_matches_golden("test.array", value, version=1)

    def test_replay_with_drift_fails(self, temp_golden, monkeypatch):
        monkeypatch.setenv("REGEN_GOLDEN", "1")
        helpers.assert_matches_golden(
            "test.array", np.array([1.0, 2.0]), version=1
        )

        monkeypatch.delenv("REGEN_GOLDEN")
        with pytest.raises(AssertionError, match="does not match golden"):
            helpers.assert_matches_golden(
                "test.array", np.array([1.0, 2.5]), version=1
            )

    def test_missing_golden_fails(self, temp_golden, monkeypatch):
        monkeypatch.delenv("REGEN_GOLDEN", raising=False)
        with pytest.raises(AssertionError, match="No golden registered"):
            helpers.assert_matches_golden(
                "test.missing", np.array([1.0]), version=1
            )

    def test_version_mismatch_fails(self, temp_golden, monkeypatch):
        monkeypatch.setenv("REGEN_GOLDEN", "1")
        helpers.assert_matches_golden(
            "test.x", np.array([1.0]), version=1
        )

        monkeypatch.delenv("REGEN_GOLDEN")
        with pytest.raises(AssertionError, match="version mismatch"):
            helpers.assert_matches_golden(
                "test.x", np.array([1.0]), version=2
            )

    def test_sha_drift_without_version_bump_fails(
        self, temp_golden, monkeypatch
    ):
        monkeypatch.setenv("REGEN_GOLDEN", "1")
        helpers.assert_matches_golden(
            "test.x", np.array([1.0]), version=1
        )

        # Tamper with the artifact directly.
        path = temp_golden / "v1" / "test.x.npz"
        np.savez(path, value=np.array([99.0]))

        monkeypatch.delenv("REGEN_GOLDEN")
        with pytest.raises(AssertionError, match="drifted"):
            helpers.assert_matches_golden(
                "test.x", np.array([1.0]), version=1
            )
