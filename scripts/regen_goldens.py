"""Regenerate every golden artifact in ``tests/golden``.

Usage::

    REGEN_GOLDEN=1 python scripts/regen_goldens.py

The script just runs the test session with ``REGEN_GOLDEN=1`` set so
every call to ``assert_matches_golden`` overwrites the artifact and
manifest entry.  Tests that read goldens then become trivially green
on the *next* run (the contract is "the bytes you write to disk are
the bytes you read back").
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    env["REGEN_GOLDEN"] = "1"
    print("[regen_goldens] Running pytest with REGEN_GOLDEN=1 ...")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_market/test_golden.py",
        "tests/test_market/",
        "-q",
        "-x",
    ]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
