"""Tests for installed package metadata."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import wxenergy as wx


def test_runtime_version_matches_distribution_metadata() -> None:
    assert wx.__version__ == version("wxenergy")


def test_typed_package_marker_is_installed() -> None:
    marker = Path(wx.__file__).with_name("py.typed")
    assert marker.is_file()
