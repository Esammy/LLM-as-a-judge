"""Smoke tests for the package surface."""

import judgekit


def test_version_is_exposed() -> None:
    assert judgekit.__version__ == "0.1.0"


def test_package_declares_its_exports() -> None:
    assert "__version__" in judgekit.__all__
