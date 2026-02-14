"""Smoke tests to verify package installation and basic imports."""

import hydro_param


def test_version():
    assert hydro_param.__version__ == "0.1.0dev0"
