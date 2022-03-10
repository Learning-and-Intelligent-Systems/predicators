"""Shared configurations for pytest.

See https://docs.pytest.org/en/6.2.x/fixture.html.
"""
import pytest

longrun = pytest.mark.skipif("not config.getoption('longrun')")


def pytest_addoption(parser):
    """Enable a command line flag for running tests decorated with @longrun."""
    parser.addoption('--longrun',
                     action='store_true',
                     dest="longrun",
                     default=False,
                     help="enable tests decorated with @longrun")
