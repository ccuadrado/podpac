"""
Test Setup
"""

import pytest
from podpac.core.settings import settings

def pytest_addoption(parser):
    """Add command line option to pytest
    Note you MUST invoke test as `pytest podpac --ci` to use these options. 
    Using only `pytest --ci` will result in an error

    Parameters
    ----------
    parser : TYPE

    """
    # config option for when we're running tests on ci
    parser.addoption("--ci", action='store_true', default=False)

def pytest_configure(config):
    """Configuration before all tests are run

    Parameters
    ----------
    config : TYPE

    """

    pass

def pytest_unconfigure(config):
    """Configuration after all tests are run

    Parameters
    ----------
    config : TYPE

    """
    pass

original_default_cache = settings['DEFAULT_CACHE']

def pytest_sessionstart(session):
    settings['DEFAULT_CACHE'] = []

def pytest_sessionfinish(session, exitstatus):
    settings['DEFAULT_CACHE'] = original_default_cache