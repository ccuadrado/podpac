"""
Podpac Module

Public API
See https://podpac.org/developer/contributing.html#public-api
for more information about import conventions

Attributes
----------
version_info : OrderedDict
    Dict with keys MAJOR, MINOR, HOTFIX depicting version
"""


# Monkey match os.makedirs for Python 2 compatibility
import sys
import os
_osmakedirs = os.makedirs
def makedirs(name, mode=511, exist_ok=False):
    try: 
        _osmakedirs(name, mode)
    except Exception as e:
        if exist_ok:
            pass
        else:
            raise e
if sys.version_info.major == 2:
    makedirs.__doc__ = os.makedirs.__doc__
    os.makedirs = makedirs
else:
    del _osmakedirs
del os
del sys

# Public API
from podpac.core.settings import settings
from podpac.core.coordinates import Coordinates, crange, clinspace
from podpac.core.node import Node, NodeException
import podpac.core.authentication as authentication
import podpac.core.utils as utils


# Organized submodules
# These files are simply wrappers to create a curated namespace of podpac modules
from podpac import algorithm
from podpac import data
from podpac import interpolators
from podpac import coordinates
from podpac import compositor
from podpac import pipeline
from podpac import datalib   # handles imports in datalib/__init__.py

## Developer API
from podpac import core

# version handling
from podpac import version
__version__ = version.version()
version_info = version.VERSION_INFO
