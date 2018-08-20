"""
Podpac Module

Public API
See https://creare-com.github.io/podpac-docs/developer/contributing.html#public-api
for more information about import conventions

Attributes
----------
version_info : TYPE
    Description
"""

# Public API
from podpac.core.units import Units, UnitsDataArray, UnitsNode
from podpac.core.coordinate import (
    Coordinate, Coord, MonotonicCoord, UniformCoord, coord_linspace)
from podpac.core.node import Node, Style
from podpac.core.algorithm.algorithm import (
    Algorithm, Arithmetic, SinCoords)
from podpac.core.algorithm.stats import (
    Min, Max, Sum, Count, Mean, Median,
    Variance, StandardDeviation, Skew, Kurtosis)
from podpac.core.algorithm.coord_select import ExpandCoordinates
from podpac.core.algorithm.signal import (
    Convolution, SpatialConvolution, TimeConvolution)
from podpac.core.data import DataSource
from podpac.core.compositor import Compositor, OrderedCompositor
from podpac.core.pipeline import Pipeline, PipelineError

from podpac.settings import CACHE_DIR

# Organized submodules
# TODO

# Developer API
import podpac.core

# version handling
from podpac import version
__version__ = version.version()
version_info = version.VERSION_INFO
del version
