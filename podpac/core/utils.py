"""
Utils Summary
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import sys
import json
import functools
import importlib
from collections import OrderedDict
import logging
from copy import deepcopy
import traitlets as tl
import numpy as np

# create log for module
_log = logging.getLogger(__name__)

import podpac
from . import settings

def common_doc(doc_dict):
    """ Decorator: replaces commond fields in a function docstring

    Parameters
    -----------
    doc_dict : dict
        Dictionary of parameters that will be used to format a doctring. e.g. func.__doc__.format(**doc_dict)
    """
    def _decorator(func):
        if func.__doc__ is None:
            return func

        func.__doc__ = func.__doc__.format(**doc_dict)
        return func
    return _decorator

def cached_property(func):
    """Summary

    Parameters
    ----------
    func : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    @property
    @functools.wraps(func)
    def f(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        cache_name = '_cached_' + func.__name__
        if hasattr(self, cache_name):
            cache_val = getattr(self, cache_name)
        else:
            cache_val = None
        if cache_val is not None:
            return cache_val
        cache_val = func(self)
        setattr(self, cache_name, cache_val)
        return cache_val
    return f


def clear_cache(self, change, attrs):
    """Summary

    Parameters
    ----------
    change : TYPE
        Description
    attrs : TYPE
        Description
    """
    if (change['old'] is None and change['new'] is not None) or \
               np.any(np.array(change['old']) != np.array(change['new'])):
        for attr in attrs:
            setattr(self, '_cached_' + attr, None)


def trait_is_defined(obj, trait):
    """Utility method to determine if trait is defined on object without
    call to default (@tl.default)

    Parameters
    ----------
    object : object
        Class with traits
    trait : str
        Class property to investigate

    Returns
    -------
    bool
        True if the trait exists on the object and is defined
        False if the trait does not exist on the object or the trait is not defined
    """
    return obj.has_trait(trait) and trait in obj._trait_values


def create_logfile(filename=settings.settings['LOG_FILE_PATH'],
                   level=logging.INFO,
                   format='[%(asctime)s] %(name)s.%(funcName)s[%(lineno)d] - %(levelname)s - %(message)s'
                   ):
    """Convience method to create a log file that only logs
    podpac related messages
    
    Parameters
    ----------
    filename : str, optional
        Filename of the log file. Defaults to ``podpac.log``
    level : int, optional
        Log level to use (0 - 50). Defaults to ``logging.INFO`` (20)
        See https://docs.python.org/3/library/logging.html#levels
    format : str, optional
        String format for log messages.
        See https://docs.python.org/3/library/logging.html#logrecord-attributes
        for creating format. Default is: 
        format='[%(asctime)s] %(name)s.%(funcName)s[%(lineno)d] - %(levelname)s - %(message)s'
    
    Returns
    -------
    logging.Logger, logging.Handler, logging.Formatter
        Returns the constructed logger, handler, and formatter for the log file
    """
    # get logger for podpac module only
    log = logging.getLogger('podpac')
    log.setLevel(level)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)    

    # create a file handler
    handler = logging.FileHandler(filename, 'a')

    # create a logging format
    # see https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    log.addHandler(handler)

    # insert log from utils into logfile
    _log.info('Logging to file {}'.format(filename))

    return log, handler, formatter


if sys.version < '3.6':
    # for Python 2 and Python < 3.6 compatibility
    class OrderedDictTrait(tl.Dict):
        """ OrderedDict trait """

        default_value = OrderedDict()
        
        def validate(self, obj, value):
            if value == {}:
                value = OrderedDict()
            elif not isinstance(value, OrderedDict):
                raise tl.TraitError(
                    "The '%s' trait of an %s instance must be an OrderedDict, but a value of %s %s was specified" % (
                        self.name, obj.__class__.__name__, value, type(value)))
            super(OrderedDictTrait, self).validate(obj, value)
            return value

else:
    OrderedDictTrait = tl.Dict

class ArrayTrait(tl.TraitType):
    """ A coercing numpy array trait. """

    def __init__(self, ndim=None, shape=None, dtype=None, dtypes=None, *args, **kwargs):
        if ndim is not None and shape is not None and len(shape) != ndim:
            raise ValueError("Incompatible ndim and shape (ndim=%d, shape=%s)" % (ndim, shape))
        if dtype is not None and not isinstance(dtype, type):
            if dtype not in np.typeDict:
                raise ValueError("Unknown dtype '%s'" % dtype)
            dtype = np.typeDict[dtype]
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        super(ArrayTrait, self).__init__(*args, **kwargs)

    def validate(self, obj, value):
        # coerce type
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except:
                raise tl.TraitError(
                    "The '%s' trait of an %s instance must be an np.ndarray, but a value of %s %s was specified" % (
                        self.name, obj.__class__.__name__, value, type(value)))

        # ndim
        if self.ndim is not None and self.ndim != value.ndim:
            raise tl.TraitError(
                "The '%s' trait of an %s instance must have ndim %d, but a value with ndim %d was specified" % (
                    self.name, obj.__class__.__name__, self.ndim, value.ndim))

        # shape
        if self.shape is not None and self.shape != value.shape:
            raise tl.TraitError(
                "The '%s' trait of an %s instance must have shape %s, but a value %s with shape %s was specified" % (
                    self.name, obj.__class__.__name__, self.shape, value, value.shape))

        # dtype
        if self.dtype is not None:
            try:
                value = value.astype(self.dtype)
            except:
                raise tl.TraitError(
                    "The '%s' trait of an %s instance must have dtype %s, but a value with dtype %s was specified" % (
                        self.name, obj.__class__.__name__, self.dtype, value.dtype))

        return value

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # podpac Coordinates objects
        if isinstance(obj, podpac.Coordinates):
            return obj.definition

        # podpac Coordinates objects
        elif isinstance(obj, podpac.Node):
            return obj.definition

        # numpy arrays
        elif isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.datetime64):
                return obj.astype(str).tolist()
            elif np.issubdtype(obj.dtype, np.timedelta64):
                f = np.vectorize(podpac.core.coordinates.utils.make_timedelta_string)
                return f(obj).tolist()
            elif np.issubdtype(obj.dtype, np.number):
                return obj.tolist()
        
        # datetime64
        elif isinstance(obj, np.datetime64):
            return obj.astype(str)

        # timedelta64
        elif isinstance(obj, np.timedelta64):
            return podpac.core.coordinates.utils.make_timedelta_string(obj)
        
        # Interpolator
        elif obj in podpac.core.data.interpolation.INTERPOLATORS:
            interpolater_class = deepcopy(obj)
            interpolator = interpolater_class()
            return interpolator.definition

        # default
        return json.JSONEncoder.default(self, obj)
