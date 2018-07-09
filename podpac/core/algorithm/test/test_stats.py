from __future__ import division, unicode_literals, print_function, absolute_import

import pytest
import numpy as np
import xarray as xr
import scipy.stats

from podpac.core.coordinate import Coordinate
from podpac.core.data.type import NumpyArray
from podpac.core.algorithm.stats import Min, Max, Sum, Count, Mean, Variance, Skew, Kurtosis, StandardDeviation
from podpac.core.algorithm.stats import Median, Percentile
from podpac.core.algorithm.stats import GroupReduce, DayOfYear

def setup_module():
    global coords, source, data
    coords = Coordinate(
        lat=(0, 1, 10),
        lon=(0, 1, 10),
        time=('2018-01-01', '2018-01-10', '1,D'),
        order=['lat', 'lon', 'time'])

    a = np.random.random(coords.shape)
    a[3, 0, 0] = np.nan
    a[0, 3, 0] = np.nan
    a[0, 0, 3] = np.nan
    source = NumpyArray(source=a, native_coordinates=coords)
    data = source.execute(coords)

class TestReduce(object):
    """ Tests the Reduce class """
    
    def test_invalid_dims(self):
        # any reduce node would do here
        node = Min(source=source)
        
        # valid dim
        node.execute(coords, {'dims': 'lat'})
        
        # invalid dim
        with pytest.raises(ValueError):
            node.execute(coords, {'dims': 'alt'})

    def test_auto_chunk(self):
        # any reduce node would do here
        node = Min(source=source)
        node.execute(coords, {'iter_chunk_size': 'auto'})

    def test_not_implemented(self):
        from podpac.core.algorithm.stats import Reduce

        node = Reduce(source=source)
        with pytest.raises(NotImplementedError):
            node.execute(coords)

    def test_chunked_fallback(self):
        from podpac.core.algorithm.stats import Reduce

        class First(Reduce):
            def reduce(self, x):
                return x.isel(**{dim:0 for dim in self.dims})

        node = First(source=source)
        
        # use reduce function
        output = node.execute(coords, {'dims': 'time'})
        
        # fall back on reduce function with warning
        with pytest.warns(UserWarning):
            output_chunked = node.execute(coords, {'dims': 'time', 'iter_chunk_size': 100})

        # should be the same
        xr.testing.assert_allclose(output, output_chunked)

class BaseTests(object):
    """ Common tests for Reduce subclasses """

    def test_full(self):
        output = self.node.execute(coords)
        # NOTE: using the numpy allclose because xarray allclose is also checking attrs (which should be ignored)
        # xr.testing.assert_allclose(output, self.expected_full)
        np.testing.assert_allclose(output.data, self.expected_full.data)

        output = self.node.execute(coords, {'dims': coords.dims})
        np.testing.assert_allclose(output.data, self.expected_full.data)

        output = self.node.execute(coords, {'iter_chunk_size': 100})
        np.testing.assert_allclose(output.data, self.expected_full.data)

        output = self.node.execute(coords, {'iter_chunk_size': 20})
        np.testing.assert_allclose(output.data, self.expected_full.data)

    def test_lat_lon(self):
        output = self.node.execute(coords, {'dims': ['lat', 'lon']})
        np.testing.assert_allclose(output.data, self.expected_latlon.data)

        output = self.node.execute(coords, {'dims': ['lat', 'lon'], 'iter_chunk_size': 100})
        np.testing.assert_allclose(output.data, self.expected_latlon.data)

    def test_time(self):
        output = self.node.execute(coords, {'dims': 'time'})
        np.testing.assert_allclose(output.data, self.expected_time.data)

        output = self.node.execute(coords, {'dims': 'time', 'iter_chunk_size': 100})
        np.testing.assert_allclose(output.data, self.expected_time.data)

        output = self.node.execute(coords, {'dims': 'time', 'iter_chunk_size': 10})
        np.testing.assert_allclose(output.data, self.expected_time.data)

class TestMin(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Min(source=source)
        cls.expected_full = data.min()
        cls.expected_latlon = data.min(dim=['lat', 'lon'])
        cls.expected_time = data.min(dim='time')

class TestMax(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Max(source=source)
        cls.expected_full = data.max()
        cls.expected_latlon = data.max(dim=['lat', 'lon'])
        cls.expected_time = data.max(dim='time')

class TestSum(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Sum(source=source)
        cls.expected_full = data.sum()
        cls.expected_latlon = data.sum(dim=['lat', 'lon'])
        cls.expected_time = data.sum(dim='time')

class TestCount(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Count(source=source)
        cls.expected_full = np.isfinite(data).sum()
        cls.expected_latlon = np.isfinite(data).sum(dim=['lat', 'lon'])
        cls.expected_time = np.isfinite(data).sum(dim='time')

class TestMean(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Mean(source=source)
        cls.expected_full = data.mean()
        cls.expected_latlon = data.mean(dim=['lat', 'lon'])
        cls.expected_time = data.mean(dim='time')

class TestVariance(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Variance(source=source)
        cls.expected_full = data.var()
        cls.expected_latlon = data.var(dim=['lat', 'lon'])
        cls.expected_time = data.var(dim='time')

class TestStandardDeviation(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = StandardDeviation(source=source)
        cls.expected_full = data.std()
        cls.expected_latlon = data.std(dim=['lat', 'lon'])
        cls.expected_time = data.std(dim='time')

class TestSkew(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Skew(source=source)
        n, m, l = data.shape
        cls.expected_full = xr.DataArray(scipy.stats.skew(data.data.reshape(n*m*l), nan_policy='omit'))
        cls.expected_latlon = scipy.stats.skew(data.data.reshape((n*m, l)), axis=0, nan_policy='omit')
        cls.expected_time = scipy.stats.skew(data, axis=2, nan_policy='omit')

class TestKurtosis(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Kurtosis(source=source)
        n, m, l = data.shape
        cls.expected_full = xr.DataArray(scipy.stats.kurtosis(data.data.reshape(n*m*l), nan_policy='omit'))
        cls.expected_latlon = scipy.stats.kurtosis(data.data.reshape((n*m, l)), axis=0, nan_policy='omit')
        cls.expected_time = scipy.stats.kurtosis(data, axis=2, nan_policy='omit')

class TestMedian(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Median(source=source)
        cls.expected_full = data.median()
        cls.expected_latlon = data.median(dim=['lat', 'lon'])
        cls.expected_time = data.median(dim='time')

@pytest.mark.skip("TODO")
class TestPercentile(BaseTests):
    @classmethod
    def setup_class(cls):
        cls.node = Percentile(source=source)
        # TODO can we replace dims_axes with reshape (or vice versa)

@pytest.mark.skip("TODO")
class TestGroupReduce(object):
    def test(self):
        pass

@pytest.mark.skip("TODO")
class TestDayOfYear(object):
    def test(self):
        pass