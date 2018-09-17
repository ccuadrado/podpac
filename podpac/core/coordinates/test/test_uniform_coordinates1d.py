
from datetime import datetime

import pytest
import traitlets as tl
import numpy as np
from numpy.testing import assert_equal

from podpac.core.coordinates.array_coordinates1d import ArrayCoordinates1d
from podpac.core.coordinates.uniform_coordinates1d import UniformCoordinates1d

class TestUniformCoordinatesCreation(object):
    def test_numerical(self):
        # ascending
        c = UniformCoordinates1d(0, 50, 10)
        a = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        assert c.start == 0
        assert c.stop == 50
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0, 50], dtype=float))
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(50, 0, -10)
        a = np.array([50, 40, 30, 20, 10, 0], dtype=float)
        assert c.start == 50
        assert c.stop == 0
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0, 50], dtype=float))
        assert c.size == 6
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_numerical_inexact(self):
        # ascending
        c = UniformCoordinates1d(0, 49, 10)
        a = np.array([0, 10, 20, 30, 40], dtype=float)
        assert c.start == 0
        assert c.stop == 49
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([0, 40], dtype=float))
        assert c.size == 5
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(50, 1, -10)
        a = np.array([50, 40, 30, 20, 10], dtype=float)
        assert c.start == 50
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([10, 50], dtype=float))
        assert c.dtype == float
        assert c.size == a.size
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True
        
    def test_datetime(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D')
        a = np.array(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-04')
        assert c.step == np.timedelta64(1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True
        
        # descending
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D')
        a = np.array(['2018-01-04', '2018-01-03', '2018-01-02', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-04')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_inexact(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D')
        a = np.array(['2018-01-01', '2018-01-03', '2018-01-05']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-06')
        assert c.step == np.timedelta64(2, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D')
        a = np.array(['2018-01-06', '2018-01-04', '2018-01-02']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-06')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-2, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_month_step(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-04-01', '1,M')
        a = np.array(['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-04-01')
        assert c.step == np.timedelta64(1, 'M')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d('2018-04-01', '2018-01-01', '-1,M')
        a = np.array(['2018-04-01', '2018-03-01', '2018-02-01', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-04-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'M')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_year_step(self):
        # ascending, exact
        c = UniformCoordinates1d('2018-01-01', '2021-01-01', '1,Y')
        a = np.array(['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2021-01-01')
        assert c.step == np.timedelta64(1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending, exact
        c = UniformCoordinates1d('2021-01-01', '2018-01-01', '-1,Y')
        a = np.array(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2021-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

        # ascending, inexact (two cases)
        c = UniformCoordinates1d('2018-01-01', '2021-04-01', '1,Y')
        a = np.array(['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2021-04-01')
        assert c.step == np.timedelta64(1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        c = UniformCoordinates1d('2018-04-01', '2021-01-01', '1,Y')
        a = np.array(['2018-04-01', '2019-04-01', '2020-04-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-04-01')
        assert c.stop == np.datetime64('2021-01-01')
        assert c.step == np.timedelta64(1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending, inexact (two cases)
        c = UniformCoordinates1d('2021-01-01', '2018-04-01', '-1,Y')
        a = np.array(['2021-01-01', '2020-01-01', '2019-01-01', '2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2021-01-01')
        assert c.stop == np.datetime64('2018-04-01')
        assert c.step == np.timedelta64(-1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

        c = UniformCoordinates1d('2021-04-01', '2018-01-01', '-1,Y')
        a = np.array(['2021-04-01', '2020-04-01', '2019-04-01', '2018-04-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2021-04-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'Y')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_numerical_size(self):
        # ascending
        c = UniformCoordinates1d(0, 10, size=20)
        assert c.start == 0
        assert c.stop == 10
        assert c.step == 10/19.
        assert_equal(c.coordinates, np.linspace(0, 10, 20))
        assert_equal(c.bounds, np.array([0, 10], dtype=float))
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == False
        assert c.is_uniform == True

        # descending
        c = UniformCoordinates1d(10, 0, size=20)
        assert c.start == 10
        assert c.stop == 0
        assert c.step == -10/19.
        assert_equal(c.coordinates, np.linspace(10, 0, 20))
        assert_equal(c.bounds, np.array([0, 10], dtype=float))
        assert c.size == 20
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == True
        assert c.is_uniform == True

    def test_datetime_size(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=10)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-10')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == False

        # descending
        c = UniformCoordinates1d('2018-01-10', '2018-01-01', size=10)
        assert c.start == np.datetime64('2018-01-10')
        assert c.stop == np.datetime64('2018-01-01')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 10
        assert c.dtype == np.datetime64
        assert c.is_descending == True

    @pytest.mark.skip("spec uncertain")
    def test_datetime_size_inexact(self):
        # ascending
        c = UniformCoordinates1d('2018-01-01', '2018-01-10', size=20)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-10')
        assert_equal(c.bounds, [np.datetime64('2018-01-01'), np.datetime64('2018-01-10')])
        assert c.size == 20
        assert c.dtype == np.datetime64
        assert c.is_descending == False

    def test_size_floating_point_error(self):
        c = UniformCoordinates1d(50.619, 50.62795, size=30)
        assert c.size == 30

    def test_numerical_singleton(self):
        # positive step
        c = UniformCoordinates1d(1, 1, 10)
        a = np.array([1], dtype=float)
        assert c.start == 1
        assert c.stop == 1
        assert c.step == 10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([1, 1], dtype=float))
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

        # negative step
        c = UniformCoordinates1d(1, 1, -10)
        a = np.array([1], dtype=float)
        assert c.start == 1
        assert c.stop == 1
        assert c.step == -10
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, np.array([1, 1], dtype=float))
        assert c.size == 1
        assert c.dtype == float
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

    def test_datetime_singleton(self):
        # positive step
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D')
        a = np.array(['2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[0, -1]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True
        
        # negative step
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D')
        a = np.array(['2018-01-01']).astype(np.datetime64)
        assert c.start == np.datetime64('2018-01-01')
        assert c.stop == np.datetime64('2018-01-01')
        assert c.step == np.timedelta64(-1, 'D')
        assert_equal(c.coordinates, a)
        assert_equal(c.bounds, a[[-1, 0]])
        assert c.size == a.size
        assert c.dtype == np.datetime64
        assert c.is_monotonic == True
        assert c.is_descending == None
        assert c.is_uniform == True

    def test_invalid_init(self):
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 0)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, -10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(50, 0, 10)
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, '2018-01-01', 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', 50, 10)

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', '2018-01-02', 10)
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0., '2018-01-01', '1,D')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-01', 50, '1,D')

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, '1,D')

        with pytest.raises(ValueError):
            UniformCoordinates1d('a', 50, 10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 'b', 10)

        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 'a')

        with pytest.raises(TypeError):
            UniformCoordinates1d()

        with pytest.raises(TypeError):
            UniformCoordinates1d(0)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, 10, size=6)

        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size=20.)
        
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 10, size='string')

        with pytest.raises(TypeError):
            UniformCoordinates1d('2018-01-10', '2018-01-01', size='1,D')

    def test_extents(self):
        # default None
        c = UniformCoordinates1d(0, 50, 10)
        assert c.extents is None

        # numerical
        c = UniformCoordinates1d(0, 50, 10, extents=[0, 55])
        assert_equal(c.extents, np.array([0, 55], dtype=float))

        # datetime
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', extents=['2018-01-01', '2019-01-06'])
        assert_equal(c.extents, np.array(['2018-01-01', '2019-01-06']).astype(np.datetime64))

        # invalid (ctype=point)
        with pytest.raises(TypeError):
            UniformCoordinates1d(0, 50, 10, ctype='point', extents=[0, 55])

        # invalid (wrong dtype)
        with pytest.raises(ValueError):
            UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', extents=[0, 55])
        
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 10, extents=['2018-01-01', '2019-03-01'])

        # invalid (shape)
        with pytest.raises(ValueError):
            UniformCoordinates1d(0, 50, 10, extents=[0])

class TestUniformCoordinatesProperties(object):
    def test_properties(self):
        # without extents
        c = UniformCoordinates1d(0, 50, 10, ctype='point')
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys'])

        # with extents
        c = UniformCoordinates1d(0, 50, 10, extents=[0, 50])
        assert isinstance(c.properties, dict)
        assert set(c.properties.keys()) == set(['ctype', 'coord_ref_sys', 'extents'])

    def test_area_bounds_point(self):
        # numerical, ascending/descending and exact/inexact
        c = UniformCoordinates1d(0, 50, 10, ctype='point')
        assert_equal(c.area_bounds, np.array([0, 50], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, ctype='point')
        assert_equal(c.area_bounds, np.array([0, 50], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, ctype='point')
        assert_equal(c.area_bounds, np.array([0, 40], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, ctype='point')
        assert_equal(c.area_bounds, np.array([10, 50], dtype=float))

        # datetime, ascending/descending and exact/inexact
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='point')
        assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))

    def test_area_bounds_explicit_extents(self):
        c = UniformCoordinates1d(0, 50, 10, extents=[-10, 10])
        assert_equal(c.area_bounds, np.array([-10, 10], dtype=float))
        
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', extents=['2016-01-01', '2021-01-01'])
        assert_equal(c.area_bounds, np.array(['2016-01-01', '2021-01-01']).astype(np.datetime64))
        
    def test_area_bounds_left(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, ctype='left')
        assert_equal(c.area_bounds, np.array([0, 60], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, ctype='left')
        assert_equal(c.area_bounds, np.array([0, 60], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, ctype='left')
        assert_equal(c.area_bounds, np.array([0, 50.0], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, ctype='left')
        assert_equal(c.area_bounds, np.array([10, 60.0], dtype=float))
        c = UniformCoordinates1d(0, 0, 10, ctype='left')
        assert_equal(c.area_bounds, np.array([0, 10], dtype=float))
        c = UniformCoordinates1d(0, 0, -10, ctype='left')
        assert_equal(c.area_bounds, np.array([0, 10], dtype=float))

        # TODO
        # # datetime, ascending/descending and exact/inexact/singleton
        # c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', ctype='left')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))

    def test_area_bounds_right(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 50], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 50], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 40], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, ctype='right')
        assert_equal(c.area_bounds, np.array([0, 50], dtype=float))
        c = UniformCoordinates1d(0, 0, 10, ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 0], dtype=float))
        c = UniformCoordinates1d(0, 0, -10, ctype='right')
        assert_equal(c.area_bounds, np.array([-10, 0], dtype=float))

        # TODO
        # # datetime, ascending/descending and exact/inexact/singleton
        # c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        # c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', ctype='right')
        # assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))

    def test_area_bounds_midpoint_numerical(self):
        # numerical, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d(0, 50, 10, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 55], dtype=float))
        c = UniformCoordinates1d(50, 0, -10, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 55], dtype=float))
        c = UniformCoordinates1d(0, 49, 10, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 45], dtype=float))
        c = UniformCoordinates1d(50, 9, -10, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([5, 55], dtype=float))
        c = UniformCoordinates1d(0, 0, 10, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 5], dtype=float))
        c = UniformCoordinates1d(0, 0, -10, ctype='midpoint')
        assert_equal(c.area_bounds, np.array([-5, 5], dtype=float))

    @pytest.mark.skip('TODO')
    def test_area_bounds_midpoint_datetime(self):
        # datetime, ascending/descending and exact/inexact/singleton
        c = UniformCoordinates1d('2018-01-01', '2018-01-04', '1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-04', '2018-01-01', '-1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-06', '2,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-05']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-06', '2018-01-01', '-2,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-02', '2018-01-06']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))
        c = UniformCoordinates1d('2018-01-01', '2018-01-01', '-1,D', ctype='midpoint')
        assert_equal(c.area_bounds, np.array(['2018-01-01', '2018-01-04']).astype(np.datetime64))

class TestUniformCoordinatesIndexing(object):
    def test_index(self):
        c = UniformCoordinates1d(0, 50, 10, name='lat', ctype='point')
        
        # int
        c2 = c[2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 20
        assert c2.step == 10

        c2 = c[-2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 40
        assert c2.stop == 40
        assert c2.step == 10

        # slice
        c2 = c[:2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 10
        assert c2.step == 10

        c2 = c[2:]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 50
        assert c2.step == 10
        
        c2 = c[::2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 50
        assert c2.step == 20
        
        c2 = c[1:-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 10
        assert c2.stop == 40
        assert c2.step == 10

        c2 = c[-3:5]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 40
        assert c2.step == 10
        
        c2 = c[::-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 0
        assert c2.step == -10
        
        # ordered array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([0, 10, 30], dtype=float))

        c2 = c[[3, 1, 0]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([30, 10, 0], dtype=float))

        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([0, 30, 10], dtype=float))

        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([0, 10, 20, 40], dtype=float))

        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

    def test_index_descending(self):
        c = UniformCoordinates1d(50, 0, -10, name='lat', ctype='point')
        
        # int
        c2 = c[2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 30
        assert c2.step == -10

        c2 = c[-2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 10
        assert c2.stop == 10
        assert c2.step == -10

        # slice
        c2 = c[:2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 40
        assert c2.step == -10

        c2 = c[2:]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 30
        assert c2.stop == 0
        assert c2.step == -10
        
        c2 = c[::2]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 50
        assert c2.stop == 0
        assert c2.step == -20
        
        c2 = c[1:-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 40
        assert c2.stop == 10
        assert c2.step == -10

        c2 = c[-3:5]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 20
        assert c2.stop == 10
        assert c2.step == -10
        
        c2 = c[::-1]
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert c2.start == 0
        assert c2.stop == 50
        assert c2.step == 10
        
        # ordered array
        c2 = c[[0, 1, 3]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([50, 40, 20], dtype=float))
        
        c2 = c[[3, 1, 0]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([20, 40, 50], dtype=float))
        
        c2 = c[[0, 3, 1]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([50, 20, 40], dtype=float))
        # boolean array
        c2 = c[[True, True, True, False, True, False]]
        assert isinstance(c2, ArrayCoordinates1d)
        assert c2.name == c.name
        assert c2.properties == c.properties
        assert_equal(c2.coordinates, np.array([50, 40, 30, 10], dtype=float))
        
        # invalid
        with pytest.raises(IndexError):
            c[0.3]

        with pytest.raises(IndexError):
            c[10]

@pytest.mark.skip('TODO')
class TestUniformCoordinatesSelection(object):
    def test_select_ascending(self):
        c = UniformCoordinates1d(20., 70., 10.)
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), UniformCoordinates1d)
        assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
        assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
        # partial, above
        s = c.select([45, 100], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 50.
        assert s.stop == 70.
        assert s.step == 10.
        
        # partial, below
        s = c.select([5, 55], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 20.
        assert s.stop == 50.
        assert s.step == 10.

        # partial, inner
        s = c.select([30., 60.], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 30.
        assert s.stop == 60.
        assert s.step == 10.

        # partial, inner exact
        s = c.select([35., 55.], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 40.
        assert s.stop == 50.
        assert s.step == 10.
        
        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_descending(self):
        c = UniformCoordinates1d(70., 20., -10.)
        
        # full and empty selection type
        assert isinstance(c.select([0, 100]), UniformCoordinates1d)
        assert isinstance(c.select([100, 200]), ArrayCoordinates1d)
        assert isinstance(c.select([0, 5]), ArrayCoordinates1d)
        
        # partial, above
        s = c.select([45, 100], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 70.
        assert s.stop == 50.
        assert s.step == -10.
        
        # partial, below
        s = c.select([5, 55], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 50.
        assert s.stop == 20.
        assert s.step == -10.

        # partial, inner
        s = c.select([30., 60.], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 60.
        assert s.stop == 30.
        assert s.step == -10.

        # partial, inner exact
        s = c.select([35., 55.], pad=0)
        assert isinstance(s, UniformCoordinates1d)
        assert s.start == 50.
        assert s.stop == 40.
        assert s.step == -10.
        
        # partial, none
        s = c.select([52, 55], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

        # partial, backwards bounds
        s = c.select([70, 30], pad=0)
        assert isinstance(s, ArrayCoordinates1d)
        assert_equal(s.coordinates, [])

    def test_select_ind_ascending(self):
        c = UniformCoordinates1d(20., 70., 10.)
        
        # partial, above
        s = c.select([45, 100], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 60., 70.])
        
        # partial, below
        s = c.select([5, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [20., 30., 40., 50])

        # partial, inner
        s = c.select([30., 60.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [30., 40., 50., 60])

        # partial, inner exact
        s = c.select([35., 55.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [40., 50.])
        
        # partial, none
        s = c.select([52, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

        # partial, backwards bounds
        s = c.select([70, 30], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

    def test_select_ind_descending(self):
        c = UniformCoordinates1d(70., 20., -10.)
        
        # partial, above
        s = c.select([45, 100], ind=True, pad=0)
        assert_equal(c.coordinates[s], [70., 60., 50.])
        
        # partial, below
        s = c.select([5, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 40., 30., 20.])

        # partial, inner
        s = c.select([30., 60.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [60., 50., 40., 30.])

        # partial, inner exact
        s = c.select([35., 55.], ind=True, pad=0)
        assert_equal(c.coordinates[s], [50., 40.])
        
        # partial, none
        s = c.select([52, 55], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

        # partial, backwards bounds
        s = c.select([70, 30], ind=True, pad=0)
        assert_equal(c.coordinates[s], [])

    def test_intersect(self):
        # MonotonicCoordinates1d other
        a = UniformCoordinates1d(10., 60., 10.)
        b = UniformCoordinates1d(45., 95., 5.)
        
        ab = a.intersect(b, pad=0)
        assert isinstance(ab, UniformCoordinates1d)
        assert ab.start == 50.
        assert ab.stop == 60.
        assert ab.step == 10.

        ba = b.intersect(a, pad=0)
        assert isinstance(ba, UniformCoordinates1d)
        assert ba.start == 45.
        assert ba.stop == 60.
        assert ba.step == 5.

        # ArrayCoordinates1d other
        c = ArrayCoordinates1d([40., 70., 50.,])
        assert isinstance(a.intersect(c), UniformCoordinates1d)
        
        # MonotonicCoordinates1d
        m = MonotonicCoordinates1d([40., 50., 70.])
        assert isinstance(a.intersect(m), UniformCoordinates1d)

@pytest.mark.skip("TODO")
class TestUniformCoordinatesConcatenation(object):
    def test_concat(self):
        a = UniformCoordinates1d(30., 60., 10.)
        b = UniformCoordinates1d(60., 30., -10.)

        # empty other
        r = a.concat(ArrayCoordinates1d())
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == a.start
        assert r.stop == a.stop
        assert r.step == a.step

        r = a._concat(ArrayCoordinates1d())
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == a.start
        assert r.stop == a.stop
        assert r.step == a.step

        # both ascending -> UniformCoordinates1d
        r = a.concat(UniformCoordinates1d(70., 100, 10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 30.
        assert r.stop == 100.
        assert r.step == 10.

        r = a.concat(UniformCoordinates1d(0., 20, 10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 0.
        assert r.stop == 60.
        assert r.step == 10.

        # both descending -> UniformCoordinates1d
        r = b.concat(UniformCoordinates1d(100., 70, -10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 100.
        assert r.stop == 30.
        assert r.step == -10.

        r = b.concat(UniformCoordinates1d(20., 0, -10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 60.
        assert r.stop == 0.
        assert r.step == -10.

        # mismatched -> UniformCoordinates1d
        r = b.concat(UniformCoordinates1d(70., 100, 10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 100.
        assert r.stop == 30.
        assert r.step == -10.

        r = b.concat(UniformCoordinates1d(0., 20, 10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 60.
        assert r.stop == 0.
        assert r.step == -10.

        r = a.concat(UniformCoordinates1d(100., 70, -10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 30.
        assert r.stop == 100.
        assert r.step == 10.

        r = a.concat(UniformCoordinates1d(20., 0, -10.))
        assert isinstance(r, UniformCoordinates1d)
        assert r.start == 0.
        assert r.stop == 60.
        assert r.step == 10.

        # separated, both ascending -> MonotonicCoordinates1d
        r = a.concat(UniformCoordinates1d(80., 100, 10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 80., 90., 100.])

        r = a.concat(UniformCoordinates1d(0., 10, 10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [0., 10., 30., 40., 50., 60.])

        # separated, both descendeng -> MonotonicCoordinates1d
        r = b.concat(UniformCoordinates1d(100., 80., -10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [100., 90., 80., 60., 50., 40., 30.])

        r = b.concat(UniformCoordinates1d(10., 0, -10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [60., 50., 40., 30., 10., 0.])

        # separated, mismatched -> MonotonicCoordinates1d
        r = b.concat(UniformCoordinates1d(80., 100, 10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [100., 90., 80., 60., 50., 40., 30.])

        r = b.concat(UniformCoordinates1d(0., 10, 10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [60., 50., 40., 30., 10., 0.])

        # separated, both descendeng -> MonotonicCoordinates1d
        r = a.concat(UniformCoordinates1d(100., 80., -10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 80., 90., 100.])

        r = a.concat(UniformCoordinates1d(10., 0, -10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [0., 10., 30., 40., 50., 60.])

        # mismatched step -> MonotonicCoordinates1d
        r = a.concat(UniformCoordinates1d(70., 100, 5.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 70., 75., 80., 85., 90., 95., 100.])

        # not aligned -> MonotonicCoordinates1d
        r = a.concat(UniformCoordinates1d(65., 100, 10.))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 65., 75., 85., 95.])

        # overlap -> ArrayCoordinates1d
        r = a.concat(UniformCoordinates1d(50., 100, 10))
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 50., 60., 70., 80., 90., 100.])

        # MonotonicCoordinates1d other
        r = a.concat(MonotonicCoordinates1d([75, 80, 90]))
        assert isinstance(r, MonotonicCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 75, 80, 90])

        r = a.concat(MonotonicCoordinates1d([55, 75, 80, 90]))
        assert isinstance(r, ArrayCoordinates1d)
        assert_equal(r.coordinates, [30., 40., 50., 60., 55, 75, 80, 90])
        
        # ArrayCoordinates1d other
        r = a.concat(ArrayCoordinates1d([75, 90, 80]))
        assert_equal(r.coordinates, [30., 40., 50., 60., 75, 90, 80])
        assert isinstance(r, ArrayCoordinates1d)

    def test_concat_equal(self):
        # empty other
        a = UniformCoordinates1d(30., 60., 10.)
        a.concat(ArrayCoordinates1d(), inplace=True)
        assert a.start == 30.
        assert a.stop == 60.
        assert a.step == 10.

        a = UniformCoordinates1d(30., 60., 10.)
        a._concat_equal(ArrayCoordinates1d())
        assert a.start == 30.
        assert a.stop == 60.
        assert a.step == 10.

        # both ascending
        a = UniformCoordinates1d(30., 60., 10.)
        a.concat(UniformCoordinates1d(70., 100, 10.), inplace=True)
        assert a.start == 30.
        assert a.stop == 100.
        assert a.step == 10.

        a = UniformCoordinates1d(30., 60., 10.)
        a.concat(UniformCoordinates1d(0., 20, 10.), inplace=True)
        assert a.start == 0.
        assert a.stop == 60.
        assert a.step == 10.

        # both descending
        b = UniformCoordinates1d(60., 30., -10.)
        b.concat(UniformCoordinates1d(100., 70, -10.), inplace=True)
        assert b.start == 100.
        assert b.stop == 30.
        assert b.step == -10.

        b = UniformCoordinates1d(60., 30., -10.)
        b.concat(UniformCoordinates1d(20., 0, -10.), inplace=True)
        assert b.start == 60.
        assert b.stop == 0.
        assert b.step == -10.

        # mismatched
        b = UniformCoordinates1d(60., 30., -10.)
        b.concat(UniformCoordinates1d(70., 100, 10.), inplace=True)
        assert b.start == 100.
        assert b.stop == 30.
        assert b.step == -10.

        b = UniformCoordinates1d(60., 30., -10.)
        b.concat(UniformCoordinates1d(0., 20, 10.), inplace=True)
        assert b.start == 60.
        assert b.stop == 0.
        assert b.step == -10.

        a = UniformCoordinates1d(30., 60., 10.)
        a.concat(UniformCoordinates1d(100., 70, -10.), inplace=True)
        assert a.start == 30.
        assert a.stop == 100.
        assert a.step == 10.

        a = UniformCoordinates1d(30., 60., 10.)
        a.concat(UniformCoordinates1d(20., 0, -10.), inplace=True)
        assert a.start == 0.
        assert a.stop == 60.
        assert a.step == 10.

        # separated -> ValueError
        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(80., 100, 10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(0., 10, 10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoordinates1d(100., 80., -10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoordinates1d(10., 0, -10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoordinates1d(80., 100, 10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            b.concat(UniformCoordinates1d(0., 10, 10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(100., 80., -10.), inplace=True)

        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(10., 0, -10.), inplace=True)

        # mismatched step -> ValueError
        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(70., 100, 5.), inplace=True)

        # not aligned -> ValueError
        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(65., 100, 10.), inplace=True)

        # overlap -> ValueError
        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(ValueError):
            a.concat(UniformCoordinates1d(50., 100, 10), inplace=True)

        # non UniformCoordinates1d other -> TypeError
        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(TypeError):
            a.concat(MonotonicCoordinates1d([75, 80, 90]), inplace=True)

        # ArrayCoordinates1d other
        a = UniformCoordinates1d(30., 60., 10.)
        with pytest.raises(TypeError):
            a.concat(ArrayCoordinates1d([75, 90, 80]), inplace=True)

@pytest.mark.skip("TODO")
class TestUniformCoordinatesAddition(object):
    def test_add(self):
        # numerical
        c = UniformCoordinates1d(20., 60., 10.)
        c2 = c.add(1)
        assert isinstance(c2, UniformCoordinates1d)
        assert c2.start == 21.
        assert c2.stop == 61.
        assert c2.step == 10.

        # simple datetime
        t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
        t2d = t.add('2,D')
        assert isinstance(t2d, UniformCoordinates1d)
        assert t2d.start == np.datetime64('2018-01-03')
        assert t2d.stop == np.datetime64('2018-01-12')
        assert t2d.step == np.timedelta64(1, 'D')

        t2d = t.add('-2,D')
        assert isinstance(t2d, UniformCoordinates1d)
        assert t2d.start == np.datetime64('2017-12-30')
        assert t2d.stop == np.datetime64('2018-01-08')
        assert t2d.step == np.timedelta64(1, 'D')

        # nominal datetime
        t2m = t.add('2,M')
        assert isinstance(t2m, UniformCoordinates1d)
        assert t2m.start == np.datetime64('2018-03-01')
        assert t2m.stop == np.datetime64('2018-03-10')
        assert t2m.step == np.timedelta64(1, 'D')

        t2y = t.add('2,Y')
        assert isinstance(t2y, UniformCoordinates1d)
        assert t2y.start == np.datetime64('2020-01-01')
        assert t2y.stop == np.datetime64('2020-01-10')
        assert t2y.step == np.timedelta64(1, 'D')

    def test_add_equal(self):
        # numerical
        c = UniformCoordinates1d(20., 60., 10.)
        c.add(1, inplace=True)
        assert c.start == 21.
        assert c.stop == 61.
        assert c.step == 10.

        # simple datetime
        t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
        t.add('2,D', inplace=True)
        assert t.start == np.datetime64('2018-01-03')
        assert t.stop == np.datetime64('2018-01-12')
        assert t.step == np.timedelta64(1, 'D')

        t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
        t.add('-2,D', inplace=True)
        assert t.start == np.datetime64('2017-12-30')
        assert t.stop == np.datetime64('2018-01-08')
        assert t.step == np.timedelta64(1, 'D')

        # nominal datetime
        t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
        t.add('2,M', inplace=True)
        assert t.start == np.datetime64('2018-03-01')
        assert t.stop == np.datetime64('2018-03-10')
        assert t.step == np.timedelta64(1, 'D')

        t = UniformCoordinates1d('2018-01-01', '2018-01-10', '1,D')
        t.add('2,Y', inplace=True)
        assert t.start == np.datetime64('2020-01-01')
        assert t.stop == np.datetime64('2020-01-10')
        assert t.step == np.timedelta64(1, 'D')