
# see test_array_coordinates1d.py
from podpac.core.coordinates.coordinates1d import Coordinates1d

class TestCoordinates1d(object):
    def test_common_api(self):
        c = Coordinates1d(name='lat')

        attrs = ['name', 'units', 'coord_ref_sys', 'ctype', 'segment_lengths', 'is_monotonic', 'is_descending', 'is_uniform',
                 'dims', 'udims', 'properties', 'coordinates', 'dtype', 'size', 'bounds', 'area_bounds', 'definition']

        for attr in attrs:
            try:
                getattr(c, attr)
            except NotImplementedError:
                pass

        try:
            c.from_definition({})
        except NotImplementedError:
            pass

        try:
            c.copy()
        except NotImplementedError:
            pass

        try:
            c.copy(name='lon', ctype='point')
        except NotImplementedError:
            pass

        try:
            c.select([])
        except NotImplementedError:
            pass

        try:
            c.select([], outer=True, return_indices=True)
        except NotImplementedError:
            pass

        try:
            c.intersect(c)
        except NotImplementedError:
            pass

        try:
            c.intersect(c, outer=True, return_indices=True)
        except NotImplementedError:
            pass

        assert c != None