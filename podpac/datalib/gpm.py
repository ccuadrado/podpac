"""Specialized PODPAC nodes to access GPM data via OpenDAP
https://pmm.nasa.gov/data-access/downloads/gpm

"""


import numpy as np
import traitlets as tl

# Internal dependencies
import podpac
from podpac.core.coordinates import Coordinates
from podpac.core.data.types import PyDAP
from podpac.core import authentication
from podpac.core.utils import common_doc
from podpac.core.data.datasource import COMMON_DATA_DOC

COMMON_DOC = COMMON_DATA_DOC.copy()
COMMON_DOC.update({
    'smap_date': 'str\n        SMAP date string',
    'np_date':   'np.datetime64\n        Numpy date object',
    'auth_class': ('EarthDataSession (Class object)\n        Class used to make an authenticated session from a'
               ' username and password (both are defined in base class)'),
    'auth_session' : ('Instance(EarthDataSession)\n        Authenticated session used to make http requests using'
                  'NASA Earth Data Login credentials'),
    'base_url' : 'str\n        Url to nsidc openDAP server', 
    'layerkey': ('str\n        Key used to retrieve data from OpenDAP dataset. This specifies the key used to retrieve '
             'the data'),
    'password': 'User\'s EarthData password',
    'username': 'User\'s EarthData username',
    'product': 'SMAP product name',
    'version': 'Version number for the SMAP product',
    'source_coordinates': '''Returns the coordinates that uniquely describe each source

    Returns
    -------
    :class:`podpac.Coordinates`
        Coordinates that uniquely describe each source''',
    'keys': '''Available layers that are in the OpenDAP dataset

    Returns
    -------
    List
        The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey.

    Notes
    -----
    This function assumes that all of the keys in the available dataset are the same for every file.
    ''',
})

GPM_BASE_URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3/'

GPM_VERSION = '.05'
GPM_PRODUCT_DICT = {
    #'<Product>.ver': ['latkey',               'lonkey',                     'rootdatakey',                       'layerkey'
    '3IMERGHHE':   ['cell_lat',             'cell_lon',                   'Analysis_Data_',                    '{rdk}sm_surface_analysis'],
    '3IMERGHHL':   ['cell_lat',             'cell_lon',                   'Geophysical_Data_',                 '{rdk}sm_surface'],
    '3IMERGHH':    ['{rdk}latitude',        '{rdk}longitude',             'Soil_Moisture_Retrieval_Data_',     '{rdk}soil_moisture']
}


@common_doc(COMMON_DOC)
class GPMSource(PyDAP):
    """Accesses GPM data given a specific openDAP URL. This is the base class giving access to GPM data, and knows how
    to extract the correct coordinates and data keys for the soil moisture data.

    Attributes
    ----------
    auth_class : {auth_class}
    auth_session : {auth_session}
    date_file_url_re : SRE_Pattern
        Regular expression used to retrieve date from self.source (OpenDAP Url)
    date_time_file_url_re : SRE_Pattern
        Regular expression used to retrieve date and time from self.source (OpenDAP Url)
    layerkey : str
        Key used to retrieve data from OpenDAP dataset. This specifies the key used to retrieve the data
    nan_vals : list
        List of values that should be treated as no-data (these are replaced by np.nan)
    rootdatakey : str
        String the prepends every or most keys for data in the OpenDAP dataset
    """

    auth_session = tl.Instance(authentication.EarthDataSession)
    auth_class = tl.Type(authentication.EarthDataSession)

    @tl.default('auth_session')
    def _auth_session_default(self):
        session = self.auth_class(
            username=self.username, password=self.password, hostname_regex=GPM_BASE_URL)
        # check url
        try:
            session.get(GPM_BASE_URL)
        except Exception as e:
            print("Unknown exception: ", e)
        return session

    #date_url_re = re.compile('[0-9]{4}\.[0-9]{2}\.[0-9]{2}')
    date_time_file_url_re = re.compile('[0-9]{8}T[0-9]{6}')
    date_file_url_re = re.compile('[0-9]{8}')

    rootdatakey = tl.Unicode()
    @tl.default('rootdatakey')
    def _rootdatakey_default(self):
        return SMAP_PRODUCT_MAP.sel(product=self.product,
                                    attr='rootdatakey').item()

    layerkey = tl.Unicode()
    @tl.default('layerkey')
    def _layerkey_default(self):
        return SMAP_PRODUCT_MAP.sel(
            product=self.product,
            attr='layerkey').item()

    nan_vals = [-9999.0]

    @property
    def product(self):
        """Returns the SMAP product from the OpenDAP Url

        Returns
        -------
        str
            {product}
        """
        src = self.source.split('/')
        return src[src.index('SMAP')+1].split('.')[0]
    
    @property
    def version(self):
        """Returns the SMAP product version from the OpenDAP Url

        Returns
        -------
        int
            {version}
        """
        src = self.source.split('/')
        return int(src[src.index('SMAP')+1].split('.')[1])
        

    @tl.default('datakey')
    def _datakey_default(self):
        return self.layerkey.format(rdk=self.rootdatakey)

    @property
    def latkey(self):
        """The key used to retrieve the latitude

        Returns
        -------
        str
            OpenDap dataset key for latitude
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='latkey') \
               .item().format(rdk=self.rootdatakey)

    @property
    def lonkey(self):
        """The key used to retrieve the latitude

        Returns
        -------
        str
            OpenDap dataset key for longitude
        """
        return SMAP_PRODUCT_MAP.sel(product=self.product, attr='lonkey').item().format(rdk=self.rootdatakey)

    @common_doc(COMMON_DOC)
    def get_native_coordinates(self):
        """{get_native_coordinates}
        """
        try:
            return self.load_cached_obj('native.coordinates')
        except:
            pass
        times = self.get_available_times()
        ds = self.dataset
        lons = np.array(ds[self.lonkey][:, :])
        lats = np.array(ds[self.latkey][:, :])
        lons[lons == self.nan_vals[0]] = np.nan
        lats[lats == self.nan_vals[0]] = np.nan
        lons = np.nanmean(lons, axis=0)
        lats = np.nanmean(lats, axis=1)
        coords = podpac.Coordinates([times, lats, lons], dims=['time', 'lat', 'lon'])
        self.cache_obj(coords, 'native.coordinates')
        return coords

    def get_available_times(self):
        """Retrieve the available times from the SMAP file. This is primarily based on the filename, but some products 
        have multiple times stored in a single file.

        Returns
        -------
        np.ndarray(dtype=np.datetime64)
            Available times in the SMAP source
        """
        m = self.date_time_file_url_re.search(self.source)
        if not m:
            m = self.date_file_url_re.search(self.source)
        times = m.group()
        times = smap2np_date(times)
        if 'SM_P_' in self.source:
            times = times + np.array([6, 18], 'timedelta64[h]')
        return times

    @common_doc(COMMON_DOC)
    def get_data(self, coordinates, coordinates_index):
        """{get_data}
        """
        # We actually ignore the time slice
        s = tuple([slc for d, slc in zip(coordinates.dims, coordinates_index)
                   if 'time' not in d])
        if 'SM_P_' in self.source:
            d = self.create_output_array(coordinates)
            am_key = self.layerkey.format(rdk=self.rootdatakey + 'AM')
            pm_key = self.layerkey.format(rdk=self.rootdatakey + 'PM') + '_pm'

            try:
                t = self.native_coordinates.coords['time'][0]
                d.loc[dict(time=t)] = np.array(self.dataset[am_key][s])
            except: 
                pass

            try: 
                t = self.native_coordinates.coords['time'][1]
                d.loc[dict(time=t)] = np.array(self.dataset[pm_key][s])
            except: 
                pass

        else:
            data = np.array(self.dataset[self.datakey][s])
            d = self.create_output_array(coordinates, data=data.reshape(coordinates.shape))

        return d


