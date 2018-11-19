"""Specialized PODPAC nodes to access GPM data via OpenDAP
https://pmm.nasa.gov/data-access/downloads/gpm

"""

import re


import numpy as np
import traitlets as tl

# Internal dependencies
import podpac
from podpac.compositor import OrderedCompositor
from podpac.coordinates import Coordinates
from podpac.data import PyDAP, COMMON_DATA_DOC
from podpac import authentication, utils

# Optional dependencies
bs4 = utils.optional_import('bs4')

GPM_DOC = COMMON_DATA_DOC.copy()
GPM_DOC.update({
    'smap_date': 'str\n        SMAP date string',
    'np_date':   'np.datetime64\n        Numpy date object',
    'auth_class': 'Class used to make an authenticated session from a username and password',
    'auth_session' : 'Authenticated session used to make http requests using NASA Earth Data Login credentials',
    'base_url' : 'Url to NSIDC openDAP server',
    'layerkey': 'Key used to retrieve data from OpenDAP dataset.',
    'password': 'User\'s EarthData password',
    'username': 'User\'s EarthData username',
    'product': 'SMAP product name',
    'version': 'Version number for the SMAP product',
    'source_coordinates':
    """Returns the coordinates that uniquely describe each source

    Returns
    -------
    :class:`podpac.Coordinates`
        Coordinates that uniquely describe each source
    """,
    'keys':
    """Layers available in the OpenDAP dataset

    Returns
    -------
    List
        The list of available keys from the OpenDAP dataset. Any of these keys can be set as self.datakey.

    Notes
    -----
    This function assumes that all of the keys in the available dataset are the same for every file.
    """,
})

GPM_BASE_URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3'
GPM_DEFAULT_PRODUCT = '3IMERGHHE'

GPM_PRODUCT_DICT = {
    '3IMERGHHE': {
        'latkey': 'cell_lat',
        'lonkey': 'cell_lon',
        'rootdatakey': 'Analysis_Data_',
        'layerkey': '{rdk}sm_surface_analysis'
    },
    '3IMERGHHL': {
        'latkey': 'cell_lat',
        'lonkey': 'cell_lon',
        'rootdatakey': 'Analysis_Data_',
        'layerkey': '{rdk}sm_surface_analysis'
    },
    '3IMERGHH': {
        'latkey': 'cell_lat',
        'lonkey': 'cell_lon',
        'rootdatakey': 'Analysis_Data_',
        'layerkey': '{rdk}sm_surface_analysis'
    }
}


@utils.common_doc(GPM_DOC)
def _get_from_url(url, auth_session):
    """Helper function to get data from an NSIDC url with error checking.

    NOTE: Could be reused with SMAP
    
    Parameters
    ----------
    url : str
        URL to server
    auth_session: : class:`podpac.authentication.EarthDataSession`
        {auth_session}
    
    Returns
    -------
    :class:`requests.Response`
        requests response class
    
    Raises
    ------
    RuntimeError
        HTTP Error returned from Server
    """
    r = auth_session.get(url)

    if r.status_code != 200:

        # if unsuccessful, try removing opendap from the URL and trying again
        r = auth_session.get(url.replace('opendap/', ''))
        if r.status_code != 200:
            raise RuntimeError('HTTP error: <{}>\n{}'.format(r.status_code, r.text[:256]))

    return r

@utils.common_doc(GPM_DOC)
def _infer_GPM_product_version(product, base_url, auth_session):
    """Helper function to automatically infer the version number of GPM
    products in case user did not specify a version, or the version changed.

    NOTE: Could be reused with SMAP

    
    Parameters
    ------------
    product: str
        Name of the GPM product (e.g. one of :attr:`GPM_PRODUCT_DICT` keys)
    base_url: str
        {base_url}
    auth_session: :class:`podpac.authentication.EarthDataSession`
        {auth_session}
    """

    # get the listing of products from the base url
    r = _get_from_url(base_url, auth_session)

    # search the html for the product
    m = re.search(product, r.text)

    # return the following 2 characters (i.e. '05')
    # NOTE: this is different from SMAP by 1 character (+4)
    return int(r.text[m.end() + 1: m.end() + 3])

@utils.common_doc(GPM_DOC)
class GPMSource(PyDAP):
    """Accesses GPM data given a specific openDAP URL. This is the base class giving access to GPM data, and knows how
    to extract the correct coordinates and data keys.
    
    Attributes
    ----------
    auth_class : :class:`podpac.authentication.EarthDataSession`
        {auth_class}
    auth_session : :class:`podpac.authentication.EarthDataSession`
        {auth_session}
    date_file_url_re : SRE_Pattern
        Regular expression used to retrieve date from self.source (OpenDAP Url)
    date_time_file_url_re : SRE_Pattern
        Regular expression used to retrieve date and time from self.source (OpenDAP Url)
    layerkey : str
        {layerkey}
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

    @utils.common_doc(GPM_DOC)
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

    @utils.common_doc(GPM_DOC)
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

@utils.common_doc(GPM_DOC)
class GPM(OrderedCompositor):
    """Compositor of all available GPM date.
    
    Attributes
    ----------
    auth_class : :class:`podpac.authentication.EarthDataSession`
        {auth_class}
    auth_session : :class:`podpac.authentication.EarthDataSession`
        {auth_session}
    base_url : str
        {base_url}
    layerkey : str
        {layerkey}
    password : str
        {password}
    product : str
        {product}
    username : str
        {username}
    version : int
        {version}
    """

    base_url = tl.Unicode(GPM_BASE_URL).tag(attr=True)
    product = tl.Enum(list(GPM_PRODUCT_DICT.keys()), default_value=GPM_DEFAULT_PRODUCT).tag(attr=True)
    auth_class = tl.Type(authentication.EarthDataSession)
    username = tl.Unicode(None, allow_none=True)
    password = tl.Unicode(None, allow_none=True)

    # attributes with defaults
    auth_session = tl.Instance(authentication.EarthDataSession)
    version = tl.Int(allow_none=True).tag(attr=True)
    layerkey = tl.Unicode()

    @tl.default('version')
    def _detect_product_version(self):
        return _infer_GPM_product_version(self.product, self.base_url, self.auth_session)

    @tl.default('auth_session')
    def _auth_session_default(self):
        return self.auth_class(username=self.username, password=self.password, product_url=self.base_url)

    @tl.default('layerkey')
    def _layerkey_default(self):
        return GPM_PRODUCT_DICT[self.product]['layerkey']

    # @tl.observe('layerkey')
    # def _layerkey_change(self, change):
    #     if change['old'] != change['new'] and change['old'] != '':
    #         for s in self.sources:
    #             s.layerkey = change['new']

    @property
    def source(self):
        """The source is used for a unique name to cache GPM products.

        Returns
        -------
        str
            The GPM product name with version
        """
        # NOTE: this is off from SMAP by 1 version character (05 instead of 005)
        return 'GPM_{}.{:02d}'.format(self.product, self.version)

    @property
    def url(self):
        """The specific url to the GPM product 
        
        Returns
        -------
        str
            The GPM product url
        """
        return '/'.join([self.base_url, self.source])
    
    def get_available_years(self):
        """Return the available folder years
        """

        r = _get_from_url(self.url, self.auth_session)
        soup = bs4.BeautifulSoup(r.text, 'lxml')
        a = soup.find_all('a')

        # search for 4 consecutive numbers in the <a> tag contents
        regex = re.compile('[0-9]{4}')
        years = [regex.match(a.get_text()).group() for a in soup.find_all('a') if regex.match(a.get_text())]
        
        return years


    def get_available_times_dates(self):
        """Returns the available folder dates in the SMAP product

        Returns
        -------
        np.ndarray
            Array of dates in numpy datetime64 format
        list
            list of dates in SMAP date format

        Raises
        ------
        RuntimeError
            HTTP Error if resource can not be accessed (see :attr:`auth_class` EarthDataSession login credentials)
        """

        

        r = _get_from_url(url, self.auth_session)
        soup = bs4.BeautifulSoup(r.text, 'lxml')
        a = soup.find_all('a')
        regex = self.date_url_re
        times = []
        dates = []
        for aa in a:
            m = regex.match(aa.get_text())
            if m:
                times.append(np.datetime64(m.group().replace('.', '-')))
                dates.append(m.group())
        times.sort()
        dates.sort()
        return np.array(times), dates
