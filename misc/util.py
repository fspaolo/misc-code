"""
Module with some high level utility functions.

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import os
import re
import numpy as np
import scipy as sp
import tables as tb
import datetime as dt

# definition of Table structures for HDF5 files

class TimeSeries(tb.IsDescription):
    sat_name = tb.StringCol(20, pos=1)
    ref_time = tb.StringCol(20, pos=2)
    date = tb.StringCol(20, pos=3)
    year = tb.Int32Col(pos=4)
    month = tb.Int32Col(pos=5)
    dh_mean = tb.Float64Col(pos=6)
    dh_error = tb.Float64Col(pos=7)
    dg_mean = tb.Float64Col(pos=8)
    dg_error = tb.Float64Col(pos=9)
    n_ad = tb.Int32Col(pos=10)
    n_da = tb.Int32Col(pos=11)


class TimeSeriesGrid(tb.IsDescription):
    sat_name = tb.StringCol(20, pos=1)
    ref_time = tb.StringCol(20, pos=2)
    date = tb.StringCol(20, pos=3)
    year = tb.Int32Col(pos=4)
    month = tb.Int32Col(pos=5)


class SecsToDateTime(object):
    """
    Converts `seconds since epoch` to `datetime` (i.e., year, month, day).

    secs : 1D array, decimal seconds.
    since_year : int, ref_epoch = <since_year>-Jan-1 00:00:00 is assumed.
    since_epoch : tuple, especifies ref_epoch as (YYYY, MM, DD, hh, mm, ss).

    Notes
    -----
    1. Matlab uses as time reference the year 0000, and Python 
       `datetime` uses the year 0001.
    2. utc85 (or ESA-time) is seconds since 1985-1-1 00:00:00,
       ICESat-time is seconds since 2000-1-1 12:00:00,
       secs00 is seconds since 2000-1-1 00:00:00.

    """
    def __init__(self, secs=0, since_year=1985, since_epoch=None):
        if np.ndim(secs) > 0:
            self.secs = np.asarray(secs)
        else:
            self.secs = secs  

        if since_epoch is None:
            # <since_year>-Jan-1 00:00:00
            ref_epoch = dt.date(since_year, 1, 1)
        else:
            # not working !!!!!!!!!!!!!!!!!!!!!
            ref_epoch = dt.datetime(since_epoch)    

        # ref_epoch in days since 0001-Jan-1 00:00:00
        ref_epoch_in_days = mpl.date2num(ref_epoch)  

        # secs/86400 -> frac days -> date
        frac_days = self.secs / (24*60*60.)
        self._datenum = ref_epoch_in_days + frac_days
        self._dates = mpl.num2date(self._datenum)

    def datenum(self, matlab=False):
        if matlab:
            # frac days since 0000-Jan-1 00:00:00
            return self._datenum + 366.
        else:
            # frac days since 0001-Jan-1 00:00:00
            return self._datenum

    def dates(self):
        return self._dates

    def years(self):
        return np.array([d.year for d in self._dates])

    def months(self):
        return np.array([d.month for d in self._dates])
        
    def days(self):
        return np.array([d.day for d in self._dates])

    def ymdhms(self):
        return np.array([(d.year, d.month, d.day, d.hour,
            d.minute, d.second) for d in self._dates])


class CircularList(list):
    """
    A list that wraps around instead of throwing an index error.

    Simple, perhaps incomplete implementation of a Circular List in 
    Python. Subclasses list and overrides __getitem__. The only 
    special behavior is that attempts to access indices which are out 
    of bounds will wrap around - accessing mylist(len(mylist)) should 
    return the first item in the list instead of an IndexError Slice 
    operations must still be 'in bounds' First tries list's 
    __getitem__. If that is not successful, it converts the index key 
    to an integer, then calculates the appropriate 'in bounds' index 
    and returns whatever is stored there. If converting the key to 
    integer fails, TypeError is raised.
    
    Works like a regular list:
    >>> cl = CircularList([1,2,3])
    >>> cl
    [1, 2, 3]
    >>> cl[0]
    1
    >>> cl[-1]
    3
    >>> cl[2]
    3

    Except wraps around:
    >>> cl[3]
    1
    >>> cl[-4]
    3
    
    Slices work
    >>> cl[0:2]
    [1, 2]
    
    but only in range.
    """
    def __getitem__(self, key):
        
        # try normal list behavior
        try:
            return super(CircularList, self).__getitem__(key)
        except IndexError:
            pass
        # key can be either integer or slice object,
        # only implementing int now.
        try:
            index = int(key)
            index = index % self.__len__()
            return super(CircularList, self).__getitem__(index)
        except ValueError:
            raise TypeError


def linear_fit(x, y, return_coef=False):
    """
    Fit a straight-line by Ordinary Least Squares.

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    if len(ind) < 2: 
        return [np.nan, np.nan]
    x, y = x[ind], y[ind]
    A = np.ones((len(x), 2))
    A[:,0] = x
    m, c = np.linalg.lstsq(A, y)[0]
    if return_coef:
        return (m, c)
    else:
        x_val = np.linspace(x.min(), x.max(), 200)
        y_fit = m*x_val + c
        return (x_val, y_fit)


def linear_fit_robust(x, y, return_coef=False):
    """
    Fit a straight-line by robust regression (M-estimate).

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    import scikits.statsmodels.api as sm
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    if len(ind) < 2: 
        return [np.nan, np.nan]
    x, y = x[ind], y[ind]
    X = sm.add_constant(x, prepend=False)
    y_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    y_fit = y_model.fit()
    if return_coef:
        if len(y_fit.params) < 2: return (y_fit.params[0], 0.)
        else: return y_fit.params[:]
    else:
        return (x, y_fit.fittedvalues)


def poly_fit(x, y, order=1, return_coef=False):
    """
    Fit a polynomial of order `order` to data points `x,y`.
    """
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    if len(ind) < 3: 
        return [np.nan, np.nan]
    x, y = x[ind], y[ind]
    coef = np.polyfit(x, y, order)
    if return_coef:
        return coef
    else:
        x_val = np.linspace(x.min(), x.max(), 200)
        y_fit = np.polyval(coef, x_val)
        return (x_val, y_fit)


def spline_interp(x, y, smooth=0.01):
    """
    Interpolate data using cubic spline of given smoothness.
    smooth : smoothness factor
    """
    from scipy.interpolate import splrep, splev
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    x, y = x[ind], y[ind]

    # find the knot points
    tck = splrep(x, y, s=smooth)

    # evaluate spline on interpolated points
    x_val = np.linspace(x.min(), x.max(), 200)
    y_fit = splev(x_val, tck)
    return (x_val, y_fit)


def get_size(arr):
    """
    Get the size in MB of a Numpy or PyTables object.

    parameters
    ----------
    arr : 1D/2D Numpy or PyTables Array.
    """
    try:
        m, n = arr.shape
    except:
        m, n = arr.shape, 1
    num_elem = m*n
    item_size = arr.dtype.itemsize
    return (item_size*num_elem/1e6)


def check_if_can_be_loaded(data, max_size=512):
    """
    Check if PyTables Array can be loaded in memory.
    """
    if get_size(data) > max_size:
        msg = 'data is larger than %d MB, not loading in-memory!' \
            % max_size
        raise MemoryError(msg)
    else:
        return True


def check_if_need_to_save(data, max_size=128):
    """
    Check when data in memory need to be flushed on disk.
    """
    data = np.asarray(data)
    if get_size(data) > max_size:
        return True
    else:
        return False


def _get_season(year, month, return_month=2):
    """
    Returns the first, second or third month of the 3-month 
    season-block, and update the `year` when needed.

    year, month : int
    return_month : 1, 2 or 3
    """
    if return_month != 1 and return_month != 2 and return_month != 3:
        raise IOError('`return_month` must be: 1, 2 or 3')
    MAM = [3, 4, 5]      # Mar/Apr/May -> Fall SH 
    JJA = [6, 7, 8]      # Jun/Jul/Aug -> winter SH
    SON = [9, 10, 11]    # Sep/Oct/Nov -> Spring SH
    DJF = [12, 1, 2]     # Dec/Jan/Feb -> summer SH
    return_month -= 1
    if month in MAM:
        return year, MAM[return_month]
    elif month in JJA:
        return year, JJA[return_month]
    elif month in SON:
        return year, SON[return_month]
    elif month in DJF:
        if month == 12 and return_month > 0:
            year += 1
        return year, DJF[return_month]
    else:
        print 'not a valid month from 1 to 12!'
        return None, None


def get_season(year, month, return_month=2):
    """
    Apply `_get_season` to a scalar or sequence. See `_get_season`.
    """
    if not np.iterable(year) or not np.iterable(month):
        year = np.asarray([year])
        month = np.asarray([month])
    ym = np.asarray([_get_season(y, m, return_month) for y, m in zip(year, month)])
    return ym[:,0], ym[:,1]


def box(region, npts=None):
    """
    Generate a box given a `region` coords = (W,E,S,N).
    """
    west, east, south, north = region
    if npts:
        n = int(npts/4.)
        x = np.empty(n*4, 'f8')
        y = np.empty(n*4, 'f8')
        lons = np.linspace(west, east, n)
        lats = np.linspace(south, north, n)
        x[:n] = lons[:]
        y[:n] = north
        x[n:n*2] = east
        y[n:n*2] = lats[::-1]
        x[n*2:n*3] = lons[::-1]
        y[n*2:n*3] = south
        x[n*3:] = west 
        y[n*3:] = lats[:]
    else:
        x = np.array([west, east, east, west, west])
        y = np.array([north, north, south, south, north])
    return [x, y]


def lon_180_360(lon, region):
    """
    Convert lon from 180 to 360 (or vice-verse) according to `region`.
    """
    l, r, b, t = region
    if l < 0:
        lon[lon>180] -= 360  # 360 -> 180
    elif r > 180:
        lon[lon<0] += 360    # 180 -> 360
    return lon


def ll2xy(lon, lat, slat=71, slon=0, hemi='s', units='km'):
    """
    Convert from 'lon,lat' to polar stereographic 'x,y'.
 
    This function converts from geodetic latitude and longitude to
    polar stereographic 'x,y' coordinates for the polar regions. The 
    equations are from Snyder, J.P., 1982, Map Projections Used by 
    the U.S. Geological Survey, Geological Survey Bulletin 1532, U.S. 
    Government Printing Office. See JPL Technical Memorandum 
    3349-85-101 for further details.
    
    Parameters
    ----------
    lon, lat : array_like (rank-1 or 2) or float 
        Geodetic longitude and latitude (degrees, -/+180 or 0/360 and -/+90).
    slat : float
        Standard latitude (e.g., 71 S), see Notes.
    slon : float
        Standard longitude (e.g., -70), see Notes.
    hemi : string
        Hemisphere: 'n' or 's' (not case-sensitive).
    units : string
        Polar Stereographic x,y units: 'm' or 'km' (not case-sensitive).
    
    Returns
    -------
    x, y : ndarray (rank-1 or 2) or float
        Polar stereographic x and y coordinates (in 'm' or 'km').

    Notes
    -----
    SLAT is is the "true" latitude in the plane projection 
    (the map), so there is no deformation over this latitude; 
    e.g., using the same SLON but changing SLAT from 70 to 71 
    degrees, will move things in polar stereo. The goal is to 
    locally preserve area and angles. Most users use 71S but 
    the sea ice people use 70S.
    
    SLON provides a "vertical" coordinate for plotting and for 
    rectangle orientation. E.g., for Arctic sea ice, NSIDC use 
    SLON=45 in order to make a grid that is optimized for where 
    sea ice occurs. CATS2008a has SLON=-70 (AP roughly up), so 
    that the grid can be long enough to include South Georgia.

    MOA Image Map (the GeoTIFF): SLAT=-71, SLON=0
    MOA mask grid (from Laurie): SLAT=-71, SLON=-70
    Scripps mask grid (from GeoTIFF): SLAT=-71, SLON=0

    History
    -------
    Written in Fortran by C.S. Morris - Apr 29, 1985
    Revised by C.S. Morris - Dec 11, 1985
    Revised by V.J. Troisi - Jan 1990
        SGN - provides hemisphere dependency (+/- 1)
    Revised by Xiaoming Li - Oct 1996
        Corrected equation for RHO
    Converted to Matlab by L. Padman - Oct 25, 2006
    Updated for slon by L. Padman - Nov 21, 2006
    Converted to Python by F.S. Paolo - Mar 23, 2010
    
    Example
    -------
    >>> lon = [-150.3, 66.2, 5.3]
    >>> lat = [70.2, 75.5, 80.3]
    >>> x, y = ll2xy(lon, lat, slat=71, slon=-70, hemi='s', units='m')

    Original (Matlab) documentation
    -------------------------------
    ARGUMENTS:                                                         
                                                                       
    Variable     I/O    Description                          
                                                                        
    lat           I     Geodetic Latitude (degrees, +90 to -90)
    lon           I     Geodetic Longitude (degrees, 0 to 360)
    SLAT          I     Standard latitude (typ. 71, or 70)
    SLON          I  
    HEMI          I     Hemisphere (char*1: 'N' or 'S' (not
                                    case-sensitive)
    x             O     Polar Stereographic X Coordinate (km)
    y             O     Polar Stereographic Y Coordinate (km)

    """
    if units != 'm':
        units = 'km'
    print 'parameters:'
    print 'standard lat:', slat
    print 'standard lon:', slon
    print 'hemisphere:', hemi
    print 'units of x,y:', units
    print "converting lon,lat -> x,y ..."

    # definition of constants:
    CDR = 57.29577951     # conversion degrees to radians (180/pi)
    E2 = 6.694379852*1e-3 # eccentricity squared
    E = np.sqrt(E2)
    PI = 3.141592654
    RE = 6378.1370        # updated 2/11/08 (see email from Shad O'Neel)
    #RE = 6378.273        # original value
 
    # if sequence convert to ndarray double
    if type(lon).__name__ in ['list', 'tuple']:
        lon = np.array(lon, 'f8') 
        lat = np.array(lat, 'f8')        

    # if ndarray convert to double if it isn't
    if type(lon).__name__ == 'ndarray' and lon.dtype != 'float64':
        lon = lon.astype(np.float64)
        lat = lat.astype(np.float64)
 
    # convert longitude
    if type(lon).__name__ == 'ndarray':  # is numpy array
        lon[lon<0] += 360.               # -/+180 -> 0/360
    elif lon < 0:                        # is scalar
        lon += 360.                    
 
    if (str.lower(hemi) == 's'):
        SGN = -1
    else:
        SGN = 1
    if (np.abs(slat) == 90):
        RHO = 2. * RE / ((1 + E)**(1 + E) * (1 - E)**(1 - E))**(E/2.)
    else:
        SL  = np.abs(slat) / CDR
        TC  = np.tan(PI/4. - SL/2.) / ((1 - E * np.sin(SL)) \
            / (1 + E * np.sin(SL)))**(E/2.)
        MC  = np.cos(SL) / np.sqrt(1 - E2 * (np.sin(SL)**2))
        RHO = RE * MC / TC
 
    lat = np.abs(lat) / CDR
    T = np.tan(PI/4. - lat/2.) / ((1 - E * np.sin(lat)) \
      / (1 + E * np.sin(lat)))**(E/2.)
 
    lon2 = -(lon - slon) / CDR
    x = -RHO * T * np.sin(lon2)  # global vars
    y =  RHO * T * np.cos(lon2)

    if units == 'm':            # computations are done in km
        x *= 1000.
        y *= 1000.

    print 'done.'
    return [x, y]

 
def xy2ll(x, y, slat=71, slon=0, hemi='s', units='km'):
    """
    Convert from polar stereographic 'x,y' to 'lon,lat'.
 
    This subroutine converts from Polar Stereographic 'x,y' coordinates 
    to geodetic longitude and latitude for the polar regions. The 
    equations are from Snyder, J.P., 1982, Map Projections Used by the 
    U.S. Geological Survey, Geological Survey Bulletin 1532, U.S. 
    Government Printing Office. See JPL Technical Memorandum 
    3349-85-101 for further details.  
 
    Parameters
    ----------
    x, y : array_like (rank-1 or 2) or float
        Polar stereographic x and y coordinates (in 'm' or 'km').
    slat : float
        Standard latitude (e.g., 71 S), see Notes.
    slon : float
        Standard longitude (e.g., -70), see Notes.
    hemi : string
        Hemisphere: 'n' or 's' (not case-sensitive).
    units : string
        Polar Stereographic x,y units: 'm' or 'km' (not case-sensitive).
 
    Returns
    -------
    lon, lat : ndarray (rank-1 or 2) or float
        Geodetic longitude and latitude (degrees, 0/360 and -/+90).
 
    Notes
    -----
    SLAT is is the "true" latitude in the plane projection 
    (the map), so there is no deformation over this latitude; 
    e.g., using the same SLON but changing SLAT from 70 to 71 
    degrees, will move things in polar stereo. The goal is to 
    locally preserve area and angles. Most users use 71S but 
    the sea ice people use 70S.
    
    SLON provides a "vertical" coordinate for plotting and for 
    rectangle orientation. E.g., for Arctic sea ice, NSIDC use 
    SLON=45 in order to make a grid that is optimized for where 
    sea ice occurs. CATS2008a has SLON=-70 (AP roughly up), so 
    that the grid can be long enough to include South Georgia.

    MOA Image Map (the GeoTIFF): SLAT=-71, SLON=0
    MOA mask grid (from Laurie): SLAT=-71, SLON=-70
    Scripps mask grid (from GeoTIFF): SLAT=-71, SLON=0

    History
    -------
    Written in Fortran by C.S. Morris - Apr 29, 1985
    Revised by C.S. Morris - Dec 11, 1985
    Revised by V.J. Troisi - Jan 1990
        SGN - provides hemisphere dependency (+/- 1)
    Converted to Matlab by L. Padman - Oct 25, 2006
    Updated for slon by L. Padman - Nov 21, 2006
    Converted to Python by F.S. Paolo - Mar 23, 2010
 
    Example
    -------
    >>> x = [-2141.06767831  1096.06628549  1021.77465469]
    >>> y = [  365.97940112 -1142.96735458   268.05756254]
    >>> lon, lat = xy2ll(x, y, slat=71, slon=-70, hemi='s', units='km')

    Original (Matlab) documentation
    -------------------------------
    ARGUMENTS:                                                           
                                                                         
    Variable     I/O    Description                          
                                                                      
    X             I     Polar Stereographic X Coordinate (km) 
    Y             I     Polar Stereographic Y Coordinate (km)
    SLAT          I     Standard latitude (typ. 71, or 70)
    SLON          I     Standard longitude
    HEMI          I     Hemisphere (char*1, 'S' or 'N', 
                                    not case-sensitive)
    lat           O     Geodetic Latitude (degrees, +90 to -90)
    lon           O     Geodetic Longitude (degrees, 0 to 360) 

    """
    if units != 'm':
        units = 'km'
    print 'parameters:'
    print 'standard lat:', slat
    print 'standard lon:', slon
    print 'hemisphere:', hemi
    print 'units of x,y:', units
    print "converting 'x,y' -> 'lon,lat' ..."

    # definition of constants:
    CDR = 57.29577951     # conversion degrees to radians (180/pi)
    E2 = 6.694379852*1e-3 # eccentricity squared
    E = np.sqrt(E2)
    PI = 3.141592654
    RE = 6378.1370        # updated 2/11/08 (see email from Shad O'Neel)
    #RE = 6378.273        # original value
 
    # if sequence convert to ndarray
    if type(x).__name__ in ['list', 'tuple']:
        x = np.array(x, 'f8')
        y = np.array(y, 'f8')
    
    # if ndarray convert to double if it isn't
    if type(x).__name__ == 'ndarray' and x.dtype != 'float64':
        x = x.astype(np.float64)
        y = y.astype(np.float64)
 
    if units == 'm':      # computations are done in km !!!
        x *= 1e-3
        y *= 1e-3

    if(str.lower(hemi) == 's'):
        SGN = -1.
    else:
        SGN = 1.
    slat = np.abs(slat)
    SL = slat / CDR
    RHO = np.sqrt(x**2 + y**2)    # if scalar, is numpy.float64
    if np.alltrue(RHO < 0.1):     # Don't calculate if "all points" on the equator
        lat = 90.0 * SGN
        lon = 0.0
        return lon, lat
    else:
        CM = np.cos(SL) / np.sqrt(1 - E2 * (np.sin(SL)**2))
        T = np.tan((PI/4.) - (SL/2.)) / ((1 - E * np.sin(SL)) \
            / (1 + E * np.sin(SL)))**(E/2.)
        if (np.abs(slat - 90.) < 1.e-5):
            T = ((RHO * np.sqrt((1 + E)**(1 + E) * (1 - E)**(1 - E))) / 2.) / RE
        else:
            T = RHO * T / (RE * CM)
 
        a1 =  5 * E2**2 / 24.
        a2 =  1 * E2**3 / 12.
        a3 =  7 * E2**2 / 48.
        a4 = 29 * E2**3 / 240.
        a5 =  7 * E2**3 / 120.
        
        CHI = (PI/2.) - 2. * np.arctan(T)
        lat = CHI + ((E2/2.) + a1 + a2) * np.sin(2. * CHI) \
              + (a3 + a4) * np.sin(4. * CHI) + a5 * np.sin(6. * CHI)
        lat = SGN * lat * CDR
        lon = -(np.arctan2(-x, y) * CDR) + slon

        #lon = SGN * (np.arctan2(SGN * x, -SGN * y) * CDR) + slon # original !!!
        #lon[lon<-180] += 360; lon[lon>180] -= 360                # keep lon to -/+180
    print 'done.'
    return [lon, lat]


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


def add_cols_to_tbl(fname, tname, cols):
    """
    Add columns to an existing table.
    """
    # Open it again in append mode
    f = tb.openFile(fname, "a")
    table = f.getNode(tname) 
    # Get a description of table in dictionary format
    descr = table.description._v_colObjects
    descr2 = descr.copy()
    # Add a column to description
    for cname, cval in cols.items():
        descr2[cname] = tb.Col.from_dtype(cval.dtype)
    # Create a new table with the new description
    table2 = f.createTable('/', tname[1:]+'2', descr2, "temporary table", tb.Filters(9))
    # Copy the user attributes
    table.attrs._f_copy(table2)
    # Fill the rows of new table with default values
    for i in xrange(table.nrows):
        table2.row.append()
    # Flush the rows to disk
    table2.flush()
    # Copy the columns of source table to destination
    for col in descr:
        getattr(table2.cols, col)[:] = getattr(table.cols, col)[:]
    # Fill the new column(s)
    for cname, cval in cols.items():
        getattr(table2.cols, cname)[:] = cval[:] 
    # Remove the original table
    table.remove()
    # Move table2 to table
    table2.move('/', tname[1:])
    # Print the new table
    print "new table with added column(s):", f
    # Finally, close the file
    f.close()


def save_arr_as_tbl(fname, tname, cols):
    """
    Given 1D arrays save (or add if file exists) a Table.

    fname : name of new or existing file.
    tname : name of new table.
    cols : a dictionary {'colname': colval, ...}.
    """
    # Create column description
    descr = {}
    for i, (cname, cval) in enumerate(cols.items()):
        descr[cname] = tb.Col.from_dtype(cval.dtype, pos=i)
    f = tb.openFile(fname, 'a')  # if doesn't exist create it
    table = f.createTable('/', tname, descr, "", tb.Filters(9))
    table.append([v for k, v in cols.items()])
    table.flush()
    print "file with new table:", f


def save_arr_as_mat(fname, arrs, complib='blosc'):
    """
    Given 1D and/or 2D arrays save as a column matrix (2D array).

    fname : name of file to be saved.
    arrs : a list with 1D/2D arrays with *same first dimension*.
    """
    nrow, ncol = 0, 0
    for a in arrs:
        if a.ndim > 1:
            ncol += a.shape[1]
        else:
            ncol += 1
        nrow = a.shape[0]
    f = tb.openFile(fname, 'w')
    atom = tb.Atom.from_dtype(np.dtype('f8'))
    shape = (nrow, ncol)
    filters = tb.Filters(complib=complib, complevel=9)
    d = f.createCArray('/','data', atom=atom, shape=shape, filters=filters)
    j1, j2 = 0, 0
    for a in arrs:
        if a.ndim > 1:
            j2 += a.shape[1]
        else:
            a = a.reshape(nrow, 1)
            j2 += a.shape[1]
        d[:,j1:j2] = a
        j1 = j2
    print "file with new array:", f
    f.close()


# find the n-tuples correspondent to each element of a list
ntuples = lambda lst, n: zip(*[lst[i:]+lst[:i] for i in range(n)])


### time conversion functions

def sec2dt(secs, since_year=1985):
    """
    Convert seconds since_year to datetime objects
    secs : float, array
    """
    dt_ref = dt.datetime(since_year, 1, 1, 0, 0)
    return [dt_ref + dt.timedelta(seconds=s) for s in secs]


def fy2ym(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    m, y = int(np.ceil(fy*12)), int(y)
    return [y, m]


def ym2fy(year, month):
    """Year, month -> decimal year."""
    year = np.asarray(year)
    month = np.asarray(month)
    fyear = year + (month - 0.5)/12.  # decimal years (for midle of a month)
    return fyear 


def y2dt(year):
    """
    Convert decimal year to `datetime` object.
    year : float or array_like of floats
    """
    if not np.iterable(year):
        year = np.asarray([year])
    ym = np.asarray([fy2ym(y) for y in year])
    dt = np.asarray([pn.datetime(y, m, 15) for y, m in ym])
    return dt


def ym2dt(year, month):
    """
    Convert year and month to `datetime` object.
    """
    return [dt.datetime(y, m, 15) for y, m in zip(year, month)]


def year2ym(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    m, y = int(np.ceil(fy*12)), int(y)
    if (m == 0): m = 1
    return [y, m]


# NOT SURE THIS FUNC IS OK. REVIEW THE ALGORITHM!
def ym2year(year, month):
    """Year, month -> decimal year."""
    year = np.asarray(year)
    month = np.asarray(month)
    fyear = year + (month - 1)/12. + 15.22/365.25  # decimal years (midle of a month)
    return fyear 


def year2int(year, day=15):
    """Decimal year to integer representation -> YYYMMDD."""
    if not np.iterable(year):
        year = np.asarray([year])
    ym = [year2ym(y) for y in year]
    return [int(y*10000 + m*100 + day) for y,m in ym]


def int2ymd(iyear):
    f, y = np.modf(iyear/10000.)
    d, m = np.modf(f*100)
    return (int(y), int(m), int(d*100))


# NOT SURE THIS FUNC IS OK. REVIEW THE ALGORITHM!
def int2year(iyear):
    """Integer representation of year to decimal year."""
    if not np.iterable(iyear):
        iyear = np.asarray([iyear])
    iyear = np.asarray([int(y) for y in iyear])
    fyear = lambda y, m, d: y + (m - 1)/12. + d/365.25
    ymd = [int2ymd(iy) for iy in iyear]
    return [fyear(y,m,d) for y,m,d in ymd]


def fy2ym(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    m, y = int(np.ceil(fy*12)), int(y)
    return [y, m]


def y2dt(year):
    """
    Convert decimal year to `datetime` object.
    year : float or array_like of floats.
    """
    if not np.iterable(year):
        year = np.asarray([year])
    ym = np.asarray([fy2ym(y) for y in year])
    dt = np.asarray([pd.datetime(y, m, 16) for y, m in ym])
    return dt


def num2dt(times):
    """
    Convert a numeric representation of time to datetime.
    times : int/float array_like representing YYYYMMDD.
    """
    return np.asarray([pd.datetime.strptime(str(int(t)), '%Y%m%d') for t in times])


def first_non_null(arr):
    """Return index of first non-null element."""
    return (i for i,elem in enumerate(arr) if ~np.isnan(elem)).next()


def first_non_null2(arr):
    ind, = np.where(~np.isnan(arr))
    if len(ind) > 0:
        return ind[0]
    else:
        return None

