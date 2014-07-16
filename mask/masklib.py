"""
Module to apply the Antarctic ice mask (MOA or Scripps).

Fernando Paolo <fpaolo@ucsd.edu>
Aug 12, 2012

"""

import os
import sys 
import math
import numpy as np
import tables as tb
import string as str
import scipy.io as io


def get_mask(maskfile, x='/x', y='/y', mask='/mask', paddzeros=0):
    """Get mask from HDF5 file."""
    print 'loading mask ...'
    fin = tb.openFile(maskfile, 'r')
    x = fin.getNode(x).read()
    y = fin.getNode(y).read()
    mask = fin.getNode(mask).read()
    print fin
    fin.close()
    if int(math.log10(x.max())+1) > 4:  # if m -> km
        x = x*1e-3
        y = y*1e-3
    if paddzeros != 0:
        x, y, mask = padd_zeros(x, y, mask, paddzeros)
    return [x, y, mask]


def padd_zeros(x, y, z, n):
    p, q = z.shape
    z2 = np.zeros((p+n*2, q+n*2), 'i1')
    x2 = np.empty(q+n*2, 'f8')
    y2 = np.empty(p+n*2, 'f8')
    x2[:n] = np.arange(x[0]-n, x[0])
    x2[-n:] = np.arange(x[-1]+1, x[-1]+n+1)
    # could it be that y isn't reversed???
    y2[:n] = np.arange(y[0]+n, y[0], -1)
    y2[-n:] = np.arange(y[-1]-1, y[-1]-n-1, -1)  # y is reversed
    x2[n:-n] = x
    y2[n:-n] = y
    z2[n:-n,n:-n] = z
    return [x2, y2, z2]


def get_subreg(x, y, xm, ym, mask, buf=0):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    j, = np.where((xmin-buf <= xm) & (xm <= xmax+buf))
    i, = np.where((ymin-buf <= ym) & (ym <= ymax+buf))
    return [xm[j[0]:j[-1]+1], ym[i[0]:i[-1]+1], mask[i[0]:i[-1]+1,j[0]:j[-1]+1]]


def digitize(x, y, xm, ym):
    """
    Digitize x/y according to xm/ym (mask coords).

    len(x) == len(y), len(xm) == or != len(ym)
    """
    print 'digitizing x,y ...'
    dx = np.abs(xm[1] - xm[0])
    dy = np.abs(ym[1] - ym[0])
    xmin, xmax = xm.min(), xm.max()
    ymin, ymax = ym.min(), ym.max()
    xmin -= dx/2.; xmax += dx/2.  # the edges
    ymin -= dy/2.; ymax += dy/2.
    x_edges = np.arange(xmin, xmax+dx, dx)
    y_edges = np.arange(ymin, ymax+dy, dy)
    j_bins = np.digitize(x, bins=x_edges)
    i_bins = np.digitize(y, bins=y_edges)
    # substitute indices out of mask bounds: 0 -> 1 and N+1 -> N
    n = len(xm)
    m = len(ym)
    j_bins[j_bins==0] = 1
    j_bins[j_bins==n+1] = n
    i_bins[i_bins==0] = 1
    i_bins[i_bins==m+1] = m
    print 'done.'
    return [j_bins-1, i_bins-1]  # shift inds to the left (from edges)


def apply_mask(lon, lat, xm, ym, mask, buf=0, **kw):
    x, y = ll2xy(lon, lat, **kw)
    x, y = np.rint(x), np.rint(y)
    xm2, ym2, mask2 = get_subreg(x, y, xm, ym, mask, buf=0)
    if ym2[0] > ym2[-1]:
        # monotonically increasing
        ym2 = ym2[::-1]  
        mask2 = mask2[::-1,:]
    jj, ii = digitize(x, y, xm2, ym2)
    flg1 = mask2[ii,jj]    # return flags (1D array)
    flg2 = np.zeros_like(flg1)
    if buf != 0:
        print 'searching %d km buffer ...' % buf
        for k, (i, j) in enumerate(zip(ii, jj)):
            flg2[k] = ~np.alltrue(mask2[i,j] == mask2[i-buf:i+buf+1,j-buf:j+buf+1])
        print 'done.'
    return [flg1, flg2]


def save_arr_as_tbl(fname, tname, cols):
    """
    Given 1D arrays save as a Table (add to file if exists).

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
