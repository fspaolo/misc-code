"""
 Bin irregularly spaced data on independent rectangular cells (regular grid).
 
 Fernando Paolo <fpaolo@ucsd.edu>
 November 06, 2010
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

def bindata(x, y, z, xi, yi, ppbin=False, binval='median'):
    """
    Bin irregularly spaced data on a regular grid (center of the bins).

    Computes the median (default) or mean value within bins defined by
    regularly spaced xi and yi coordinates (the grid defining the bins).
    
    Parameters
    ----------
    x, y : ndarray (1D)
        The idependent variables x- and y-axis of the grid.
    z : ndarray (1D)
        The dependent variable in the form z = f(x,y).
    xi, yi : ndarray (1D)
        The coordinates defining the x- and y-axis of the grid.
    ppbin : boolean, optional
        The function returns `bins` variable (see below for description): 
        [False | True].
    binval : string, optional
        The statistical operator used to compute the value of each 
        bin: ['median' | 'mean'].
   
    Returns
    -------
    grid : ndarray (2D)
        The evenly binned data. The value of each cell is the median
        (or mean) value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points per bin. Returns only if `ppbin` is set
        to True.

    Revisions
    ---------
    2010-11-06 Fernando Paolo, Initial version 
    """

    if binval == 'median': 
        median = True
    else:
        median = False

    # make the grid
    nrow = yi.shape[0]
    ncol = xi.shape[0]
    grid = np.empty((nrow,ncol), dtype=xi.dtype) 
    if ppbin: bins = np.copy(grid)

    # step size
    dx = xi[1]-xi[0]
    dy = yi[1]-yi[0]
    hx = dx/2.
    hy = dy/2.

    # bin data
    for row in xrange(nrow):
        for col in xrange(ncol):
            xc = xi[col]          # xc,yc = center of the bin
            yc = yi[row]
            ind, = np.where((xc-hx <= x) & (x < xc+hx) & \
                            (yc-hy <= y) & (y < yc+hy))
            npts = len(ind)
            if npts > 0:
                if median:
                    grid[row,col] = np.median(z[ind])
                else:
                    grid[row,col] = np.mean(z[ind])
                if ppbin: bins[row,col] = npts
            else:
                grid[row,col] = np.nan
                if ppbin: bins[row,col] = 0

    # return the grid
    if ppbin:
        return grid, bins
    else:
        return grid


def plotbins(xi, yi, grid, cmap=cm.Spectral_r):
    """
    Plot bins (the grid) with coordinated x- and y-axis.
    """
    if xi.shape[0] < 2 or yi.shape[0] < 2:
        print 'x- or y-axis too small: N data < 2'
        sys.exit()
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]
    left, right, bottom, top = xi.min(), xi.max(), yi.min(), yi.max()
    extent = (left-dx/2., right+dx/2., bottom-dy/2., top+dy/2.)
    plt.imshow(grid, extent=extent, aspect='auto', origin='lower', \
               cmap=cmap, interpolation='nearest')


def ll2xy(lat, lon, slat=71, slon=-70, hemi='s'):
    """
    Converts from 'lat,lon' to polar stereographic 'x,y'.
     
    This function converts from geodetic latitude and longitude to
    polar stereographic 'x,y' coordinates for the polar regions. The 
    equations are from Snyder, J.P., 1982, Map Projections Used by 
    the U.S. Geological Survey, Geological Survey Bulletin 1532, U.S. 
    Government Printing Office. See JPL Technical Memorandum 
    3349-85-101 for further details.
    
    Parameters
    ----------
    lat, lon : array_like or float 
        Geodetic latitude and longitude (degrees, -/+90 and -/+180 | 0/360).
    slat : float
        Standard latitude (e.g., 71).
    slon : float
        Standard longitude (e.g., -70).
    hemi : string
        Hemisphere: 'n' | 's', not case-sensitive.
    
    Returns
    -------
    x, y : ndarray or float
        Polar stereographic x and y coordinates (km).
    
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
    >>> lat = [70.2, 75.5, 80.3]
    >>> lon = [-150.3, 66.2, 5.3]
    >>> x, y = ll2xy(lat, lon, 71, -70, 's')
    """
    
    print 'lat,lon -> x,y ...'
    # definition of constants:
    CDR = 57.29577951     # conversion degrees to radians (180/pi)
    E2 = 6.694379852*1e-3 # eccentricity squared
    E = np.sqrt(E2)
    PI = 3.141592654
    RE = 6378.1370        # updated 2/11/08 (see email from Shad O'Neel)
    #RE = 6378.273        # original value

    if type(lat).__name__ != 'ndarray':
        lat = np.array(lat, 'f8')        # convert to ndarray
    if type(lon).__name__ != 'ndarray':
        lon = np.array(lon, 'f8') 

    if lon.ndim > 0: 
        lon[lon<0] += 360                # -/+180 -> 0/360
    elif lon < 0: 
        lon += 360

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

    lon = -(lon - slon) / CDR
    x = -RHO * T * np.sin(lon)
    y =  RHO * T * np.cos(lon)
    return x, y


def xy2ll(x, y, slat=71, slon=-70, hemi='s'):
    """
    Converts from polar stereographic 'x,y' to 'lat,lon'.

    This subroutine converts from Polar Stereographic 'x,y' coordinates 
    to geodetic latitude and longitude for the polar regions. The 
    equations are from Snyder, J.P., 1982, Map Projections Used by the 
    U.S. Geological Survey, Geological Survey Bulletin 1532, U.S. 
    Government Printing Office.  See JPL Technical Memorandum 
    3349-85-101 for further details.  

    Parameters
    ----------
    x, y : array_like or float
        Polar stereographic x and y coordinates (km).
    slat : float
        Standard latitude (e.g., 71).
    slon : float
        Standard longitude (e.g., -70).
    hemi : string
        Hemisphere: 'n' | 's', not case-sensitive.

    Returns
    -------
    lat, lon : ndarray or float
        Geodetic latitude and longitude (degrees, -/+90 and 0/360).

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
    >>> lat, lon = xy2ll(x, y, 71, -70, 's')
    """

    print 'x,y -> lat,lon ...'
    # definition of constants:
    CDR = 57.29577951     # conversion degrees to radians (180/pi)
    E2 = 6.694379852*1e-3 # eccentricity squared
    E = np.sqrt(E2)
    PI = 3.141592654
    RE = 6378.1370        # updated 2/11/08 (see email from Shad O'Neel)
    #RE = 6378.273        # original value

    if type(x).__name__ != 'ndarray':
        x = np.array(x, 'f8')          # convert to ndarray
    if type(y).__name__ != 'ndarray':
        y = np.array(y, 'f8')

    if(str.lower(hemi) == 's'):
        SGN = -1
    else:
        SGN = 1
    slat = np.abs(slat)
    SL = slat / CDR
    RHO = np.sqrt(x**2 + y**2)
    if RHO.all() < 0.1:           
        # Don't calculate if on the equator
        lat = 90. * SGN
        lon = 0.0
        return lat, lon
    else:
        CM = np.cos(SL) / np.sqrt(1 - E2 * (np.sin(SL)**2))
        T = np.tan((PI/4.) - (SL/2.)) / ((1 - E * np.sin(SL)) \
          / (1 + E * np.sin(SL)))**(E/2.)
        if (np.abs(slat - 90) < 1.e-5):
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
        # original
        #lon = SGN * (np.arctan2(SGN * x, -SGN * y) * CDR) + slon 
        # keep lon to +/-180
        #lon[lon<-180] += 360; lon[lon>180] -= 360
    return lat, lon
