"""
 Class to apply the MODIS Antarctic ice mask (from a Matlab file: *.mat).

 For documentation on each method see the docstrings.

 Fernando Paolo <fpaolo@ucsd.edu>
 March 24, 2010
"""

import numpy as np
import string as str
import scipy.io as io
from sys import exit

try:
    import masksearch as m  # C function for speed up!
    cmodule = True
    print 'C module imported!'
except:
    cmodule = False
    print "couln't import C module!"

class Mask:
    """
    Class to apply the MODIS Antarctic ice mask.
    """
    def __init__(self, maskfile=None):
        if maskfile is not None:
            self.getmask(maskfile)    # see bellow
        else:
            self.maskfile = maskfile


    def getmask(self, maskfile):
        """
        Get mask from Matlab file.
        """
        try:
            print 'loading mask file:', maskfile, '...'
            mfile = io.loadmat(maskfile, squeeze_me=True, struct_as_record=True)
            MASK = mfile['MASK'].item()
            m_mask = MASK[7]
            x_mask = MASK[0]
            y_mask = MASK[1]
        except:
            print 'error: getmask: something wrong with the file: %s' % maskfile
            exit()
        print 'MASK: flags [2D] =', np.shape(m_mask), ', y [1D] =', \
              np.shape(y_mask), ', x [1D] =', np.shape(x_mask)
              
        self.m_mask = m_mask
        self.x_mask = x_mask
        self.y_mask = y_mask
        self.maskfile = maskfile


    def mapll(self, lat, lon, slat=71, slon=-70, hemi='s'):
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
        >>> m = Mask()
        >>> lat = [70.2, 75.5, 80.3]
        >>> lon = [-150.3, 66.2, 5.3]
        >>> x, y = m.mapll(lat, lon, 71, -70, 's')
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
        x = -RHO * T * np.sin(lon)  # global vars
        y =  RHO * T * np.cos(lon)
        return x, y
     
     
    def mapxy(self, x, y, slat=71, slon=-70, hemi='s'):
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
        >>> m = Mask()
        >>> x = [-2141.06767831  1096.06628549  1021.77465469]
        >>> y = [  365.97940112 -1142.96735458   268.05756254]
        >>> lat, lon = m.mapxy(x, y, 71, -70, 's')
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
            x = np.array(x, 'f8')          # convert to ndarray double
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
            # keep lon to -/+180
            #lon[lon<-180] += 360; lon[lon>180] -= 360
        return lat, lon


    def applymask(self, data, latcol=2, loncol=3, slat=71, slon=-70, \
                  hemi='s', border=7):
        """
        Apply the mask to a data set given the conditions.
        
        Parameters
        ----------
        data : 2D ndarray
            The data to apply the mask.
        latcol : integer
            Column of latitude in the data matrix.
        loncol : integer
            Column of longitude in the data matrix.
        slat : float
            Standard latitude (e.g., 71).
        slon : float
            Standard longitude (e.g., -70).
        hemi : string
            Hemisphere ('n' or 's', not case-sensitive).
        border: integer
            Distance from the border (in km).
        
        Returns
        -------
        out : 2D ndarray
            The original data with two extra columns containing the flags:
            col1 (mask) 0=land/1=water/2=ice-shelf, col2 (border) 0/1.
        
        Example
        -------
        >>> m = Mask('maskfilename')
        >>> outdata = m.applymask(data, latcol=1, loncol=2, border=7)
        """
        if type(data).__name__ != 'ndarray':
            print "error: applymask: 'data' must be ndarray:"
            print 'you may want to do the following:'
            print '>>> import numpy as np'
            print '>>> data = np.array(data)'
            exit()
        if self.maskfile is None:
            print 'error: applymask: get mask file first:'
            print '>>> m = Mask()'
            print ">>> m.getmask('fname')"
            print 'then you can do:'
            print '>>> m.applymask(data, latcol, loncol,...)'
            exit()
     
        m_mask = self.m_mask.astype('i2')
        x_mask = self.x_mask.astype('i2')
        y_mask = self.y_mask.astype('i2')

        x, y = self.mapll(data[:,latcol], data[:,loncol], slat, slon, hemi)
     
        ndata = data.shape[0]
        flags = np.ones((ndata,2), 'i2') # 1 = water
        flags[:,1] = 0                   # 0 = no-border
        x = np.rint(x).astype('i2')      # round to nearest integer !
        y = np.rint(y).astype('i2')
        R = border
     
        # data bounds
        x_min = x.min(); x_max = x.max()
        y_min = y.min(); y_max = y.max()

        # shortens the mask for faster searching
        resize = False
        if x_mask[0] < x_min-R:                 # mask larger than data+R
            jmin, = np.where(x_mask == x_min-R) 
            resize = True
        else:                                   # data larger than mask
            jmin = 0                            # use mask bounds
        if y_mask[0] < y_min-R:
            imin, = np.where(y_mask == y_min-R) 
            resize = True
        else:
            imin = 0
        if x_mask[-1] > x_max+R:
            jmax, = np.where(x_mask == x_max+R) 
            resize = True
        else:
            jmax = x_mask.shape[0]-1
        if y_mask[-1] > y_max+R:
            imax, = np.where(y_mask == y_max+R)
            resize = True
        else:
            imax = y_mask.shape[0]-1

        if resize:
            x_mask = x_mask[jmin:jmax+1]
            y_mask = y_mask[imin:imax+1]
            m_mask = m_mask[imin:imax+1, jmin:jmax+1]
        
        print 'applying mask ...'

        # C function for speed up-----------------------------------
        if cmodule:
            m.mask_search(x, y, x_mask, y_mask, m_mask, flags, R)
        #-----------------------------------------------------------
        else:
            # search the flag in the mask
            for i in xrange(ndata):
                x_i = x[i]; y_i = y[i]

                if x_mask[0]+R < x_i and x_i < x_mask[-1]-R and \
                   y_mask[0]+R < y_i and y_i < y_mask[-1]-R:
                    
                    row, = np.where(y_mask == y_i)
                    col, = np.where(x_mask == x_i)
                    f =  m_mask[row,col]          # 0=land/1=water/2=ice-shelf
                    flags[i,0] = f                

                    # neighboring values on a square 2Rx2R -> border flag: 0/1
                    if (m_mask[row-R:row+R+1, col-R:col+R+1] == f).all():  
                        flags[i,1] = 0     # if all True
                    else:                                             
                        flags[i,1] = 1     # else is border

        data = np.column_stack((data, flags))  # add colum with flags
     
        return data


    def plotmask(self, region=None, resolution=20, slat=71, slon=-70, hemi='s'):
        """
        Plot the mask.
        """
        try:
            import enthought.mayavi.mlab as ml 
            mayavi = True
            print 'using mayavi'
        except:
            import pylab as pl
            mayavi = False
            print 'using matplotlib'

        if self.maskfile is None:
            print 'error: plotmask: get mask file first:'
            print '>>> m = Mask()'
            print ">>> m.getmask('fname')"
            print 'then you can do:'
            print ">>> m.plotmask(region='left/right/bottom/upper', resolution=20)"
            exit()
     
        m_mask = self.m_mask
        x_mask = self.x_mask
        y_mask = self.y_mask
        
        if region is not None:
            left, right, bottom, upper = str.split(region, '/')
            left, bottom = self.mapll(bottom, left, slat, slon, hemi)
            right, upper = self.mapll(upper, right, slat, slon, hemi)
            jmin, = np.where(x_mask == np.rint(left))
            jmax, = np.where(x_mask == np.rint(right))
            imin, = np.where(y_mask == np.rint(bottom))
            imax, = np.where(y_mask == np.rint(upper))
            #x_mask = x_mask[jmin:jmax+1:resolution]
            #y_mask = y_mask[imin:imax+1:resolution]
            m_mask = m_mask[imin:imax+1:resolution, jmin:jmax+1:resolution]
        else:
            #x_mask = x_mask[::resolution]
            #y_mask = y_mask[::resolution]
            m_mask = m_mask[::resolution,::resolution]
        
        print 'plotting mask ...'
        if mayavi:
            ml.figure()
            ml.imshow(m_mask)
            ml.show()
        else:
            pl.figure()
            pl.imshow(m_mask, origin='lower', interpolation='nearest')
            pl.show()
        print 'done!'
