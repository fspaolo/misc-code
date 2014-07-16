"""
Module to apply the MODIS-based Antarctic ice mask.

For documentation on each method see the docstrings.

Fernando Paolo <fpaolo@ucsd.edu>
March 24, 2010

"""

import numpy as np
import tables as tb
import string as str
import scipy.io as io
import sys 
import os

try:
    import masksearch as m   # C function for speed up!
    cmodule = True
    print 'C module for speed-up imported!'
except:
    cmodule = False
    print "couln't import C module, using pure Python instead!"


class Mask(object):
    """Class to apply the MODIS Antarctic ice mask.
    """
    def __init__(self, maskfile=None):
        if maskfile is not None:
            if os.path.splitext(maskfile)[1] == '.mat':
                self.getmask_mat(maskfile)    # Matlab file 
            else:
                self.getmask(maskfile)        # HDF5 files
        else:
            self.maskfile = None 

    def __str__(self):
        return '\nMODIS MOA-based mask of 1km resolution: %s\n' % self.maskfile

    def getmask_mat(self, maskfile):
        """Get mask from Matlab file (*.mat).
        """
        try:
            print 'loading mask file:', maskfile, '...'
            mfile = io.loadmat(maskfile, squeeze_me=True, struct_as_record=True)
            MASK = mfile['MASK'].item()
            self.m_mask = MASK[7]
            self.x_mask = MASK[0]
            self.y_mask = MASK[1]
            self.maskfile = maskfile
            print 'MASK: flags [2D] =', np.shape(self.m_mask), ', y [1D] =', \
                np.shape(self.y_mask), ', x [1D] =', np.shape(self.x_mask)
        except:
            print 'error: getmask: something wrong with the file: %s' % maskfile
            raise
    
    def getmask(self, maskfile):
        """Get mask from HDF5 file (*.h5).
        """
        try:
            print 'loading mask file:', maskfile, '...'
            self.fmask = tb.openFile(maskfile, 'r')  # don't forget to close!
            self.m_mask = self.fmask.root.mask.read()
            self.x_mask = self.fmask.root.x.read()
            self.y_mask = self.fmask.root.y.read()
            self.maskfile = maskfile
            print 'MASK: flags [2D] =', np.shape(self.m_mask), ', y [1D] =', \
                np.shape(self.y_mask), ', x [1D] =', np.shape(self.x_mask)
        except:
            print 'error: getmask: something wrong with the file: %s' % maskfile
            raise
              
    def closemask(self):
        """Close the opened HDF5 mask file.
        """
        try:
            if self.fmask.isopen: 
                self.fmask.close()
        except:
            pass

    def mapll(self, lon, lat, slat=71, slon=-70, hemi='s', units='km'):
        """Convert from 'lon,lat' to polar stereographic 'x,y'.
     
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
            Standard latitude (e.g., 71 or 70 deg for N/S).
        slon : float
            Standard longitude (e.g., -70).
        hemi : string
            Hemisphere: 'n' or 's', not case-sensitive.
        units : string
            Polar Stereographic x,y units: 'm' or 'km', not case-sensitive.
        
        Returns
        -------
        x, y : ndarray (rank-1 or 2) or float
            Polar stereographic x and y coordinates (in 'm' or 'km').
        
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
        >>> lon = [-150.3, 66.2, 5.3]
        >>> lat = [70.2, 75.5, 80.3]
        >>> x, y = m.mapll(lon, lat, slat=71, slon=70, hemi='s', units='m')

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
        print "converting from 'lon,lat' to 'x,y' ..."

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
     
        lon = -(lon - slon) / CDR
        x = -RHO * T * np.sin(lon)  # global vars
        y =  RHO * T * np.cos(lon)

        if units == 'm':            # computations are done in km
            x *= 1000.
            y *= 1000.

        return x, y
     
    def mapxy(self, x, y, slat=71, slon=-70, hemi='s', units='km'):
        """Convert from polar stereographic 'x,y' to 'lon,lat'.
     
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
            Standard latitude (e.g., 71 or 70 for N/S).
        slon : float
            Standard longitude (e.g., -70).
        hemi : string
            Hemisphere: 'n' or 's', not case-sensitive.
        units : string
            Polar stereographic x,y units: 'm' or 'km', not case-sensitive.
     
        Returns
        -------
        lon, lat : ndarray (rank-1 or 2) or float
            Geodetic longitude and latitude (degrees, 0/360 and -/+90).
     
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
        >>> lon, lat = m.mapxy(x, y, slat=71, slon=-70, hemi='s', units='km')

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
        print "converting from 'x,y' to 'lon,lat' ..."

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
        return lon, lat

    def applymask(self, data, latcol=2, loncol=3, slat=71, slon=-70, \
                  hemi='s', border=3):
        """Apply the MOA mask to a data set given the conditions.
        
        Parameters
        ----------
        data : 2D ndarray
            The data to apply the mask.
        latcol : integer
            Column of latitude in the data matrix.
        loncol : integer
            Column of longitude in the data matrix.
        slat : float
            Standard latitude (e.g., 71 or 70 for N/S).
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
            col1 (mask) 0=land/1=water/2=ice-shelf, col2 (is border) 0=no/1=yes.
        
        Example
        -------
        >>> m = Mask('maskfilename')
        >>> dataout = m.applymask(data, latcol=1, loncol=2, border=3)

        """
        #if type(data).__name__ != 'ndarray':
        #    print "error: applymask: 'data' must be ndarray:"
        #    print 'you may want to do the following:'
        #    print '>>> import numpy as np'
        #    print '>>> data = np.array(data)'
        #    sys.exit()

        if self.maskfile is None:
            print 'error: applymask: get mask file first:'
            print '>>> m = Mask()'
            print ">>> m.getmask('fname')"
            print 'then you can do:'
            print '>>> m.applymask(data, latcol, loncol,...)'
            sys.exit()
     
        #ind, = np.where((-90 <= data[:,latcol]) & (data[:,latcol] >= 90) &
        #                (-180 <= data[:,loncol]) & (data[:,loncol] >= 360))
        x, y = self.mapll(data[:,loncol], data[:,latcol], slat=slat, slon=slon, hemi=hemi)
     
        ndata = data.shape[0]
        flags = np.ones((ndata,2), 'i2') # 1 = water
        flags[:,1] = 0                   # 0 = is not the border of the mask
        x = np.rint(x)                   # round to nearest integer !!!!!
        y = np.rint(y)
        R = border
     
        # data bounds
        x_min = x.min(); x_max = x.max()
        y_min = y.min(); y_max = y.max()

        # shortens the mask (2D ndarray) for faster searching (if needed)
        resize = False
        if self.x_mask[0] < x_min-R:                 # mask larger than data+R
            jmin, = np.where(self.x_mask == x_min-R) 
            resize = True
        else:                                   # data larger than mask
            jmin = 0                            # use mask bounds
        if self.y_mask[0] < y_min-R:
            imin, = np.where(self.y_mask == y_min-R) 
            resize = True
        else:
            imin = 0
        if self.x_mask[-1] > x_max+R:
            jmax, = np.where(self.x_mask == x_max+R) 
            resize = True
        else:
            jmax = self.x_mask.shape[0]-1
        if self.y_mask[-1] > y_max+R:
            imax, = np.where(self.y_mask == y_max+R)
            resize = True
        else:
            imax = self.y_mask.shape[0]-1

        '''
        if resize:
            x_mask = x_mask[jmin:jmax+1]
            y_mask = y_mask[imin:imax+1]
            m_mask = m_mask[imin:imax+1, jmin:jmax+1]
        '''
        
        print 'applying mask ...'

        # C function for speed up-----------------------------------

        if cmodule:
            m.mask_search(x.astype('i2'), 
                          y.astype('i2'), 
                          self.x_mask[jmin:jmax+1].astype('i2'), 
                          self.y_mask[imin:imax+1].astype('i2'), 
                          self.m_mask[imin:imax+1, jmin:jmax+1].astype('i2'), 
                          flags, 
                          R)

        #-----------------------------------------------------------

        else:
            # search the flag in the mask
            for i in xrange(ndata):
                x_i = x[i]; y_i = y[i]

                if self.x_mask[jmin]+R < x_i and x_i < self.x_mask[jmax+1]-R and \
                   self.y_mask[imin]+R < y_i and y_i < self.y_mask[imax+1]-R:
                    
                    row, = np.where(self.y_mask[imin:imax+1] == y_i)
                    col, = np.where(self.x_mask[jmin:jmax+1] == x_i)
                    f = self.m_mask[imin+row,jmin+col]  # 0=land/1=water/2=ice-shelf
                    flags[i,0] = f                

                    # neighboring values on a square 2Rx2R -> border flag: 0/1
                    if np.alltrue(self.m_mask[imin+row-R:imin+row+R+1, 
                                              jmin+col-R:jmin+col+R+1] == f):  
                        flags[i,1] = 0     # if all True
                    else:                                             
                        flags[i,1] = 1     # else is border

        #-----------------------------------------------------------

        dataout = np.column_stack((data, flags))  # add colum with flags
        return dataout

    def get_mask2d(self, lon, lat, slat=71, slon=-70, hemi='s', border=0, mapll=True):
        """
        Get the MOA mask for a given grid of lon/lat.
        
        Parameters
        ----------
        lon,lat : 2D ndarrays
            Grids with lons and lats to apply the mask.
        slat : float
            Standard latitude (e.g., 71 or 70 for N/S).
        slon : float
            Standard longitude (e.g., -70).
        hemi : string
            Hemisphere ('n' or 's', not case-sensitive).
        border: integer
            Distance from the border to flag points (in km).
        
        Returns
        -------
        mask : 2D ndarray
            Mask with flags: 0=land/1=water/2=ice-shelf/3=border
        
        Example
        -------
        >>> m = Mask('maskfilename')
        >>> mask = m.get_mask2d(lon, lat, border=3)

        """
        if self.maskfile is None:
            print 'error: applymask: get mask file first:'
            print '>>> m = Mask()'
            print ">>> m.getmask('fname')"
            print 'then you can do:'
            print '>>> m.get_mask2d(lon, lat,...)'
            sys.exit()

        lon, lat = np.asarray(lon), np.asarray(lat)

        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
     
        if mapll:
            x, y = self.mapll(lon, lat, slat=slat, slon=slon, hemi=hemi)
        else:
            x, y = lon, lat
     
        ny, nx = lon.shape
        mask = np.empty((ny,nx), 'i2')
        mask.fill(-1)
        x = np.rint(x)                   # round to nearest integer !!!!!
        y = np.rint(y)
        R = border
     
        # data bounds
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()

        # shortens the mask (2D ndarray) for faster searching (if needed)
        resize = False
        if self.x_mask[0] < x_min-R:                 # mask larger than data+R
            jmin, = np.where(self.x_mask == x_min-R) 
            resize = True
        else:                                        # data larger than mask
            jmin = 0                                 # use mask bounds
        if self.y_mask[0] < y_min-R:
            imin, = np.where(self.y_mask == y_min-R) 
            resize = True
        else:
            imin = 0
        if self.x_mask[-1] > x_max+R:
            jmax, = np.where(self.x_mask == x_max+R) 
            resize = True
        else:
            jmax = self.x_mask.shape[0]-1
        if self.y_mask[-1] > y_max+R:
            imax, = np.where(self.y_mask == y_max+R)
            resize = True
        else:
            imax = self.y_mask.shape[0]-1

        print 'applying mask ...'

        # C function for speed up
        #-------------------------------------------------------------
        '''
        if cmodule:
            m.mask_search(x.astype('i2'), 
                          y.astype('i2'), 
                          self.x_mask[jmin:jmax+1].astype('i2'), 
                          self.y_mask[imin:imax+1].astype('i2'), 
                          self.m_mask[imin:imax+1, jmin:jmax+1].astype('i2'), 
                          flags, 
                          R)
        '''
        #-----------------------------------------------------------
        if False:
            pass
        else:
            # search the flag in the mask
            for i in xrange(ny):
                for j in xrange(nx):
                    x_ij = x[i,j]
                    y_ij = y[i,j]
                    if not self.x_mask[jmin]+R < x_ij and x_ij < self.x_mask[jmax+1]-R and \
                       self.y_mask[imin]+R < y_ij and y_ij < self.y_mask[imax+1]-R:
                        continue
                    row, = np.where(self.y_mask[imin:imax+1] == y_ij)
                    col, = np.where(self.x_mask[jmin:jmax+1] == x_ij)
                    f = self.m_mask[imin+row,jmin+col]  # 0=land/1=water/2=ice-shelf

                    # neighboring values on a square 2Rx2R -> border
                    if np.alltrue(self.m_mask[imin+row-R:imin+row+R+1, 
                                              jmin+col-R:jmin+col+R+1] == f):  
                        mask[i,j] = f
                    else:
                        mask[i,j] = 3     # is border
        return mask

    def plotmask(self, region=None, resolution=20, slat=71, slon=-70, hemi='s'):
        """Plot the mask.
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
            print ">>> m.plotmask(region='left/right/bottom/top', resolution=20)"
            sys.exit()
     
        m_mask = self.m_mask
        x_mask = self.x_mask
        y_mask = self.y_mask
        
        if region is not None:
            left, right, bottom, top = str.split(region, '/')
            left, bottom = self.mapll(left, bottom, slat=slat, slon=slon, hemi=hemi)
            right, top = self.mapll(right, top, slat=slat, slon=slon, hemi=hemi)
            jmin, = np.where(x_mask == np.rint(left))
            jmax, = np.where(x_mask == np.rint(right))
            imin, = np.where(y_mask == np.rint(bottom))
            imax, = np.where(y_mask == np.rint(top))
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
