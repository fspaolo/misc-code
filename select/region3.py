#!/usr/bin/env python
"""
Separate several data files in contiguous geographic sectors. 

Given a `range` and `step` size in degrees for *longitude*,
the program separates several input data files into the respective 
contiguous sectors (in individual files), within specified
latitude boundaries.

Notes
-----
`left` and `bottom` are inclusive, `right` and `top` are not.

"""
# Fernando <fpaolo@ucsd.edu>
# November 4, 2010

import os
import sys
import argparse as ap
import numpy as np
import tables as tb

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-r', dest='region', nargs=4, type=float, 
    metavar=('L', 'R', 'B', 'T'),
    help='longitude and latitude range: left right bottom top')
parser.add_argument('-d', dest='step', nargs=2, type=float, 
    metavar=('DX', 'DY'),
    help='step size for lon and lat (in degrees) [default: 10 10]')
parser.add_argument('-l', dest='overlap', nargs=4, type=float, 
    metavar=('DL', 'DR', 'DB', 'DT'),
    help='amount of overlapping on each side (in degrees) [default: 0 0 0 0]')
parser.add_argument('-x', dest='loncol', default=3, type=int,
    help='column of longitude in the file (0,1,..) [default: 3]')
parser.add_argument('-y', dest='latcol', default=2, type=int,
    help='column of latitude in the file (0,1,..) [default: 2]')
parser.add_argument('-m', dest='maskcol', default=9, type=int,
    help='column of mask flags in the file (0,1,..) [default: 9]')
parser.add_argument('-f', dest='flag', default=1, type=int,
    help='mask flag value: != flag (0,1,..) [default: 1=ocean]')
parser.add_argument('-s', dest='suffix', default='_reg',
    help='suffix for output file name [default: _reg]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default: HDF5]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')

args = parser.parse_args()
files = args.file
left, right, bottom, top = args.region
step = args.step
overlap = args.overlap
loncol = args.loncol
latcol = args.latcol
maskcol = args.maskcol
flag = args.flag
suffix = args.suffix
ascii = args.ascii
verbose = args.verbose

if step is None:
    dx, dy = 10., 10.
else:
    dx, dy = step

if overlap is None:
    dl, dr, db, dt = 0., 0., 0., 0.
else:
    dl, dr, db, dt = overlap

if ascii:
    ext = '.txt'
else:
    ext = '.h5'

def lon_180_to_360(lon):
    if isinstance(lon, np.ndarray):
        lon[lon<0] += 360
    elif lon < 0: 
        lon += 360
    return lon

class CheckBounds(object):
    """Check the correct input of region coordinates.
    """
    def __init__(self, left, right, bottom, top):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self._check_bounds()
   
    def _check_bounds(self):
        if self.left >= self.right or self.bottom >= self.top:
            self._print_msg(1)
        if self.left < 0 and (left < -180 or right > 180):  # -180/180
            self._print_msg(2)
        elif self.left >= 0 and right > 360:                # 0/360
            self._print_msg(2)

    def _print_msg(self, msg):
        if msg == 1:
            print 'error: region must be: `left` < `right` and `bottom` < `top`'
        elif msg == 2:
            print 'error: longitude or latitude out of bounds:'
        else:
            pass
        print 'left/right:', self.left, self.right
        print 'bottom/top:', self.bottom, self.top
        sys.exit()

class SubRegion(object):
    """Define a geographic subregion with data points.
    """
    def __init__(self, lon, lat, fmask, flag, left, right, 
                 bottom, top, dl, dr):
        self.lon = lon
        self.lat = lat
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.dl = dl
        self.dr = dr
        self.fmask = fmask
        self.flag = flag

    def find_pts(self, pts):
        """Find all the pts in a subregion with `borders`.
        """
        # conditions
        lons0 = (self.left <= self.lon) & (self.lon < self.right) 
        lons1 = (self.left+self.dl <= self.lon) & (self.lon < self.right) # w/o dl
        lons2 = (360-self.dl <= self.lon) & (self.lon < 360)
        lons3 = (self.left <= self.lon) & (self.lon < self.right-self.dr) # w/o dr
        lons4 = (0 <= self.lon) & (self.lon < self.dr)
        lats0 = (self.bottom <= self.lat) & (self.lat < self.top)
        fmask = (self.fmask != self.flag)

        # subregion starting at 0 
        if self.left < 0 and self.right <= 360:
            if pts == 'all':
                ind, = np.where( (lons1 | lons2) & lats0 )
            elif pts == 'ice':
                ind, = np.where( (lons1 | lons2) & lats0 & fmask )
        # subregion ending at 360 
        elif self.left >= 0 and self.right > 360:
            if pts == 'all':
                ind, = np.where( (lons3 | lons4) & lats0 )
            elif pts == 'ice':
                ind, = np.where( (lons3 | lons4) & lats0 & fmask )
        # any subregion
        else:
            if pts == 'all':
                ind, = np.where( lons0 & lats0 )
            elif pts == 'ice':
                ind, = np.where( lons0 & lats0 & fmask )
        return ind

    def find_max_lat(self, ind):
        if ind.shape[0]:
            return self.lat[ind].max()    # max lat over land
        else:
            return -90. 

#--------------------------------------------------------------------

CheckBounds(left, right, bottom, top)

left = lon_180_to_360(left)
right = lon_180_to_360(right)

if ascii:
    print 'reading and writing ASCII'
else:
    print 'reading and writing HDF5'

print 'processing files: %d ...' % len(files)
print 'region:', left, right, bottom, top
print 'step size:', dx, dy
print 'overlap:', dl, dr, db, dt
print 'lon,lat columns: %d,%d' % (loncol, latcol)

n_files = 0
n_pts = 0
n_validpts = 0
for f in files:
    if verbose: 
        print 'file:', f
    if ascii:
        data = np.loadtxt(f)
    else:
        h5f = tb.openFile(f, 'r')
        #data = h5f.root.data.read()  # in-memory
        data = h5f.root.data          # out-of-memory

    lon = data[:,loncol]
    lat = data[:,latcol]
    fmask = data[:,maskcol]
    lon = lon_180_to_360(lon)
    n_pts += data.shape[0]

    i_region = 0
    for i_lon in np.arange(left, right, dx):      # lons
        i_left = i_lon - dl
        i_right = i_lon + dx + dr
        i_bottom = -90 # i_lat - db
        i_top = 90 # i_lat + dy + dt

        if verbose: 
            print 'subregion %02d:' % i_region, i_left, i_right, i_bottom, i_top

        # warning: this may not test *all* the cases!!!

        s = SubRegion(lon, lat, fmask, flag, i_left, i_right, i_bottom, i_top, dl, dr)
        ind = s.find_pts('ice')
        s.top = s.find_max_lat(ind) + 1.5    # set upper boundary
        '''
        if i_region == 16: s.top = -66
        if i_region == 29: s.top = -62.4
        if i_region == 31: s.top = -75
        '''
        ind = s.find_pts('all')

        if ind.shape[0]:
            outfile = '%s%s%02d%s' % (os.path.splitext(f)[0], 
                                      suffix, i_region, ext)
            n_validpts += ind.shape[0]
            if ascii:
                np.savetxt(outfile, data[ind,:], fmt='%f')
                n_files += 1
            else:
                fout = tb.openFile(outfile, 'w')
                atom = tb.Atom.from_dtype(data.dtype)
                shape = data[ind,:].shape 
                filters = tb.Filters(complib='blosc', complevel=9)
                dout = fout.createCArray(fout.root, 'data', atom=atom, 
                                         shape=shape, filters=filters)
                dout[:] = data[ind,:] 
                fout.close()
                n_files += 1

        i_region += 1

    if not ascii:
        h5f.close()

print 'done.'
print 'points read:', n_pts
print 'valid points:', n_validpts
print 'files created:', n_files
print 'output ext: %s' % (suffix + ext)
