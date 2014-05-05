#!/usr/bin/env python
"""
Separate several data files in contiguous geographic regions. 

Given a `range` and `step` size in degrees for longitude and latitude,
the program separates several input data files into the respective 
contiguous geographic regions (in individual files).

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
parser.add_argument('-s', dest='suffix', default='_',
    help='suffix for output file name [default: _NN]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')

args = parser.parse_args()
files = args.file
left, right, bottom, top = args.region
step = args.step
overlap = args.overlap
loncol = args.loncol
latcol = args.latcol
suffix = args.suffix
verbose = args.verbose

if step is None:
    dx, dy = 10., 10.
else:
    dx, dy = step

if overlap is None:
    dl, dr, db, dt = 0., 0., 0., 0.
else:
    dl, dr, db, dt = overlap

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
        if (self.left < 0) and (left < -180 or right > 180):  # -180/180
            self._print_msg(2)
        elif (self.left >= 0) and (right > 360):              # 0/360
            self._print_msg(2)

    def _print_msg(self, msg):
        if msg == 1:
            print 'error: region must be: left < right, bottom < top'
        elif msg == 2:
            print 'error: longitude or latitude out of bounds:'
        else:
            pass
        print 'left/right:', self.left, self.right
        print 'bottom/top:', self.bottom, self.top
        sys.exit()

def find_pts(left, right, bottom, top, dl, dr):
    """Find all the pts in the subregion with the borders.
    """
    # subregion starting at 0 
    if left < 0 and right <= 360:
        # take out the border on the left
        left += dl  
        ind, = np.where( ( ((left <= lon) & (lon < right)) | \
                           ((360-dl <= lon) & (lon < 360)) ) & \
                         ( (bottom <= lat) & (lat < top)  ) )
    # subregion finishing at 360 
    elif left >= 0 and right > 360:
        # take out the border on the right 
        right -= dr  
        ind, = np.where( ( ((left <= lon) & (lon < right)) | \
                           ((0 <= lon) & (lon < dr)) ) & \
                         ( (bottom <= lat) & (lat < top)  ) )
    # any subregion
    else:
        ind, = np.where((left <= lon) & (lon < right) & \
                        (bottom <= lat) & (lat < top))     
    return ind

#--------------------------------------------------------------------

CheckBounds(left, right, bottom, top)

left = lon_180_to_360(left)
right = lon_180_to_360(right)

print 'processing files: %d ...' % len(files)
print 'region:', left, right, bottom, top
print 'step size:', dx, dy
print 'overlap:', dl, dr, db, dt
print 'lon,lat columns: %d,%d' % (loncol, latcol)

n_files = 0
n_pts = 0
n_validpts = 0
for f in files:
    if verbose: print 'file:', f
    h5f = tb.openFile(f, 'r')
    #data = h5f.root.data.read()  # in-memory
    data = h5f.root.data          # out-of-memory

    lon = data[:,loncol]
    lat = data[:,latcol]
    lon = lon_180_to_360(lon)
    n_pts += data.shape[0]

    i_region = 1
    for i_lat in np.arange(bottom, top, dy):
        for i_lon in np.arange(left, right, dx):
            i_left = i_lon - dl
            i_right = i_lon + dx + dr
            i_bottom = i_lat - db
            i_top = i_lat + dy + dt

            if verbose: 
                print 'subregion:', i_left, i_right, i_bottom, i_top

            # warning: this does not test *all* the cases!!!

            # NO NEED TO CHECK LAT BOUNDS!

            ind = find_pts(i_left, i_right, i_bottom, i_top, dl, dr)

            if ind.shape[0]:
                outfile = '%s%s%02d%s' % (os.path.splitext(f)[0], 
                                          suffix, i_region, ext)
                n_validpts += ind.shape[0]
                i_region += 1
                # save
                fout = tb.openFile(outfile, 'w')
                atom = tb.Atom.from_dtype(data.dtype)
                shape = data[ind,:].shape 
                filters = tb.Filters(complib='zlib', complevel=9)
                dout = fout.createCArray(fout.root, 'data', atom=atom, 
                                         shape=shape, filters=filters)
                dout[:] = data[ind,:] 
                fout.close()
                n_files += 1
    h5f.close()

print 'done.'
print 'points read:', n_pts
print 'valid points:', n_validpts
print 'files created:', n_files
print 'output ext: %s' % (suffix+'NN'+ext)
