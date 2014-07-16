#!/usr/bin/env python
"""
Select data points in a given geographic region. 

Extract all the points lying within a rectangular geographic
area: left, right, bottom, top (all inclusive).

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
parser.add_argument('-r', dest='N', nargs=4, type=float, 
    metavar=('L', 'R', 'B', 'T'),
    help='geographic coordinates to extract: left right bottom top')
parser.add_argument('-x', dest='loncol', default=3, type=int,
    help='column of longitude in the file (0,1,..) [default: 3]')
parser.add_argument('-y', dest='latcol', default=2, type=int,
    help='column of latitude in the file (0,1,..) [default: 2]')
parser.add_argument('-s', dest='suffix', default='_reg',
    help='suffix for output file name [default: _reg]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default: HDF5]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')

args = parser.parse_args()
files = args.file
left, right, bottom, top = args.N
loncol = args.loncol
latcol = args.latcol
suffix = args.suffix
ascii = args.ascii
verbose = args.verbose

if ascii:
    ext = '.txt'
else:
    ext = '.h5'

if ascii:
    print 'reading and writing ASCII'
else:
    print 'reading and writing HDF5'

print 'processing files: %d ...' % len(files)
print 'region:', left, right, bottom, top
print 'lon,lat columns: %d,%d' % (loncol, latcol)

if left < 0: 
    left += 360
if right < 0: 
    right += 360
    
nfiles = 0
npts = 0
nvalidpts = 0
for f in files:
    if verbose: print 'file:', f
    if ascii:
        data = np.loadtxt(f)
    else:
        h5f = tb.openFile(f)
        #data = h5f.root.data.read()  # in-memory
        data = h5f.root.data          # out-of-memory

    lon = data[:,loncol]
    lat = data[:,latcol]
    lon[lon<0] += 360
    npts += data.shape[0]

    ind, = np.where(( (270 <= lon) & (lon < 320) & (-90 <= lat) & (lat < -74.2) ) | \
                    ( (320 <= lon) & (lon < 340) & (-90 <= lat) & (lat < -77) ))

    if ind.shape[0]:
        nvalidpts += ind.shape[0]
        if ascii:
            np.savetxt(os.path.splitext(f)[0] + suffix + ext, 
                       data[ind,:], fmt='%f')
            nfiles += 1
        else:
            fout = tb.openFile(os.path.splitext(f)[0] + suffix + ext, 'w')
            atom = tb.Atom.from_dtype(data.dtype)
            shape = data[ind,:].shape 
            filters = tb.Filters(complib='blosc', complevel=9)
            dout = fout.createCArray(fout.root, 'data', atom=atom, 
                                     shape=shape, filters=filters)
            dout[:] = data[ind,:] 
            fout.close()

            #h5f = tb.openFile(os.path.splitext(f)[0] + suffix + ext, 'w')
            #h5f.createArray(h5f.root, 'data', data)
            #h5f.close()
            nfiles += 1
    if not ascii:
        h5f.close()

print 'done.'
print 'points read:', npts
print 'valid points:', nvalidpts
print 'files created:', nfiles
print 'output ext: %s' % (suffix+ext)
