#!/usr/bin/env python
"""
Separate several data files in contiguous geographic sectors. 

Given a `range` and `step` size in degrees for *longitude*,
the program separates several input data files into the respective 
contiguous sectors (in individual files), within specified
latitude boundaries.

Notes
-----
`left` and `right` are inclusive.

"""
# Fernando <fpaolo@ucsd.edu>
# October 29, 2012

import os
import sys
import argparse as ap
import numpy as np
import tables as tb

from funcs import *

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-r', dest='range', nargs=2, type=float, 
    metavar=('L', 'R'), help='longitude range: left right')
parser.add_argument('-d', dest='step', type=float, default=90,
    metavar=('DX'), help='size of sectors (in deg), default 90')
parser.add_argument('-l', dest='overlap', default=.5, type=float, metavar=('DL'),
    help='extra space for overlap on each side (in deg), default 0,5')
parser.add_argument('-x', dest='loncol', default=3, type=int,
    help='column of longitude in the file (0,1,..), default 3')
parser.add_argument('-y', dest='latcol', default=2, type=int,
    help='column of latitude in the file (0,1,..), default 2')

args = parser.parse_args()
files = args.file
x1, x2 = args.range
dx = args.step
buf = args.overlap
loncol = args.loncol
latcol = args.latcol


print 'processing files: %d ...' % len(files)
print 'lon,lat columns: %d,%d' % (loncol, latcol)

nfiles = 0
npts = 0
nvalidpts = 0
for f in files:
    # input
    fin = tb.openFile(f, 'r')
    data = fin.getNode('/data')
    lon = data[:,loncol]
    lat = data[:,latcol]
    npts += data.shape[0]

    # processing
    sectors = define_sectors(x1, x2, dx=dx, buf=buf)
    results = get_sectors(lon, sectors)

    if len(results) < 1: 
        continue

    # output
    for r in results:
        ind, secnum = r
        fname_out = '%s_%02d.h5' % (os.path.splitext(f)[0], secnum)
        save_arr(fname_out, data[ind,:])
        nvalidpts += ind.shape[0]
        nfiles += 1

close_files()

print 'done.'
print 'points read:', npts
print 'valid points:', nvalidpts
print 'files created:', nfiles
print 'output ext: *_NN.h5'
