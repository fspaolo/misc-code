#!/usr/bin/env python
"""
 Convert ASCII data files (tables) to HDF5 format (2D array) and vice-versa.

 The program recognizes the type of input files (ASCII or HDF5) 
 and performs the conversion accordingly.

 Fernando Paolo <fpaolo@ucsd.edu>
 January 1, 2010
"""

import argparse as ap
import numpy as np
import tables as tb 
import mimetypes as mt
import sys
import os

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+',
    help='file[s] to convert [ex: /path/to/files/*.ext]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
    const=True, help='load data in-memory (faster) [default: on-disk]')
parser.add_argument('-c', dest='ncol', default=9,
    help='number of columns of ASCII file [default: 9]')
parser.add_argument('-l', dest='complib', default='blosc',
    help='compression library to be used: zlib, lzo, bzip2, blosc [default: blosc]')

args = parser.parse_args()
files = args.file
verbose = args.verbose
inmemory = args.inmemory
ncol = args.ncol
complib = args.complib

if inmemory:
    print 'working: in-memory'
else:
    print 'working: out-of-memory'

print 'files to convert: %d ' % len(files)

mime, _ = mt.guess_type(files[0])

if mime == 'text/plain':
    print 'ASCII -> HDF5 ...'
    for f in files:
        if os.path.getsize(f) == 0: continue  # empty files
        if verbose: print 'file:', f
        data = np.loadtxt(f)                  # data in-memory
        h5f = tb.openFile(os.path.splitext(f)[0] + '.h5', 'w')
        if inmemory: 
            h5f.createArray(h5f.root, 'data', data)
        else:                                 # keep data on-disk
            atom = tb.Float64Atom()
            shape = data.shape
            filters = tb.Filters(complib=complib, complevel=5)
            dout = h5f.createCArray(h5f.root,'data', atom=atom, shape=shape,
                                    filters=filters)
            dout[:] = data
        h5f.close()
else:
    print 'HDF5 -> ASCII ...'
    for f in files:
        if os.path.getsize(f) == 0:     # empty files
            continue  
        if verbose: 
            print 'file:', f
        h5f = tb.openFile(f, 'r')
        if inmemory:                    # load data in-memory
            data = h5f.root.data.read()  
        else:                           # keep data on-disk
            data = h5f.root.data         
        np.savetxt(os.path.splitext(f)[0] + '.txt', data, fmt='%f')
        h5f.close()

print 'done.'
