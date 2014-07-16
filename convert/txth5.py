#!/usr/bin/env python
"""Convert ASCII (column) data files to HDF5, and vice-versa.

The program recognizes the type of input file (ASCII or HDF5) 
and performs the conversion accordingly. If the name of the `fields` 
are passed, the columns in the ASCII file will be converted to 1D
arrays given the respective names. Otherwise a 2D array named `data` 
(mirroring the ASCII data) is created.

Examples
--------
To see the available options::

$ python txth5.py -h

To convert several ASCII files given the name of each field (column)::

$ python txth5.py -f lon,lat,elev /path/to/files/*.txt

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# January 1, 2010

import os
import sys
import argparse as ap
import numpy as np
import tables as tb 
import mimetypes as mt

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+',
    help='file[s] to convert [ex: /path/to/files/*.ext]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
    const=True, help='load data in-memory (faster) [default: on-disk]')
parser.add_argument('-l', dest='complib', default='zlib',
    help='compression library to be used: zlib, lzo, bzip2, blosc [default: zlib]')

args = parser.parse_args()
files = args.files
verbose = args.verbose
inmemory = args.inmemory
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
        if os.path.getsize(f) == 0:           # empty files
            continue  
        if verbose: print 'file:', f
        data = np.loadtxt(f)                  # load data in-memory
        h5f = tb.openFile(os.path.splitext(f)[0] + '.h5', 'w')
        if inmemory: 
            h5f.createArray(h5f.root, 'data', data)
        else:                                 # keep data on-disk
            atom = tb.Float64Atom()
            shape = data.shape
            filters = tb.Filters(complib=complib, complevel=9)
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
