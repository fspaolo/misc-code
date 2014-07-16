#!/usr/bin/env python
"""
 Change the compression library of HDF5 files. 

 Fernando Paolo <fpaolo@ucsd.edu>
 May 21, 2011
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
parser.add_argument('-l', dest='complib', default='zlib',
    help='compression library: zlib, bz2, blosc [default: zlib]')

args = parser.parse_args()
files = args.file
verbose = args.verbose
inmemory = args.inmemory
complib = args.complib

if inmemory:
    print 'working: in-memory'
else:
    print 'working: out-of-memory'

print 'files to convert: %d ' % len(files)

for f in files:

    if os.path.getsize(f) == 0: 
        continue  # empty files
    if verbose: 
        print 'file:', f

    fin = tb.openFile(f, 'r')
    if inmemory:
        # load data in-memory
        data = fin.root.data.read()  
    else:
        # keep data on-disk
        data = fin.root.data         

    fout = tb.openFile(os.path.splitext(f)[0] + '_zlib.h5', 'w')
    if inmemory:
        fout.createArray(fout.root, 'data', data)
    else:
        # keep data on-disk
        atom = tb.Float64Atom()
        shape = data.shape
        filters = tb.Filters(complib=complib, complevel=9)
        dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                filters=filters)
        dout[:] = data[:]

    try:
        fout.close()
        fin.close()
    except:
        pass

print 'output ext:', '_' + complib
print 'done.'
