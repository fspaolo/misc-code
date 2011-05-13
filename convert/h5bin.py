#!/usr/bin/env python
"""
 Convert HDF5 data files (2D array) to plain Binary (stream of bytes).

 If the extension of the input file is '.h5' the program performs convertion
 from HDF5 to plain Binary. For any other extention it is assume convertion
 from plain Binary to HDF5.

 Fernando Paolo <fpaolo@ucsd.edu>
 May 12, 2011
"""

import argparse as ap
import numpy as np
import tables as tb 
import mimetypes as mt
import sys
import os

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+',
    help='file[s] to convert [ex: /path/to/files/*.ext]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
    const=True, help='load data in-memory (faster) [default: on-disk]')
parser.add_argument('-c', dest='ncol', type=int, default=10,
    help='number of elements per record (columns) [default: 9]')
parser.add_argument('-d', dest='dtype', default='f8',
    help='data type: i2, i4, f4, f8, ... [default: f8]')

args = parser.parse_args()

_, ext_in = os.path.splitext(args.files[0])

print 'files to convert:', len(args.files)

def bin_info(ncol, dtype, ext_in):
    if ext_in == '.h5':
        print 'output binary data record: %s x %s' % (ncol, dtype)
        f = open('bin_info.txt', 'w')
        f.write('binary data record: %s x %s' % (ncol, dtype))
        f.close()
    else:
        print 'input binary data record: %s x %s' % (ncol, dtype)

if ext_in == '.h5':
    print 'converting HDF5 -> BIN ...'
    for f in args.files:
        if args.verbose: 
            print 'file:', f
        if os.path.getsize(f) == 0:         # check for empty files
            if args.verbose: 
                print 'empty file'
            continue                    
        fin = tb.openFile(f, 'r')
        if args.inmemory:                   # load data in-memory
            data = fin.root.data.read()  
        else:                               # keep data on-disk
            data = fin.root.data
        ncol = data.shape[1]
        dtype = data.dtype.str[1:]
        fname = os.path.splitext(f)[0]
        data[:].tofile('%s.bin' % fname)
        fin.close()
    print 'last output ->', f.split('.')[0] + '.bin'
else:
    print 'converting BIN -> HDF5 ...'
    for f in args.files:
        if args.verbose: 
            print 'file:', f
        if os.path.getsize(f) == 0:         # check for empty files
            if args.verbose: 
                print 'empty file'
            continue                    
        h5f = tb.openFile(f.split('.')[0] + '.h5', 'w')
        ncol = args.ncol
        dtype = args.dtype
        data = np.fromfile(f, dtype=dtype)  # in-memory
        shape = (data.shape[0]/ncol, ncol)
        if args.inmemory:
            h5f.createArray(h5f.root, 'data', data.reshape(shape))
        else:                               # keep data on-disk
            atom = tb.Atom.from_dtype(np.dtype(dtype))
            filters = tb.Filters(complib='blosc', complevel=5)
            dout = h5f.createCArray(h5f.root,'data', atom=atom, shape=shape,
                                    filters=filters)
            print dout.shape, data.shape, shape
            dout[:] = data.reshape(shape)
        h5f.close()
    print 'last output ->', f.split('.')[0] + '.h5'

bin_info(ncol, dtype, ext_in)
print 'done.'
