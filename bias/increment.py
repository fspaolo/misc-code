#!/usr/bin/env python
doc = """\
 Program to apply a bias correction to several files.

 Two ways of do it:

 1) Add extra column with 'bias' value to original file [default]
 2) Increment the 'bias' value to an specified data column
"""
"""
 Notes:

 [Add values]

 Seasat          =  0.32 +/- 0.12 m
 Geosat GM/ERM   =  0.11 +/- 0.01 m
 ERS-1           = -0.45 +/- 0.03 m
 ERS-2           = -0.05 +/- 0.01 m
 Envisat         = -0.45 +/- 0.03 m
 GFO             =  ???

 Seasat to ERS1C bias = 0.77 + 0.15
 Seasat to Envisat bias = 0.77 + 0.15

 Fernando Paolo <fpaolo@ucsd.edu>
 August 9, 2010
"""

import numpy as np
import tables as tb
import argparse as ap
import sys
import os

# command line arguments
parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter, description=doc)
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to increment')
parser.add_argument('-i', dest='inc', type=float, required=True, 
    help='value to be incremented')
parser.add_argument('-c', dest='col', type=int, default=False, 
    help='column to be incremented (0,1..) [default: add extra col]')
parser.add_argument('-e', dest='ext', default='_bias', 
    help='add name to file extension [default: _bias]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default: HDF5]')

args =  parser.parse_args()
files = args.file
inc = args.inc
col = args.col
ext = args.ext
ascii = args.ascii

print 'processing files: %d ...' % len(files)
if ascii:
    print 'reading and writing ASCII'
else:
    print 'reading and writing HDF5'

print 'increment value: %f' % inc
if col == False:
    print 'adding extra column with increment value'
else:
    print 'column to be incremented: %d' % col

for f in files:
    if ascii:
        data = np.loadtxt(f)
    else:
        fin = tb.openFile(f)
        data = fin.root.data.read()
        #data = fin.root.data
        if data.shape[0]:
            fout = tb.openFile(os.path.splitext(f)[0] + ext + '.h5', 'w')
            shape = (data.shape[0], data.shape[1] + 1) # add extra column
            atom = tb.Atom.from_dtype(data.dtype)
            filters = tb.Filters(complib='zlib', complevel=9)
            dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                     filters=filters)
    if col == False:
        if ascii:
            increment = np.empty(data.shape[0], dtype=data.dtype)
            increment[:] = inc
            data = np.column_stack((data, increment))  # add last colum with flags
        else:
            dout[:,:-1] = data[:,:]                    # add data
            dout[:,-1] = inc                           # add increment column
            fout.close()
    else:
        if ascii:
            data[:,col] += inc                         # add increment to a column
            np.savetxt(os.path.splitext(f)[0] + ext + '.txt', data, fmt='%f')
        else:
            dout[:,col] += inc
            fout.close()
    
    if not ascii:
        fin.close()

print 'done!'
if ascii:
    print 'output ext: %s.txt' % ext
else:
    print 'output ext: %s.h5' % ext
