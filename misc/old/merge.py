#!/usr/bin/env python
"""
Merge several HDF5 or ASCII files in 90-day periods.

Fernando <fpaolo@ucsd.edu>
November 4, 2010
"""

import os
import sys
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-o', dest='outfile', default='all',
    help="output file name *without* extension ['all']")  
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default HDF5]')

args = parser.parse_args()

if args.ascii:
    args.outfile += '.txt'
else:
    args.outfile += '.h5'

print 'merging files: %d ...' % len(args.file)
if args.ascii:
    print 'reading and writing ASCII files'
else:
    print 'reading and writing HDF5 files'

isfirst = True
for f in args.file:
    if args.ascii:
        data = np.loadtxt(f)
    else:
        h5f = tb.openFile(f)
        data = h5f.root.data.read()
        h5f.close()

    if isfirst:
        DATA = np.empty((0, data.shape[1]), 'f8')
        isfirst = False

    DATA = np.vstack((DATA, data))  # <<<<< in-memory (see out-of-memory)

if args.ascii:
    path, file = os.path.split(f)
    np.savetxt(os.path.join(path, args.outfile), DATA, fmt='%f')
else:
    path, file = os.path.split(f)
    #h5f = tb.openFile(os.path.join(path, args.outfile), 'w')
    #h5f.createArray(h5f.root, 'data', DATA)
    #h5f.close()

    filters = tb.Filters(complib='blosc', complevel=9)
    atom = tb.Atom.from_dtype(DATA.dtype)
    shape = DATA.shape
    fout = tb.openFile(os.path.join(path, args.outfile), 'w')
    dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                             filters=filters)
    dout[:] = DATA[:]
    fout.close()


print 'done!'
