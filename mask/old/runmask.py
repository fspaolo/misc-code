"""
Script to use the class Mask and apply it to a several HDF5/ASCII files.

Fernando Paolo <fpaolo@ucsd.edu>
October 29, 2010
"""

import numpy as np
import tables as tb
import argparse as ap
import sys
import os

import libmask  # Mask module

BUFFER = 50  # km, third flag with data buffer. Use `None` to disable

np.seterr(all='warn')  # warn floating point errors

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-f', dest='mfile', default='mask_ice_1km_2008_0410c.h5', 
    help='name of mask file (.h5 or .mat) [mask_ice_1km_2008_0410c.h5]')  
parser.add_argument('-b', dest='border', default=3, type=int,
    help='cut-off distance in km from any border [default: 3 km]')  
parser.add_argument('-x', dest='loncol', default=3, type=int,
    help='column of longitude in the file (0,1,..) [default: 3]')  
parser.add_argument('-y', dest='latcol', default=2, type=int,
    help='column of latitude in the file (0,1,..) [default: 2]')  
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='read and write ASCII files [default: HDF5]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
    const=True, help='load data in-memory (faster) [default: on-disk]')

args = parser.parse_args()
files = args.file
maskfile = args.mfile
border = args.border
loncol = args.loncol
latcol = args.latcol
ascii = args.ascii
inmemory = args.inmemory

if ascii:
    ext = '_mask.txt'
else:
    ext = '_mask.h5'

print 'processing files: %d ...' % len(files)
print 'border: %d km' % border
if ascii:
    print 'reading and writing ASCII'
else:
    print 'reading and writing HDF5'

m = libmask.Mask(maskfile)  # load mask file only once

for f in files:
    print 'file:', f
    if ascii:
        data = np.loadtxt(f)
    else:
        fin = tb.openFile(f)
        if inmemory:
            data = fin.root.data.read()  # in-memory
        else:
            data = fin.root.data         # out-of-memory

    try:
        dataout = m.applymask(data, latcol=latcol, loncol=loncol, \
                              slat=71, slon=-70, hemi='s', border=border)
    except:
        print '\n***** something went wrong!\n'
        print '***** file:', f
        continue

    if ascii:
        np.savetxt(os.path.splitext(f)[0] + ext, dataout, fmt='%f')
    else:
        fout = tb.openFile(os.path.splitext(f)[0] + ext, 'w')
        atom = tb.Atom.from_dtype(dataout.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        dout = fout.createCArray(fout.root,'data', atom=atom, shape=dataout.shape,
                                 filters=filters)
        dout[:] = dataout
        fout.close()

    if not ascii and fin.isopen: 
        fin.close()

m.closemask()  # close HDF5 mask file
print 'done.'
print 'output ext: %s' % ext
