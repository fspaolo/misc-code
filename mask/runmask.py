"""
 Script to use the class mask and apply it to a several HDF5/ASCII files.

 Fernando Paolo <fpaolo@ucsd.edu>
 October 29, 2010
"""

import numpy as np
import tables as tb
import argparse as ap
from sys import exit
import os

import libmask

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-m', dest='mfile', default='mask_ice_1km_2008_0410c.mat', 
                    help='name of mask file (.mat) [mask_ice_1km_2008_0410c.mat]')  
parser.add_argument('-b', dest='border', default=7, type=int,
                    help='cut-off distance in km from any border [7 km]')  
parser.add_argument('-x', dest='loncol', default=3, type=int,
                    help='column of longitude in the file (0,1,..) [3]')  
parser.add_argument('-y', dest='latcol', default=2, type=int,
                    help='column of latitude in the file (0,1,..) [2]')  
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
                    const=True, help='reads and writes ASCII files [default HDF5]')

args = parser.parse_args()
files = args.file
maskfile = args.mfile
border = args.border
loncol = args.loncol
latcol = args.latcol
ascii = args.ascii

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
        h5f = tb.openFile(f)
        data = h5f.root.data.read()
        h5f.close()

    try:
        data = m.applymask(data, latcol=latcol, loncol=loncol, border=border)
    except:
        print '\n********** something went wrong **********\n'
        continue

    if ascii:
        np.savetxt(os.path.splitext(f)[0] + ext, data, fmt='%f')
    else:
        h5f = tb.openFile(os.path.splitext(f)[0] + ext, 'w')
        h5f.createArray(h5f.root, 'data', data)
        h5f.close()

print 'done!'
print 'output ext: %s' % ext
