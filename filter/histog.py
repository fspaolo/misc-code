"""
 Compute histogram of selected data column on HDF5 file.

 Fernando Paolo <fpaolo@ucsd.edu>
 August 24, 2010
"""

import numpy as np
import tables as tb
import argparse as ap
import pylab as pl
import sys
import os

# command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5 file[s] to compute histogram')
parser.add_argument('-c', dest='col', type=int, default=4, 
                    help='column to compute histogram [4]')
args =  parser.parse_args()
files = args.file
col = args.col

print 'processing files: %d...' % len(files)
print 'histogram of column: %d' % col

left = 282    # region
right = 305
bottom = -78
top = -60

for f in files:
    h5f = tb.openFile(f)
    data = h5f.root.data.read()
    h5f.close()

    y = data[:,2]
    x = data[:,3]
    z = data[:,col]

    ind, = np.where((left <= x) & (x <= right) & \
		    (bottom <= y) & (y <= top) & \
		    (-10 <= z) & (z <= 10) & (z != 0))
    z = z[ind] * 100 # cm
    
    mean = np.mean(z)
    std = np.std(z)
    print 'file: %s' % f 
    print 'mean: %f' % mean
    print 'std: %f' % std
    n, bins, patches = pl.hist(z, 80, facecolor='blue', alpha=0.7)

    #h5f = tb.openFile(os.path.splitext(f)[0] + ext + '.h5', 'w')
    #h5f.createArray(h5f.root, 'data', data)
    #h5f.close()

pl.title('Retracking correction from 2nd ramp (cm)')
pl.xlabel('Retracking correction (cm)')
pl.ylabel('Number of points')
pl.show()
print 'done!'
#print 'output extension: %s.h5' % ext
