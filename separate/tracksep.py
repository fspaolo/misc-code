#!/usr/bin/env python
doc = """\
 Separate satellite tracks in ascending and descending using the latitude.

 Two ways:

 1) Add extra column with flags: 0=ascending, 1=descending [default]
 2) Separated tracks in different output files: file_asc.h5, file_des.h5
"""

# Fernando Paolo <fpaolo@ucsd.edu>
# November 1, 2010

import numpy as np
import tables as tb
import argparse as ap
from sys import exit
import os

# parse command line arguments
parser = ap.ArgumentParser(description=doc)
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-y', dest='latcol', default=2, type=int,
                    help='column of latitude in the file (0,1,..) [default: 2]')  
parser.add_argument('-f', dest='trackfiles', default=False, 
                    action='store_const', const=True, 
                    help='separate tracks in files [default: add column with flag]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
                    const=True, help='reads and writes ASCII files [default: HDF5]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
                    const=True, help='load data in-memory (faster) ' 
                    '[default: out-of-memory]')

args = parser.parse_args()
files = args.file
latcol = args.latcol
trackfiles = args.trackfiles
ascii = args.ascii

print 'processing files: %d ...' % len(files)
if ascii:
    print 'reading and writing ASCII'
else:
    print 'reading and writing HDF5'

if trackfiles:
    print "separate tracks in files: 'asc' and 'des'"
else:
    print 'separate tracks using flags: 0=asc, 1=des'

for f in files:
    if ascii:
        data = np.loadtxt(f)
    else:
        h5f = tb.openFile(f)
        data = h5f.root.data.read()
        h5f.close()

    nrow = data.shape[0]
    ncol = data.shape[1]
    
    print 'separating tracks ...'  

    #---------------------------------------------------------------
    if trackfiles:
        des = np.empty((nrow, ncol), 'f8')
        asc = np.empty((nrow, ncol), 'f8')
        i_d = 0
        i_a = 0
        for i in xrange(nrow-1):       # <<<<< optimize whit Cython?
            lat1 = data[i,latcol]
            lat2 = data[i+1,latcol]
            if lat1 < lat2:            # ascending
                asc[i_a,:] = data[i,:]
                i_a += 1
            elif lat1 >= lat2:         # descending
                des[i_d,:] = data[i,:]
                i_d += 1
    #---------------------------------------------------------------
    else:
        flag = np.empty(nrow, 'i4')
        for i in xrange(nrow-1): 
            lat1 = data[i,latcol]
            lat2 = data[i+1,latcol]
            if lat1 < lat2:     
                flag[i] = 0            # ascending
            elif lat1 >= lat2: 
                flag[i] = 1            # descending
        data = np.column_stack((data, flag))  # add last colum with flags
    #---------------------------------------------------------------
    
    if trackfiles:
        if ascii:
            np.savetxt(os.path.splitext(f)[0] + '_des.txt', des[:i_d,:], fmt='%f')
            np.savetxt(os.path.splitext(f)[0] + '_asc.txt', asc[:i_a,:], fmt='%f')
        else:
            h5f = tb.openFile(os.path.splitext(f)[0] + '_des.h5', 'w')
            h5f.createArray(h5f.root, 'data', des[:i_d,:])
            h5f.close()
            h5f = tb.openFile(os.path.splitext(f)[0] + '_asc.h5', 'w')
            h5f.createArray(h5f.root, 'data', asc[:i_a,:])
            h5f.close()
    else:
        if ascii:
            np.savetxt(os.path.splitext(f)[0] + '.txt', data, fmt='%f')
        else:
            h5f = tb.openFile(os.path.splitext(f)[0] + '.h5', 'w')
            h5f.createArray(h5f.root, 'data', data)
            h5f.close()

print 'done!'
if trackfiles:
    if ascii:
        print 'output ext: *_asc.txt, *_des.txt'
    else:
        print 'output ext: *_asc.h5, *_des.h5'
else:
    print 'column w/flag added to original file (last one):'
    print '0 = ascending'
    print '1 = descending'
