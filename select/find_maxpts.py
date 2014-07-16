#!/usr/bin/env python
doc = """\
 Select and find the 3 files with more ocean- or ice-mode data points.

 example:

     $ find_maxpts.py -s '_05 _06 _07 _08' /path/to/files/*.ext
"""
"""
 Fernando Paolo <fpaolo@ucsd.edu>
 June 28, 2011
"""

import numpy as np
import tables as tb
import argparse as ap
import sys
import os

# command line arguments
parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter, description=doc)
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to increment')
parser.add_argument('-s', dest='substrings', default='', help='sequence of substrings '
                    "to match in file name [ex: -s 'str1 str2 str3']")
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
                    const=True, help='reads and writes ASCII files [default: HDF5]')
parser.add_argument('-i', dest='ice', default=False, action='store_const',
                    const=True, help='find ice-mode data points [default: ocean-mode]')

args =  parser.parse_args()
files = args.file
subs = args.substrings.split(' ')
ascii = args.ascii
ice = args.ice
COLFMODE = 6    # column of mode flag

print 'reading files: %d ...' % len(files)
if ascii:
    print 'in ASCII format'
else:
    print 'in HDF5 format'
if ice:
    print 'find ice-mode'
else:
    print 'find ocean-mode'

max_oc = np.zeros(3, 'i4')
max_ice = np.zeros(3, 'i4')
max_coarse = np.zeros(3, 'i4')
max_pts = np.zeros(3, 'i4')
file_names = np.array([None, None, None])

for f in files:

    isInFile = [(sub in f) for sub in subs]          # check substrings in filename

    if np.any(isInFile):

        if ascii:
            data = np.loadtxt(f)
        else:
            fin = tb.openFile(f)
            #data = fin.root.data.read()
            data = fin.root.data                     # out-of-memory
            
        i_oc, = np.where(data[:,COLFMODE] == 0)      # ocean-mode 
        i_ice, = np.where(data[:,COLFMODE] == 1)     # ice-mode 
        i_coarse, = np.where(data[:,COLFMODE] == 2)  # coarse-mode (RA2)
        n_oc = i_oc.shape[0]
        n_ice = i_ice.shape[0]
        n_coarse = i_coarse.shape[0]
        n_pts = data.shape[0]                        # n_pts == nrow

        if n_pts:
            if ice:                                  # check for ice-mode
                i_min = max_ice.argmin()
                if max_ice[i_min] < n_ice:
                    max_oc[i_min] = n_oc 
                    max_ice[i_min] = n_ice 
                    max_coarse[i_min] = n_coarse
                    max_pts[i_min] = n_pts 
                    file_names[i_min] = f
            else:                                    # check for ocean-mode
                i_min = max_oc.argmin()
                if max_oc[i_min] < n_oc:
                    max_oc[i_min] = n_oc 
                    max_ice[i_min] = n_ice 
                    max_coarse[i_min] = n_coarse
                    max_pts[i_min] = n_pts 
                    file_names[i_min] = f

        if not ascii:
            fin.close()

    else:
        continue

print 'done.'

def display(*args):
    for j_oc, j_ice, j_coarse, j_pts, j_fname in \
        zip(max_oc[ii], max_ice[ii], max_coarse[ii], max_pts[ii], file_names[ii]):
        print 'ocean: %d  ice: %d  coarse: %d  (total: %d)' \
              % (j_oc, j_ice, j_coarse, j_pts)
        print 'file:', j_fname
        print

if ice and not max_ice.all() == 0:
    ii = max_ice.argsort()    # sort
    ii = ii[::-1]             # reverse
    print '============'
    print 'max ice-mode'
    print '============'
    display(ii, max_oc, max_ice, max_coarse, max_pts, file_names)
elif not max_oc.all() == 0:
    ii = max_oc.argsort()     # sort
    ii = ii[::-1]             # reverse
    print '=============='
    print 'max ocean-mode'
    print '=============='
    display(ii, max_oc, max_ice, max_coarse, max_pts, file_names)
else:
    print 'no files found!'
