#!/usr/bin/env python
doc = """\
 Finds the 3 files with more data points on it.

 Ex: find_maxoc.py -s '_05 _06 _07 _08' /path/to/files/*.ext
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
parser.add_argument('-s', dest='substrings', default='', help='sequence of substrings'
    "to match in file name [ex: -s 'str1 str2 str3']")
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default: HDF5]')

args =  parser.parse_args()
files = args.file
subs = args.substrings.split(' ')
ascii = args.ascii

print 'reading files: %d ...' % len(files)
if ascii:
    print 'reading ASCII format'
else:
    print 'reading HDF5 format'

max_rows = np.zeros(3, 'i4')
file_names = np.array([None, None, None])

for f in files:

    isInFile = [(sub in f) for sub in subs]

    if np.any(isInFile):

        if ascii:
            data = np.loadtxt(f)
        else:
            fin = tb.openFile(f)
            #data = fin.root.data.read()
            data = fin.root.data            # out-of-memory
            
        nrow = data.shape[0]                # nrow == # of points

        if nrow:
            i_min = max_rows.argmin()
            if max_rows[i_min] < nrow:
                max_rows[i_min] = nrow
                file_names[i_min] = f

        if not ascii:
            fin.close()
    else:
        continue

print 'done.'

print 'npt_oc', 'npt_ice', 'fname:'
ii = max_rows.argsort()
for npt, fname in zip(max_rows[ii], file_names[ii]):
    print npt, fname
