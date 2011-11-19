#!/usr/bin/env python
"""
Merge several 1-month HDF5 files in 3-month blocks.

Fernando <fpaolo@ucsd.edu>
November 18, 2011
"""

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5/ASCII file[s] to read')

args = parser.parse_args()
files = args.files

def get_season(month):
    """Given a month finds the 3-month block of respective season.
    """
    MAM = [3, 4, 5]      # Mar/Apr/May -> Fall SH 
    JJA = [6, 7, 8]      # Jun/Jul/Aug -> winter SH
    SON = [9, 10, 11]    # Sep/Oct/Nov -> Spring SH
    DJF = [12, 1, 2]     # Dec/Jan/Feb -> summer SH

    if month in MAM:
        return MAM
    elif month in JJA:
        return JJA
    elif month in SON:
        return SON
    elif month in DJF:
        return DJF
    else:
        print 'not a valid month from 1 to 12!'
        return None, None, None  # none existent range

def get_month(fname):
    name = os.path.basename(fname)
    t = re.findall('\d\d\d\d\d\d+', name)[-1]
    return int(t[4:6])

def get_fname_out(fname, season, nfiles, i):
    path, name = os.path.split(fname)
    sat = name.split('_')[0]
    date = re.findall('\d\d\d\d\d\d+', name)[-1][:6] # first file
    if i == nfiles-1:
        date += '%02d' % season[0]
    elif i == nfiles-2:
        date += '%02d' % season[1]
    else:
        date += '%02d' % season[2]
    fname_out = ''.join([sat, '_', date, '.h5'])
    return os.path.join(path, fname_out)


print 'reading files: %d ...' % len(files)

files.sort(key=lambda s: [re.findall('\d\d\d\d\d\d+', s)]) # sort !

nfiles = len(files)
season = []
for i, fname in enumerate(files):

    # get data out-of-memory
    fin = tb.openFile(fname)
    data = fin.root.data    

    month = get_month(fname)

    if month not in season:
        season = get_season(month)
        fname_out = get_fname_out(fname, season, nfiles, i)
        try:
            fout.close()
        except:
            pass
        fout = tb.openFile(fname_out, 'w')
        shape = (0, data.shape[1])
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        dout = fout.createEArray(fout.root, 'data', atom=atom, 
                                 shape=shape, filters=filters)
    dout.append(data[:])
    fin.close()

try:
    fout.close()
except:
    pass
print 'done.'
