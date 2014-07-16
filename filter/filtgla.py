"""
Filter selected flags.

Fernando Paolo <fpaolo@ucsd.edu>
August 25, 2012
"""

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap

# edit here
#------------------------------------------------------------------------

# column of each parameter
ext = '_filt.h5'
col = {'gain': 5, 'wenergy': 6, 'reflect': 7, 'icesvar': 8}

# `Campaign`: (Minimum_uncorrected_reflectance, Maximum_fit_variance)
camp = {'20031027': (0.0375, 0.0375), # 2a   
        '20040305': (0.0125, 0.03),   # 2b 
        '20040604': (0.0125, 0.02),   # 2c
        '20041021': (0.0375, 0.04),   # 3a
        '20050307': (0.0375, 0.038),  # 3b
        '20050606': (0.025, 0.0425),  # 3c 
        '20051107': (0.025, 0.0225),  # 3d
        '20060311': (0.025, 0.035),   # 3e
        '20060610': (0.025, 0.0275),  # 3f
        '20061111': (0.025, 0.025),   # 3g
        '20070329': (0.05, 0.03),     # 3h
        '20071019': (0.05, 0.0275),   # 3i
        '20080305': (0.05, 0.03),     # 3j
        '20081012': (0.025, 0.025),   # 3k
        '20081206': (0.025, 0.02),    # 2d
        '20090326': (0.1, 0.02),      # 2e
        '20091006': (0.075, 0.02)}    # 2f

#------------------------------------------------------------------------

parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 Table file(s) to read')
args = parser.parse_args()


def save_arr(fname, data):
    """
    Open/Create specific output file and save specific variables.
    """
    if '.txt' in fname:
        np.savetxt(fname, data, fmt='%f')
    elif '.h5' in fname:
        fout = tb.openFile(fname, 'w')
        shape = data.shape
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='zlib', complevel=9)
        dout = fout.createCArray('/','data', atom=atom, 
                                 shape=shape, filters=filters)
        dout[:] = data
        fout.close()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


files = args.files

print 'filtering files:', len(files), '...'

nfiles = 0
npoints = 0
nfiltered = 0
for f in files:

    ### input
    fin = tb.openFile(f)
    data = fin.getNode('/data')
    nrows = data.shape[0]
    #gain = data[:,col['gain']]
    #wenerg = data[:,col['wenerg']]
    reflect = data[:,col['reflect']]
    icesvar = data[:,col['icesvar']]
    campaign = re.findall('\d\d\d\d\d\d\d\d', f)[0]

    # filtering ------------------------------------------------------

    npoints += nrows 
    if nrows < 1: continue

    reflect_val, icesvar_val = camp[campaign]      # one val per campaign
    ind, = np.where( (reflect_val <= reflect) & (icesvar <= icesvar_val) )

    print campaign, reflect_val, icesvar_val
    nfiltered += nrows - ind.shape[0]

    #------------------------------------------------------------------

    ### output
    if ind.shape[0] < 1: continue
    fname_out = os.path.splitext(f)[0] + ext
    save_arr(fname_out, data[ind,:])
    nfiles += 1

close_files()

perc = ((np.float(nfiltered) / npoints) * 100)
nleft = npoints - nfiltered
print 'done.'
print 'files created:', nfiles
print 'total points:', npoints 
print 'filtered out: %d (%.1f%%)' % (nfiltered, perc)
print 'points left: %d (%.1f%%)' % (nleft, 100-perc)
