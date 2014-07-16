"""
Filter selected flags.

Fernando Paolo <fpaolo@ucsd.edu>
August 25, 2012
"""

import os
import sys
import numpy as np
import tables as tb
import argparse as ap

# edit here
#------------------------------------------------------------------------

# column of each parameter
col = {'fmode': 6, 'fret': 7, 'fprob': 8, 'fmask': 9, 'fbord': 10, 'fbuf': 11, 'ftrk':12}
ext = '_float.h5'

#------------------------------------------------------------------------

parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 2D file(s) to read')
parser.add_argument('-s', dest='struct', default='idr', type=str,
                    help='data structure: idr/gla, default idr')  
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
struct = args.struct

print 'data structure:', struct
print 'filtering files:', len(files), '...'

nfiles = 0
npoints = 0
nfiltered = 0
for f in files:

    # input
    #-----------------------------------------------------------------

    fin = tb.openFile(f)
    data = fin.getNode('/data')
    nrows = data.shape[0]
    if struct == 'idr':
        fret = data[:,col['fret']]
        fprob = data[:,col['fprob']]
    elif struct == 'gla':
        pass
        #reflect = data[:,col['reflect']]
        #icesvar = data[:,col['icesvar']]
    else:
        raise IOError('-s must be idr/gla')
    fmask = data[:,col['fmask']]
    #fbuf = data[:,col['fbuf']]
    #fbord = data[:,col['fbord']]  # not needed if `fbuf` is used

    #-----------------------------------------------------------------
    
    npoints += nrows 
    if nrows < 1:
        continue

    if struct == 'idr':
        ind, = np.where( \
            (fret == 1) & (fprob == 0) & \
            #( (fmask == 2) | (fmask == 3) | (fmask == 4) | (fbuf == 1) ) \
            ( (fmask == 4) ) \
            )
    else:       # gla
        ind, = np.where( \
                #( (fmask == 2) | (fmask == 3) | (fmask == 4) | (fbuf == 1) ) \
            ( (fmask == 4) ) \
            )

    nfiltered += nrows - ind.shape[0]

    #-----------------------------------------------------------------

    # save
    if ind.shape[0] > 0:
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
