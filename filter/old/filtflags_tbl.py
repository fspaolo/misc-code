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

#------------------------------------------------------------------------
# edit here
#------------------------------------------------------------------------

# accept selected (key) values!
column = {'fmode': 6, 'fret': 7, 'fprob': 8, 'fmask': 9, 'fbord': 10, 'fsep':11}
flag = {'fmode': None, 'fret': 1, 'fprob': 0, 'fmask': 2, 'fbord': None, 'fsep' : None}  
ext = '_shelf.h5'

#------------------------------------------------------------------------

parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 Table file(s) to read')
parser.add_argument('-s', dest='struct', default='idr', type=str,
                    help='data (Table) structure: idr or gla [default: idr]')  
args = parser.parse_args()


def save_tbl(fname, table, rows=None, complib='zlib'):
    """
    Create/Reopen a file and save an existing table.
    """
    filters = tb.Filters(complib=complib, complevel=9)
    f = tb.openFile(fname, 'a')  # if doesn't exist create it
    t = f.createTable('/', table.name, table.description, '', filters)
    if rows is None:
        t.append(table[:])
    else:
        t.append(table[rows])
    t.flush()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


files = args.files
struct = args.struct

print 'table structure:', struct
print 'filtering files:', len(files), '...'

npoints = 0
nfiltered = 0
for f in files:
    # input
    #-----------------------------------------------------------------
    fin = tb.openFile(f)
    if struct == 'idr':
        fret = fin.root.idr.cols.fret[:]
        fprob = fin.root.idr.cols.fprob[:]
        nrows = fin.root.idr.nrows
    elif struct == 'gla':
        nrows = fin.root.gla.nrows
        pass
    else:
        raise IOError('-s must be idr/gla')
    fmask = fin.root.mask.cols.fmask[:]
    fbuff = fin.root.mask.cols.fbuff[:]
    #-----------------------------------------------------------------
    
    npoints += nrows 

    if nrows > 0:
        if struct == 'idr':
            ind, = np.where( \
                (fret == 1) & (fprob == 0) & \
                ( (fmask == 2) | (fmask == 3) | (fmask == 4) | (fbuff == 1) ) \
                )
        else:
            ind, = np.where( \
                ( (fmask == 2) | (fmask == 3) | (fmask == 4) | (fbuff == 1) ) \
                )

        nfiltered += nrows - ind.shape[0]

        if ind.shape[0] > 0:
            fname_out = os.path.splitext(f)[0] + ext
            # tables to save
            #---------------------------------------------------------
            if struct == 'idr':
                table1_out = fin.root.idr
            else:
                table1_out = fin.root.gla
            table2_out = fin.root.mask
            table3_out = fin.root.trk

            save_tbl(fname_out, table1_out, ind)
            save_tbl(fname_out, table2_out, ind)
            save_tbl(fname_out, table3_out, ind)
            #---------------------------------------------------------

perc = ((np.float(nfiltered) / npoints) * 100)
nleft = npoints - nfiltered
print 'done!'
print 'total points:', npoints 
print 'filtered out: %d (%.1f%%)' % (nfiltered, perc)
print 'points left: %d (%.1f%%)' % (nleft, 100-perc)
close_files()
