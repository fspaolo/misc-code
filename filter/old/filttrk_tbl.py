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
    f = tb.openFile(fname, 'a')          # if doesn't exist create it
    t = f.createTable('/', table.name, table.description, '', filters)
    if rows is None:
        t.append(table[:])
    else:
        t.append(table[rows])
    t.flush()
    f.close()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


files = args.files
struct = args.struct

print 'table structure:', struct
print 'filtering files:', len(files), '...'

nasc = 0
ndes = 0
for f in files:
    # input
    #-----------------------------------------------------------------
    fin = tb.openFile(f)
    ftrk = fin.root.trk.cols.ftrk[:]
    nrows = ftrk.shape[0]
    #-----------------------------------------------------------------

    if nrows > 0:
        i_a, = np.where(ftrk == 0)
        i_d, = np.where(ftrk == 1)

        if (i_a.shape[0] > 0) and (i_d.shape[0] > 0):
            fname_a = os.path.splitext(f)[0] + '_a.h5' 
            fname_d = os.path.splitext(f)[0] + '_d.h5' 

            # tables to save
            #---------------------------------------------------------
            if struct == 'idr':
                t1 = fin.root.idr
            else:
                t1 = fin.root.gla
            t2 = fin.root.mask
            t3 = fin.root.trk

            save_tbl(fname_a, t1, i_a)
            save_tbl(fname_a, t2, i_a)
            save_tbl(fname_a, t3, i_a)
            save_tbl(fname_d, t1, i_d)
            save_tbl(fname_d, t2, i_d)
            save_tbl(fname_d, t3, i_d)
            #---------------------------------------------------------

            nasc += i_a.shape[0]
            ndes += i_d.shape[0]

print 'done!'
print 'ascending tracks:', nasc
print 'descending tracks:', ndes
close_files()
