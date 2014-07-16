#!/usr/bin/env python
"""
Convert HDF5 data files (Tables) to flat Binary (stream of bytes).

Fernando Paolo <fpaolo@ucsd.edu>
Aug 31, 2012
"""

import os
import sys
import numpy as np
import tables as tb 

args = sys.argv[1:]


def join_tbl_to_arr(fname, tables, arr_name='data'):
    """
    Join several tables (w/same nrows) into one big array.

    tables : sequece of table objects.
    """
    ncols = 0
    nrows = tables[0].nrows
    for tbl in tables:
        ncols += len(tbl.colnames)
    # create the array that will contain the tables
    f = tb.openFile(fname, 'a')
    shape = (nrows, ncols)
    atom = tb.Atom.from_type('float64')
    filt = tb.Filters(complib='blosc', complevel=9)
    arr = f.createCArray('/', arr_name, atom=atom, shape=shape, filters=filt)
    # copy the columns of each table to destination
    j = 0
    for tbl in tables:
        for col in tbl.colnames:
            print 'col:', col
            arr[:,j] = getattr(tbl.cols, col)[:]
            j += 1
    print "new array in:", f
    f.close()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


def main(files):
    print 'files to convert:', len(files)
    for fname in files:
        print 'file:', fname
        fname_out = os.path.splitext(fname)[0]
        if os.path.getsize(fname) == 0:     # check for empty files
            print 'empty file'
            continue                    
        fin = tb.openFile(fname, 'r')
        tables = [t for t in fin.root]      # get tables from input file
        print 'joining tables:', tables
        jtable = join_tbl_to_arr(fname_out+'_ar.h5', tables)      # 2D array
    print 'done.'
    print 'last output ->', fname_out+'_ar.h5'
    close_files()


main(args)
