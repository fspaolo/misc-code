#!/usr/bin/env python
"""
Convert HDF5 data files (Tables) to flat Binary (stream of bytes).

Fernando Paolo <fpaolo@ucsd.edu>
Aug 31, 2012
"""

import os
import sys
import argparse as ap
import numpy as np
import tables as tb 
import mimetypes as mt

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+',
    help='file[s] to convert [ex: /path/to/files/*.ext]')
parser.add_argument('-d', dest='double', default=False, action='store_const',
    const=True, help='for homogeneus binary dtype (all doubles), default original mix types')
parser.add_argument('-b', dest='big_endian', default=False, action='store_const',
    const=True, help='if data is big-endian, default little-endian')
args = parser.parse_args()


class GLA(tb.IsDescription):
    orbit = tb.Int32Col(pos=0)
    secs00 = tb.Float64Col(pos=1)
    lat = tb.Float64Col(pos=2)
    lon = tb.Float64Col(pos=3)
    elev = tb.Float64Col(pos=4)
    agc = tb.Float64Col(pos=5)
    energy = tb.Float64Col(pos=6)
    txenergy = tb.Float64Col(pos=7)
    reflect = tb.Float64Col(pos=8)
    fbuff = tb.Int8Col(pos=9)
    fmask = tb.Int8Col(pos=10)
    fbord = tb.Int8Col(pos=11)
    ftrk = tb.UInt8Col(pos=12)


def bin_info(fname, data):
    shape = data.shape
    dtype = data.dtype
    if len(shape) == 2:
        nrows, ncols = shape
    else:
        nrows, ncols = shape[0], 1
    print 'binary file:'
    print 'number of records:', nrows
    print 'dtype of records %d x %s:' % (ncols, str(dtype))
    f = open(fname, 'w')
    f.write('binary data record:\n%d x (%d x %s)\n' % (nrows, ncols, str(dtype)))
    f.close()


def join_tbl(fname, tname, descr, tables):
    """
    Join several tables (w/same nrows) into one.

    tables : sequece of table objects.
    """
    f = tb.openFile(fname, 'a')
    # create a new table with the new description
    table = f.createTable('/', tname, descr, "join table", tb.Filters(9))
    # fill the rows of new table with default values
    nrows = tables[0].nrows
    for i in xrange(nrows):
        table.row.append()
    # flush the rows to disk
    table.flush()
    # copy the columns of each table to destination
    for tbl in tables:
        for col in tbl.colnames:
            print 'col:', col
            getattr(table.cols, col)[:] = getattr(tbl.cols, col)[:]
    # print the new table
    print "new join table in:", f
    #f.close()
    return table


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
    print "new array:", f
    #f.close()
    return arr


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


def main(args):
    files = args.files
    double = args.double
    big_endian = args.big_endian
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
        if double:
            jtable = join_tbl_to_arr(fname_out+'.tmp', tables)      # 2D array
        else:
            jtable = join_tbl(fname_out+'.tmp', 'tmp', GLA, tables) # table
        print 'converting hdf5 -> bin ...'
        jtable[:].tofile(fname_out)
        fin.close()
        bin_info(fname_out+'.info', jtable)
        os.system('rm *.tmp')
        print "-"*10
    print 'done.'
    print 'last output ->', fname_out 
    close_files()


main(args)
