#!/usr/bin/env python
"""
Convert HDF5 data files (2D array) to flat Binary (stream of bytes).

If the extension of the input file is '.h5' the program performs convertion
from HDF5 to flat Binary. For any other extention it is assume convertion
from flat Binary to HDF5.

Fernando Paolo <fpaolo@ucsd.edu>
May 12, 2011
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
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
    const=True, help='load data in-memory (faster) [default: on-disk]')
parser.add_argument('-c', dest='ncol', type=int, default=10,
    help='number of elements per record (columns) [default: 9]')
parser.add_argument('-d', dest='dtype', default='f8',
    help='data type: i2, i4, f4, f8, ... [default: f8]')
parser.add_argument('-e', dest='ext', default=None,
    help='output file extension [default: .bin or .h5]')
parser.add_argument('-b', dest='big_endian', default=False, action='store_const',
    const=True, help='if data is big-endian [default: little-endian]')
args = parser.parse_args()

big_endian = args.big_endian
files = args.files
ext = args.ext
verbose = args.verbose
inmemory = args.inmemory
ncol = args.ncol
dtype = args.dtype

dt = np.dtype([('f1', '>i4'), ('f2', '>f8'), ('f3', '>f8'), 
               ('f4', '>f8'), ('f5', '>f8'), ('f6', '>f8'), 
               ('f7', '>i2'), ('f8', '>i2'), ('f9', '>i2')]) 

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


def bin_info(ncol, dtype, ext_in):
    if ext_in == '.h5':
        print 'output binary data record: %s x %s' % (ncol, dtype)
        f = open('bin_info.txt', 'w')
        f.write('binary data record: %s x %s' % (ncol, dtype))
        f.close()
    else:
        print 'input binary data record: %s x %s' % (ncol, dtype)


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
    print "new join table:", f
    # finally, close the file
    #f.close()
    return table


def tbl_to_bin(fname):
    print 'converting HDF5 -> BIN ...'
    print 'file:', fname
    if os.path.getsize(fname) == 0:     # check for empty files
        print 'empty file'
        continue                    
    fin = tb.openFile(fname, 'r')
    tables = [t for t in fin.root]      # get tables from input file
    jtable = join_tbl(fname + '.tmp', 'tmp', Descrt, tables)
    jtable.read().tofile(os.path.splitext(fname)[0] + '.bin')
    fin.close()
    print 'last output ->', f.split('.')[0] + ext_out 


print 'files to convert:', len(files)

if ext_in == '.h5':
    for fname in files:
        h5_to_bin(fname)
    print 'last output ->', fname.split('.')[0] + bin

bin_info(ncol, dtype, ext_in)
print 'done.'
