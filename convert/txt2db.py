#!/usr/bin/env python
"""Create Table and add ASCII data files to an HDF5 database.

Note that a `description` class of the table has to be provided.
"""
# Fernando Paolo <fpaolo@ucsd.edu>
# January 1, 2010

import os
import sys
import mimetypes as mt
import argparse as ap
import numpy as np
import tables as tb 

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+',
    help='file[s] to convert [ex: /path/to/files/*.ext]')
parser.add_argument('-v', dest='verbose', default=False, action='store_const',
    const=True, help='for verbose [default: run silent]')
parser.add_argument('-d', dest='dbfile', default='data.h5',
     help='name of HDF5 database file [default: data.h5]')
parser.add_argument('-t', dest='tname', required=True,
     help='name of the table to be created')
parser.add_argument('-n', dest='node', required=True,
     help='node on which to create the table')
parser.add_argument('-c', dest='usecols', default=(),
    help='define in which `order` to add the ASCII columns, ex: -c 3,0,1 '
    '[default: as they are]')
parser.add_argument('-l', dest='complib', default='zlib',
    help='compression library to be used: zlib, lzo, bzip2, blosc ' 
    '[default: zlib]')

args = parser.parse_args()
files = args.files
dbfile = args.dbfile
tname = args.tname
node = args.node
usecols = args.usecols
complib = args.complib

class elevation(tb.IsDescription):
    orbit = tb.Int32Col(pos=1)
    utc85 = tb.Float64Col(pos=2)
    lon = tb.Float64Col(pos=3)
    lat = tb.Float64Col(pos=4)
    elev = tb.Float64Col(pos=5)
    agc = tb.Float64Col(pos=6)
    fmode = tb.Int8Col(pos=7)
    fret = tb.Int8Col(pos=8)
    fprob = tb.Int8Col(pos=9)


def files_to_table(fnames, table):
    for f in fnames:
        data = np.loadtxt(f, dtype=table.dtype, usecols=usecols)  # data in-memory
        table.append(data)
    table.flush()


def main():
    print 'files to add: %d ' % len(files)
    filters = tb.Filters(complib=complib, complevel=9)
    h5f = tb.openFile(dbfile, 'a')
    group = h5f.getNode(node)
    table = h5f.createTable(group, tname, elevation, filters=filters)
    files_to_table(files, table)
    h5f.flush()
    h5f.close()
    print 'done.'

if __name__ == '__main__':
    main()
