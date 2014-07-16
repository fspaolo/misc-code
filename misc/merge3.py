#!/usr/bin/env python
"""
Merge several HDF5 or ASCII files.

"""
# Fernando <fpaolo@ucsd.edu>
# November 2, 2012 

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 2D file[s] to merge')
parser.add_argument('-o', dest='fnameout', default='junk.h5', 
    help='name of file out')
args = parser.parse_args()

class TimeSeriesGrid(tb.IsDescription):
    satname = tb.StringCol(20, pos=1)
    time1 = tb.Int32Col(pos=2)
    time2 = tb.Int32Col(pos=3)

def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 

def get_fname_out(stem, fnamein, pref='', suf=''):
    path = os.path.split(fnamein)[0]
    return os.path.join(path, ''.join([pref, stem, suf, '.h5']))

def get_shape_out(fname):
    f = tb.openFile(fname, 'r')
    dh_mean = f.getNode('/dh_mean')
    i, j, k = dh_mean.shape
    dtype = dh_mean.dtype
    f.close()
    return (i, j, k, dtype)

files = args.files
fnameout = args.fnameout
#files = sys.argv[1:]
#files = sort()

nt, ny, nx, dtype = get_shape_out(files[0])
N = 136

fout = tb.openFile(fnameout, 'w')
filters = tb.Filters(complib='zlib', complevel=9)
atom = tb.Atom.from_dtype(dtype)
chunkshape = (1,ny,nx)  # chunk = slab to be saved
title = ''
g = '/'

tableo = fout.createTable(g, 'table', TimeSeriesGrid, title, filters)
lono = fout.createCArray(g, 'lon', atom, (nx,), title, filters)
lato = fout.createCArray(g, 'lat', atom, (ny,), title, filters)
x_edgeso = fout.createCArray(g, 'x_edges', atom, (nx+1,), title, filters)
y_edgeso = fout.createCArray(g, 'y_edges', atom, (ny+1,), title, filters)
dh_meano = fout.createCArray(g, 'dh_mean', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
dh_erroro = fout.createCArray(g, 'dh_error', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
dh_error2o = fout.createCArray(g, 'dh_error2', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
dg_meano = fout.createCArray(g, 'dg_mean', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
dg_erroro = fout.createCArray(g, 'dg_error', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
dg_error2o = fout.createCArray(g, 'dg_error2', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
n_ado = fout.createCArray(g, 'n_ad', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)
n_dao = fout.createCArray(g, 'n_da', atom, (N,ny,nx), title, filters, chunkshape=chunkshape)

for n, fnamein in enumerate(files):
    fin = tb.openFile(fnamein, 'r')
    dg_error = fin.getNode('/dg_error')
    dg_error2 = fin.getNode('/dg_error2')
    dg_mean = fin.getNode('/dg_mean')
    dh_error = fin.getNode('/dh_error')
    dh_error2 = fin.getNode('/dh_error2')
    dh_mean = fin.getNode('/dh_mean')
    lat = fin.getNode('/lat')
    lon = fin.getNode('/lon')
    n_ad = fin.getNode('/n_ad')
    n_da = fin.getNode('/n_da')
    table = fin.getNode('/table')
    x_edges = fin.getNode('/x_edges')
    y_edges = fin.getNode('/y_edges')

    tableo.append([(table.cols.satname[0], table.cols.time1[0], table.cols.time2[0])]) 
    tableo.flush()
    dh_meano[n,...] = dh_mean[:]
    dh_erroro[n,...] = dh_error[:]
    dh_error2o[n,...] = dh_error2[:]
    dg_meano[n,...] = dg_mean[:]
    dg_erroro[n,...] = dg_error[:]
    dg_error2o[n,...] = dg_error2[:]
    n_ado[n,...] = n_ad[:]
    n_dao[n,...] = n_da[:]

    print n, fnamein

lono[:] = lon[:]
lato[:] = lat[:]
x_edgeso[:] = x_edges[:]
y_edgeso[:] = y_edges[:]

close_files()
print 'done.'
print 'output ->', fnameout
