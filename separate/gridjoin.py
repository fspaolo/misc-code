#!/usr/bin/env python
"""
Join several sub-grids (HDF5 files) in space and time into a 3d array.

Join all files that have a common (given) pattern in the name. The patterns
may be numbers and/or characters. Example: 'YYYYMMDD', where YYYY is year, MM
is month and DD is day. Attention, *The pattern is hard coded!*

Notes
-----
Run the code on each satellite independently. Do the merging at the end.

To avoid Unix limitation on number of cmd args, pass instead of file names a
string as "/path/to/data/file_*_name.ext"

Example
-------
python gridjoin.py -r 0 360 -82 -62 -d .75 .25 \
    "/data/alt/ra/ers1/hdf/antarctica/xovers/ers1_*_grid.h5"

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# Jan 21, 2013

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap
from glob import glob

from funcs import lon_180_to_360, define_sectors

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='files with sub-grids to merge (HDF5 2D arrays)')
parser.add_argument('-r', dest='region', nargs=4, type=float, 
    metavar=('l', 'r', 'b', 't'), default=(0, 360, -82, -62),
    help='coordinates of full-grid domain (edges): left right bottom top')
parser.add_argument('-d', dest='delta', nargs=2, type=float, 
    metavar=('dx', 'dy'), default=(1.2, 0.4),
    help='size of grid cells: dx dy (deg), default 1.2 x 0.4')
args = parser.parse_args()


class GetData(object):
    def __init__(self, fname):
        fin = tb.openFile(fname)
        self.file = fin
        self.time1 = fin.getNode('/time1')[:]
        self.time2 = fin.getNode('/time2')[:]
        self.lon = fin.getNode('/lon')[:]
        self.lat = fin.getNode('/lat')[:]
        self.x_edges = fin.getNode('/x_edges')[:]
        self.y_edges = fin.getNode('/y_edges')[:]
        self.dh_mean = fin.getNode('/dh_mean')[:]
        self.dh_error = fin.getNode('/dh_error')[:]
        self.dh_error2 = fin.getNode('/dh_error2')[:]
        self.dg_mean = fin.getNode('/dg_mean')[:]
        self.dg_error = fin.getNode('/dg_error')[:]
        self.dg_error2 = fin.getNode('/dg_error2')[:]
        self.n_ad = fin.getNode('/n_ad')[:]
        self.n_da = fin.getNode('/n_da')[:]


class OutputContainers(object):
    def __init__(self, fname_out, shape):
        fout = tb.openFile(fname_out, 'w')
        nt, ny, nx = shape
        filters = tb.Filters(complib='zlib', complevel=9)
        atom = tb.Atom.from_type('float64', dflt=np.nan) # dflt is important!
        chunkshape = (1, ny, nx)                         # chunk (slab to be saved)
        title = ''
        self.file = fout
        self.time1 = fout.createCArray('/', 'time1', atom, (nt,), 
            title, filters)
        self.time2 = fout.createCArray('/', 'time2', atom, (nt,), 
            title, filters)
        self.lon = fout.createCArray('/', 'lon', atom, (nx,), 
            title, filters)
        self.lat = fout.createCArray('/', 'lat', atom, (ny,), 
            title, filters)
        self.x_edges = fout.createCArray('/', 'x_edges', atom, (nx+1,), 
            title, filters)
        self.y_edges = fout.createCArray('/', 'y_edges', atom, (ny+1,), 
            title, filters)
        self.dh_mean = fout.createCArray('/', 'dh_mean', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.dh_error = fout.createCArray('/', 'dh_error', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.dh_error2 = fout.createCArray('/', 'dh_error2', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.dg_mean = fout.createCArray('/', 'dg_mean', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.dg_error = fout.createCArray('/', 'dg_error', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.dg_error2 = fout.createCArray('/', 'dg_error2', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.n_ad = fout.createCArray('/', 'n_ad', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)
        self.n_da = fout.createCArray('/', 'n_da', atom, (nt,ny,nx), 
            title, filters, chunkshape=chunkshape)


def get_times(fnames):
    times = np.unique([re.findall('\d\d\d\d\d\d+_\d\d\d\d\d\d+', f)[0] for f in fnames])
    t1 = np.asarray([int(t.split('_')[0]) for t in times])
    t2 = np.asarray([int(t.split('_')[-1]) for t in times])
    return [t1, t2]


def get_fname_out(fnames):
    fname = fnames[0]
    times = np.ravel([re.findall('\d\d\d\d\d\d+', f) for f in fnames])
    tminmax_new = '_'.join([min(times), max(times)])
    tminmax_old = re.findall('\d\d\d\d\d\d+_\d\d\d\d\d\d+', fname)[0]
    nreg = re.findall('_\d\d_', fname)[0]
    fname = fname.replace(tminmax_old, tminmax_new)
    fname = fname.replace(nreg, '_')
    fname = fname.replace('grid', 'grids') 
    return fname


def get_grid_coords(region, dx, dy):
    """Generate coords for each grid-cell (center)."""
    l, r, b, t = region
    hx, hy = dx/2., dy/2.
    x_coords = np.arange(l, r, dx) + hx
    y_coords = np.arange(b, t, dy) + hy
    return [x_coords.round(6), y_coords.round(6)]


def get_grid_edges(x_coords, y_coords):
    dx = x_coords[1] - x_coords[0]
    dy = y_coords[1] - y_coords[0]
    x_edges = np.append(x_coords, x_coords[-1] + dx) 
    y_edges = np.append(y_coords, y_coords[-1] + dy) 
    return [x_edges - dx/2, y_edges - dy/2]


def get_indices(lons, x_coords):
    """Get full-grid indices of respective subgrid coords."""
    j1, = np.where(lons.min().round(6) == x_coords)
    j2, = np.where(lons.max().round(6) == x_coords)
    return (j1, j2)


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


def main(args):
    if len(args.files) > 1:
        files = args.files
    else:
        # use 'glob' instead
        files = glob(args.files[0])   

    L, R, B, T = args.region  # full-grid 2d (edges)
    dx = args.delta[0]
    dy = args.delta[1]

    files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s)) # important!
    time1, time2 = get_times(files)                          # full-grid 2d
    fname_out = get_fname_out(files)                         # full-grid 2d
    x_coords, y_coords = get_grid_coords((L,R,B,T), dx, dy)  # full-grid 2d
    x_edges, y_edges = get_grid_edges(x_coords, y_coords)    # full-gird 2d

    # construct full-grid 3d
    nt = len(time1)
    ny = (T - B) / dy
    nx = (R - L) / dx

    dout = OutputContainers(fname_out, (nt,ny,nx))

    dout.time1[:] = time1[:]
    dout.time2[:] = time2[:]
    dout.lon[:] = x_coords[:]
    dout.lat[:] = y_coords[:]
    dout.x_edges[:] = x_edges[:]
    dout.y_edges[:] = y_edges[:]

    for fname in files:
        din = GetData(fname)    # in-memory
        t1, t2 = int(din.time1), int(din.time2)

        i, = np.where((t1 == time1) & (t2 == time2))
        k1, k2 = get_indices(din.lon, x_coords)

        dout.dh_mean[i,:,k1:k2+1] = din.dh_mean
        dout.dh_error[i,:,k1:k2+1] = din.dh_error
        dout.dh_error2[i,:,k1:k2+1] = din.dh_error2
        dout.dg_mean[i,:,k1:k2+1] = din.dg_mean
        dout.dg_error[i,:,k1:k2+1] = din.dg_error
        dout.dg_error2[i,:,k1:k2+1] = din.dg_error2
        dout.n_ad[i,:,k1:k2+1] = din.n_ad
        dout.n_da[i,:,k1:k2+1] = din.n_da

        dout.file.flush()
        din.file.close()

        print 'file:', fname
        print 'grid, t1, t2:', i, k1, k2

    close_files()
    print 'file out ->', fname_out


if __name__ == '__main__':
    main(args)
