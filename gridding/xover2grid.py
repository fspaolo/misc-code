#!/usr/bin/env python
"""
Take crossover files as input and construct dh time series for
every bin on a grid --> output: several grids.

Example
-------
$ python xover2grid.py ~/data/fris/xover/seasonal/ers1_199*_tide.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap
import matplotlib.pyplot as plt

from funcs import *

sys.path.append('/Users/fpaolo/code/misc')
from util import *

#---------------------------------------------------------------------
# global variables
#---------------------------------------------------------------------

ABSVAL = 10  # m, to edit abs(dh)
MAX_SIZE_DATA = 512  # MB, to load data in memory

#---------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='crossover files to read (HDF5 2D array)')
parser.add_argument('-r', dest='region', nargs=4, type=float, 
    metavar=('l', 'r', 'b', 't'), default=(-100, -20, -82, -75),
    help='coordinates of grid domain: left right bottom top')
parser.add_argument('-d', dest='delta', nargs=2, type=float, 
    metavar=('dx', 'dy'), default=(1.2, 0.4),
    help='size of grid cells: dx dy (deg) [default: 1 1/3]')

args = parser.parse_args()

def main(args):

    files = args.files
    x_range = (args.region[0], args.region[1])
    y_range = (args.region[2], args.region[3])
    dx = args.delta[0]
    dy = args.delta[1]

    files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s))

    isfirst = True
    for pos, fname in enumerate(files):

        f = tb.openFile(fname)
        check_if_can_be_loaded(f.root.data, MAX_SIZE_DATA)
        data = f.root.data.read()      # in-memory --> faster!

        # 1. FILTER DATA FIRST

        fmode1 = data[:,10]
        fmode2 = data[:,11]
        fbord1 = data[:,18]
        fbord2 = data[:,19]

        '''
        condition = ((fmode1 == fmode2) & (fbord1 == 0) & (fbord2 == 0)) 
        '''
        condition = ((fmode1 == 1) & (fmode2 == 1) & \
                     (fbord1 == 0) & (fbord2 == 0))    # ice mode
        '''
        condition = ((fmode1 == 0) & (fmode2 == 0) & \
                     (fbord1 == 0) & (fbord2 == 0))    # fine mode
        '''
        ind, = np.where(condition)

        if len(ind) < 1:    # go to next file
            f.close()
            continue

        data = data[ind,:]

        d = {}
        d['lon'] = data[:,0]
        d['lat'] = data[:,1]
        d['h1'] = data[:,6]
        d['h2'] = data[:,7]
        d['g1'] = data[:,8]
        d['g2'] = data[:,9]
        d['ftrack1'] = data[:,20]
        d['ftrack2'] = data[:,21]
        d['tide1'] = data[:,24]
        d['tide2'] = data[:,25]
        d['load1'] = data[:,26]
        d['load2'] = data[:,27]

        # 2. APPLY CORRECTIONS 

        d = apply_tide_and_load_corr(d)

        del data, fmode1, fmode2, fbord1, fbord2
        del d['tide1'], d['tide2'], d['load1'], d['load2']

        #-----------------------------------------------------------------

        # digitize lons and lats
        x_edges = np.arange(x_range[0], x_range[-1]+dx, dx)
        y_edges = np.arange(y_range[0], y_range[-1]+dy, dy)
        j_bins = np.digitize(d['lon'], bins=x_edges)
        i_bins = np.digitize(d['lat'], bins=y_edges)
        nx, ny = len(x_edges)-1, len(y_edges)-1
        hx, hy = dx/2., dy/2.
        lon = (x_edges + hx)[:-1]
        lat = (y_edges + hy)[:-1]

        # output grids 
        dh_mean = np.empty((ny,nx), 'f8') * np.nan
        dh_error = np.empty_like(dh_mean) * np.nan
        dg_mean = np.empty_like(dh_mean) * np.nan
        dg_error = np.empty_like(dh_mean) * np.nan
        n_ad = np.empty((ny,nx), 'i4')
        n_da = np.empty_like(n_ad)

        #-----------------------------------------------------------------

        # calculations per grid cell
        for i in xrange(ny):
            for j in xrange(nx):
                '''
                ind, = np.where((x_edges[j] <= d['lon']) & \
                                (d['lon'] < x_edges[j+1]) & \
                                (y_edges[i] <= d['lat']) & \
                                (d['lat'] < y_edges[i+1]))
                '''
                # single grid cell
                ind, = np.where((j_bins == j+1) & (i_bins == i+1))

                # separate --> asc/des, des/asc 
                dh_ad, dh_da = compute_dh_ad_da(d['h1'][ind], d['h2'][ind], 
                               d['ftrack1'][ind], d['ftrack2'][ind])

                ### dh TS

                # filter
                #dh_ad = std_iterative_editing(dh_ad, nsd=3)
                #dh_da = std_iterative_editing(dh_da, nsd=3)
                dh_ad = abs_value_editing(dh_ad, absval=ABSVAL)
                dh_da = abs_value_editing(dh_da, absval=ABSVAL)

                # mean values
                #dh_mean[i,j] = compute_ordinary_mean(dh_ad, dh_da, useall=False) 
                dh_mean[i,j] = compute_weighted_mean(dh_ad, dh_da, useall=False) 
                dh_error[i,j] = compute_weighted_mean_error(dh_ad, dh_da, useall=False) 
                #dh_error[i,j] = compute_wingham_error(dh_ad, dh_da, useall=False) 
                n_ad[i,j] = len(dh_ad)
                n_da[i,j] = len(dh_da)

                ### dAGC TS

                dg_ad, dg_da = compute_dh_ad_da(d['g1'][ind], d['g2'][ind], 
                               d['ftrack1'][ind], d['ftrack2'][ind])

                #dg_ad = std_iterative_editing(dg_ad, nsd=3)
                #dg_da = std_iterative_editing(dg_da, nsd=3)

                dg_mean[i,j] = compute_weighted_mean(dg_ad, dg_da, useall=False) 
                dg_error[i,j] = compute_weighted_mean_error(dg_ad, dg_da, useall=False) 

        #-----------------------------------------------------------------

        # get time info for every grid
        sat_name, ref_time, year, month = get_time_from_fname(fname)  

        # save the grids
        if isfirst:
            isfirst = False
            fname_out = get_fname_out(files, sufix='dh_grids.h5')
            title = 'FRIS Elevation Change Grids'
            filters = tb.Filters(complib='blosc', complevel=9)
            atom = tb.Atom.from_dtype(dh_mean.dtype)
            ni, nj = dh_mean.shape
            db = tb.openFile(fname_out, 'w')

            g = db.createGroup('/', 'fris')
            t1 = db.createTable(g, 'ts', TimeSeriesGrid, title, filters)
            e1 = db.createEArray(g, 'dh_mean', atom, (ni, nj, 0), '', filters)
            e2 = db.createEArray(g, 'dh_error', atom, (ni, nj, 0), '', filters)
            e3 = db.createEArray(g, 'dg_mean', atom, (ni, nj, 0), '', filters)
            e4 = db.createEArray(g, 'dg_error', atom, (ni, nj, 0), '', filters)
            e5 = db.createEArray(g, 'n_ad', atom, (ni, nj, 0), '', filters)
            e6 = db.createEArray(g, 'n_da', atom, (ni, nj, 0), '', filters)

            c1 = db.createCArray(g, 'x_edges', atom, (nj+1,), '', filters)
            c2 = db.createCArray(g, 'y_edges', atom, (ni+1,), '', filters)
            c3 = db.createCArray(g, 'lon', atom, (nj,), '', filters)
            c4 = db.createCArray(g, 'lat', atom, (ni,), '', filters)

            c1[:] = x_edges
            c2[:] = y_edges
            c3[:] = lon
            c4[:] = lat

        t1.append([(sat_name, ref_time, year, month)])
        e1.append(dh_mean.reshape(ni, nj, 1))
        e2.append(dh_error.reshape(ni, nj, 1))
        e3.append(dg_mean.reshape(ni, nj, 1))
        e4.append(dg_error.reshape(ni, nj, 1))
        e5.append(n_ad.reshape(ni, nj, 1))
        e6.append(n_da.reshape(ni, nj, 1))
        t1.flush()

        f.close()

    db.flush()
    db.close()

    print_info(x_edges, y_edges, lon, lat, dx, dy)

    plot_grids(x_edges, y_edges, dh_mean, dg_mean, n_ad, n_da)
    plt.show()

    print 'file out -->', fname_out


if __name__ == '__main__':
    main(args)
