#!/usr/bin/env python
"""
Take crossover files as input and construct dh time series for
a given box region.

Example
-------
$ python thisprog.py /path/to/data/ers1_*_tide.h5

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

ABSVAL = 10  # m, absolute value to edit dh
MAX_SIZE_DATA = 512  # MB, to load data in memory

#---------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='crossover files to read (HDF5 2D array)')
parser.add_argument('-r', dest='region', nargs=4, type=float, 
    metavar=('L', 'R', 'B', 'T'), default=(-61.7, -60.4, -78.2, -77.8),
    help='coordinates of `box` region: left right bottom top')

args = parser.parse_args()

def main(args):

    files = args.files
    left, right, bottom, top = args.region

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

        if len(ind) < 1:
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
        #print 'calculating box region ...'

        # calculations per box (given region)
        ind, = np.where((left <= d['lon']) & (d['lon'] <= right) & \
                        (bottom <= d['lat']) & (d['lat'] <= top))

        # dh TS

        # separate --> asc/des, des/asc 
        dh_ad, dh_da = compute_dh_ad_da(d['h1'][ind], d['h2'][ind], 
                       d['ftrack1'][ind], d['ftrack2'][ind])

        # filter
        #dh_ad = std_iterative_editing(dh_ad, nsd=3)
        #dh_da = std_iterative_editing(dh_da, nsd=3)
        #dh_ad = abs_value_editing(dh_ad, absval=ABSVAL)
        #dh_da = abs_value_editing(dh_da, absval=ABSVAL)

        # mean values
        #dh_mean = compute_ordinary_mean(dh_ad, dh_da, useall=False) 
        dh_mean = compute_weighted_mean(dh_ad, dh_da, useall=False) 
        dh_error = compute_weighted_mean_error(dh_ad, dh_da, useall=False) 
        #dh_error = compute_wingham_error(dh_ad, dh_da, useall=False) 
        n_ad = len(dh_ad)
        n_da = len(dh_da)

        # dAGC TS

        dg_ad, dg_da = compute_dh_ad_da(d['g1'][ind], d['g2'][ind], 
                       d['ftrack1'][ind], d['ftrack2'][ind])

        #dg_ad = std_iterative_editing(dg_ad, nsd=3)
        #dg_da = std_iterative_editing(dg_da, nsd=3)

        dg_mean = compute_weighted_mean(dg_ad, dg_da, useall=False) 
        dg_error = compute_weighted_mean_error(dg_ad, dg_da, useall=False) 

        #-----------------------------------------------------------------

        #print 'npts ad/da:', n_ad, n_da 
        #print 'dh mean:', dh_mean

        if dh_mean is None or dh_mean == np.nan:
            f.close()
            continue

        # get time info for every <dh>
        sat_name, ref_time, year, month = get_time_from_fname(fname)  

        if isfirst:
            isfirst = False
            fname_out = get_fname_out(files, sufix='dh_box.h5')
            title = 'FRIS Time Series'
            filters = tb.Filters(complib='blosc', complevel=9)
            db = tb.openFile(fname_out, 'w')
            g = db.createGroup('/', 'tseries')
            t = db.createTable(g, 'ts', TimeSeries, title, filters)

            # in-memory container
            #dataout = np.empty((len(files), len(t.cols)), dtype=t.dtype)

        t.append([(sat_name, ref_time, year, month, dh_mean, dh_error, 
            dg_mean, dg_error, n_ad, n_da)])
        t.flush()

        f.close()

        #if pos == 20: break
        #------------------------------------------------------------------

    plot_ts(t)
    plt.show()

    print 'done.'

    db.flush()
    db.close()

    print 'file out -->', fname_out

if __name__ == '__main__':
    main(args)
