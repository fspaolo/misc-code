#!/usr/bin/env python
"""
FINDS OFFSET OF ONE VARIABLE ONLY !!!

Run the code as many times as variables to crosscalibrate, e.g., 
'dh_mean', 'dg_mean', 'dh_mean_mixed_const', 'dh_mean_short_const'.

The input array must be 3d (t,y,x) containing all the different satellite 
time-series together as a continuous array per location (y,x).

Example
-------
python getoffset.py \
        /data/alt/ra/ers1/hdf/antarctica/xovers/all_19920716_20111015_shelf_tide_grids_mts.h5

Notes
-----
To speed up the process move the file to be read to a local dir. 

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 8, 2012

from __future__ import division
import sys
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as si
import altimpy as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

FNAME_OUT = '/Users/fpaolo/data/shelves/offset_absval.h5.byfirst'

PLOT = False
PLOT_TS = False
SAVE_TO_FILE = True
DETREND = False                       # True for time-series offset only
FILTER = False                        # HP-filter before computing offset
SAT_BIAS = True                      # True=inter-satellite bias, False=time-series offset
LINEAR_FIT = False                     # True = fits a line to the overlaps, False = uses absolute vals
SUBSET = False                         # subset a small region
SAT_NAMES = ['ers1', 'ers2', 'envi']  # important for the order!
VAR_TO_CALIBRATE = 'dh_mean'
#VAR_TO_CALIBRATE = 'dg_mean'
#VAR_TO_CALIBRATE = 'dh_mean_mixed_const'
#VAR_TO_CALIBRATE = 'dh_mean_short_const'

LON, LAT = 186.375, -79.375  # check this before and after bs corr

#-------------------------------------------------------------------------

def detrend(y):
    """With support for NaNs."""
    y2 = y.copy()
    i_notnan, = np.where(~np.isnan(y))
    y2[i_notnan] = si.detrend(y[i_notnan])
    return y2


def main():

    fname_in = sys.argv[1] 

    din = GetData(fname_in, 'a')
    satname = din.satname
    time = change_day(din.time, 15)      # change all days (e.g. 14,15,16,17) to 15
    ts = getattr(din, VAR_TO_CALIBRATE)
    err = din.dh_error
    n_ad = din.n_ad
    n_da = din.n_da
    lon = din.lon
    lat = din.lat
    din.file.close()
    t = ap.num2year(time)

    if SUBSET: # get subset
        ts, lon2, lat2 = ap.get_subset(ap.amundsen, ts, lon, lat)
        err, lon2, lat2 = ap.get_subset(ap.amundsen, err, lon, lat)
        n_ad, lon2, lat2 = ap.get_subset(ap.amundsen, n_ad, lon, lat)
        n_da, lon2, lat2 = ap.get_subset(ap.amundsen, n_da, lon, lat)
        lon, lat = lon2, lat2

    xx, yy = np.meshgrid(lon, lat)
    nt, ny, nx = ts.shape
    offset_12 = np.full((ny,nx), np.nan)
    offset_23 = np.full((ny,nx), np.nan)

    print 'cross-calibrating time series:', VAR_TO_CALIBRATE

    isfirst = True

    if SAT_NAMES is None:
        satnames = np.unique(din.satname)

    # iterate over every grid cell (all times)
    no_overlap_12 = 0
    no_overlap_23 = 0
    for i in xrange(ny):
        for j in xrange(nx):

            if 0:
                i, j = ap.find_nearest2(xx, yy, (LON,LAT))
                i -= 0
                j += 0

            print 'grid-cell:', i, j

            ts_ij = ts[:,i,j]
            if np.isnan(ts_ij).all(): continue

            # get all time series (all sats) in one df (per grid-cell)
            var = create_df_with_sats(time, ts_ij, satname, SAT_NAMES)

            if DETREND:
                var = var.apply(detrend)

            if FILTER:
                var = var.apply(ap.hp_filt, lamb=7, nan=True)

            if PLOT_TS and (var.count().sum() > 10):
                print 'grid-cell:', i, j
                var.plot(linewidth=3, figsize=(9, 3), legend=False)
                plt.title('Elevation change, dh  (lon=%.2f, lat=%.2f)' % (xx[i,j], yy[i,j]))
                plt.ylabel('m')
                plt.show()

            # compute offset (if ts overlap)
            #---------------------------------------------------
            x = pd.notnull(var)
            overlap_12 = x['ers1'] & x['ers2']
            overlap_23 = x['ers2'] & x['envi']

            if np.sometrue(overlap_12):
                if SAT_BIAS:
                    s1 = var['ers1'][overlap_12]
                    s2 = var['ers2'][overlap_12]
                    if LINEAR_FIT:
                        # using linear fit
                        s1 = s1[s1.notnull() & s2.notnull()]
                        s2 = s2[s1.notnull() & s2.notnull()]
                        if len(s1) > 1 and len(s2) > 1:
                            s1.index, s1[:] = ap.linear_fit(ap.date2year(s1.index), s1.values)
                            s2.index, s2[:] = ap.linear_fit(ap.date2year(s2.index), s2.values)
                            offset = (s1.values[-1] - s1.values[0]) - (s2.values[-1] - s2.values[0])
                        else:
                            pass
                    else:
                        # using absolute values
                        s1 = ap.referenced(s1, to='first')
                        s2 = ap.referenced(s2, to='first')
                        s1[0], s2[0] = np.nan, np.nan # remove first values
                        offset = np.nanmean(s1 - s2)

                    #pd.concat((s1, s2), axis=1).plot(marker='o')
                else:
                    offset = np.nanmean(var['ers1'] - var['ers2'])
                offset_12[i,j] = offset
            else:
                no_overlap_12 += 1

            if np.sometrue(overlap_23):
                if SAT_BIAS:
                    s2 = var['ers2'][overlap_23]
                    s3 = var['envi'][overlap_23]
                    if LINEAR_FIT:
                        s2 = s2[s2.notnull() & s3.notnull()]
                        s3 = s3[s2.notnull() & s3.notnull()]
                        if len(s2) > 1 and len(s3) > 1:
                            s2.index, s2[:] = ap.linear_fit(ap.date2year(s2.index), s2.values)
                            s3.index, s3[:] = ap.linear_fit(ap.date2year(s3.index), s3.values)
                            offset = (s2.values[-1] - s2.values[0]) - (s3.values[-1] - s3.values[0])
                        else:
                            pass
                    else:
                        s2 = ap.referenced(s2, to='first')
                        s3 = ap.referenced(s3, to='first')
                        s2[0], s3[0] = np.nan, np.nan
                        offset = np.nanmean(s2 - s3)
                    #pd.concat((s2, s3), axis=1).plot(marker='o')
                    #plt.show()
                else:
                    offset = np.nanmean(var['ers2'] - var['envi'])
                offset_23[i,j] = offset 
            else:
                no_overlap_23 += 1

            #---------------------------------------------------

    mean_offset_12 = np.nanmean(offset_12)
    median_offset_12 = np.nanmedian(offset_12)
    mean_offset_23 = np.nanmean(offset_23)
    median_offset_23 = np.nanmedian(offset_23)

    if SAVE_TO_FILE:
        fout = tb.open_file(FNAME_OUT, 'w')
        fout.create_array('/', 'lon', lon)
        fout.create_array('/', 'lat', lat)
        fout.create_array('/', 'offset_12', offset_12)
        fout.create_array('/', 'offset_23', offset_23)
        fout.close()

    if PLOT:
        plt.figure()
        plt.subplot(211)
        offset_12 = ap.median_filt(offset_12, 3, 3)
        plt.imshow(offset_12, origin='lower', interpolation='nearest', vmin=-.5, vmax=.5)
        plt.title('ERS1-ERS2')
        plt.colorbar(shrink=0.8)
        plt.subplot(212)
        offset_23 = ap.median_filt(offset_23, 3, 3)
        plt.imshow(offset_23, origin='lower', interpolation='nearest', vmin=-.5, vmax=.5)
        plt.title('ERS2-Envisat')
        #plt.colorbar(shrink=0.3, orientation='h')
        plt.colorbar(shrink=0.8)
        plt.figure()
        plt.subplot(121)
        o12 = offset_12[~np.isnan(offset_12)]
        plt.hist(o12, bins=100)
        plt.title('ERS1-ERS2')
        ax = plt.gca()
        ap.intitle('mean/median = %.2f/%.2f m' % (mean_offset_12, median_offset_12), 
                    ax=ax, loc=2)
        plt.xlim(-1, 1)
        plt.subplot(122)
        o23 = offset_23[~np.isnan(offset_23)]
        plt.hist(o23, bins=100)
        plt.title('ERS2-Envisat')
        ax = plt.gca()
        ap.intitle('mean/median = %.2f/%.2f m' % (mean_offset_23, median_offset_23),
                    ax=ax, loc=2)
        plt.xlim(-1, 1)
        plt.show()

    print 'calibrated variable:', VAR_TO_CALIBRATE
    print 'no overlaps:', no_overlap_12, no_overlap_23
    print 'mean offset:', mean_offset_12, mean_offset_23
    print 'median offset:', median_offset_12, median_offset_23
    print 'out file ->', FNAME_OUT


if __name__ == '__main__':
    main()
