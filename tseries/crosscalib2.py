#!/usr/bin/env python
"""
CROSS-CALIBRATES ONE VARIABLE ONLY !!!

Run the code as many times as variables to crosscalibrate, e.g., 
'dh_mean', 'dg_mean', 'dh_mean_mixed_const', 'dh_mean_short_const'.

Cross-calibrate the time series and weight-average the overlapping parts. The
input array must be 3d (t,y,x) containing all the different satellite 
time-series together as a continuous array per location (y,x).

Example
-------
python crosscalib2.py \
        /data/alt/ra/ers1/hdf/antarctica/xovers/all_19920716_20111015_shelf_tide_grids_mts.h5

Notes
-----
To speed up the process move the file to be read to a local dir. 

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 8, 2012

import sys
import pandas as pd
import matplotlib.pyplot as plt

from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = True
SAVE_TO_FILE = False
SAT_NAMES = ['ers1', 'ers2', 'envi']  # important for the order!
#VAR_TO_CALIBRATE = 'dh_mean'
#VAR_CALIBRATED = 'dh_mean_xcal'
#VAR_TO_CALIBRATE = 'dg_mean'
#VAR_CALIBRATED = 'dg_mean_xcal'
VAR_TO_CALIBRATE = 'dh_mean_mixed_const'
VAR_CALIBRATED = 'dh_mean_mixed_const_xcal'
#VAR_TO_CALIBRATE = 'dh_mean_short_const'
#VAR_CALIBRATED = 'dh_mean_short_const_xcal'

#-------------------------------------------------------------------------

def main():

    fname_in = sys.argv[1] 

    din = GetData(fname_in, 'a')
    satname = din.satname
    time = change_day(din.time, 15)      # change all days (e.g. 14,15,16,17) to 15
    ts = getattr(din, VAR_TO_CALIBRATE)
    n_ad = din.n_ad
    n_da = din.n_da
    nt, ny, nx = ts.shape                # i,j,k = t,y,x

    print 'calibrating time series:', VAR_TO_CALIBRATE

    isfirst = True

    if SAT_NAMES is None:
        satnames = np.unique(din.satname)

    # iterate over every grid cell (all times)
    #-----------------------------------------------------------------

    count = 0
    no_overlap = 0
    for i in xrange(ny):
        for j in xrange(nx):
            print 'grid-cell:', i, j

            ts_ij = ts[:,i,j]
            n_ad_ij = n_ad[:,i,j]
            n_da_ij = n_da[:,i,j]
            wij = pd.DataFrame()

            if np.alltrue(np.isnan(ts_ij)): continue

            # get all time series (all sats) in one df (per grid-cell)
            var = create_df_with_sats(time, ts_ij, satname, SAT_NAMES)
            nad = create_df_with_sats(time, n_ad_ij, satname, SAT_NAMES)
            nda = create_df_with_sats(time, n_da_ij, satname, SAT_NAMES)
            nij = nad.combineAdd(nda)

            # 1) [no!] remove 'initial zeros' for cross-calibration
            '''
            if not np.alltrue(np.isnan(var.values)):
                for col in var.columns[1:]:      # all but first col
                    ind = first_non_null(var[col])
                    if (ind is not None) and (var[col][ind] == 0): 
                        var[col][ind]= np.nan
            '''

            # 2) only crosscalibrate if time series overlap
            x = pd.notnull(var)
            overlap_12 = x['ers1'] & x['ers2']
            overlap_23 = x['ers2'] & x['envi']
            if np.sometrue(overlap_12) and np.sometrue(overlap_23):
                # add offset (mean of diff) of overlaping parts
                var['ers2'] += (var['ers1'] - var['ers2']).mean()
                var['envi'] += (var['ers2'] - var['envi']).mean()
            else:
                # if any don't overlap, discard all time series!
                var[:] = np.nan
                no_overlap += 1
                print 'NO OVERLAPPING FOUND:', i, j

            if PLOT and (var.count().sum() > 10):
                print 'grid-cell:', i, j
                nij.plot(legend=False)
                var.plot(linewidth=3, figsize=(9, 3), legend=False)
                plt.title('Elevation change, dh')
                plt.ylabel('m')
                plt.savefig('crosscalib_dh.png')
                plt.show()

            # 3) average the calibrated cols (only the overlaps)
            # weights for the overlapping parts
            wij['w1'] = (nij['ers1'] / (nij['ers1'] + nij['ers2'])).fillna(1)
            wij['w2'] = (nij['ers2'] / (nij['ers1'] + nij['ers2'])).fillna(1)
            wij['w3'] = (nij['ers2'] / (nij['ers2'] + nij['envi'])).fillna(1)
            wij['w4'] = (nij['envi'] / (nij['ers2'] + nij['envi'])).fillna(1)
            # weighted mean of the overlapping parts
            wsum = (wij['w1'] * var['ers1']).add(wij['w2'] * var['ers2'], fill_value=0)
            wsum = (wij['w3'] * wsum).add(wij['w4'] * var['envi'], fill_value=0)
            var[VAR_CALIBRATED] = wsum
            #var[VAR_CALIBRATED] = var.mean(axis=1)

            if PLOT and (var.count().sum() > 10):
                print 'grid-cell:', i, j
                var[VAR_CALIBRATED].plot(linewidth=3, figsize=(9, 3))
                plt.title('Elevation change, dh')
                plt.ylabel('m')
                plt.savefig('crosscalib_dh.png')
                plt.show()

            # CODE REVISED TILL HERE ON DEC 18, 2012 -> CODE OK!

            if isfirst:
                # get sat/time info for TS (only once)
                time_xcal = np.asarray([int(t.strftime('%Y%m%d')) \
                                        for t in var.index])

            # save one TS per grid cell at a time
            #---------------------------------------------------------

            ##########################################################
            # TODO: apparently some problem with memory (on Triton)? #
            ##########################################################

            if not SAVE_TO_FILE: continue

            N = len(var)  # <<<<<<< important!
            if isfirst:
                print 'Opening output file.'
                # open output file
                isfirst = False
                # 'dflt=NaN' is important!
                atom = tb.Atom.from_type('float64', dflt=np.nan)  
                filters = tb.Filters(complib='zlib', complevel=9)
                try:
                    t = din.file.createCArray('/', 'time_xcal', atom, (N,), '', 
                                               filters)
                    t[:] = time_xcal
                except:
                    pass
                c1 = din.file.createCArray('/', VAR_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)

            c1[:,i,j] = var[VAR_CALIBRATED].values

            din.file.flush()
            print 'saved time series:', i,j

    din.file.close()

    print 'discarded time series with no overlap:', no_overlap
    print 'calibrated variable name:', VAR_CALIBRATED
    print 'out file:', fname_in


if __name__ == '__main__':
    main()
