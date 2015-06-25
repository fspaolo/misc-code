#!/usr/bin/env python
"""
PROPAGATE ERROR AND NUMBER OF OBS DUE TO CROSS-CALIBRATION !!!

Example
-------
python crosscalib3.py ~/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 8, 2012

import sys
import pandas as pd
import matplotlib.pyplot as plt

from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = False
SAVE_TO_FILE = True
SAT_NAMES = ['ers1', 'ers2', 'envi']  # important for the order!
ERR1_CALIBRATED = 'dh_error_xcal'
ERR2_CALIBRATED = 'dh_error2_xcal'
ERR3_CALIBRATED = 'dg_error_xcal'
ERR4_CALIBRATED = 'dg_error2_xcal'
NAD_CALIBRATED = 'n_ad_xcal'
NDA_CALIBRATED = 'n_da_xcal'

#-------------------------------------------------------------------------

def main():

    fname_in = sys.argv[1] 

    din = GetData(fname_in, 'a')
    satname = din.satname
    time = change_day(din.time, 15)      # change all days (e.g. 14,15,16,17) to 15
    dh_err = din.dh_error
    dh_err2 = din.dh_error2
    dg_err = din.dg_error
    dg_err2 = din.dg_error2
    n_ad = din.n_ad
    n_da = din.n_da
    nt, ny, nx = dh_err.shape            # i,j,k = t,y,x

    print 'calibrating time series...'

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

            dh_err_ij = dh_err[:,i,j]
            dh_err2_ij = dh_err2[:,i,j]
            dg_err_ij = dg_err[:,i,j]
            dg_err2_ij = dg_err2[:,i,j]
            n_ad_ij = n_ad[:,i,j]
            n_da_ij = n_da[:,i,j]
            wij = pd.DataFrame()

            if np.alltrue(np.isnan(dh_err_ij)): continue

            # get all time series (all sats) in one df (per grid-cell)
            err1 = create_df_with_sats(time, dh_err_ij, satname, SAT_NAMES)
            err2 = create_df_with_sats(time, dh_err2_ij, satname, SAT_NAMES)
            err3 = create_df_with_sats(time, dg_err_ij, satname, SAT_NAMES)
            err4 = create_df_with_sats(time, dg_err2_ij, satname, SAT_NAMES)
            nad = create_df_with_sats(time, n_ad_ij, satname, SAT_NAMES)
            nda = create_df_with_sats(time, n_da_ij, satname, SAT_NAMES)
            nij = nad.combineAdd(nda)

            # 2) only crosscalibrate if time series overlap
            x = pd.notnull(err1)
            overlap_12 = x['ers1'] & x['ers2']
            overlap_23 = x['ers2'] & x['envi']
            if np.sometrue(overlap_12) and np.sometrue(overlap_23):
                pass
            else:
                # if any don't overlap, discard all time series!
                err1[:] = np.nan
                err2[:] = np.nan
                err3[:] = np.nan
                err4[:] = np.nan
                nad[:] = np.nan
                nda[:] = np.nan
                no_overlap += 1
                print 'NO OVERLAPPING FOUND:', i, j

            if PLOT and (err1.count().sum() > 10):
                print 'grid-cell:', i, j
                err1.plot(linewidth=3, figsize=(9, 3), legend=False)
                plt.title('Elevation change, dh')
                plt.ylabel('m')
                plt.savefig('crosscalib_dh.png')
                plt.show()

            # weight-average the calibrated cols (only the overlaps)

            # weights for the overlapping parts
            wij['w1'] = (nij['ers1'] / (nij['ers1'] + nij['ers2'])).fillna(1)
            wij['w2'] = (nij['ers2'] / (nij['ers1'] + nij['ers2'])).fillna(1)
            wij['w3'] = (nij['ers2'] / (nij['ers2'] + nij['envi'])).fillna(1)
            wij['w4'] = (nij['envi'] / (nij['ers2'] + nij['envi'])).fillna(1)

            # weighted-mean standard error of the overlapping parts
            # se_mean = sqrt(w1**2 * se1**2 + w2**2 * se2**2 + ...)
            err1[ERR1_CALIBRATED] = err1.mean(axis=1)
            err2[ERR2_CALIBRATED] = err2.mean(axis=1)
            err3[ERR3_CALIBRATED] = err3.mean(axis=1)
            err4[ERR4_CALIBRATED] = err4.mean(axis=1)

            idx = overlap_12  # ers1-ers2

            wsum = np.sqrt(wij['w1'][idx]**2 * err1['ers1'][idx]**2).add(
                           wij['w2'][idx]**2 * err1['ers2'][idx]**2)
            err1[ERR1_CALIBRATED][idx] = wsum

            wsum = np.sqrt(wij['w1'][idx]**2 * err2['ers1'][idx]**2).add(
                           wij['w2'][idx]**2 * err2['ers2'][idx]**2)
            err2[ERR2_CALIBRATED][idx] = wsum

            wsum = np.sqrt(wij['w1'][idx]**2 * err3['ers1'][idx]**2).add(
                           wij['w2'][idx]**2 * err3['ers2'][idx]**2)
            err3[ERR3_CALIBRATED][idx] = wsum

            wsum = np.sqrt(wij['w1'][idx]**2 * err4['ers1'][idx]**2).add(
                           wij['w2'][idx]**2 * err4['ers2'][idx]**2)
            err4[ERR4_CALIBRATED][idx] = wsum

            idx = overlap_23  # ers2-envi

            wsum = np.sqrt(wij['w3'][idx]**2 * err1['ers2'][idx]**2).add(
                           wij['w4'][idx]**2 * err1['envi'][idx]**2)
            err1[ERR1_CALIBRATED][idx] = wsum

            wsum = np.sqrt(wij['w3'][idx]**2 * err2['ers2'][idx]**2).add(
                           wij['w4'][idx]**2 * err2['envi'][idx]**2)
            err2[ERR2_CALIBRATED][idx] = wsum

            wsum = np.sqrt(wij['w3'][idx]**2 * err3['ers2'][idx]**2).add(
                           wij['w4'][idx]**2 * err3['envi'][idx]**2)
            err3[ERR3_CALIBRATED][idx] = wsum

            wsum = np.sqrt(wij['w3'][idx]**2 * err4['ers2'][idx]**2).add(
                           wij['w4'][idx]**2 * err4['envi'][idx]**2)
            err4[ERR4_CALIBRATED][idx] = wsum

            # weighted-mean number of observations
            wsum = (wij['w1'] * nad['ers1']).add(wij['w2'] * nad['ers2'], fill_value=0)
            wsum = (wij['w3'] * wsum).add(wij['w4'] * nad['envi'], fill_value=0)
            nad[NAD_CALIBRATED] = wsum

            wsum = (wij['w1'] * nda['ers1']).add(wij['w2'] * nda['ers2'], fill_value=0)
            wsum = (wij['w3'] * wsum).add(wij['w4'] * nda['envi'], fill_value=0)
            nda[NDA_CALIBRATED] = wsum

            if PLOT and (err1.count().sum() > 10):
                print 'grid-cell:', i, j
                err1.plot(linewidth=3, figsize=(9, 3))
                plt.title('Elevation change, dh')
                plt.ylabel('m')
                plt.savefig('crosscalib_dh.png')
                plt.show()

            # CODE REVISED TILL HERE ON DEC 18, 2012 -> CODE OK!

            # save one TS per grid cell at a time
            #---------------------------------------------------------

            if not SAVE_TO_FILE: continue

            N = len(err1)  # <<<<<<< important!
            if isfirst:
                # open or create output file
                isfirst = False
                # 'dflt=NaN' is important!
                atom = tb.Atom.from_type('float64', dflt=np.nan)  
                filters = tb.Filters(complib='zlib', complevel=9)
                c1 = din.file.create_carray('/', ERR1_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)
                c2 = din.file.create_carray('/', ERR2_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)
                c3 = din.file.create_carray('/', ERR3_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)
                c4 = din.file.create_carray('/', ERR4_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)
                c5 = din.file.create_carray('/', NAD_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)
                c6 = din.file.create_carray('/', NDA_CALIBRATED, atom, 
                                           (N,ny,nx), '', filters)

            c1[:,i,j] = err1[ERR1_CALIBRATED].values
            c2[:,i,j] = err2[ERR2_CALIBRATED].values
            c3[:,i,j] = err3[ERR3_CALIBRATED].values
            c4[:,i,j] = err4[ERR4_CALIBRATED].values
            c5[:,i,j] = nad[NAD_CALIBRATED].values
            c6[:,i,j] = nda[NDA_CALIBRATED].values

            din.file.flush()
            print 'saved time series:', i,j

    din.file.close()

    print 'discarded time series with no overlap:', no_overlap
    print 'calibrated variable name: errors and # observations'
    print 'out file:', fname_in


if __name__ == '__main__':
    main()
