#!/usr/bin/env python
"""
Perform cross-calibration of time series. The input array must be 3d (t,y,x)
containing all the different satellite time-series together as a continuous
array per location (y,x).

Example
-------
python crosscalib.py /home/fpaolo/data/all/all_19920516_20111215_grids_mean.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 8, 2012

import sys
import argparse as ap
import matplotlib.pyplot as plt

from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = True
SAVE_TO_FILE = False
SAT_NAMES = ['ers1', 'ers2', 'envi']  # important for the order!
VAR_TO_CALIBRATE = 'dh_mean'

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs=1, 
    help='HDF5 files with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: same as input file]')

args = parser.parse_args()


def main(args):

    fname_in = args.file[0]
    fname_out = get_fname_out(fname_in, fname_out=args.fname_out, 
                              suffix='calib.h5')

    # input data --> dictionary and file
    #-----------------------------------------------------------------

    din = GetData(fname_in, 'a')
    nt, ny, nx = getattr(din, VAR_TO_CALIBRATE).shape  # i,j,k = t,y,x
    din.time = change_day(din.time, 15)    # change all days (e.g. 14,15,16,17) to 15
                                           # to match different satellites obs.

    print 'calibrating time series ...'

    # iterate over every grid cell (all times): j,k = y,x
    #-----------------------------------------------------------------

    isfirst = True

    if SAT_NAMES is None:
        satnames = np.unique(din.satname)

    count = 0
    for i in xrange(ny):
        for j in xrange(nx):

            ts_ij = din.dh_mean[:,i,j]  # all ts in one grid-cell
            if np.alltrue(np.isnan(ts_ij)): continue

            # get all time series (all sats) in one df (per grid-cell)
            df1 = create_df_with_sats(din.time, din.dh_mean[:,i,j], 
                din.satname, SAT_NAMES)
            df2 = create_df_with_sats(din.time, din.dg_mean[:,i,j], 
                din.satname, SAT_NAMES)

            # remove 'initial zeros' for cross-calibration
            if not np.alltrue(np.isnan(df1.values)):
                for col in df1.columns[1:]:      # all but first col
                    ind = first_non_null(df1[col])
                    if (ind is not None) and (df1[col][ind] == 0): 
                        df1[col][ind]= np.nan
                    ind = first_non_null(df2[col])
                    if (ind is not None) and (df2[col][ind] == 0): 
                        df2[col][ind] = np.nan

            ##############################################
            # TODO: check if there is overlap between TS #
            ##############################################

            # cross-calibrate: add offset (mean of diff) of overlaping parts
            df1['ers2'] += (df1['ers1'] - df1['ers2']).mean()
            df1['envi'] += (df1['ers2'] - df1['envi']).mean()
            df2['ers2'] += (df2['ers1'] - df2['ers2']).mean()
            df2['envi'] += (df2['ers2'] - df2['envi']).mean()

            if PLOT and (df1.count().sum() > 10):
                print 'grid-cell:', i, j
                df1.plot(linewidth=3, figsize=(9, 3), legend=False)
                plt.title('Elevation change, dh')
                plt.ylabel('m')
                plt.savefig('crosscalib_dh.png')
                df2.plot(linewidth=3, figsize=(9, 3), legend=False)
                plt.title('Backscatter change, dAGC')
                plt.ylabel('dB')
                plt.savefig('crosscalib_dg.png')
                plt.show()

            # average the calibrated cols (this will average only the overlaps)
            df1['dh_mean_all'] = df1.mean(axis=1)
            df2['dg_mean_all'] = df2.mean(axis=1)

            ######################################
            # TODO: how to propagate the errors? #
            ######################################

            if PLOT and (df1.count().sum() > 10):
                print 'grid-cell:', i, j
                df1['dh_mean_all'].plot(linewidth=3, figsize=(9, 3))
                plt.title('Elevation change, dh')
                plt.ylabel('m')
                plt.savefig('crosscalib_dh.png')
                plt.show()
                df2['dg_mean_all'].plot(linewidth=3, figsize=(9, 3))
                plt.title('Backscatter change, dAGC')
                plt.ylabel('dB')
                plt.savefig('crosscalib_dg.png')
                plt.show()

            # CODE REVISED TILL HERE ON DEC 18, 2012 -> CODE OK!

            if isfirst:
                # get sat/time info for TS (only once)
                time = np.asarray([int(t.strftime('%Y%m%d')) for t in df1.index])

            # save one TS per grid cell at a time
            #---------------------------------------------------------

            ##########################################################
            # TODO: apparently some problem with memory (on Triton)? #
            ##########################################################

            if not SAVE_TO_FILE: continue

            N = len(df1)  # <<<<<<< important!
            if isfirst:
                # open or create output file
                isfirst = False
                atom = tb.Atom.from_type('float64', dflt=np.nan)  # dflt is important!
                filters = tb.Filters(complib='zlib', complevel=9)
                t = din.file.createCArray('/', 'time_all', atom, (N,), '', filters)
                c1 = din.file.createCArray('/', 'dh_mean_all', atom, (N,ny,nx), '', filters)
                c2 = din.file.createCArray('/', 'dg_mean_all', atom, (N,ny,nx), '', filters)
                t[:] = time

            c1[:,i,j] = df1['dh_mean_all'].values
            c2[:,i,j] = df2['dg_mean_all'].values

            din.file.flush()
            print 'saved time series:', i,j

    din.file.close()

    print 'out file -->', fname_in


if __name__ == '__main__':
    main(args)
