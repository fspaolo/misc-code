#!/usr/bin/env python
"""
Reference and weigh-average the multi-reference time series.

Notes
-----
* Process each satellite independently and merge the files later on.
* Indices: i,j,k = t,y,x
* Process file on a local dir for speed!
* Warning! If using 'reference by first', the non-null diagonal will slightly
  "improve" the average by introducing one repeated value per column (in the
  diagonal). To avoid this, the diagonal should be changed to NaNs. On the
  other hand, this improvement (by one repeated observation) is negligible!

* Reference by offset (default) using two-sided matrix:

            1           2           3           4           5

    a     Daa         h12+Daa     h13+Daa     h14+Daa     h15+Daa
    b    -h12+Dab     Dab         h23+Dab     h24+Dab     h25+Dab
    c    -h13+Dac    -h23+Dab     Dac         h34+Dac     h35+Dac
    d    -h14+Dad    -h24+Dad    -h34+Dad     Dad         h45+Dad
    e    -h15+Dae    -h25+Dae    -h35+Dae    -h45+Dae     Dae

    where h21 = -h12 (and so on); and the offset between a(t) and b(t) is 
    Dab = E[a(t)-b(t)]. Note that Daa = 0.

Examples
--------
$ python averagets2.py /data/alt/ra/ers1/hdf/antarctica/xovers/*_grids.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as arg
import altimpy as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = True
SAVE_TO_FILE = False
FULL_MATRIX = True  #True/False = two-sided/one-sided matrix

#-------------------------------------------------------------------------

# parse command line arguments
parser = arg.ArgumentParser()
parser.add_argument('file_in', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/<file_in>_mts.h5]')
parser.add_argument('-s', dest='suffix', default='mts.h5',
    help='suffix for output file [default: mts.h5]')

args = parser.parse_args()

def main(args):

    fname_in = args.file_in[0]
    fname_out = args.fname_out
    suffix = args.suffix

    fname_out = get_fname_out(fname_in, fname_out=fname_out, suffix=suffix)

    ### input data
    din = GetData(fname_in)  # all in-memory (load it once!)
    _, ny, nx = din.dh_mean.shape

    print 'processing time series ...'

    # for every grid-cell pull all ts and construct one df (matrix) 
    #---------------------------------------------------------------------

    isfirst = True
    for i in xrange(ny):
        for j in xrange(nx):

            # test all time series in one grid-cell
            if np.alltrue(np.isnan(din.dh_mean[:,i,j])): continue

            # grid-cell coords for plotting
            #x, y = np.meshgrid(din.lon, din.lat)
            #i, j = ap.find_nearest2(x, y, (191.500 , -81.400))
            lat_i, lon_i = din.lat[i], din.lon[j]

            # SOMETHING VERY WRONG HERE (SEE FIX BELOW)!!!!!!
            # create 1 DF for every grid cell: one-sided matrix
            dh_mean_ij = create_df_with_ts(din.time1, din.time2, 
                din.dh_mean[:,i,j], months=3)
            dh_error_ij = create_df_with_ts(din.time1, din.time2, 
                din.dh_error[:,i,j], months=3)
            dh_error2_ij = create_df_with_ts(din.time1, din.time2, 
                din.dh_error2[:,i,j], months=3)
            dg_mean_ij = create_df_with_ts(din.time1, din.time2, 
                din.dg_mean[:,i,j], months=3) 
            dg_error_ij = create_df_with_ts(din.time1, din.time2, 
                din.dg_error[:,i,j], months=3)
            dg_error2_ij = create_df_with_ts(din.time1, din.time2, 
                din.dg_error2[:,i,j], months=3)
            n_ad_ij = create_df_with_ts(din.time1, din.time2, 
                din.n_ad[:,i,j], months=3)
            n_da_ij = create_df_with_ts(din.time1, din.time2, 
                din.n_da[:,i,j], months=3)

            # TEMP FIX (SOMETHING WRONG WITH THE CREATION OF THE DATA FRAMES)!
            dh_mean_ij = redo_df(dh_mean_ij)
            dg_mean_ij = redo_df(dg_mean_ij)
            dh_error_ij = redo_df(dh_error_ij)
            dg_error_ij = redo_df(dg_error_ij)
            dh_error2_ij = redo_df(dh_error2_ij)
            dg_error2_ij = redo_df(dg_error2_ij)
            n_ad_ij = redo_df(n_ad_ij)
            n_da_ij = redo_df(n_da_ij)
            n_ij = n_ad_ij.combineAdd(n_da_ij)

            '''
            print dh_mean_ij.count().sum()
            print n_ij.count().sum()
            print n_ad_ij.count().sum(), n_da_ij.count().sum()
            print dh_mean_ij.to_string()
            print dh_error2_ij.to_string()
            print n_ij.to_string()
            '''

            if np.alltrue(dh_mean_ij.isnull()): continue

            # use two-sided matrix -> lower = -upper
            if FULL_MATRIX:
                ap.low2upp(dh_mean_ij.values, k=-1, mult=-1) # mult = -1 (< 0)!
                ap.low2upp(dg_mean_ij.values, k=-1, mult=-1)
                ap.low2upp(dh_error_ij.values, k=-1, mult=1) # mult = 1 (> 0)!
                ap.low2upp(dg_error_ij.values, k=-1, mult=1)
                ap.low2upp(dh_error2_ij.values, k=-1, mult=1)
                ap.low2upp(dg_error2_ij.values, k=-1, mult=1)
                ap.low2upp(n_ad_ij.values, k=-1, mult=1)
                ap.low2upp(n_da_ij.values, k=-1, mult=1)
                ap.low2upp(n_ij.values, k=-1, mult=1)

            '''
            if PLOT and (dh_mean_ij.count().sum() > 100):
                print dh_mean_ij.T.to_string()
                pass
            '''

            # select the reference time series
            col_ref, ts_ref = ap.select_ref(dh_mean_ij, dynamic=True)

            # remove time series with no overlap to the reference
            cols = ap.find_non_overlap(dh_mean_ij, col_ref)
            dh_mean_ij[cols] = np.nan
            dg_mean_ij[cols] = np.nan
            dh_error_ij[cols] = np.nan
            dg_error_ij[cols] = np.nan
            dh_error2_ij[cols] = np.nan
            dg_error2_ij[cols] = np.nan 
            n_ad_ij[cols] = np.nan
            n_da_ij[cols] = np.nan
            n_ij[cols] = np.nan
            if len(cols) != 0:
                print 'non-overlapping columns:', len(cols)

            '''
            print 'ONE LOOP'
            print dh_error_ij[dh_error_ij.columns[1]].plot()
            print n_ij[n_ij.columns[1]].plot()
            plt.show()
            '''

            '''
            if PLOT and (dh_mean_ij.count().sum() > 100):
                plt.figure()
                ap.hinton(dh_mean_ij.T.values)
                plt.figure()
                plt.spy(dh_mean_ij.T.values)
                dh_mean_ij.plot(legend=False)
                plt.title('lon, lat = %.3f, %.3f' % (lon_i, lat_i))
                plt.show()
            '''

            '''
            # (no!) filter out time series with less than 50% non-null values
            filter_length(dh_mean_ij, .5)
            filter_length(dg_mean_ij, .5)
            '''

            # reference all time series to a common epoch
            ap.ref_by_offset(dh_mean_ij, col_ref)
            ap.ref_by_offset(dg_mean_ij, col_ref)
            '''
            # Warning! In ref_by_first the non-zero diagonal will "improve" the
            # average by introducing repeated values (in the diagonal)
            ap.ref_by_first(dh_mean_ij, dynamic_ref=True)
            ap.ref_by_first(dg_mean_ij, dynamic_ref=True)
            '''

            # propagate error/obs due to referencing
            ap.prop_err_by_offset(dh_error_ij, col_ref)
            ap.prop_err_by_offset(dh_error2_ij, col_ref)
            ap.prop_err_by_offset(dg_error_ij, col_ref)
            ap.prop_err_by_offset(dg_error2_ij, col_ref)
            ap.prop_obs_by_offset(n_ad_ij, col_ref)
            ap.prop_obs_by_offset(n_da_ij, col_ref)
            ap.prop_obs_by_offset(n_ij, col_ref)

            # compute average time series
            dh_mean_i = ap.weighted_average(dh_mean_ij, n_ij)
            dg_mean_i = ap.weighted_average(dg_mean_ij, n_ij)
            dh_error_i = ap.weighted_average_error(dh_error_ij, n_ij)
            dh_error2_i = ap.weighted_average_error(dh_error2_ij, n_ij)
            dg_error_i = ap.weighted_average_error(dg_error_ij, n_ij)
            dg_error2_i = ap.weighted_average_error(dg_error2_ij, n_ij)
            n_ad_i = ap.weighted_average(n_ad_ij, n_ij)
            n_da_i = ap.weighted_average(n_da_ij, n_ij)

            # reference only the final product (at the end of the processing)
            '''
            dh_mean_i = ap.referenced(dh_mean_i, to='first')
            dg_mean_i = ap.referenced(dg_mean_i, to='first')
            '''

            # plot figures
            #-------------------------------------------------------------

            if PLOT and (dh_mean_ij.count().sum() > 100):
                plt.figure()
                n_i = n_ad_i.add(n_da_i)
                n_ad_i.plot(legend=False)
                n_da_i.plot(legend=False)
                n_i.plot(legend=False)
                plt.show()
                print 'coords:', lon_i, ',', lat_i
                plt.figure()
                dh_mean_ij.plot(legend=False)
                plt.title('lon, lat = %.3f, %.3f' % (lon_i, lat_i))
                plt.figure()
                ap.hinton(dh_mean_ij.T.values)
                plt.figure()
                plt.spy(dh_mean_ij.T.values)
                plot_tseries(dh_mean_i, dh_error_i, dg_mean_i, dg_error_i)
                plt.title('lon, lat = %.3f, %.3f' % (lon_i, lat_i))
                plt.show()

            # REVISED TILL HERE ON NOV 8, 2012 -> CODE OK!

            # get sat/time info for TS (only once)
            if isfirst:
                time2 = [int(t.strftime('%Y%m%d')) for t in dh_mean_i.index]

            # save one TS per grid cell at a time
            #-------------------------------------------------------------

            if not SAVE_TO_FILE: continue

            if isfirst:
                # create output file
                isfirst = False
                nt = len(dh_mean_i)
                dout = OutputContainers(fname_out, (nt,ny,nx), (nt,1,1))

                # save info
                dout.time[:] = time2  # only time2 needed!
                dout.satname[:] = din.satname
                dout.lon[:] = din.lon
                dout.lat[:] = din.lat
                dout.x_edges[:] = din.x_edges
                dout.y_edges[:] = din.y_edges

            # save time series
            dout.dh_mean[:,i,j] = dh_mean_i.values
            dout.dh_error[:,i,j] = dh_error_i.values
            dout.dh_error2[:,i,j] = dh_error2_i.values
            dout.dg_mean[:,i,j] = dg_mean_i.values
            dout.dg_error[:,i,j] = dg_error_i.values
            dout.dg_error2[:,i,j] = dg_error2_i.values
            dout.n_ad[:,i,j] = n_ad_i.values
            dout.n_da[:,i,j] = n_da_i.values

    close_files()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
