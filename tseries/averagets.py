#!/usr/bin/env python
"""
Notes
-----
* Process one satellite at a time and merge the files later on.
* Indices: i,j,k = t,y,x

Example
-------
$ python averagets.py ~/data/fris/xover/seasonal/ers1_*_grids.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = True
SAVE_TO_FILE = False
NODE_NAME = ''

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file_in', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/<file_in>_mean.h5]')
parser.add_argument('-s', dest='suffix', default='mean.h5',
    help='suffix for output file [default: mean.h5]')

args = parser.parse_args()

def main(args):

    fname_in = args.file_in[0]
    fname_out = args.fname_out
    suffix = args.suffix

    fname_out = get_fname_out(fname_in, fname_out=fname_out, suffix=suffix)

    # input data
    #---------------------------------------------------------------------

    d, fin = get_data_from_file(fname_in, NODE_NAME)
    _, ny, nx = d['dh_mean'].shape            # ni,nj,nk = nt,ny,nx

    print 'processing time series ...'

    # iterate over every grid cell (all times): j,i = x,y
    #-----------------------------------------------------------------

    isfirst = True

    for i in xrange(ny):
        for j in xrange(nx):
            #j, i = 35, 13 # ex: full grid cell (fris)
            #j, i = 25, 13 # ex: first ts missing (fris)

            # create 1 DF for every grid cell 
            dh_mean_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['dh_mean'][:,i,j], months=3) 
            dh_error_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['dh_error'][:,i,j], months=3)
            dh_error2_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['dh_error2'][:,i,j], months=3)
            dg_mean_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['dg_mean'][:,i,j], months=3) 
            dg_error_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['dg_error'][:,i,j], months=3)
            dg_error2_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['dg_error2'][:,i,j], months=3)
            n_ad_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['n_ad'][:,i,j], months=3) 
            n_da_ij = create_df_with_ts(d['time1'], d['time2'], 
                d['n_da'][:,i,j], months=3) 
            n_ij = n_ad_ij.combineAdd(n_da_ij)

            # reference the TS dynamicaly

            reference_ts(dh_mean_ij, by='offset', dynamic_ref=True)
            reference_ts(dg_mean_ij, by='offset', dynamic_ref=True)
            propagate_error(dh_error_ij, by='offset', dynamic_ref=True)
            propagate_error(dh_error2_ij, by='offset', dynamic_ref=True)
            propagate_error(dg_error_ij, by='offset', dynamic_ref=True)
            propagate_error(dg_error2_ij, by='offset', dynamic_ref=True)
            propagate_num_obs(n_ij, by='offset', dynamic_ref=True)

            # compute average TS

            dh_mean_i = weighted_average(dh_mean_ij, n_ij)
            dg_mean_i = weighted_average(dg_mean_ij, n_ij)
            dh_error_i = weighted_average_error(dh_error_ij, n_ij)   # if independent errors
            dh_error2_i = weighted_average_error(dh_error2_ij, n_ij)
            dg_error_i = weighted_average_error(dg_error_ij, n_ij)
            dg_error2_i = weighted_average_error(dg_error2_ij, n_ij)
            #dh_error_i = weighted_average(dh_error_ij, n_ij)        # if correlated errors
            #dh_error2_i = weighted_average(dh_error2_ij, n_ij)
            #dg_error_i = weighted_average(dg_error_ij, n_ij)
            #dg_error2_i = weighted_average(dg_error2_ij, n_ij)
            n_ad_i = average_obs(n_ad_ij)
            n_da_i = average_obs(n_da_ij)

            dh_mean_i = reference_to_first(dh_mean_i)
            dg_mean_i = reference_to_first(dg_mean_i)

            # plot figures

            if PLOT and (dh_mean_ij.count().sum() > 100):
                plot_df(dh_mean_ij, matrix=False, legend=False, rot=45)
                plot_tseries(dh_mean_i, dh_error2_i, dg_mean_i, dg_error2_i)

            # REVISED TILL HERE ON NOV 8, 2012 -> OK!

            # revise from here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # get sat/time info for TS (only once)

            if isfirst:
                satname2 = np.empty(len(dh_mean_i), 'S20')
                time1 = np.empty(len(dh_mean_i), 'i4')
                satname2.fill(d['satname'])
                time2 = [int(t.strftime('%Y%m%d')) for t in dh_mean_i.index]
                time1.fill(time2[0])

            # save one TS per grid cell at a time
            #---------------------------------------------------------

            if not SAVE_TO_FILE: continue

            if isfirst:
                # open or create output file
                isfirst = False
                N = len(dh_mean_i)
                dout, fout = create_output_containers(fname_out, 
                    dh_mean_i, (N,ny,nx), NODE_NAME, (N,1,1)) # <-- chunkshape

                # save info
                dout['table'].append(np.rec.array(
                    [satname2, time1, time2], dtype=dout['table'].dtype))
                dout['table'].flush()
                dout['lon'][:] = d['lon'][:]
                dout['lat'][:] = d['lat'][:]
                dout['x_edges'][:] = d['x_edges'][:]
                dout['y_edges'][:] = d['y_edges'][:]

            # save time series
            dout['dh_mean'][:,i,j] = dh_mean_i.values
            dout['dh_error'][:,i,j] = dh_error_i.values
            dout['dh_error2'][:,i,j] = dh_error2_i.values
            dout['dg_mean'][:,i,j] = dg_mean_i.values
            dout['dg_error'][:,i,j] = dg_error_i.values
            dout['dg_error2'][:,i,j] = dg_error2_i.values
            dout['n_ad'][:,i,j] = n_ad_i.values
            dout['n_da'][:,i,j] = n_da_i.values

    fout.flush()
    fout.close()
    fin.close()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
