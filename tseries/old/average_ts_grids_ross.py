#!/usr/bin/env python
"""
Notes
-----
* Process one satellite at a time and merge the files later on.
* indices: i,j,k = t,x,y

Example
-------
$ python average_ts_grids.py ~/data/fris/xover/seasonal/ers1_19920608_19960606_grids.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

SAVE_TO_FILE = True
NODE_NAME = 'ross'
PLOT = False

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file_in', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/<file_in>_mean.h5]')

args = parser.parse_args()

def main(args):

    fname_in = args.file_in[0]
    fname_out = args.fname_out
    suffix = '%s_mean.h5' % NODE_NAME
    fname_out = get_fname_out(fname_in, fname_out=fname_out, sufix=suffix)

    # input data
    #---------------------------------------------------------------------

    d = {}
    file_in = tb.openFile(fname_in)
    data = file_in.getNode('/', NODE_NAME)
    table = data.table

    d['sat_name'] = table.cols.sat_name
    d['ref_time'] = table.cols.ref_time
    d['date'] = table.cols.date
    d['year'] = table.cols.year
    d['month'] = table.cols.month
    d['dh_mean'] = data.dh_mean
    d['dh_error'] = data.dh_error
    d['dg_mean'] = data.dg_mean
    d['dg_error'] = data.dg_error
    d['n_ad'] = data.n_ad
    d['n_da'] = data.n_da
    d['x_edges'] = data.x_edges
    d['y_edges'] = data.y_edges
    d['lon'] = data.lon
    d['lat'] = data.lat

    #---------------------------------------------------------------------

    sat_missions = np.unique(d['sat_name'])
    _, ny, nx =  d['dh_mean'].shape    # i,j,k = nt,ny,nx

    print 'processing time series ...'

    isfirst = True
    # iterate over every sat mission
    for sat in sat_missions:

        # iterate over every grid cell (all times): j,i = x,y
        #-----------------------------------------------------------------

        for i in xrange(ny):
            for j in xrange(nx):
                #j, i = 35, 13 # ex: full grid cell
                #j, i = 25, 13 # ex: first ts missing
                #j, i = 25, 0  # ex: first ts missing

                # create 1 DF for every grid cell 
                dh_mean_ij = create_df_with_ts(table, d['dh_mean'], sat,
                                    j, i, is_seasonal=True, ref_month=2)
                dh_error_ij = create_df_with_ts(table, d['dh_error'], sat,
                                    j, i, is_seasonal=True, ref_month=2)
                dg_mean_ij = create_df_with_ts(table, d['dg_mean'], sat,
                                    j, i, is_seasonal=True, ref_month=2)
                dg_error_ij = create_df_with_ts(table, d['dg_error'], sat,
                                    j, i, is_seasonal=True, ref_month=2)
                n_ad_ij = create_df_with_ts(table, d['n_ad'], sat,
                                    j, i, is_seasonal=True, ref_month=2)
                n_da_ij = create_df_with_ts(table, d['n_da'], sat,
                                    j, i, is_seasonal=True, ref_month=2)
                n_ij = n_ad_ij.combineAdd(n_da_ij)

                # reference the TS dynamicaly

                reference_ts(dh_mean_ij, by='bias', dynamic_ref=True)
                reference_ts(dg_mean_ij, by='bias', dynamic_ref=True)
                propagate_error(dh_error_ij, by='bias', dynamic_ref=True)
                propagate_error(dg_error_ij, by='bias', dynamic_ref=True)
                propagate_num_obs(n_ij, by='bias', dynamic_ref=True)

                # compute average TS

                dh_mean_i = weighted_average(dh_mean_ij, n_ij)
                dg_mean_i = weighted_average(dg_mean_ij, n_ij)
                dh_error_i = weighted_average_error(dh_error_ij, n_ij)
                dg_error_i = weighted_average_error(dg_error_ij, n_ij)
                n_ad_i = average_obs(n_ad_ij)
                n_da_i = average_obs(n_da_ij)

                # plot figures

                if PLOT:
                    plot_df(dh_mean_ij, matrix=False)
                    plot_tseries(dh_mean_i, dh_error_i, dg_mean_i, dg_error_i)

                # get sat/time info for TS (only once)

                if isfirst:
                    sat_name2 = np.empty(len(dh_mean_i), 'S10')
                    ref_time = np.empty(len(dh_mean_i), 'S10')
                    sat_name2.fill(sat)
                    ref_time.fill(dh_mean_i.index[0])
                    date = [dt.date for dt in dh_mean_i.index]
                    year = [dt.year for dt in dh_mean_i.index]
                    month = [dt.month for dt in dh_mean_i.index]

                # save one TS per grid cell at a time
                #---------------------------------------------------------

                if not SAVE_TO_FILE: continue

                if isfirst:
                    # open or create output file
                    isfirst = False
                    N = len(dh_mean_i)
                    dout, file_out = create_output_containers(
                                     fname_out, dh_mean_i, (N,ny,nx), 
                                     NODE_NAME, (N,1,1)) # <-- chunkshape

                    # save info
                    dout['table'].append(np.rec.array(
                        [sat_name2, ref_time, date, year, month], 
                        dtype=dout['table'].dtype))
                    dout['table'].flush()
                    dout['lon'][:] = d['lon'][:]
                    dout['lat'][:] = d['lat'][:]
                    dout['x_edges'][:] = d['x_edges'][:]
                    dout['y_edges'][:] = d['y_edges'][:]

                # save time series
                dout['dh_mean'][:,i,j] = dh_mean_i.values
                dout['dh_error'][:,i,j] = dh_error_i.values
                dout['dg_mean'][:,i,j] = dg_mean_i.values
                dout['dg_error'][:,i,j] = dg_error_i.values
                dout['n_ad'][:,i,j] = n_ad_i.values
                dout['n_da'][:,i,j] = n_da_i.values

    file_out.flush()
    file_out.close()
    file_in.close()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
