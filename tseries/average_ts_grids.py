"""
Notes
-----
* Process one satellite at a time and merge the files later on.
* indices: i,j,k = t,x,y

Ecample
-------
$ python average_ts_grids.py ~/data/fris/xover/seasonal/ers1_19920608_19960606_dh_grids.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

SAVE_TO_FILE = True
NODE_NAME = 'fris'

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file_in', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/sat_tmin_tmax_dh_grids.h5]')

args = parser.parse_args()

def main(args):

    fname_in = args.file_in[0]
    fname_out = get_fname_out(fname_in, fname_out=args.fname_out, sufix='mean.h5')

    # input data
    #---------------------------------------------------------------------

    d = {}
    file_in = tb.openFile(fname_in)
    data = file_in.getNode('/', NODE_NAME)
    table = data.table

    d['sat_name'] = table.cols.sat_name
    d['ref_time'] = table.cols.ref_time
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
    _, nx, ny =  d['dh_mean'].shape    # i,j,k = t,x,y

    isfirst = True
    # iterate over every sat mission
    for sat in sat_missions:

        # iterate over every grid cell (all times): j,k = x,y
        #-----------------------------------------------------------------

        for k in xrange(ny):
            for j in xrange(nx):
                #j, k = 35, 13 # ex: full grid cell
                #j, k = 25, 13 # ex: first ts missing
                #j, k = 25, 0  # ex: first ts missing

                # create 1 DF for every grid cell 
                dh_mean_ij = create_df_with_ts(table, d['dh_mean'], sat,
                                    j, k, is_seasonal=True, ref_month=2)
                dh_error_ij = create_df_with_ts(table, d['dh_error'], sat,
                                    j, k, is_seasonal=True, ref_month=2)
                dg_mean_ij = create_df_with_ts(table, d['dg_mean'], sat,
                                    j, k, is_seasonal=True, ref_month=2)
                dg_error_ij = create_df_with_ts(table, d['dg_error'], sat,
                                    j, k, is_seasonal=True, ref_month=2)
                n_ad_ij = create_df_with_ts(table, d['n_ad'], sat,
                                    j, k, is_seasonal=True, ref_month=2)
                n_da_ij = create_df_with_ts(table, d['n_da'], sat,
                                    j, k, is_seasonal=True, ref_month=2)
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

                #plot_df(dh_mean_ij, matrix=False)
                #plot_tseries(dh_mean_i, dh_error_i, dg_mean_i, dg_error_i)

                # save one TS per grid cell at a time
                #---------------------------------------------------------

                if not SAVE_TO_FILE: continue

                if isfirst:
                    # open or create output file
                    isfirst = False
                    atom = tb.Atom.from_dtype(dh_mean_i.values.dtype)
                    ni, nj, nk = len(dh_mean_i), nx, ny
                    dout, file_out = create_output_containers(fname_out, atom, (ni,nj,nk))

                    # get sat/time info for TS
                    sat_name2 = np.empty(ni, 'S10')
                    sat_name2.fill(sat)
                    ref_time = np.empty(ni, 'S10')
                    ref_time.fill(dh_mean_i.index[0])
                    year = [dt.year for dt in dh_mean_i.index]
                    month = [dt.month for dt in dh_mean_i.index]

                    # save info
                    dout['table'].append(np.rec.array([sat_name2, ref_time, year, month], 
                                         dtype=dout['table'].dtype))
                    dout['table'].flush()
                    dout['x_edges'][:] = d['x_edges'][:]
                    dout['y_edges'][:] = d['y_edges'][:]
                    dout['lon'][:] = d['lon'][:]
                    dout['lat'][:] = d['lat'][:]

                # save time series
                dout['dh_mean'][:,j,k] = dh_mean_i.values
                dout['dh_error'][:,j,k] = dh_error_i.values
                dout['dg_mean'][:,j,k] = dg_mean_i.values
                dout['dg_error'][:,j,k] = dg_error_i.values
                dout['n_ad'][:,j,k] = n_ad_i.values
                dout['n_da'][:,j,k] = n_da_i.values

    file_out.flush()
    file_out.close()
    file_in.close()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
