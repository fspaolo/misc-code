"""
$ pwd
/Users/fpaolo/code/bindata

$ python average_ts_grids.py ~/data/fris/xover/seasonal/ers1_19920608_19960606_dh_grids.h5
"""

from funcs import *

sys.path.append('/Users/fpaolo/code/misc')
from util import *

filename_out = 'tseries_mean.h5'
save_to_file = True
nodename = 'fris'


def main(argv):

    files = argv[1:]
    fname = files[0]

    # load data
    #---------------------------------------------------------------------

    d = {}
    db = tb.openFile(fname)
    data = db.getNode('/', nodename)
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
    ny, nx, nz =  d['dh_mean'].shape

    isfirst = True
    # iterate over every sat mission
    for sat in sat_missions:

        # iterate over every grid cell (all times)
        #-----------------------------------------------------------------

        for i in xrange(ny):
            for j in xrange(nx):
                #i, j = 13, 35 # ex: full grid cell
                #i, j = 13, 25 # ex: first ts missing
                #i, j = 0, 25 # ex: first ts missing

                # create 1 DF for every grid cell 
                dh_mean_ij = create_df_with_ts(table, d['dh_mean'], sat,
                                    i, j, is_seasonal=True, ref_month=2)
                dh_error_ij = create_df_with_ts(table, d['dh_error'], sat,
                                    i, j, is_seasonal=True, ref_month=2)
                dg_mean_ij = create_df_with_ts(table, d['dg_mean'], sat,
                                    i, j, is_seasonal=True, ref_month=2)
                dg_error_ij = create_df_with_ts(table, d['dg_error'], sat,
                                    i, j, is_seasonal=True, ref_month=2)
                n_ad_ij = create_df_with_ts(table, d['n_ad'], sat,
                                    i, j, is_seasonal=True, ref_month=2)
                n_da_ij = create_df_with_ts(table, d['n_da'], sat,
                                    i, j, is_seasonal=True, ref_month=2)
                n_ij = n_ad_ij.combineAdd(n_da_ij)

                # reference the TS dynamicaly

                reference_ts(dh_mean_ij, by='bias', dynamic_ref=True)
                reference_ts(dg_mean_ij, by='bias', dynamic_ref=True)
                propagate_error(dh_error_ij, by='bias', dynamic_ref=True)
                propagate_error(dg_error_ij, by='bias', dynamic_ref=True)
                propagate_num_obs(n_ij, by='bias', dynamic_ref=True)

                # compute average TS

                dh_mean_j = weighted_average(dh_mean_ij, n_ij)
                dg_mean_j = weighted_average(dg_mean_ij, n_ij)
                dh_error_j = weighted_average_error(dh_error_ij, n_ij)
                dg_error_j = weighted_average_error(dg_error_ij, n_ij)
                n_ad_j = average_obs(n_ad_ij)
                n_da_j = average_obs(n_da_ij)

                # plot figures

                #plot_df(dh_mean_ij, matrix=False)
                #plot_tseries(dh_mean_j, dh_error_j, dg_mean_j, dg_error_j)

                # save one TS per grid cell at a time
                #---------------------------------------------------------

                if not save_to_file: continue

                if isfirst:
                    # open or create output file
                    isfirst = False
                    fname_out, sat_name = get_fname_out(fname, filename_out)
                    title = 'FRIS Average Time Series'
                    filters = tb.Filters(complib='blosc', complevel=9)
                    atom = tb.Atom.from_dtype(dh_mean_j.values.dtype)
                    ni, nj, nk = ny, nx, len(dh_mean_j)
                    db = tb.openFile(fname_out, 'w')

                    g = db.createGroup('/', 'fris')
                    t1 = db.createTable(g, 'table', TimeSeriesGrid, title, filters)

                    c1 = db.createCArray(g, 'dh_mean', atom, (ni, nj, nk), '', filters)
                    c2 = db.createCArray(g, 'dh_error', atom, (ni, nj, nk), '', filters)
                    c3 = db.createCArray(g, 'dg_mean', atom, (ni, nj, nk), '', filters)
                    c4 = db.createCArray(g, 'dg_error', atom, (ni, nj, nk), '', filters)
                    c5 = db.createCArray(g, 'n_ad', atom, (ni, nj, nk), '', filters)
                    c6 = db.createCArray(g, 'n_da', atom, (ni, nj, nk), '', filters)

                    c7 = db.createCArray(g, 'x_edges', atom, (nj+1,), '', filters)
                    c8 = db.createCArray(g, 'y_edges', atom, (ni+1,), '', filters)
                    c9 = db.createCArray(g, 'lon', atom, (nj,), '', filters)
                    c10 = db.createCArray(g, 'lat', atom, (ni,), '', filters)

                    # get sat/time info for TS
                    sat_name2 = np.empty(nk, 'S10')
                    sat_name2.fill(sat_name)
                    ref_time = np.empty(nk, 'S10')
                    ref_time.fill(dh_mean_j.index[0])
                    year = [dt.year for dt in dh_mean_j.index]
                    month = [dt.month for dt in dh_mean_j.index]

                    # save info
                    t1.append(np.rec.array([sat_name2, ref_time, year, month], dtype=t1.dtype))
                    t1.flush()

                    c7[:] = d['x_edges'][:]
                    c8[:] = d['y_edges'][:]
                    c9[:] = d['lon'][:]
                    c10[:] = d['lat'][:]

                # save time series
                c1[i,j,:] = dh_mean_j.values
                c2[i,j,:] = dh_error_j.values
                c3[i,j,:] = dg_mean_j.values
                c4[i,j,:] = dg_error_j.values
                c5[i,j,:] = n_ad_j.values
                c6[i,j,:] = n_da_j.values

                '''
                fname_out, sat_name = get_fname_out(fname, filename_out)
                title = 'FRIS Average Time Series'
                filters = tb.Filters(complib='blosc', complevel=9)
                db = tb.openFile(fname_out, 'a')
                try:
                    t = db.createTable('/', 'ts_mean', TimeSeries, title, filters)
                except:
                    t = db.getNode('/', 'ts_mean')

                sat = np.empty(len(dh_j), 'S10')
                sat[:] = sat_name
                year = [d.year for d in dates2]
                month = [d.month for d in dates2]

                data = np.rec.array([sat, dates2, year, month, dh_j, se_j], dtype=t.dtype)
                t.append(data)
                t.flush()
                db.close()
                '''
    db.flush()
    db.close()

    print 'out file -->', fname_out

if __name__ == '__main__':
    sys.exit(main(sys.argv))
