"""
Module containing functions and classes used by:

average_ts_grids.py
cross_calib_grids.py

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 19, 2011

import os
import sys
import re
import numpy as np
import tables as tb
import pandas as pn
import matplotlib.pyplot as plt

sys.path.append('/Users/fpaolo/code/misc')
from util import *
from viz import plot_matrix


def get_dtime_range_from_files(files):
    """
    Get time range in a `datetime` object from input files.
    """
    times = [re.findall('\d\d\d\d\d\d+', fname) for fname in files]
    t_1 = [t1 for t1, t2 in times]
    t_2 = [t2 for t1, t2 in times]
    t_min, t_max =  min(t_1), max(t_2)
    y1, m1 = int(t_min[:4]), int(t_min[4:6])
    y2, m2 = int(t_max[:4]), int(t_max[4:6])
    print 't_min, t_max:', t_min, t_max
    # seasonal
    if len(t_min) > 6 and len(t_max) > 6:  
        y1, m1 = get_season(y1, m1, return_month=1)
        y2, m2 = get_season(y2, m2, return_month=1)
        start = '%d/%d/%d' % (y1, m1, 15)
        end = '%d/%d/%d' % (y2, m2, 15)
        return pn.DateRange(start, end, 
            offset=pn.datetools.DateOffset(months=3))
    # monthly
    else:
        start = '%d/%d/%d' % (y1, m1, 15)
        end = '%d/%d/%d' % (y2, m2, 15)
        return pn.DateRange(start, end, 
            offset=pn.datetools.DateOffset(months=1))


def get_dtime_range(ref_time, year, month, is_seasonal=False, ref_month=2):
    """
    Get full (continuous) time range in a `datetime` object.

    ref_time : array of str
    year, month : array of int
    is_seasonal : [seasonal | monthly] TS
    ref_month : reference month for 3-month season-block
    """
    time_min = ref_time[0]
    y1, m1 = int(time_min[:4]), int(time_min[4:6])
    y2, m2 = year[-1], month[-1]
    if is_seasonal:  
        # seasonal TS
        y1, m1 = get_season(y1, m1, return_month=ref_month)
        y2, m2 = get_season(y2, m2, return_month=ref_month)
        start = '%d/%d/%d' % (y1, m1, 15)
        end = '%d/%d/%d' % (y2, m2, 15)
        dt = pn.DateRange(start, end, 
            offset=pn.datetools.DateOffset(months=3))
        print 'time range (seasonal): %d/%d to %d/%d' % (y1, m1, y2, m2)
        print 'length: %d seasons' % len(dt)
        print 'season-block referenced to month [1|2|3]: %d' % ref_month
        return dt
    else:
        # monthly TS
        start = '%d/%d/%d' % (y1, m1, 15)
        end = '%d/%d/%d' % (y2, m2, 15)
        dt = pn.DateRange(start, end, 
            offset=pn.datetools.DateOffset(months=1))
        print 'time range (monthly): %d/%d to %d/%d' % (y1, m1, y2, m2)
        print 'length: %d months' % len(dt)
        return dt


def year_month_to_dtime(year, month, is_seasonal=False, ref_month=2):
    """
    Convert year and month to `datetime` object.

    year, month : int or array_like of int
    """
    if not np.iterable(year) or not np.iterable(month):
        year = np.asarray([year])
        month = np.asarray([month])
    if is_seasonal:
        year, month = get_season(year, month, return_month=ref_month)
    return np.asarray([pn.datetime(y, m, 15) for y, m in zip(year, month)])


def ref_time_to_dtime(ref_time, is_seasonal=False, ref_month=2):
    """
    Convert `ref_time` to `datetime` object.

    ref_time : str or array_like of str
    """
    if np.ndim(ref_time) == 0:
        ref_time = np.asarray([ref_time])
    ym = np.asarray([(int(t[:4]), int(t[4:6])) for t in ref_time])
    if is_seasonal:
        ym[:,0], ym[:,1] = get_season(ym[:,0], ym[:,1], return_month=ref_month)
    return np.asarray([pn.datetime(y, m, 15) for y, m in ym])


def create_df_with_ts(table, data3d, sat_name, j, k, **kw):
    """
    Create a DataFrame with the different `ref_time` TS.
    """
    # load table and data for 1-grid-cell/1-sat/all-times
    i = table.getWhereList('sat_name == "%s"' % sat_name)
    t = table.readCoordinates(i)
    d = data3d[i,j,k]

    dtime_range = get_dtime_range(t['ref_time'], t['year'], t['month'], **kw)
    dtime_ref = ref_time_to_dtime(t['ref_time'], **kw)
    dtime_ts = year_month_to_dtime(t['year'], t['month'], **kw)

    df = pn.DataFrame(index=dtime_range, columns=dtime_range)

    # fill df with 1 `ref_time` TS at a time
    if not np.alltrue(np.isnan(d)):
        for dt_ref in dtime_range:
            ind, = np.where(dtime_ref == dt_ref)
            df[dt_ref] = pn.Series(d[ind], index=dtime_ts[ind])
            df[dt_ref][dt_ref] = 0.
    return df


def reference_ts(df, by='bias', dynamic_ref=True):
    """
    Reference all TS to the *selected reference* TS by:
    - using the `first` time of each TS (Davis)
    - computing the `bias` to the ref TS (Paolo)

    The selected reference TS can be:
    - the one with the first reference time (first TS)
    - the one with the maximum non-null entries (dynamic)
    """
    if np.alltrue(np.isnan(df.values)): return

    if by=='first':
        ts_ref = df[df.columns[0]][1:]  # use first TS as ref
        cols = df.columns[1:]
        for c, i in zip(cols, ts_ref):
            df[c] += i    # add the element `i` to entire ts

    elif by=='bias':
        if dynamic_ref:
            # use TS with max non-null entries as ref
            ts_ref = df[df.columns[df.count().argmax()]]  
        else:
            # use TS with first ref_time as ref
            ts_ref = df[df.columns[0]]
        for c in df.columns:
            df[c] += np.mean(ts_ref - df[c])  # add a `bias` to entire ts


def propagate_error(df, by='bias', dynamic_ref=True):
    """
    Propagate the error of the ts_ref due to referencing (doing
    squared root of squared sum) by `first` or `bias` (depending 
    on the referencing method used).

    For `dynamic_ref` see the `reference_ts` function.
    """
    if np.alltrue(np.isnan(df.values)): return

    if by == 'first':
        # use TS with first ref_time as ref
        error_ref = df[df.columns[0]][1:]    # all elems but first one
        cols = df.columns[1:]                # all TS but the first one
        for c, e in zip(cols, error_ref):
            df[c] = np.sqrt(e**2 + df[c]**2)

    elif by == 'bias':
        if dynamic_ref:
            # use TS with max non-null entries as ref
            col_ref = df.columns[df.count().argmax()]
        else:
            # use TS with first ref_time as ref
            col_ref = df.columns[0]
        error_ref = df[col_ref]
        cols = df.columns[df.columns!=col_ref]    # all TS but TS_ref
        for c in cols:
            # only use the *coincident* non-null entries
            ind = error_ref.notnull() == df[c].notnull()
            e_mean = error_ref[ind].mean()
            df[c] = np.sqrt(e_mean**2 + df[c]**2)


def propagate_num_obs(df, by='bias', dynamic_ref=True):
    """
    Propagate the num of obs of the ts_ref due to referencing:
    by `first` or `bias` (depending on the referencing method used).

    For `dynamic_ref` see the `reference_ts` function.
    """
    if np.alltrue(np.isnan(df.values)): return

    df[df==0] = np.nan    # important for calculations!

    if by == 'first':
        # use TS with first ref_time as ref
        n_ref = df[df.columns[0]][1:]    # all elems but the first one
        cols = df.columns[1:]            # all TS but the first one
        for c, n in zip(cols, n_ref):
            df[c] += n

    elif by == 'bias':
        if dynamic_ref:
            # use TS with max non-null entries as ref
            col_ref = df.columns[df.count().argmax()]
        else:
            # use TS with first ref_time as ref
            col_ref = df.columns[0]
        n_ref = df[col_ref]
        cols = df.columns[df.columns!=col_ref]    # all TS but TS_ref

        for c in cols:
            # only use the *coincident* non-null entries
            ind = n_ref.notnull() == df[c].notnull()
            n_mean = np.round(n_ref[ind].mean())
            df[c] += n_mean


def weighted_average(df, n_df):
    """
    Calculate the unbiased weighted average TS, weighted by
    the number of observations.
    """
    if np.alltrue(np.isnan(df.values)):
        ts_mean = df.sum(axis=1)      # colapse matrix -> series
    else:
        # weights for averaging
        n_j = n_df.sum(axis=1)        # total #obs per row (col in matrix)
        w_ij = n_df.div(n_j, axis=0)  # one weight per element (sum==1)
        ts_mean = (w_ij * df).sum(axis=1)
        ts_mean[0] = 0
        #print w_ij.sum(axis=1)
    return ts_mean


def weighted_average_error(df, n_df):
    """
    Calculate the unbiased weighted average TS for the estandard 
    error, weighted by the number of observations.
    """
    if np.alltrue(np.isnan(df.values)):
        ts_mean = df.sum(axis=1)      # colapse matrix -> series
    else:
        # weights for averaging
        n_j = n_df.sum(axis=1)        # total #obs per row (col in matrix)
        w_ij = n_df.div(n_j, axis=0)  # one weight per element (sum==1)
        ts_mean = np.sqrt( ((w_ij * df)**2).sum(axis=1) )
    return ts_mean


def average_obs(n_df):
    """
    Calculate the average number of observations in the resulting
    average TS from the individual TS.
    """
    n_df[n_df==0] = np.nan
    return n_df.mean(axis=1)


def plot_df(df, matrix=True):
    if not np.alltrue(np.isnan(df.values)):
        df.plot()
        if np.ndim(df) == 2 and matrix: 
            plot_matrix(df.T)
        plt.show()


def plot_tseries(ts1, err1, ts2, err2):
    if not np.alltrue(np.isnan(ts1.values)):
        fig = plt.figure()
        ax1 = fig.add_subplot(211, axisbg='#FFF8DC')
        plt.errorbar(ts1.index, ts1, yerr=err1)
        plt.ylabel('dh (m)')
        ax2 = fig.add_subplot(212, axisbg='#FFF8DC')
        plt.errorbar(ts2.index, ts2, yerr=err2)
        plt.ylabel('dAGC (m)')
        fig.autofmt_xdate()
        plt.show()


def get_dframe_from_array(fname):
    """
    Load 2D Array PyTables to DataFrame, from file.
    """
    f = tb.openFile(fname)
    data = f.root.data[:]
    f.close()
    year_month = [(get_season(y, m, return_month=1)) for y, m in data[:,:2]]
    dates = [pn.datetime(y, m, 15) for y, m in year_month]
    return pn.DataFrame(data, index=dates, columns=XXX)


def get_dframe_from_table(fname, data, cols):
    """
    Load DataFrame from a Table/Array PyTables.
    """
    t1, t2 = re.findall('\d\d\d\d\d\d+', fname)
    if len(t1) > 6 and len(t2) > 6:
        year_month = [(get_season(y, m, return_month=1)) \
                      for y, m in data[:][['year', 'month']]]
    else:
        year_month = [(y, m) for y, m in data[:][['year', 'month']]]
    dates = [pn.datetime(y, m, 15) for y, m in year_month]
    return pn.DataFrame(data, index=dates, columns=cols)


def get_fname_out(file_in, fname_out=None, sufix=None):
    """
    Construct the output file name with the min and max times 
    from the input file.
    """
    if fname_out is None:
        path, ext = os.path.splitext(file_in)  # path from input file
        fname_out = '_'.join([path, sufix])
    return fname_out


def create_output_containers(fname_out, atom, (ni,nj,nk)):
    # open or create output file
    title = 'FRIS Average Time Series'
    filters = tb.Filters(complib='blosc', complevel=9)
    file_out = tb.openFile(fname_out, 'w')
    
    dout = {}
    g = file_out.createGroup('/', 'fris')
    dout['table'] = file_out.createTable(g, 'table', TimeSeriesGrid, title, filters)
    
    dout['dh_mean'] = file_out.createCArray(g, 'dh_mean', atom, (ni,nj,nk), '', filters)
    dout['dh_error'] = file_out.createCArray(g, 'dh_error', atom, (ni,nj,nk), '', filters)
    dout['dg_mean'] = file_out.createCArray(g, 'dg_mean', atom, (ni,nj,nk), '', filters)
    dout['dg_error'] = file_out.createCArray(g, 'dg_error', atom, (ni,nj,nk), '', filters)
    dout['n_ad'] = file_out.createCArray(g, 'n_ad', atom, (ni,nj,nk), '', filters)
    dout['n_da']= file_out.createCArray(g, 'n_da', atom, (ni,nj,nk), '', filters)
    
    dout['x_edges'] = file_out.createCArray(g, 'x_edges', atom, (nj+1,), '', filters)
    dout['y_edges'] = file_out.createCArray(g, 'y_edges', atom, (nk+1,), '', filters)
    dout['lon'] = file_out.createCArray(g, 'lon', atom, (nj,), '', filters)
    dout['lat'] = file_out.createCArray(g, 'lat', atom, (nk,), '', filters)

    return dout, file_out
    
