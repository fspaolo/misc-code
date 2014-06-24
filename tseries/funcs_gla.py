"""
Module containing functions and classes used by:

averagets.py
crosscalib.py

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


class TimeSeriesGrid(tb.IsDescription):
    satname = tb.StringCol(20, pos=1)
    time1 = tb.Int32Col(pos=2)
    time2 = tb.Int32Col(pos=3)


def get_data_from_file(fname_in, node_name='dh_mean', mode='r'):
    fin = tb.openFile(fname_in, mode)
    data = fin.root
    d = {}
    d['dh_mean'] = fin.getNode('/', node_name)[:]
    d['table'] = data.table
    d['satname'] = d['table'].cols.satname[:]
    d['time1'] = d['table'].cols.time1[:]
    d['time2'] = d['table'].cols.time2[:]
    try:
        d['dh_mean_corr'] = data.dh_mean_corr[:]
    except:
        pass
    d['dh_error'] = data.dh_error[:]
    d['dh_error2'] = data.dh_error2[:]
    d['dg_mean'] = data.dg_mean[:]
    d['dg_error'] = data.dg_error[:]
    d['dg_error2'] = data.dg_error2[:]
    d['n_ad'] = data.n_ad[:]
    d['n_da'] = data.n_da[:]
    d['x_edges'] = data.x_edges[:]
    d['y_edges'] = data.y_edges[:]
    d['lon'] = data.lon[:]
    d['lat'] = data.lat[:]
    return [d, fin]


def fy_to_ym(fyear):
    """Decimal year -> year, month."""
    fy, y  = np.modf(fyear)
    m, y = int(np.ceil(fy*12)), int(y)
    return [y, m]


def ym_to_fy(year, month):
    """Year, month -> decimal year."""
    year = np.asarray(year)
    month = np.asarray(month)
    fyear = year + (month - 0.5)/12.  # decimal years (for midle of a month)
    return fyear 


def year_to_dtime(year):
    """
    Convert decimal year to `datetime` object.
    year : float or array_like of floats
    """
    if not np.iterable(year):
        year = np.asarray([year])
    ym = np.asarray([fy_to_ym(y) for y in year])
    dt = np.asarray([pn.datetime(y, m, 15) for y, m in ym])
    return dt


def int2dtime(itimes):
    """
    Convert an integer representation of time YYYYMMDD to datetime.
    """
    return np.asarray([pn.datetime.strptime(str(it), '%Y%m%d') for it in itimes])


def time1_to_dtime(time1):
    """
    Convert `time1` to `datetime` object.
    time1 : str or array_like of str --> 'yyyy-mm-dd'
    """
    if np.ndim(time1) == 0:
        time1 = np.asarray([time1])
    #ym = np.asarray([(int(t[:4]), int(t[4:6])) for t in time1]) # for 'yyyymmdd'
    ym = np.asarray([(int(t.split('-')[0]), int(t.split('-')[1])) for t in time1])
    dt = np.asarray([pn.datetime(y, m, 15) for y, m in ym])
    return dt


'''
def full_dtime_range(dt1, dt2, months=3):
    """
    Get full (continuous) datetime range.
    """
    offset = pn.datetools.DateOffset(months=months)
    return pn.DateRange(dt1, dt2, offset=offset)
'''

def full_dtime_range(dt1, dt2, **kw):
    """
    Get full (continuous?) datetime range.
    """
    dt3 = np.unique(np.append(dt1, dt2))
    dt3.sort()
    return dt3


def create_df_with_ts(time1, time2, ts, **kw):
    """
    Create a DataFrame with all the different time1-TS.
    time1, time2 : integer representation of time: YYYYMMDD
    ts : time series for 1-grid-cell/1-sat/all-times
    """
    dtime1 = int2dtime(time1)
    dtime2 = int2dtime(time2)
    dtime_range = full_dtime_range(dtime1, dtime2, **kw) # <<< specific for GLA
    #dtime_range = full_dtime_range(dtime1.min(), dtime2.max(), **kw)

    df = pn.DataFrame(index=dtime_range, columns=dtime_range) # empty DF

    # fill df with 1 `time1-TS` at a time
    if not np.alltrue(np.isnan(ts)):
        for dt_ref in dtime_range:
            ind, = np.where(dtime1 == dt_ref)
            df[dt_ref] = pn.Series(ts[ind], index=dtime2[ind])
            if not np.alltrue(df[dt_ref].isnull().values):
                df[dt_ref][dt_ref] = 0.
    return df


def create_df_with_sats(time1, time2, ts, satname, satnames, **kw):
    """
    Create a DataFrame with one TS per `satellite`.
    Note: i,j = y,x <<<<< important!
    """
    dtime1 = int2dtime(time1)
    dtime2 = int2dtime(time2)
    #dtime_range = full_dtime_range(dtime1.min(), dtime2.max(), **kw)
    dtime_range = full_dtime_range(dtime1, dtime2, **kw)

    df = pn.DataFrame(index=dtime_range, columns=satnames) # empty DF

    # fill df with 1 `satname` TS at a time
    if not np.alltrue(np.isnan(ts)):
        for sat in satnames:
            ind, = np.where(sat == satname)
            df[sat] = pn.Series(ts[ind], index=dtime2[ind])
    return df


def reference_ts(df, by='offset', dynamic_ref=True):
    """
    Reference all TS to the *selected reference* TS by:
    - using the `first` time of each TS (Davis)
    - computing the `offset` to the ref TS (Paolo)

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

    elif by=='offset':
        if dynamic_ref:
            # use TS with max non-null entries as ref
            ts_ref = df[df.columns[df.count().argmax()]]  
        else:
            # use TS with first time1 as ref
            ts_ref = df[df.columns[0]]
        for c in df.columns:
            df[c] += np.mean(ts_ref - df[c])  # add a `offset` to entire ts


def propagate_error(df, by='offset', dynamic_ref=True):
    """
    Propagate the error of the ts_ref to other ts due to referencing 
    by `first` or `offset` (depending on the referencing method used).
    error: squared root of squared sum of `e_mean_ref` (coincident ts points)
    and e_i.

    For `dynamic_ref` see the `reference_ts` function.
    """
    if np.alltrue(np.isnan(df.values)): return

    if by == 'first':
        # use TS with first time1 as ref
        error_ref = df[df.columns[0]][1:]    # all elems but first one
        cols = df.columns[1:]                # all TS but the first one
        for c, e in zip(cols, error_ref):
            df[c] = np.sqrt(e**2 + df[c]**2)

    elif by == 'offset':
        if dynamic_ref:
            # use TS with max non-null entries as ref
            col_ref = df.columns[df.count().argmax()]
        else:
            # use TS with first time1 as ref
            col_ref = df.columns[0]
        error_ref = df[col_ref]                   # TS_ref
        cols = df.columns[df.columns!=col_ref]    # all TS but TS_ref
        for c in cols:
            # only use the *coincident* non-null entries
            ind, = np.where(error_ref.notnull() & df[c].notnull())
            # for every ts `e_mean` is different (coincident vals only)
            e_mean = error_ref[ind].mean() 
            df[c] = np.sqrt(e_mean**2 + df[c]**2)


def propagate_num_obs(df, by='offset', dynamic_ref=True):
    """
    Propagate the num of obs of the ts_ref due to referencing:
    by `first` or `offset` (depending on the referencing method used).

    For `dynamic_ref` see the `reference_ts` function.
    """
    if np.alltrue(np.isnan(df.values)): return

    if by == 'first':
        # use TS with first time1 as ref
        nobs_ref = df[df.columns[0]][1:]    # all elems but the first one
        cols = df.columns[1:]               # all TS but the first one
        for c, n in zip(cols, nobs_ref):
            df[c] += n                      # don't like this way of C.Davis !!!

    elif by == 'offset':
        if dynamic_ref:
            # use TS with max non-null entries as ref
            col_ref = df.columns[df.count().argmax()]
        else:
            # use TS with first time1 as ref
            col_ref = df.columns[0]
        nobs_ref = df[col_ref]
        cols = df.columns[df.columns!=col_ref]    # all TS but TS_ref

        for c in cols:
            # only use the *coincident* non-null entries
            ind, = np.where(nobs_ref.notnull() & df[c].notnull())
            # for every ts `nobs_mean` is different (coincident vals only)
            nobs_mean_ref = nobs_ref[ind].mean()
            nobs_mean_ts = df[c][ind].mean()
            nobs_mean = np.mean([nobs_mean_ref, nobs_mean_ts]).round()
            df[c][df[c]==0] = nobs_mean           # the 0 of TS_ref is not included !!!


def weighted_average(df, nobs_df):
    """
    Calculate the unbiased weighted average TS, weighted by
    the number of observations.
    """
    if np.alltrue(np.isnan(df.values)):
        ts_mean = df.sum(axis=1)            # if nothing, colapse matrix -> series
    else:
        # weights for averaging
        nobs_df[nobs_df==0] = 1             # to ensure dh=0 (n_obs=0) enters in the average
        nobs_j = nobs_df.sum(axis=1)        # total #obs per row (col in matrix)
        w_ij = nobs_df.div(nobs_j, axis=0)  # one weight per element (sum==1)
        ts_mean = (w_ij * df).sum(axis=1)
        #print w_ij.sum(axis=1)
    return ts_mean


def weighted_average_error(df, nobs_df):
    """
    Calculate the unbias weighted average TS for the estandard 
    error, weighted by the number of observations.
    """
    if np.alltrue(np.isnan(df.values)):
        ts_mean = df.sum(axis=1)            # if nothing, colapse matrix -> series
    else:
        # weights for averaging
        nobs_df[nobs_df==0] = 1             # to ensure the dh=0 (n_obs=0) enters the average
        nobs_j = nobs_df.sum(axis=1)        # total #obs per row (col in matrix)
        w_ij = nobs_df.div(nobs_j, axis=0)  # one weight per element (sum==1)
        ts_mean = np.sqrt( (w_ij**2 * df**2).sum(axis=1) )
    return ts_mean


def average_obs(nobs_df):
    """
    Calculate the average number of observations in the resulting
    average TS from the individual TS.
    """
    #nobs_df[nobs_df==0] = np.nan
    return nobs_df.mean(axis=1)


def reference_to_first(ts):
    ind, = np.where(ts.notnull())
    if len(ind) > 0:
        ts -= ts[ind[0]]
    return ts 


def plot_df(df, matrix=True, **kwds):
    if not np.alltrue(np.isnan(df.values)):
        df.plot(**kwds)
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


def get_fname_out(file_in, fname_out=None, suffix=None):
    if fname_out is None:
        path, ext = os.path.splitext(file_in)  # path from input file
        fname_out = '_'.join([path, suffix])
    return fname_out


def create_output_containers(fname_out, any_data, shape, node_name, chunkshape):
    # open or create output file
    file_out = tb.openFile(fname_out, 'w')
    filters = tb.Filters(complib='zlib', complevel=9)
    atom = tb.Atom.from_dtype(any_data.dtype)
    N, ny, nx = shape
    cs = chunkshape 
    title = ''
    dout = {}
    if node_name:
        g = file_out.createGroup('/', node_name)
    else:
        g = '/' 
    dout['table'] = file_out.createTable(g, 'table', 
                                 TimeSeriesGrid, title, filters)
    dout['lon'] = file_out.createCArray(g, 'lon', 
                                  atom, (nx,), '', filters)
    dout['lat'] = file_out.createCArray(g, 'lat', 
                                  atom, (ny,), '', filters)
    dout['x_edges'] = file_out.createCArray(g, 'x_edges', 
                                  atom, (nx+1,), '', filters)
    dout['y_edges'] = file_out.createCArray(g, 'y_edges', 
                                  atom, (ny+1,), '', filters)
    dout['dh_mean'] = file_out.createCArray(g, 'dh_mean', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dh_error'] = file_out.createCArray(g, 'dh_error', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dh_error2'] = file_out.createCArray(g, 'dh_error2', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dg_mean'] = file_out.createCArray(g, 'dg_mean', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dg_error'] = file_out.createCArray(g, 'dg_error', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['dg_error2'] = file_out.createCArray(g, 'dg_error2', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['n_ad'] = file_out.createCArray(g, 'n_ad', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    dout['n_da'] = file_out.createCArray(g, 'n_da', 
                                  atom, (N,ny,nx), '', filters, chunkshape=cs)
    return dout, file_out
    

### plotting functions


def hinton(W, max_weight=None, ax=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import NullLocator
    W = W.T
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(W).max())/np.log(2))
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    for (x,y),w in np.ndenumerate(W):
        if w > 0: color = 'white'
        else:     color = 'black'
        size = np.sqrt(np.abs(w))
        rect = Rectangle([x - size / 2, y - size / 2], size, size,
            facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.autoscale_view()
    # Reverse the yaxis limits
    ax.set_ylim(*ax.get_ylim()[::-1])


def add_inner_title(ax, title='', loc=1, size=None, **kwargs):
    """
    Add title inside the figure. Same locations as `label`.

    Example
    -------
    fig = plt.figure()
    ax = fig.add_subplot((111))
    ax = add_inner_title(ax, 'title', 3)
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size, pad=0., 
        borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    at.patch.set_alpha(0.5)
    return at


def plot_matrix(mat, title='', loc=1, plot=None, **kw):
    """
    Plot the representation of a matrix: matshow, hinton or spy.

    plot : can be 'matshow', 'hinton' or 'spy'. If `None` (default) 
    plots matshow and hinton diagrams.
    """
    from matplotlib.ticker import NullLocator
    if plot is None:
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.matshow(mat)
        hinton(mat, ax=ax2)
        t = add_inner_title(ax1, title, loc=loc)
        t = add_inner_title(ax2, title, loc=loc)
    else:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot((111))
        if plot == 'matshow':
            ax.matshow(mat)
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        elif plot == 'hinton':
            hinton(mat, ax=ax)
        elif plot == 'spy':
            ax.spy(mat, precision=0.1, markersize=6, **kw)
        else:
            raise ValueError('wrong argument `plot=%s`' % plot)
        t = add_inner_title(ax, title, loc=loc)
    return fig
