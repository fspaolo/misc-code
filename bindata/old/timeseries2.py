"""
$ pwd
/Users/fpaolo/code/bindata

$ python timeseries2.py /Users/fpaolo/data/fris/xover/monthly3m/ers2_19950507_200306_t.h5.bin3.single /Users/fpaolo/data/fris/xover/seasonal/ers2_19950911_20030606_t.h5.bin3.single
"""

import os
import sys
import re
import numpy as np
import tables as tb
import pandas as pn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullLocator

filename_out = 'tseries_mean.h5'
save_to_file = False

class TimeSeries(tb.IsDescription):
    #time = tb.StringCol(20, pos=1)
    sat = tb.StringCol(10, pos=1)
    time = tb.StringCol(20, pos=2)
    #t_ref = tb.StringCol(10, pos=2)
    year = tb.Int32Col(pos=3)
    month = tb.Int32Col(pos=4)
    dh = tb.Float64Col(pos=5)
    se_dh = tb.Float64Col(pos=6)
    #dg = tb.Float64Col(pos=7)
    #se_dg = tb.Float64Col(pos=8)
    #n_ad = tb.Int32Col(pos=9)
    #n_da = tb.Int32Col(pos=10)


def get_season(year, month, get_month=0):
    """
    Returns the first, second or third month of the season-block.
    """
    MAM = [3, 4, 5]      # Mar/Apr/May -> Fall SH 
    JJA = [6, 7, 8]      # Jun/Jul/Aug -> winter SH
    SON = [9, 10, 11]    # Sep/Oct/Nov -> Spring SH
    DJF = [12, 1, 2]     # Dec/Jan/Feb -> summer SH
    year = int(year)
    if month in MAM:
        return year, MAM[get_month]
    elif month in JJA:
        return year, JJA[get_month]
    elif month in SON:
        return year, SON[get_month]
    elif month in DJF:
        if month == 12 and get_month > 0:
            year += 1
        return year, DJF[get_month]
    else:
        print 'not a valid month from 1 to 12!'
        return None, None


def get_time_range_from_files(files):
    """Get time range from input files."""
    times = [re.findall('\d\d\d\d\d\d+', fname) for fname in files]
    t_1 = [t1 for t1, t2 in times]
    t_2 = [t2 for t1, t2 in times]
    t_min, t_max =  min(t_1), max(t_2)
    #times = np.array(times, 'i4').reshape(len(times), 2)
    #t_min, t_max = str(times.min()), str(times.max())
    y1, m1 = int(t_min[:4]), int(t_min[4:6])
    y2, m2 = int(t_max[:4]), int(t_max[4:6])
    # seasonal
    if len(t_min) > 6 and len(t_max) > 6:  
        y1, m1 = get_season(y1, m1, get_month=1)
        y2, m2 = get_season(y2, m2, get_month=1)
        start = '%d/%d/%d' % (y1, m1, 15)
        end = '%d/%d/%d' % (y2, m2, 15)
        return pn.DateRange(start, end, 
            offset=pn.datetools.DateOffset(months=3))
    # monthly
    else:
        print t_min
        start = '%d/%d/%d' % (y1, m1, 15)
        end = '%d/%d/%d' % (y2, m2, 15)
        return pn.DateRange(start, end, 
            offset=pn.datetools.DateOffset(months=1))


def get_frame_from_array(fname):
    """Load data from 2D Array."""
    f = tb.openFile(fname)
    data = f.root.data[:]
    f.close()
    year_month = [(get_season(y, m, get_month=1)) for y, m in data[:,:2]]
    dates = [pn.datetime(y, m, 15) for y, m in year_month]
    return pn.DataFrame(data, index=dates, columns=XXX)


def get_frame_from_table(fname, data, cols):
    """Load DataFrame from a Table."""
    t1, t2 = re.findall('\d\d\d\d\d\d+', fname)
    if len(t1) > 6 and len(t2) > 6:
        year_month = [(get_season(y, m, get_month=1)) \
                      for y, m in data[:][['year', 'month']]]
    else:
        year_month = [(y, m) for y, m in data[:][['year', 'month']]]
    dates = [pn.datetime(y, m, 15) for y, m in year_month]
    return pn.DataFrame(data, index=dates, columns=cols)


def get_fname_out(fname, filename_out):
    path, name = os.path.split(fname)
    sat_name = name.split('_')[0]
    return os.path.join(path, filename_out), sat_name


def add_inner_title(ax, title, loc, size=None, **kwargs):
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


def plot_matrices(mat):
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax1.matshow(mat)
    ax2.spy(mat, precision=0.1, markersize=5)
    hinton(mat, ax=ax3)
    return fig


def plot_matrix(mat, title='', loc=1, plot='matshow'):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot((111))
    if plot == 'matshow':
        ax.matshow(mat)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
    else:
        hinton(mat, ax=ax)
    t = add_inner_title(ax, title, loc=loc)
    return fig


#---------------------------------------------------------------------

def main(argv):

    fig = plt.figure()
    # plot time series
    ax1 = fig.add_subplot(211, axisbg='#FFF8DC')
    plt.ylim(-0.45, 1.0)
    plt.ylabel('Elevation change (m)')
    ax2 = fig.add_subplot(212, axisbg='#FFF8DC')
    plt.ylim(-0.42, 1.0)
    d1, d2 = pn.datetime(1995, 5, 8), pn.datetime(2003, 7, 1)
    plt.xlim(d1, d2)
    plt.ylabel('Elevation change (m)')
    t = add_inner_title(ax1, '1-month average (single grid cell)', loc=3)
    t = add_inner_title(ax2, '3-month average (single grid cell)', loc=3)
    fig.autofmt_xdate()
    axs = (ax1, ax2)

    for k, files in enumerate(argv[1:]):
        files = [files]
        dates2 = get_time_range_from_files(files)
        print dates2

        fname = files[0]
        db = tb.openFile(fname)
        data = db.root.tseries.ts
        cols = data.colnames
        t_ref = np.unique(data.cols.t_ref)
        nrow = len(t_ref) + 1
        ncol = len(dates2)
        dh_ij = np.empty((nrow, ncol), 'f8') * np.nan
        se_ij = dh_ij.copy()
        n_ij = dh_ij.copy()

        # fill matrix with one TS at a time
        for i, r in enumerate(t_ref):
            d = data.readWhere('t_ref == "%s"' % r)
            df = get_frame_from_table(fname, d, cols)
            # align TS 
            df = df.reindex(index=dates2)
            dh_ij[i,:] = df['dh']
            se_ij[i,:] = df['se_dh']
            n_ij[i,:] = df['n_ad'] + df['n_da']

        np.fill_diagonal(dh_ij, 0.0)
        np.fill_diagonal(se_ij, 0.0)
        np.fill_diagonal(n_ij, 0.0)

        #dh_ij[5,:] = np.nan     # remove wrong TS

        dh_ij_orig = dh_ij.copy()
        se_ij_orig = se_ij.copy()
        n_ij_orig = n_ij.copy()

        # reference the TS 
        #-----------------------------------
        # My method
        dh_ij = np.ma.masked_invalid(dh_ij)
        ts0 = dh_ij[0,:]
        for i, ts in enumerate(dh_ij[1:]):
            bias = np.mean(ts0 - ts)
            dh_ij[i+1,:] += bias
            print i+1, bias
            if i == nrow-2: 
                break
        #-----------------------------------
        '''
        # Davis method
        dh_ij = np.ma.masked_invalid(dh_ij)
        for i, t in enumerate(dh_ij[0,1:]):
            dh_ij[i+1,:] += t
            print i+1, t
            if i == nrow-2: 
                break
        '''
        se_ij = np.ma.masked_invalid(se_ij)
        for i, e in enumerate(se_ij[0,1:]):
            se_ij[i+1,:] = np.sqrt(e**2 + se_ij[i+1,:]**2)
            if i == nrow-2: 
                break
        n_ij = np.ma.masked_invalid(n_ij)
        for i, n in enumerate(n_ij[0,1:]):
            n_ij[i+1,:] += n
            if i == nrow-2: 
                break
        #-----------------------------------

        # weights for averaging
        n_j = n_ij.sum(axis=0)
        w_ij = n_ij/n_j

        # average TS
        #dh_j = dh_ij.mean(axis=0)
        dh_j, sum_of_weights = np.ma.average(dh_ij, 
            weights=w_ij, axis=0, returned=True)
        dh_j[0] = 0.0
        se_j = np.sqrt(np.ma.average(se_ij**2, weights=w_ij**2, axis=0))

        n_j = np.ma.masked_invalid(n_j)
        n_mean = n_j.mean()

        #------------------------------------------------------------------
        # figures
        #------------------------------------------------------------------

        # plot matrices
        #plot_matrices(dh_ij)
        #plot_matrix(dh_ij_orig, 'Before referencing', loc=3)
        #plt.savefig('mat_before.pdf', dpi=150, bbox_inches='tight') 
        #plot_matrix(dh_ij, 'After referencing', loc=3)
        #plt.savefig('mat_after.pdf', dpi=150, bbox_inches='tight') 
        
        #for ts_i in dh_ij_orig:
        se_ij_orig = np.ma.masked_invalid(se_ij_orig)
        for ts_i, se_i in zip(dh_ij_orig, se_ij_orig):
            axs[k].plot(dates2, ts_i, 'b', linewidth=3)
            axs[k].fill_between(dates2, ts_i+se_i, ts_i-se_i, color='gray', alpha=0.5)
            #axs[k].errorbar(dates2, ts_i, yerr=se_i, color='b', linewidth=2)
        ##t = add_inner_title(axs[k], '# pts: %d' % n_mean, loc=3)
        #for ts in dh_ij:
        #    ax2.plot(dates2, ts, linewidth=2)
        #ax2.plot(dates2, dh_j, 'w', linewidth=5)
        #ax2.plot(dates2, dh_j, 'k', linewidth=3)

        plt.savefig('ts_month_season.pdf', dpi=150, bbox_inches='tight') 
        os.system('cp ts_month_season.pdf ~/posters/agu2011/figures/')

        # errorbars
        '''
        plt.figure()
        plt.errorbar(dates2, dh_j, yerr=se_j, color='k', linewidth=2)
        plt.grid(True)
        '''

    db.close()
    plt.show()

    #-----------------------------------------------------------------
    # save data
    #-----------------------------------------------------------------

    if not save_to_file:
        return

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

    print 'out file -->', fname_out

if __name__ == '__main__':
    sys.exit(main(sys.argv))
