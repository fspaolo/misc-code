"""
Module containing functions and classes used by:

xover2grid.py
xover2box.py

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 14, 2011

import os
import sys
import re
import numpy as np
import scipy as sp
import tables as tb
import datetime as dt
import matplotlib.pyplot as plt

sys.path.append('/Users/fpaolo/code/misc')
from util import *


def compute_dh_ad_da(h1, h2, ftrack1, ftrack2, return_index=False):
    '''
    Take arrays `h1` and `h2` and compute `dh_ad` and `dh_da`.
    '''
    dh = h2 - h1                                       # always t2 - t1
    i_ad, = np.where((ftrack2 == 0) & (ftrack1 == 1))  # dh_ad
    i_da, = np.where((ftrack2 == 1) & (ftrack1 == 0))  # dh_da
    if return_index:
        return dh, i_ad, i_da
    else:
        return dh[i_ad], dh[i_da]


def compute_ordinary_mean(x1, x2, useall=False):
    '''
    Mean of the average values of the arrays `x1` and `x2`: 
    (<x1> + <x2>)/2.

    If `useall=True` returns <x1>|<x2> as the mean value if one 
    of the arrays is empty, otherwise returns NaN.
    '''
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        mean = (x1.mean() + x2.mean())/2.
    elif useall and n1 > 0:
        mean = x1.mean()
    elif useall and n2 > 0:
        mean = x2.mean()
    else:
        mean = np.nan
    return mean 


def compute_weighted_mean(x1, x2, useall=False):
    '''
    Weighted mean of the average values of the arrays `x1` and `x2`: 
    (n1 <x1> + n2 <x2>)/(n1 + n2)

    If `useall=True` returns <x1>|<x2> as the mean value if one 
    of the arrays is empty, otherwise returns NaN.
    '''
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        mean = (n1*x1.mean() + n2*x2.mean())/(n1 + n2)
    elif useall and n1 > 0:
        mean = x1.mean()
    elif useall and n2 > 0:
        mean = x2.mean()
    else:
        mean = np.nan
    return mean


def compute_weighted_mean_error(x1, x2, useall=False):
    """
    Weighted mean standard error for the stimated <dh> according 
    Davis et. al. (2001, 2004, 2006).

    If `useall=True` returns std1|std2 as the mean error if one 
    of the arrays is empty, otherwise returns NaN.
    """
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        error = np.sqrt(n1*x1.var() + n2*x2.var())/(n1 + n2)
    elif useall and n1 > 0:
        error = x1.std()/n1 
    elif useall and n2 > 0:
        error = x2.std()/n2
    else:
        error = np.nan
    return error


def compute_standard_error(x1, x2, useall=False):
    """
    Standard error for the stimated <dh>: err = std/sqrt(n)

    If `useall=True` returns std1|std2 as the mean error if one 
    of the arrays is empty, otherwise returns NaN.
    """
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        error = ((x1.std() + x2.std())/2.)/np.sqrt(n1 + n2)
    elif useall and n1 > 0:
        error = x1.std()/np.sqrt(n1) 
    elif useall and n2 > 0:
        error = x2.std()/np.sqrt(n2)
    else:
        error = np.nan
    return error


def compute_wingham_error(x1, x2, useall=False):
    """
    Error for the stimated <dh> according Wingham et. al. (2009). 

    If `useall=True` returns std1|std2 as the error if one 
    of the arrays is empty, otherwise returns NaN.
    """
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        error = np.abs(x1.mean() - x2.mean())
    elif useall and n1 > 0:
        error = x1.std()/n1
    elif useall and n2 > 0:
        error = x2.std()/n2
    else:
        error = np.nan
    return error


def std_iterative_editing(x, nsd=3, return_index=False):
    """
    Iterates filtering out all values greater than 3-std.
    """
    niter = 0
    while True: 
        sd = x[~np.isnan(x)].std()  # ignore NaNs
        i, = np.where(np.abs(x) > nsd*sd)
        if len(i) > 0:
            x[i] = np.nan
            niter += 1
        else:
            break
    if return_index:
        return np.where(~np.isnan(x))[0]
    else:
        return x[~np.isnan(x)]


def abs_value_editing(x, absval, return_index=False):
    """
    Filter out all values greater than `absval`.
    """
    i, = np.where(np.abs(x) <= absval)
    if return_index:
        return i 
    else:
        return x[i]


def apply_tide_and_load_corr(d):
    d['h1'] = d['h1'] - d['tide1'] + d['load1']
    d['h2'] = d['h2'] - d['tide2'] + d['load2']
    return d


def get_time_from_fname(fname):
    """
    Get satellite name, reference time, year and month
    of elevation change from the input file name.
    """
    sat_name, ref_time = os.path.basename(fname).split('_')[:2]
    t1, t2 = re.findall('\d\d\d\d\d\d+', fname) 
    year, month = np.int32(t2[:4]), np.int32(t2[4:6])
    return sat_name, ref_time, year, month


def get_fname_out(files, fname_out=None, prefix=None, sufix=None):
    """
    Construct the output file name with the min and max times 
    from the input files.
    """
    path, name = os.path.split(files[0])  # path from any file
    if fname_out is None:
        times = [re.findall('\d\d\d\d\d\d+', fname) for fname in files]
        t_1 = [t1 for t1, t2 in times]
        t_2 = [t2 for t1, t2 in times]
        t_min, t_max =  min(t_1), max(t_2)
        if prefix is None:
            prefix = name.split('_')[0]  # sat name
        name = '_'.join([prefix, t_min, t_max, sufix])
    else:
        name = fname_out
    return os.path.join(path, name)


def create_output_containers(fname_out, atom, (nj,nk)):
    # open or create output file
    title = 'FRIS Average Time Series'
    filters = tb.Filters(complib='blosc', complevel=9)
    file_out = tb.openFile(fname_out, 'w')
    
    dout = {}
    g = file_out.createGroup('/', 'fris')
    dout['table'] = file_out.createTable(g, 'table', TimeSeriesGrid, title, filters)
    
    dout['dh_mean'] = file_out.createEArray(g, 'dh_mean', atom, (0, nj, nk), '', filters)
    dout['dh_error'] = file_out.createEArray(g, 'dh_error', atom, (0, nj, nk), '', filters)
    dout['dg_mean'] = file_out.createEArray(g, 'dg_mean', atom, (0, nj, nk), '', filters)
    dout['dg_error'] = file_out.createEArray(g, 'dg_error', atom, (0, nj, nk), '', filters)
    dout['n_ad'] = file_out.createEArray(g, 'n_ad', atom, (0, nj, nk), '', filters)
    dout['n_da'] = file_out.createEArray(g, 'n_da', atom, (0, nj, nk), '', filters)

    dout['x_edges'] = file_out.createCArray(g, 'x_edges', atom, (nj+1,), '', filters)
    dout['y_edges'] = file_out.createCArray(g, 'y_edges', atom, (nk+1,), '', filters)
    dout['lon'] = file_out.createCArray(g, 'lon', atom, (nj,), '', filters)
    dout['lat'] = file_out.createCArray(g, 'lat', atom, (nk,), '', filters)

    return dout, file_out


def need_to_save_now(pos, fname, files):
    """
    Find when a TS is formed and need to be saved. 

    A TS is formed when (one of the followings):
    1) current file is the last one
    2) current sat_name is different from next sat_name
    3) current ref_time is different from next ref_time
    """
    try:
        fname_next = files[pos+1]  # next file
    except:
        return True                # last file
    name = os.path.basename(fname)
    name_next = os.path.basename(fname_next)
    sat_name, reftime = name.split('_')[:2]
    sat_next, reftime_next = name_next.split('_')[:2]
    if sat_name != sat_next or reftime != reftime_next:
        return True
    else:
        return False


# warnning: it is switched --> y, x !!!
def bin_by_mean(lon, lat, z, bins=10, range=None):
    bins = bins[::-1] 
    range = range[::-1]
    wsum, _ = np.histogramdd((lat, lon), weights=z, bins=bins, range=range)
    ppbin, edges = np.histogramdd((lat, lon), bins=bins, range=range) 
    #ppbin[ppbin==0] = np.nan
    #ppbin = np.ma.masked_equal(ppbin, 0)
    return (wsum/ppbin), ppbin, edges[1], edges[0]


def print_info(x_edges, y_edges, lon, lat, dx, dy):
    print '='*70
    print 'grid domain (edges): l/r/b/t = %.1f/%.1f/%.1f/%.1f deg' \
        % (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    print 'grid domain (cells): l/r/b/t = %.1f/%.1f/%.1f/%.1f deg' \
        % (lon[0], lon[-1], lat[0], lat[-1])
    print 'grid spacing (cells): dx x dy = %.1f x %.1f deg' % (dx, dy)
    print 'grid size (elements): nx x ny = %d x %d' % (len(lon), len(lat))
    print '='*70


def plot_ts(table):
    """
    Plot dh and dAGC time series and the correlation dAGC x dh.
    """
    sys.path.append('/Users/fpaolo/code/misc')
    from util import poly_fit
    # load data from Table
    year = table.cols.year[:] 
    month = table.cols.month[:] 
    dh_mean = table.cols.dh_mean[:] 
    dh_error = table.cols.dh_error[:] 
    dg_mean = table.cols.dg_mean[:] 
    dg_error = table.cols.dg_error[:] 
    dates = [dt.datetime(y, m, 15) for y, m in zip(year, month)]
    # plot TS
    fig = plt.figure()
    plt.subplot(211)
    plt.errorbar(dates, dh_mean, yerr=dh_error, linewidth=2)
    plt.ylabel('dh (m)')
    plt.subplot(212)
    plt.errorbar(dates, dg_mean, yerr=dg_error, linewidth=2)
    plt.ylabel('dAGC (dB)')
    fig.autofmt_xdate()
    # plot correlation
    dg_fit, dh_fit, _ = poly_fit(dg_mean, dh_mean)
    plt.figure()
    plt.plot(dg_mean, dh_mean, 'o')
    plt.plot(dg_fit, dh_fit, linewidth=2.5)
    plt.xlabel('dAGC (dB)')
    plt.ylabel('dh (m)')
    corr = np.corrcoef(dg_mean, dh_mean)[0,1]
    print 'correlation = %.2f' % corr


def plot_grids(x, y, g1, g2, g3, g4):
    """
    Plot the 4 calculated grids: dh, dAGC, n_ad, n_da.

    Notes
    -----
    `pcolor` cannot have NaN, use `masked array` instead.
    `pcolor` does not preserve the lon,lat aspect ratio like `imshow`.
    """
    #sys.path.append('/Users/fpaolo/code/misc')
    #import viz
    #cmap = viz.colormap('rgb')
    g1 = np.ma.masked_invalid(g1)
    g2 = np.ma.masked_invalid(g2)
    g3 = np.ma.masked_invalid(g3)
    g4 = np.ma.masked_invalid(g4)
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    xx, yy = np.meshgrid(x, y)
    fig = plt.figure()
    plt.subplot(211)
    plt.pcolor(xx, yy, g1)
    plt.colorbar()
    plt.subplot(212)
    plt.pcolor(xx, yy, g2)
    plt.colorbar()
    #viz.colorbar(fig, cmap, (-2,2))
    fig = plt.figure()
    plt.subplot(211)
    plt.pcolor(xx, yy, g3)
    plt.colorbar()
    plt.subplot(212)
    plt.pcolor(xx, yy, g4)
    plt.colorbar()
