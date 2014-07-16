"""
Module with some high level utility functions.

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import os
import re
import numpy as np
import scipy as sp
import tables as tb
import datetime as dt

# definition of Table structures for HDF5 files

class TimeSeries(tb.IsDescription):
    sat_name = tb.StringCol(10, pos=1)
    ref_time = tb.StringCol(10, pos=2)
    date = tb.StringCol(10, pos=3)
    year = tb.Int32Col(pos=4)
    month = tb.Int32Col(pos=5)
    dh_mean = tb.Float64Col(pos=6)
    dh_error = tb.Float64Col(pos=7)
    dg_mean = tb.Float64Col(pos=8)
    dg_error = tb.Float64Col(pos=9)
    n_ad = tb.Int32Col(pos=10)
    n_da = tb.Int32Col(pos=11)


class TimeSeriesGrid(tb.IsDescription):
    sat_name = tb.StringCol(10, pos=1)
    ref_time = tb.StringCol(10, pos=2)
    date = tb.StringCol(10, pos=3)
    year = tb.Int32Col(pos=4)
    month = tb.Int32Col(pos=5)


def linear_fit(x, y, return_coef=False):
    """
    Fit a straight-line by Ordinary Least Squares.

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    x, y = x[ind], y[ind]
    A = np.ones((len(x), 2))
    A[:,0] = x
    m, c = np.linalg.lstsq(A, y)[0]
    if return_coef:
        return (m, c)
    else:
        x_val = np.linspace(x.min(), x.max(), 200)
        y_fit = m*x_val + c
        return (x_val, y_fit)


def linear_fit_robust(x, y, return_coef=False):
    """
    Fit a straight-line by robust regression (M-estimate).

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    import scikits.statsmodels.api as sm
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    x, y = x[ind], y[ind]
    X = sm.add_constant(x, prepend=False)
    y_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    y_fit = y_model.fit()
    if return_coef:
        if len(y_fit.params) < 2: return (y_fit.params[0], 0.)
        else: return y_fit.params[:]
    else:
        return (x, y_fit.fittedvalues)


def poly_fit(x, y, order=1, return_coef=False):
    """
    Fit a polynomial of order `order` to data points `x,y`.
    """
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    x, y = x[ind], y[ind]
    coef = np.polyfit(x, y, order)
    if return_coef:
        return coef
    else:
        x_val = np.linspace(x.min(), x.max(), 200)
        y_fit = np.polyval(coef, x_val)
        return (x_val, y_fit)


def spline_interp(x, y, smooth=0.01):
    """
    Interpolate data using cubic spline of given smoothness.
    smooth : smoothness factor
    """
    from scipy.interpolate import splrep, splev
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y)))
    x, y = x[ind], y[ind]

    # find the knot points
    tck = splrep(x, y, s=smooth)

    # evaluate spline on interpolated points
    x_val = np.linspace(x.min(), x.max(), 200)
    y_fit = splev(x_val, tck)
    return (x_val, y_fit)


def get_size(arr):
    """
    Get the size in MB of a Numpy or PyTables object.

    parameters
    ----------
    arr : 1D/2D Numpy or PyTables Array.
    """
    try:
        m, n = arr.shape
    except:
        m, n = arr.shape, 1
    num_elem = m*n
    item_size = arr.dtype.itemsize
    return (item_size*num_elem/1e6)


def check_if_can_be_loaded(data, max_size=512):
    """
    Check if PyTables Array can be loaded in memory.
    """
    if get_size(data) > max_size:
        msg = 'data is larger than %d MB, not loading in-memory!' \
            % max_size
        raise MemoryError(msg)
    else:
        return True


def check_if_need_to_save(data, max_size=128):
    """
    Check when data in memory need to be flushed on disk.
    """
    data = np.asarray(data)
    if get_size(data) > max_size:
        return True
    else:
        return False


def _get_season(year, month, return_month=2):
    """
    Returns the first, second or third month of the 3-month 
    season-block, and update the `year` when needed.

    year, month : int
    return_month : 1, 2 or 3
    """
    if return_month != 1 and return_month != 2 and return_month != 3:
        raise IOError('`return_month` must be: 1, 2 or 3')
    MAM = [3, 4, 5]      # Mar/Apr/May -> Fall SH 
    JJA = [6, 7, 8]      # Jun/Jul/Aug -> winter SH
    SON = [9, 10, 11]    # Sep/Oct/Nov -> Spring SH
    DJF = [12, 1, 2]     # Dec/Jan/Feb -> summer SH
    return_month -= 1
    if month in MAM:
        return year, MAM[return_month]
    elif month in JJA:
        return year, JJA[return_month]
    elif month in SON:
        return year, SON[return_month]
    elif month in DJF:
        if month == 12 and return_month > 0:
            year += 1
        return year, DJF[return_month]
    else:
        print 'not a valid month from 1 to 12!'
        return None, None


def get_season(year, month, return_month=2):
    """
    Apply `_get_season` to a scalar or sequence. See `_get_season`.
    """
    if not np.iterable(year) or not np.iterable(month):
        year = np.asarray([year])
        month = np.asarray([month])
    ym = np.asarray([_get_season(y, m, return_month) for y, m in zip(year, month)])
    return ym[:,0], ym[:,1]


def year_month_to_dtime(year, month):
    """
    Convert year and month to `datetime` object.
    """
    return [dt.datetime(y, m, 15) for y, m in zip(year, month)]


def year_month_to_dtime(year, month):
    """
    Convert year and month to `datetime` object.
    """
    return [dt.datetime(y, m, 15) for y, m in zip(year, month)]


def box(region):
    """
    Generate a box given `region` = (W,E,S,N).
    """
    west, east, south, north = region
    x = np.array([west, east, east, west, west])
    y = np.array([north, north, south, south, north])
    return x, y
