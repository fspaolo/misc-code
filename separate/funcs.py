#!/usr/bin/env python
"""
Functions to be used by:

    timesep.py
    timesep_arr.py
    timesep_tbl.py

Fernando Paolo <fpaolo@ucsd.edu>
Aug 22, 2012
"""

import numpy as np
import tables as tb
import argparse as ap
import datetime as dt

 
def sec2dt(secs, since_year=1985):
    dt_ref = dt.datetime(since_year, 1, 1, 0, 0)
    return np.asarray([dt_ref + dt.timedelta(seconds=s) for s in secs])


def what_season(month):
    """
    Given a month finds the 3-month block of the respective season.
    """
    season1 = [12, 1, 2]     # Dec-Jan-Feb -> summer SH
    season2 = [3, 4, 5]      # Mar-Apr-May -> Autumn SH 
    season3 = [6, 7, 8]      # Jun-Jul-Aug -> winter SH
    season4 = [9, 10, 11]    # Sep-Oct-Nov -> Spring SH
    if month in season1:
        return season1
    elif month in season2:
        return season2
    elif month in season3:
        return season3
    elif month in season4:
        return season4
    else:
        print 'not a valid month from 1 to 12!'
        return [None, None, None]    # none existent range


def get_seasons(years, months):
    """
    Find the 3-month block referent to the seasons.

    Return a list with: `indices`, `year` and `month` (middle-month of 
    the season, m2), of all data points belonging to each season, 
    e.g., [(ind_1, y_1, m2_1), (ind_2, y_2, m2_2), ...].
    Note: data must be ordered in time.
    """
    res = []
    ii = 0
    while True:
        first_valid_month = months[ii]               # take the first month
        year = years[ii]                             # take the respective year
        m1, m2, m3 = what_season(first_valid_month)  # month -> season (3-month block)
        if first_valid_month == 12:
            ind, = np.where( ((years == year) & (months == m1)) | \
                             ((years == year+1) & ((months == m2) | (months == m3))) )
            year += 1
        elif first_valid_month == 1 or first_valid_month == 2:
            ind, = np.where( (years == year) & \
                             ((months == m2) | (months == m3)) )
        else:
            ind, = np.where( (years == year) & \
                             ((months == m1) | (months == m2) | (months == m3)) )
        res.append((ind, year, m2))
        ii = ind[-1] + 1                             # month of next season
        if ii == months.shape[0]: 
            break
    return res 


def get_times(files, timecol, since_year):
    """
    Find min and max times of a set of files.
    """
    tmin, tmax = np.inf, 0
    for fname in files:
        f = tb.openFile(fname, 'r')
        data = f.getNode('/data')
        secs = data[:,timecol]
        t_min, t_max = secs.min(), secs.max()
        if t_min < tmin:
            tmin = t_min
        if t_max > tmax:
            tmax = t_max
        f.close()
    tmin, tmax = sec2dt([tmin, tmax], since_year)
    print 'data time range:', tmin, tmax
    return [tmin, tmax]


def define_windows(t1, t2, n=3):
    """
    N-month windows at 1-month steps.
    """
    step = dt.timedelta(seconds=2629800)   # 1 month (according astronomy)
    window = n*step                        # n months
    print 'total months:', (t2-t1).total_seconds()/step.total_seconds()
    print 'window size: {0} months'.format(n)
    t_beg, t_end = t1, t2
    windows = []
    while True:
        t2 = t1 + window
        windows.append((t1, t2))
        t1 += step
        if t2 >= t_end: break
    print 'num windows:', len(windows) 
    print 'windows time range:', t_beg, t2
    return windows


def get_windows(dtimes, windows):
    """
    Find all data points within the given windows.

    Return a list with: `indices` and `datetime` (center of the
    window), of all data points belonging to each given window.
    windows : set of windows defined as (t1, t2)
    result : [(inds1, dt1), (inds2, dt2),...]
    Note : data must be ordered in time.
    """
    result = []
    for t1, t2 in windows:
        ind, = np.where((dtimes >= t1) & (dtimes <= t2))
        if len(ind) > 0:
            tm = t1 + (t2-t1)/2
            result.append((ind, tm))
    return result


def get_window(years, months, m1, m2):
    """
    Return the month window m1-m2 for every year.
    """
    res = []
    for year in np.unique(years):
        ind, = np.where( (years == year) & (m1 <= months) & (months <= m2) )
        res.append((ind, year, [m1, m2]))
    return res


def get_months(years, months):
    """
    Return the months in every year.
    """
    res = []
    ii = 0
    while True:
        first_valid_month = months[ii]               # take the first month
        year = years[ii]                             # take the respective year
        ind, = np.where( (years == year) & (months == first_valid_month) )
        res.append((ind, year, first_valid_month))
        ii = ind[-1] + 1                             # next month
        if ii == months.shape[0]: 
            break
    return res


def get_years(years):
    res = []
    for year in np.unique(years):
        ind, = np.where(years == year)
        res.append((ind, year, None))
    return res


def print_dates(dates, N):
    """Just to test the date output."""
    for j in dates[:N]:
        print j.year, j.month, j.day, j.hour, j.minute, j.second, j.microsecond
    print ''


def save_arr(fname, data):
    """
    Open/Create specific output file and save specific variables.
    """
    if '.txt' in fname:
        np.savetxt(fname, data, fmt='%f')
    elif '.h5' in fname:
        fout = tb.openFile(fname, 'w')
        shape = data.shape
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='zlib', complevel=9)
        dout = fout.createCArray('/','data', atom=atom, 
                                 shape=shape, filters=filters)
        dout[:] = data
        fout.close()


def save_tbl(fname, table, rows=None, complib='zlib'):
    """
    Create/Reopen a file and save an existing table.
    """
    filters = tb.Filters(complib=complib, complevel=9)
    f = tb.openFile(fname, 'a')  # if doesn't exist create it
    t = f.createTable('/', table.name, table.description, '', filters)
    if rows is None:
        t.append(table[:])
    else:
        t.append(table[rows])
    t.flush()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 
