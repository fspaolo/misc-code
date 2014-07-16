#!/usr/bin/env python
"""
Functions to be used by:

    timesep.py
    timesep_arr.py
    timesep_tbl.py

Fernando Paolo <fpaolo@ucsd.edu>
Aug 22, 2012
"""

import datetime as dt
import numpy as np
import tables as tb
import argparse as ap
import matplotlib.dates as mpl
from itertools import cycle

 
class SecsToDateTime2(object):
    """
    Converts `seconds since epoch` to `datetime` (i.e., year, month, day).

    secs : 1D array, decimal seconds.
    since_year : int, ref_epoch = <since_year>-Jan-1 00:00:00 is assumed.
    since_epoch : tuple, especifies ref_epoch as (YYYY, MM, DD, hh, mm, ss).

    Notes
    -----
    1. Matlab uses as time reference the year 0000, and Python 
       `datetime` uses the year 0001.
    2. utc85 (or ESA-time) is seconds since 1985-1-1 00:00:00,
       ICESat-time is seconds since 2000-1-1 12:00:00,
       secs00 is seconds since 2000-1-1 00:00:00.

    """
    def __init__(self, secs=0, since_year=1985, since_epoch=None):
        if np.ndim(secs) > 0:
            self.secs = np.asarray(secs)
        else:
            self.secs = secs  

        if since_epoch is None:
            # <since_year>-Jan-1 00:00:00
            ref_epoch = dt.date(since_year, 1, 1)
        else:
            # not working!
            ref_epoch = dt.datetime(since_epoch)    

        # ref_epoch in days since 0001-Jan-1 00:00:00
        ref_epoch_in_days = mpl.date2num(ref_epoch)  

        # secs/86400 -> frac days -> date
        frac_days = self.secs / (24*60*60.)
        self._datenum = ref_epoch_in_days + frac_days
        self._dates = mpl.num2date(self._datenum)

    def datenum(self, matlab=False):
        if matlab:
            # frac days since 0000-Jan-1 00:00:00
            return self._datenum + 366.
        else:
            # frac days since 0001-Jan-1 00:00:00
            return self._datenum

    def dates(self):
        return self._dates

    def years(self):
        return np.array([d.year for d in self._dates])

    def months(self):
        return np.array([d.month for d in self._dates])
        
    def days(self):
        return np.array([d.day for d in self._dates])

    def ymdhms(self):
        return np.array([(d.year, d.month, d.day, d.hour,
            d.minute, d.second) for d in self._dates])


class SecsToDatetime(object):
    """
    Converts `seconds since epoch` to `datetime` (i.e., year, month, day).

    secs : 1D array, decimal seconds.
    since_year : int, ref_epoch = <since_year>-Jan-1 00:00:00 is assumed.
    since_epoch : tuple, especifies ref_epoch as (YYYY, MM, DD, hh, mm, ss).

    Notes
    -----
    1. Matlab uses as time reference the year 0000, and Python 
       `datetime` uses the year 0001.
    2. utc85 (or ESA-time) is seconds since 1985-1-1 00:00:00,
       ICESat-time is seconds since 2000-1-1 12:00:00,
       secs00 is seconds since 2000-1-1 00:00:00.

    """
    def __init__(self, secs=0, since_year=1985, since_epoch=None):
        if np.ndim(secs) > 0:
            self.secs = np.asarray(secs)
        else:
            self.secs = secs  

        if since_epoch is None:
            # <since_year>-Jan-1 00:00:00
            ref_epoch = dt.datetime(since_year, 1, 1, 0, 0, 0)
        else:
            # not working!
            ref_epoch = dt.datetime(since_epoch)    

        # ref_epoch in days since 0001-Jan-1 00:00:00
        ref_epoch_in_days = mpl.date2num(ref_epoch)  

        # secs/86400 -> frac days -> date
        frac_days = self.secs / (24*60*60.)
        self._datenum = ref_epoch_in_days + frac_days
        self._dates = mpl.num2date(self._datenum)

    def datenum(self, matlab=False):
        if matlab:
            # frac days since 0000-Jan-1 00:00:00
            return self._datenum + 366.
        else:
            # frac days since 0001-Jan-1 00:00:00
            return self._datenum

    def dates(self):
        return self._dates

    def years(self):
        return np.array([d.year for d in self._dates])

    def months(self):
        return np.array([d.month for d in self._dates])
        
    def days(self):
        return np.array([d.day for d in self._dates])

    def ymdhms(self):
        return np.array([(d.year, d.month, d.day, d.hour,
            d.minute, d.second) for d in self._dates])


def sec2dt(secs, since_year=1985):
    dt_ref = dt.datetime(since_year, 1, 1, 0, 0)
    return [dt_ref + dt.timedelta(seconds=s) for s in secs]


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


ntuples = lambda lst, n: zip(*[lst[i:]+lst[:i] for i in range(n)])


def get_moving_windows(years, months, n=3):
    """
    Find all n-month windows with 1-month time steps.

    Return a list with: `indices`, `year` and `month` (first-month of 
    the n-month window), of all data points belonging to *each* 
    n-month block: [(ind_1, y_1, m1_1), (ind_2, y_2, m1_2), ...].
    Note: data must be ordered in time.

    WARNING: for now only works for n=3!
    """
    m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    windows = ntuples(m, n)                         # define the n-month windows
    m1 = months[0]                                  # take the first valid month
    year = years[0]                                 # take the respective year
    window = windows[m1-1]                          # take the respective window
    res = []
    while True:
        m1, m2, m3 = window
        if m1 == 11:
            ind, = np.where( ((years == year) & ((months == m1) | (months == m2))) | \
                             ((years == year+1) & (months == m3)) )
        elif m1 == 12:
            ind, = np.where( ((years == year) & (months == m1)) | \
                             ((years == year+1) & ((months == m2) | (months == m3))) )
        else:
            ind, = np.where( (years == year) & \
                             ((months == m1) | (months == m2) | (months == m3)) )
        res.append((ind, year, m1))
        k = ind[-1] + 1                             # next valid index
        if k == months.shape[0]: 
            break
        m1, m2, m3 = window[m2-1]                   # next window
        year = years[k]                             # next year
    return res 


def get_window(years, months, m1, m2):
    """
    Return the month window m1-m2 for every year.
    """
    res = []
    for year in np.unique(years):
        ind, = np.where( (years == year) & (m1 <= months) & (months <= m2) )
        res.append((ind, year, [m1, m2]))
    return res


def get_intervals(datenum, t1, t2):
    """
    Return the time intervals defined by t1 and t2 (list of tuples).
    """
    res = []
    dn1 = [dt.datetime(y, m, d).toordinal() for y, m, d in t1]
    dn2 = [dt.datetime(y, m, d).toordinal() for y, m, d in t2]
    for d1, d2 in zip(dn1, dn2):
        ind, = np.where( (d1 <= datenum) & (datenum <= d2) )
        d = d1 + (d2 - d1)/2.
        t = dt.datetime.fromordinal(int(d)).strftime('%Y%m%d')
        res.append((ind, t))
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


def time_sep(sep='years', **kw):
    if sep == 'months':
        res = get_months(**kw)
    elif sep == 'seasons':
        res = get_seasons(**kw)
    elif sep == 'window':
        res = get_window(**kw)
    elif sep == 'intervals':
        res = get_intervals(**kw)
    else:
        res = get_years(**kw)
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
        filters = tb.Filters(complib='blosc', complevel=9)
        dout = fout.createCArray(fout.root,'data', atom=atom, 
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
