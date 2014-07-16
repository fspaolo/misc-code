#!/usr/bin/env python
doc = """\
Given `utc85` (ESA time) separate files by: years, seasons, months or month-window.
""" 
"""
Fernando Paolo <fpaolo@ucsd.edu>
October 22, 2010
"""

import os
os.environ['MPLCONFIGDIR'] = '/var/tmp'    # to ensure a writable dir when 
                                           # accessing Xgrid as "nobody"
import sys
import datetime as dt
import numpy as np
import tables as tb
import argparse as ap
import matplotlib.dates as mpl

# parse command line arguments
parser = ap.ArgumentParser(description=doc)
group = parser.add_mutually_exclusive_group()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-c', dest='utc85col', type=int, default=1, 
    help='column of utc85 variable (0,1..) [default: 1]')
group.add_argument('-m', dest='month', action='store_const', const=True,
    help='separate by month [default: by year]')
group.add_argument('-s', dest='season', action='store_const', const=True,
    help='separate by season [default: by year]')
group.add_argument('-w', dest='window', nargs=2, type=int, metavar=('MON1', 'MON2'),
    help='separate by month-window (MON=1,2..12, increasing order)')
parser.add_argument('-d', dest='dir', action='store_const', const=True, \
    default=False, help='create directories for output files [default: no]')
parser.add_argument('-p', dest='N', type=int, default=None, 
    help='print on the screen N dates of each file and exit')

args = parser.parse_args()

SEP = '_'  # symbol to separate the suffix: '_', '-', ...

MONTH_NAME = {1:'01', 2:'02', 3:'03', 4:'04', 5:'05', 6:'06',
              7:'07', 8:'08', 9:'09', 10:'10', 11:'11', 12:'12'}
### functions to EDIT

class Input(object):
    """
    [TO EDIT] Handles the program input.

    - Get specific variables from specific file type and structure
    - Prepare specific input for the processing
    """
    def __init__(self, fname, col=1, N=None):
        self.fname = fname
        self.col = col
        self.N = N

    def read_file(self):
        """
        Get and return specific variables from specific file type.
        """
        if '.h5' in self.fname:
            h5f = tb.openFile(self.fname, 'r')
            #data = h5f.root.data.read()    # in-memory
            data = h5f.root.data            # out-of-memory
        if data.shape[0] < 1:
            return None 
        secs = data[:,self.col]
        dt = SecsToDateTime(secs, since_year=1985)
        if self.N is not None:
            print self.fname
            print_dates(dt.dates(), self.N)     # to test the output
            h5f.close()
            return None 
        # dates -> months, years
        return [dt.months(), dt.years(), data, h5f]  

    
def get_from_file(fname):
    """
    [TO EDIT] Get and return specific variables from specific file types.

    Returns the specified `data` and `file object`.
    """
    if '.txt' in fname:
        data = np.loadtxt(fname)
    elif '.h5' in fname:
        h5f = tb.openFile(fname, 'r')
        #data = h5f.root.data.read()  # in-memory
        data = h5f.root.data          # out-of-memory
    if data.shape[0] < 1:
        return [None, h5f]
    else:
        return [data, h5f]
    

class Output(object):
    """
    [TO EDIT] Handle the program output.

    - Output file type and structure
    - What variables to save
    """
   pass


class SecsToDateTime(object):
    """Converts `seconds since epoch` to `datetime` (i.e., year, month, day).

    Note1: Matlab uses as time reference the year 0000, and Python 
    `datetime` uses the year 0001.
    Note2: UTC85 (or ESA time) is seconds since 1985-1-1 00:00:00.
    """
    def __init__(self, secs=0, since_year=1985):
        # utc85 is seconds since 1985-Jan-1 00:00:00
        if np.ndim(secs) > 0:
            self.secs = np.asarray(secs)
        else:
            self.secs = secs  

        # 1985-Jan-1 00:00:00 in days (since year 0001-Jan-1 00:00:00)
        REF_EPOCH_IN_DAYS = mpl.date2num(dt.date(since_year, 1, 1))  

        # secs/86400 -> frac days -> date
        FRAC_DAYS = self.secs / 86400.
        self._dates = mpl.num2date(REF_EPOCH_IN_DAYS + FRAC_DAYS)

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
    Return a list with: indices, year and month (middle-month of 
    the season), of all data points belonging to each season, e.g., 
    [(ind_1, y_1, m2_1), (ind_2, y_2, m2_2), ...].
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


def get_window(years, months, m1, m2):
    res = []
    for year in np.unique(years):
        ind, = np.where( (years == year) & (m1 <= months) & (months <= m2) )
        res.append((ind, year, [m1, m2]))
    return res


def get_months(years, months):
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


def time_sep(years, months=None, sep='years', window=None):
    if sep == 'months':
        res = get_months(years, months)
    elif sep == 'seasons':
        res = get_seasons(years, months)
    elif sep == 'window':
        m1, m2 = window
        res = get_window(years, months, m1, m2)
    else:
        res = get_years(years)
    return res


def print_dates(dates, N):
    """Just to test the date output."""
    for j in dates[:N]:
        print j.year, j.month, j.day, j.hour, j.minute, j.second, j.microsecond
    print ''



def save_to_file(fname_out, data):
    """
    [TO EDIT] Open/Create specific output file and save specific variables.
    """
    if '.txt' in fname_out:
        np.savetxt(fname_out, data, fmt='%f')
    elif '.h5' in fname_out:
        fout = tb.openFile(fname_out, 'w')
        shape = data.shape
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        dout = fout.createCArray(fout.root,'data', atom=atom, 
                                 shape=shape, filters=filters)
        dout[:] = data
        fout.close()


def main(args):
                                                
    files = args.file
    col = args.utc85col
    month = args.month
    window = args.window
    season = args.season
    dir = args.dir
    N = args.N

    print 'files to read:', len(files)
    print 'time (utc85) column:', col
    print 'create directories:', dir

    if month is not None:
        sep = 'months'
        print 'separate by: months'
    elif season is not None:
        sep = 'seasons'
        print 'separate by: seasons'
    elif window is not None:
        sep = 'window'
        print 'separate by: window', window
    else:
        sep = 'years'
        print 'separate by: years'

    print 'reading and processing files ...'

    # input
    #-----------------------------------------------------------------

    ndirs = 0
    nfiles = 0
    for f in files:

        In = Input(f, col, N)
        d = In.read_file()
        if d is None: continue
        months, years, data, h5f = d

        # processing
        #-------------------------------------------------------------

        res = time_sep(years, months, sep=sep, window=window)

        # output -> dir or file
        #-------------------------------------------------------------

        path, file_name = os.path.split(f)
        fname, ext = os.path.splitext(file_name)

        # one output per result
        for r in res:
            ind, year, month = r[:]

            if ind.shape[0] < 1: 
                continue

            data_out = data[ind,:]

            if month is None:
                month_name = ''
            elif np.ndim(month) > 0:
                month_name = MONTH_NAME[month[0]] + MONTH_NAME[month[1]]
            else:
                month_name = MONTH_NAME[month]

            if dir:
                dname = str(year) + month_name
                outdir = os.path.join(path, dname) 
                if not os.path.exists(outdir):         # if dir doesn't exist
                    os.mkdir(outdir)                   # create one
                    ndirs += 1
                fname_out = os.path.join(outdir, file_name) 
            else:
                file_name = ''.join([fname, SEP, str(year), month_name, ext])
                fname_out = os.path.join(path, file_name)

            save_to_file(fname_out, data_out)
            nfiles += 1

        try:
            h5f.close()
        except:
            pass

    print 'done!'
    if dir is True:
        print 'directories created:', ndirs
    print 'files created:', nfiles


if __name__ == '__main__':
    main(args)
