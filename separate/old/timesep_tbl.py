#!/usr/bin/env python
doc = """\
Given time (in secs) separate files by: years, seasons, months or month-window.
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

from funcs import *

# ICESat campaigns
T1 = [(2003,10,4), (2004,2,17), (2004,5,18), (2004,10,3), 
      (2005,2,17), (2005,5,20), (2005,10,21), (2006,2,22),
      (2006,5,24), (2006,10,25), (2007,3,12), (2007,10,2), 
      (2008,2,17), (2008,10,4), (2008,11,25), (2009,3,9)]

T2 = [(2003,11,19), (2004,3,21), (2004,6,21), (2004,11,8),
      (2005,3,24), (2005,6,23), (2005,11,24), (2006,3,28),
      (2006,6,26), (2006,11,27), (2007,4,14), (2007,11,5),
      (2008,3,21), (2008,10,19), (2008,12,17), (2009,4,11)]

SEP = '_'  # symbol to separate the suffix, e.g. '_', '-'

MONTH_NAME = {1:'01', 2:'02', 3:'03', 4:'04', 5:'05', 6:'06',
              7:'07', 8:'08', 9:'09', 10:'10', 11:'11', 12:'12'}

# parse command line arguments
parser = ap.ArgumentParser(description=doc)
group = parser.add_mutually_exclusive_group()
parser.add_argument('files', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-c', dest='utc85col', type=int, default=1, 
    help='column of utc85 variable (0,1..) [default: 1]')
group.add_argument('-m', dest='months', action='store_const', const=True,
    default=False, help='separate by months [default: by years]')
group.add_argument('-s', dest='seasons', action='store_const', const=True,
    default=False, help='separate by seasons [default: by years]')
group.add_argument('-i', dest='intervals', action='store_const', const=True,
    default=False, help='separate by time intervals [default: by years]')
group.add_argument('-w', dest='window', nargs=2, type=int, metavar=('MON1', 'MON2'),
    help='separate by month-window (MON=1,2..12, increasing order)')
parser.add_argument('-d', dest='usedir', action='store_const', const=True, \
    default=False, help='create directories for output files [default: no]')
parser.add_argument('-p', dest='N', type=int, default=0, 
    help='print on the screen N dates of each file and exit')
args = parser.parse_args()


def main(args):
    files = args.files
    months = args.months
    seasons = args.seasons
    intervals = args.intervals
    window = args.window
    usedir = args.usedir
    N = args.N

    print 'files to read:', len(files)
    print 'create directories:', usedir

    if months:
        print 'separate by: months'
    elif seasons:
        print 'separate by: seasons'
    elif intervals:
        print 'separate by: intervals'
    elif window is not None:
        print 'separate by: window', window
    else:
        print 'separate by: years'

    print 'reading and processing files ...'

    # input
    #-----------------------------------------------------------------

    ndirs = 0
    nfiles = 0
    for fname in files:

        f = tb.openFile(fname, 'r')
        try:
            secs = f.root.idr.cols.utc85[:]
            since_year = 1985
        except:
            secs = f.root.idr.cols.utc00[:]
            since_year = 2000

        if secs.shape[0] < 1: continue

        # processing
        #-------------------------------------------------------------

        dt = SecsToDateTime(secs, since_year=since_year)
        if months:
            res = get_months(dt.years(), dt.months())
        elif seasons:
            res = get_seasons(dt.years(), dt.months())
        elif intervals:
            res = get_intervals(dt.datenum(), T1, T2)
        elif window is not None:
            m1, m2 = window
            res = get_window(dt.years(), dt.months(), m1, m2)
        else:
            res = get_years(dt.years())

        # output -> dir or file
        #-------------------------------------------------------------

        path, file_name = os.path.split(fname)
        fname2, ext = os.path.splitext(file_name)

        # one output per result
        for r in res:
            ind, year, month = r[:]

            if ind.shape[0] < 1: 
                continue

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
                file_name = ''.join([fname2, SEP, str(year), month_name, ext])
                fname_out = os.path.join(path, file_name)

            # tables to save
            try:
                table1_out = f.root.idr[ind]
            except:
                table1_out = f.root.gla[ind]
            table2_out = f.root.mask[ind]

            save_tbl(fname_out, table1_out)
            save_tbl(fname_out, table2_out)
            nfiles += 1

        for fid in tb.file._open_files.values():
            fid.close() 

    print 'done!'
    if dir is True:
        print 'directories created:', ndirs
    print 'files created:', nfiles


if __name__ == '__main__':
    main(args)
