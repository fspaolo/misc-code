#!/usr/bin/env python
doc = """\
Given time (in secs) separate files by time intervals (windows). 
""" 
"""
Fernando Paolo <fpaolo@ucsd.edu>
October 20, 2012
"""

import os
os.environ['MPLCONFIGDIR'] = '/var/tmp'    # to ensure a writable dir when 
                                           # accessing Xgrid as "nobody"
import sys
import numpy as np
import tables as tb
import argparse as ap
import datetime as dt

from funcs import *

# ICESat campaigns:
# 2a,2b,2c,3a,3b,3c,3d,3e,3f,3g,3g,3i,3j,3k,2d,2e,3f
T1 = [(2003,10,4), (2004,2,17), (2004,5,18), (2004,10,3), 
      (2005,2,17), (2005,5,20), (2005,10,21), (2006,2,22),
      (2006,5,24), (2006,10,25), (2007,3,12), (2007,10,2), 
      (2008,2,17), (2008,10,4), (2008,11,25), (2009,3,9), 
      (2009,9,30)]
# one extra day added so it ends at midnight
T2 = [(2003,11,20), (2004,3,22), (2004,6,22), (2004,11,9),
      (2005,3,25), (2005,6,24), (2005,11,25), (2006,3,29),
      (2006,6,27), (2006,11,28), (2007,4,15), (2007,11,6),
      (2008,3,22), (2008,10,20), (2008,12,18), (2009,4,12), 
      (2009,10,12)]

SEP = '_'  # symbol to separate the suffix, e.g. '_', '-'

# parse command line arguments
parser = ap.ArgumentParser(description=doc)
group = parser.add_mutually_exclusive_group()
parser.add_argument('files', nargs='+', help='HDF5 file(s) to read')
parser.add_argument('-t', dest='timecol', type=int, default=1, 
    help='column of time variable in seconds (0,1..), default 1')
parser.add_argument('-y', dest='refyear', type=int, default=1985, 
    help='reference year for seconds since <refyear>-Jan-1, default 1985')
group.add_argument('-s', dest='seasons', action='store_const', const=True,
    default=False, help='separate by seasons, default by years')
group.add_argument('-i', dest='intervals', action='store_const', const=True,
    default=False, help='separate by time intervals, default by years')
group.add_argument('-w', dest='windows', action='store_const', const=True,
    default=False, help='separate by moving windows, default by years')
parser.add_argument('-d', dest='usedir', action='store_const', const=True, \
    default=False, help='create directories for output files, default no')
parser.add_argument('-p', dest='N', type=int, default=0, 
    help='print on the screen N dates of each file and exit')
parser.add_argument('-r', dest='range', nargs=2, default=(0, 0), 
    help='the `t_beg` and `t_end` of dataset (in secs), default 0 0')
args = parser.parse_args()


def main(args):
    files = args.files
    timecol = args.timecol
    since_year = args.refyear
    seasons = args.seasons
    intervals = args.intervals
    mov_windows = args.windows
    usedir = args.usedir
    print_n = args.N
    tmin, tmax = args.range

    print 'files to read:', len(files)
    print 'create directories:', usedir

    if seasons:
        print 'separate by: seasons'
    elif intervals:
        print 'separate by: intervals'
    elif mov_windows:
        print 'separate by: windows'
    else:
        ValueError('not a valid separation option.')

    print 'reading and processing files ...'

    # input
    #-----------------------------------------------------------------

    if intervals:
        windows = [(dt.datetime(t1[0], t1[1], t1[2]), \
                   dt.datetime(t2[0], t2[1], t2[2]))  \
                   for t1, t2 in zip(T1, T2)]

    if mov_windows:
        #tmin, tmax = get_times(files, timecol, since_year)   # dt1, dt2 of database
        tmin, tmax = sec2dt([tmin, tmax], since_year)
        tmin = dt.datetime(tmin.year, tmin.month, 1)   # always start at the beg
        windows = define_windows(tmin, tmax, n=3)      # windows between t1 and t2

    ndirs = 0
    nfiles = 0
    for fname in files:

        f = tb.openFile(fname, 'r')
        data = f.getNode('/data')
        secs = data[:,timecol]
        dtimes = sec2dt(secs, since_year=since_year)

        if len(dtimes) < 1: continue

        # processing
        #-------------------------------------------------------------

        if print_n != 0:
            print_dates(dtimes, print_n)
            sys.exit()
        if intervals or mov_windows:
            results = get_windows(dtimes, windows)

        # output -> dir or file
        #-------------------------------------------------------------

        if len(results) < 1: 
            continue

        path, fname2 = os.path.split(fname)
        fname3, ext = os.path.splitext(fname2)

        # one output per result
        for r in results:
            ind, dtime = r
            strtime = dtime.strftime('%Y%m%d')

            # dir
            if usedir:
                outdir = os.path.join(path, strtime)
                if not os.path.exists(outdir):    # if dir doesn't exist
                    os.mkdir(outdir)              # create one
                    ndirs += 1
                fname_out = os.path.join(outdir, fname2)
            # file
            else:
                fname4 = ''.join([fname3, SEP, strtime, ext])
                fname_out = os.path.join(path, fname4)

            # save
            save_arr(fname_out, data[ind,:])
            nfiles += 1

            #---------------------------------------------------------

    for fid in tb.file._open_files.values():
        fid.close() 

    print 'done!'
    if dir is True:
        print 'directories created:', ndirs
    print 'files created:', nfiles


if __name__ == '__main__':
    main(args)
