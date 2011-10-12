#!/usr/bin/env python
doc = """\
 Given utc85 time separate files by: years, seasons, months or month-window.
""" 
"""
 Fernando Paolo <fpaolo@ucsd.edu>
 October 22, 2010
"""

import os
os.environ['MPLCONFIGDIR'] = '/var/tmp'    # to ensure a writable dir when "nobody"

import numpy as np
import tables as tb
import argparse as ap
import datetime as dt
import matplotlib.dates as mpl
import sys

# parse command line arguments
parser = ap.ArgumentParser(description=doc)
group = parser.add_mutually_exclusive_group()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-c', dest='utc85col', type=int, default=1, 
    help='column of utc85 variable (0,1..) [default: 1]')
group.add_argument('-m', dest='month', action='store_const', const=True,
    help='separate by month [default: by year]')
group.add_argument('-w', dest='window', nargs=2, type=int, metavar=('MON1', 'MON2'),
    help='separate by month-window (MON=1,2..12, increasing order)')
group.add_argument('-s', dest='season', action='store_const', const=True,
    help='separate by season [default: by year]')
parser.add_argument('-d', dest='dir', action='store_const', const=True, \
    default=False, help='create directories for output files [default: no]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
    const=True, help='reads and writes ASCII files [default: HDF5]')

args = parser.parse_args()
files = args.file
col = args.utc85col
month = args.month
window = args.window
season = args.season
dir = args.dir
ascii = args.ascii

#mname = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
#         7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
mname = {1:'01', 2:'02', 3:'03', 4:'04', 5:'05', 6:'06',
         7:'07', 8:'08', 9:'09', 10:'10', 11:'11', 12:'12'}

if ascii:
    print 'reading and writing ASCII'
else:
    print 'reading and writing HDF5'

print 'files to read:', len(files)
print 'time (utc85) column:', col
if month is not None:
    print 'separate by: month'
elif window is not None:
    M1, M2 = window
    print 'separate by: month-window', mname[M1], mname[M2]
elif season is not None:
    print 'separate by: season'
else:
    print 'separate by: year'
print 'create directories:', dir


class Utc85:
    """Converts utc85 to convenient date formats: year, month, day... 
    """
    def __init__(self, utc85=0):
        if type(utc85).__name__ != 'ndarray':
            self.utc85 = np.array([utc85])
        else:
            self.utc85 = utc85

        # 1-Jan-1985 00:00:00h in days (since 1-Jan-0001)
        self.DAYS85 = mpl.date2num(dt.date(1985, 1, 1))  

        # utc85/86400 -> frac days -> date
        self.fdays = self.utc85 / 86400.                    #<<<<< need to write
        self.date = mpl.num2date(self.DAYS85 + self.fdays)  #<<<<< this better!

    def getyear(self):
        return np.array([d.year for d in self.date])

    def getmonth(self):
        return np.array([d.month for d in self.date])
        
    def getday(self):
        return np.array([d.day for d in self.date])


def what_season(month):
    """Given a month finds the 3-month block of the season.
    """
    season1 = [12, 1, 2]     # Dec--Feb -> summer SH
    season2 = [3, 4, 5]      # Mar--May -> Autumn SH 
    season3 = [6, 7, 8]      # Jun--Aug -> winter SH
    season4 = [9, 10, 11]    # Sep--Nov -> Spring SH

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
        return None, None, None    # none existent range


def print_date(utc85, N):
    """Just to test the date output.
    """
    for j in utc85.date[:N]:
        print j.year, j.month, j.day, j.hour, j.minute, j.second, j.microsecond
    print


def main():
                                                
    print 'reading and processing files ...'

    ndirs = 0
    nfiles = 0
    for f in files:
        if ascii:
            data = np.loadtxt(f)
        else:
            fin = tb.openFile(f, 'r')
            #data = fin.root.data.read()  # in-memory
            data = fin.root.data          # out-of-memory
     
        if data.shape[0] < 1:
            print 'no data points in the file!'
            continue

        utc85 = Utc85(data[:,col])

        #print f
        #print_date(utc85, 10)         # to test the output
        #continue

        ### convert utc85 column to epochs
        if month is not None or window is not None or season is not None:
            epochs = utc85.getmonth()           # utc85 -> months
        else:
            epochs = utc85.getyear()            # utc85 -> years

        if not np.add.reduce(epochs) > 0:       # if no epochs 
            continue                            # go to next file (!!!)

        while True:

            ### find specific epoch

            if window is not None:
                # the month-window
                IND, = np.where((M1 <= epochs) & (epochs <= M2))
            elif season is not None:
                # first season occurrence
                nonzeros = epochs[epochs!=0]
                EPOCH = nonzeros[0]             # take the first non 0 epoch
                M1, M2, M3 = what_season(EPOCH) # 3-month season
                IND, = np.where((epochs == M1) | (epochs == M2) | (epochs == M3))
                epochs[IND] = 0                 # mark wiht zero all taken epochs
            else:
                # first month or year occurrence
                nonzeros = epochs[epochs!=0]
                EPOCH = nonzeros[0]             
                IND, = np.where(epochs == EPOCH)
                epochs[IND] = 0                

            path, file = os.path.split(f)
            fname, ext = os.path.splitext(file)

            ### output directory or file

            if dir is True:                     # create dir
                if month is not None:
                    dirname = mname[EPOCH]
                elif window is not None:
                    dirname = mname[M1] + mname[M2]
                elif season is not None:
                    dirname = mname[M1] + mname[M3]
                else:
                    dirname = str(EPOCH)
                outdir = os.path.join(path, dirname) 
                if not os.path.exists(outdir):  # if dir doesn't exist
                    os.mkdir(outdir)            # create one
                    ndirs += 1
                outfile = os.path.join(outdir, file) 

            else:                               # no dir
                if month is not None:
                    fname += ('_' + mname[EPOCH] + ext)
                elif window is not None:
                    fname += ('_' + mname[M1] + mname[M2] + ext)
                elif season is not None:
                    fname += ('_' + mname[M1] + mname[M3] + ext)
                else:
                    fname += ('_' + str(EPOCH) + ext)
                outfile = os.path.join(path, fname)

            ### save data

            if IND.shape[0] > 0:
                if ascii:
                    np.savetxt(outfile, data[IND,:], fmt='%f')
                    nfiles += 1
                else:
                    fout = tb.openFile(outfile, 'w')
                    shape = data[IND,:].shape
                    atom = tb.Atom.from_dtype(data.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                             filters=filters)
                    dout[:] = data[IND,:] 
                    fout.close()
                    nfiles += 1

            if window is not None:
                break   
            elif np.add.reduce(epochs) == 0: 
                break   
            else:
                continue

        if not ascii:
            fin.close()

    print 'done!'
    if dir is True:
        print 'directories created:', ndirs
    print 'files created:', nfiles


if __name__ == '__main__':
    main()
