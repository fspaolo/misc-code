import sys
import tables as tb
import argparse as ap
import datetime as dt

from funcs import *

parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 file(s) to read')
parser.add_argument('-t', dest='timecol', type=int, default=1, 
    help='column of time variable in seconds (0,1..), default 1')
parser.add_argument('-y', dest='refyear', type=int, default=1985, 
    help='reference year for seconds since <refyear>-Jan-1, default 1985')
args = parser.parse_args()

files = args.files
timecol = args.timecol
since_year = args.refyear


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
    print 'dataset time range (secs):', tmin, tmax
    return [tmin, tmax]

tmin, tmax = get_times(files, timecol, since_year)
