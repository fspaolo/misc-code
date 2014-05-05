#!/usr/bin/env python
doc = """\
Separate satellite tracks in asc/des using latitude and time.

Two output options:

1) Add extra column with flags: 0=asc/1=des [default]
2) Separate tracks in different files: filein_a.ext, filein_d.ext
"""
"""
Example
-------
To separate several track files in asc and des files (in-memory):

    $ python tracksep.py -m -f file1.h5 file2.h5 file3.h5

Note
----
Works for Southern Emisphere. For NH a small modification is needed!

Fernando Paolo <fpaolo@ucsd.edu>
November 1, 2010

Adapted to operate with Tables on Aug 22, 2012
"""

import os
import sys
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter, description=doc)
parser.add_argument('files', nargs='+', help='HDF5 file(s) to read')
parser.add_argument('-y', dest='latcol', default=2, type=int,
                    help='column of latitude in the file (0,1,..), default 2')  
parser.add_argument('-t', dest='timecol', default=1, type=int,
                    help='column of time in the file (0,1,..), default 1')  
parser.add_argument('-l', dest='timelag', default=10, type=int,
                    help='max time lag between track points, default 10 sec')  
parser.add_argument('-f', dest='trkfiles', default=False, 
                    action='store_const', const=True, 
                    help='separate tracks in files, default add column with flags')
args = parser.parse_args()


try:
    import _tracksep as mod       # C function for speed up!
    cmodule = True
    print 'C module imported!'
except:
    cmodule = False
    print "couln't import C module!"
    print 'using pure python instead'


def tracksep_indices(lat, time, timelag=10):
    """
    Finds the indices for asc/des tracks using latitude and time.
    """
    N = lat.shape[0]
    i_asc = np.zeros(N, np.bool_)                           # data in-memory
    i_des = np.zeros(N, np.bool_)                           # all false

    if cmodule:                                             # C function
        mod.tracksep_indices(lat, time, i_asc, i_des)
    else:                                                   # Py function
        i_beg = 0
        for i_end in xrange(N-1):

            # if break in time or last track
            if (time[i_end+1] - time[i_end]) > timelag or (i_end == N-2):

                i_min = np.argmin(lat[i_beg:i_end+1])
                i_min += i_beg

                if (i_beg != i_min) and (i_min != i_end):   # asc + des
                    if lat[i_beg] < lat[i_min]:  
                        i_asc[i_beg:i_min+1] = True         # first segment asc
                        i_des[i_min+1:i_end+1] = True       # second segment des
                    else:                       
                        i_des[i_beg:i_min+1] = True
                        i_asc[i_min+1:i_end+1] = True 
                elif i_beg == i_min:                        # all asc
                    i_asc[i_beg:i_end+1] = True 
                elif i_min == i_end:                        # all des
                    i_des[i_beg:i_end+1] = True 
                else:
                    pass                                    # all remains false
                i_beg = i_end + 1

        i_asc[-1] = i_asc[-2]                               # set last element
        i_des[-1] = i_des[-2]
    i_asc, = np.where(i_asc == True)                        # bool -> indices 
    i_des, = np.where(i_des == True)
    return i_asc, i_des


def tracksep_flags(lat, time, timelag=10):
    """
    Compute flags (0/1) for asc/des tracks using latitude and time.
    """
    N = lat.shape[0]
    flags = np.empty(N, np.uint8)                           # data in-memory

    if cmodule:                                             # C function
        mod.tracksep_flags(lat, time, flags)
    else:                                                   # Py function
        i_beg = 0
        for i_end in xrange(N-1):

            # if break in time or last track
            if (time[i_end+1] - time[i_end]) > timelag or (i_end == N-2):

                i_min = np.argmin(lat[i_beg:i_end+1])
                i_min += i_beg

                if (i_beg != i_min) and (i_min != i_end):   # asc + des
                    if lat[i_beg] < lat[i_min]:  
                        flags[i_beg:i_min+1] = 0            # first segment asc
                        flags[i_min+1:i_end+1] = 1          # second segment des
                    else:                       
                        flags[i_beg:i_min+1] = 1
                        flags[i_min+1:i_end+1] = 0
                elif i_beg == i_min:                        # all asc
                    flags[i_beg:i_end+1] = 0 
                elif i_min == i_end:                        # all des
                    flags[i_beg:i_end+1] = 1 
                else:
                    flags[i_beg:i_end+1] = 2 
                i_beg = i_end + 1

        flags[-1] = flags[-2]                               # set last element
    return flags


def add_arr_as_tbl(fid, tname, cols):
    """
    Given 1D arrays add a Table to an existing file.

    fid : file object in append mode ('a').
    tname : name of new table.
    cols : a dictionary containing 'col_names' and 'col_vals'.
    """
    # Create column description
    descr = {}
    for i, (cname, cval) in enumerate(cols.items()):
        descr[cname] = tb.Col.from_dtype(cval.dtype, pos=i)
    table = fid.createTable('/', tname, descr, "", tb.Filters(9))
    table.append([v for k, v in cols.items()])
    table.flush()
    print "file with new table:", fid


def main(args):
    files = args.files
    trkfiles = args.trkfiles
    latcol = args.latcol
    timecol = args.timecol 
    timelag = args.timelag    # in seconds

    print 'processing files: %d ...' % len(files)
    if trkfiles:
        print "separate tracks in files: *_a.ext and *_d.ext"
    else:
        print 'separate tracks using flags: 0=asc, 1=des'
    
    for f in files:
        # input
        #-------------------------------------------------------------

        fin = tb.openFile(f)
        data = fin.getNode('/data')  # on-disk
        lat = data[:,latcol]         # in-memory
        time = data[:,timecol]

        #-------------------------------------------------------------

        if lat.shape[0] < 3:    # need at least 3 points
            print f
            print 'no track -> the file has less than 3 data points'
            continue

        print 'separating tracks ...'  

        if trkfiles:
            i_asc, i_des = tracksep_indices(lat, time, timelag=timelag)
        else:
            flags = tracksep_flags(lat, time, timelag=timelag)

        # output
        #-------------------------------------------------------------
     
        fname = os.path.splitext(f)[0]
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        if trkfiles:
            # create two output files
            shape = data[i_asc,:].shape
            fout = tb.openFile(fname + '_a.h5', 'w')
            dout = fout.createCArray('/','data', atom=atom, shape=shape, filters=filters)
            dout[:] = data[i_asc,:] 
     
            shape = data[i_des,:].shape
            fout = tb.openFile(fname + '_d.h5', 'w')
            dout = fout.createCArray('/','data', atom=atom, shape=shape, filters=filters)
            dout[:] = data[i_des,:] 
        else:
            shape = (data.shape[0], data.shape[1]+1)
            fout = tb.openFile(fname + '_track.h5', 'w')
            dout = fout.createCArray('/','data', atom=atom, shape=shape, filters=filters)
            dout[:,:-1] = data[:]
            dout[:,-1] = flags

        for fid in tb.file._open_files.values():
            fid.close() 

        #---------------------------------------------------------------
        
    print 'done.'
    if trkfiles:
        print 'output ext: *_a.h5 and *_d.h5'
    else:
        print 'column w/flags added to original data file:'
        print '0 = ascending'
        print '1 = descending'


if __name__ == '__main__':
    status = main(args)
    sys.exit(status)
