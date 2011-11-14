#!/usr/bin/env python
doc = """\
Separate satellite tracks in asc/des using `latitude` and `time`.

Two output options:

1) Add extra column with flags: 0=ascending, 1=descending [default]
2) Separated tracks in different output files: file_asc.ext, file_des.ext
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
"""

import os
import sys
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter, description=doc)
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to read')
parser.add_argument('-y', dest='latcol', default=2, type=int,
                    help='column of latitude in the file (0,1,..) [default: 2]')  
parser.add_argument('-t', dest='timecol', default=1, type=int,
                    help='column of time in the file (0,1,..) [default: 1]')  
parser.add_argument('-l', dest='timelag', default=10, type=int,
                    help='max time lag between track points [default: 10 sec]')  
parser.add_argument('-f', dest='trackfiles', default=False, 
                    action='store_const', const=True, 
                    help='separate tracks in files [default: add column with flags]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
                    const=True, help='reads and writes ASCII files [default: HDF5]')
parser.add_argument('-m', dest='inmemory', default=False, action='store_const',
                    const=True, help='load data in-memory (faster) ' 
                    '[default: out-of-memory]')

args = parser.parse_args()
LATCOL = args.latcol
TIMECOL = args.timecol 
TIMELAG = args.timelag  # in seconds

try:
    import _tracksep as C       # C function for speed up!
    cmodule = True
    print 'C module imported!'
except:
    cmodule = False
    print "couln't import C module!"
    print 'using pure python instead'


def tracksep_indices(data, LATCOL, inmemory):
    """Find the indices in `data` for ascending/descending tracks.
    """
    N = data.shape[0]
    i_asc = np.zeros(N, np.bool_)                           # data in-memory
    i_des = np.zeros(N, np.bool_)                           # all false

    if cmodule and inmemory:                                # C function
        C.tracksep_indices(data, LATCOL, i_asc, i_des)
    else:                                                   # Py function
        i_beg = 0
        for i_end in xrange(N-1):

            if (data[i_end+1,TIMECOL] - data[i_end,TIMECOL]) > TIMELAG or i_end == N-2:  
                # if break in time or last track

                i_min = np.argmin(data[i_beg:i_end+1,LATCOL])
                i_min = np.unravel_index(i_min, data[i_beg:i_end+1,LATCOL].shape)[0]
                i_min += i_beg

                if i_beg != i_min and i_min != i_end:       # asc + des
                    if data[i_beg,LATCOL] < data[i_min,LATCOL]:  
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
    return i_asc, i_des


def tracksep_flags(data, LATCOL, inmemory):
    """Compute flags (0|1) in `data` for ascending/descending tracks.
    """
    N = data.shape[0]
    flags = np.empty(N, np.uint8)                           # data in-memory

    i_beg = 0
    for i_end in xrange(N-1):

        if (data[i_end+1,TIMECOL] - data[i_end,TIMECOL]) > TIMELAG or i_end == N-2:  
            # if break in time or last track

            i_min = np.argmin(data[i_beg:i_end+1,LATCOL])
            i_min = np.unravel_index(i_min, data[i_beg:i_end+1,LATCOL].shape)[0]
            i_min += i_beg

            if i_beg != i_min and i_min != i_end:           # asc + des
                if data[i_beg,LATCOL] < data[i_min,LATCOL]:  
                    flags[i_beg:i_min+1] = 0                # first segment asc
                    flags[i_min+1:i_end+1] = 1              # second segment des
                else:                       
                    flags[i_beg:i_min+1] = 1
                    flags[i_min+1:i_end+1] = 0
            elif i_beg == i_min:                            # all asc
                flags[i_beg:i_end+1] = 0 
            elif i_min == i_end:                            # all des
                flags[i_beg:i_end+1] = 1 
            else:
                flags[i_beg:i_end+1] = 2 
            i_beg = i_end + 1

    flags[-1] = flags[-2]                                   # set last element
    return flags


def main():
    
    print 'processing files: %d ...' % len(args.file)
    if args.ascii:
        print 'reading and writing ASCII'
    else:
        print 'reading and writing HDF5'
    if args.trackfiles:
        print "separate tracks in files: '_asc' and '_des'"
    else:
        print 'separate tracks using flags: 0=asc, 1=des'
    if args.inmemory:
        print 'in-memory'
    
    for f in args.file:
        if args.ascii:
            data = np.loadtxt(f)
        else:
            fin = tb.openFile(f)
            if args.inmemory:
                data = fin.root.data.read()    # in-memory
            else:
                data = fin.root.data           # out-of-memory
     
        if data.shape[0] < 3:                  # need at least 3 points
            print f
            print 'no track -> file with less than 3 points'
            continue
        else:
            fname = os.path.splitext(f)[0]
     
        print 'separating tracks ...'  
        #---------------------------------------------------------------
        if args.trackfiles:
            i_asc, i_des = tracksep_indices(data, LATCOL, args.inmemory)
     
            # save files
            if args.ascii:
                np.savetxt(fname + '_asc.txt', data[i_asc,:], fmt='%f')
                np.savetxt(fname + '_des.txt', data[i_des,:], fmt='%f')
            else:
                filters = tb.Filters(complib='blosc', complevel=9)
                atom = tb.Atom.from_dtype(data.dtype)
     
                shape = data[i_asc,:].shape
                fout = tb.openFile(fname + '_asc.h5', 'w')
                dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                         filters=filters)
                dout[:] = data[i_asc,:] 
                fout.close()
     
                shape = data[i_des,:].shape
                fout = tb.openFile(fname + '_des.h5', 'w')
                dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                         filters=filters)
                dout[:] = data[i_des,:] 
                fout.close()
        #---------------------------------------------------------------
        else:
            flags = tracksep_flags(data, LATCOL, args.inmemory)
     
            # save files
            if args.ascii:
                data = np.column_stack((data, flag))  # add last colum with flags
                np.savetxt(fname + '_sep.txt', data, fmt='%f')
            else:
                filters = tb.Filters(complib='blosc', complevel=9)
                atom = tb.Atom.from_dtype(data.dtype)
     
                shape = (data.shape[0], data.shape[1] + 1)
                fout = tb.openFile(fname + '_sep.h5', 'w')
                dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
                                         filters=filters)
                dout[:,:-1] = data[:] 
                dout[:,-1] = flags[:] 
                fout.close()
        #---------------------------------------------------------------
        
        if not args.ascii and fin.isopen:
            fin.close()
    
    print 'done.'
    if args.trackfiles:
        if args.ascii:
            print 'output ext: *_asc.txt, *_des.txt'
        else:
            print 'output ext: *_asc.h5, *_des.h5'
    else:
        print 'column w/flags (last one) added to original data:'
        print '0 = ascending'
        print '1 = descending'
        if args.ascii:
            print 'output ext: *_sep.txt'
        else:
            print 'output ext: *_sep.h5'


if __name__ == '__main__':
    status = main()
    sys.exit(status)
