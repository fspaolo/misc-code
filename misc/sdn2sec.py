#!/usr/bin/env python

import os
import sys
import tables as tb

# Matlab Serial Date Number is frac days since year 0000
# convert from SDN to secs since 1-Jan-2000 00:00:00
# 730486.50 is t_matlab for '1-Jan-2000 12:00:00' (the zero time for ICESat)

SDNREF = 730486.

files = sys.argv[1:]
print 'converting %d files ...' % len(files)
for fname in files:
    f = tb.openFile(fname, 'a')
    time = f.root.idr.cols.utc00    # <<<<<<<<<<<<<<< EDIT 
    sdn = time[:] 
    secs = (sdn - SDNREF)*24*60*60  
    time[:] = secs[:]
    f.flush()
    for fid in tb.file._open_files.values():
        fid.close() 
print 'done.'
