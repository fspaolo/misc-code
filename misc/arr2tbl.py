#!/usr/bin/env python
# Fernando Paolo <fpaolo@ucsd.edu>
# Augut 9, 2012

import os
import sys
import numpy as np
import tables as tb

class IDR(tb.IsDescription):
    orbit = tb.Int32Col(pos=1)
    secs85 = tb.Float64Col(pos=2)
    lat = tb.Float64Col(pos=3)
    lon = tb.Float64Col(pos=4)
    elev = tb.Float64Col(pos=5)
    agc = tb.Float64Col(pos=6)
    fmode = tb.Int8Col(pos=7)
    fret = tb.Int8Col(pos=8)
    fprob = tb.Int8Col(pos=9)


files = sys.argv[1:]
print 'converting %d files (2D arr -> table) ...' % len(files)
for fname in files:
    fin = tb.openFile(fname)
    data = fin.root.data[:]
    fout = tb.openFile(os.path.splitext(fname)[0] + '_tb.h5', 'w')
    filters = tb.Filters(complib='zlib', complevel=9)
    t = fout.createTable('/', 'idr', IDR, '', filters)
    t.append([data[:,0].astype(np.int32), 
              data[:,1].astype(np.float64), 
              data[:,2].astype(np.float64), 
              data[:,3].astype(np.float64), 
              data[:,4].astype(np.float64), 
              data[:,5].astype(np.float64), 
              data[:,6].astype(np.int8),
              data[:,7].astype(np.int8),
              data[:,8].astype(np.int8),
              ])
    t.flush()
    fout.close()
    fin.close()
print 'done.'
