#!/usr/bin/env python
# Fernando Paolo <fpaolo@ucsd.edu>
# Augut 9, 2012

import os
import sys
import tables as tb
from scipy.io import loadmat

class GLA(tb.IsDescription):
    Time = tb.Float64Col(pos=1)
    Longitude = tb.Float64Col(pos=2)
    Latitude = tb.Float64Col(pos=3)
    East = tb.Float64Col(pos=4)
    North = tb.Float64Col(pos=5)
    H_ell = tb.Float64Col(pos=6)
    Geoid = tb.Float64Col(pos=7)
    Gain = tb.Float64Col(pos=8)
    Tx_energy = tb.Float64Col(pos=9)
    Energy = tb.Float64Col(pos=10)
    Reflect = tb.Float64Col(pos=11)
    Id_nr = tb.Float64Col(pos=12)
    Orbitnr = tb.Float64Col(pos=13)
    Stripnr = tb.Float64Col(pos=14)
    Reftracknr = tb.Float64Col(pos=15)
    H_orto = tb.Float64Col(pos=16)
    H_dem = tb.Float64Col(pos=17)
    Region = tb.Float64Col(pos=18)
    Time_num = tb.Float64Col(pos=19)
    Pressure = tb.Float64Col(pos=20)
    Icesvar = tb.Float64Col(pos=21)
    Satcorr = tb.Float64Col(pos=22)
    Satco_prod = tb.Float64Col(pos=23)
    Load_tide = tb.Float64Col(pos=24)
    Ocean_tide = tb.Float64Col(pos=25)


files = sys.argv[1:]
print 'converting %d .mat files ...' % len(files)
for fname in files:
    gla = loadmat(fname)['GLA']  # 2D array
    fout = tb.openFile(os.path.splitext(fname)[0] + '.h5', 'w')
    filters = tb.Filters(complib='zlib', complevel=9)
    t = fout.createTable('/', 'gla', GLA, '', filters)
    t.append(gla)
    t.flush()
    fout.close()
print 'done.'
