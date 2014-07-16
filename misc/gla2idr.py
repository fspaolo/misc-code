#!/usr/bin/env python

import os
import sys
import numpy as np
import tables as tb
import pyhdf.SD as sd
from glob import glob

#files = sys.argv[1:]
files = glob('/data/alt/tmp/*.HDF')

# H_ell += Ocean_tide + Load_tide

REFSDN = 730486.    # this value is for midnight (00:00:00)

class GLA(tb.IsDescription):
    orbit = tb.Int32Col(pos=0)
    secs00 = tb.Float64Col(pos=1)
    lat = tb.Float64Col(pos=2)
    lon = tb.Float64Col(pos=3)
    elev = tb.Float64Col(pos=4)
    agc = tb.Float64Col(pos=5)
    txenergy = tb.Float64Col(pos=6)
    energy = tb.Float64Col(pos=6)
    reflect = tb.Float64Col(pos=7)


def to_table(fname):
    fin = tb.openFile(fname, 'a')
    fout = tb.openFile(os.path.splitext(fname)[0] + '_tb.h5', 'w')
    filters = tb.Filters(complib='zlib', complevel=9)
    t = fout.createTable('/', 'gla', GLA, '', filters)

    orbit = fin.root.gla.cols.Orbitnr[:]
    secs00 = fin.root.gla.cols.Time_num[:]
    lat = fin.root.gla.cols.Latitude[:]
    lon = fin.root.gla.cols.Longitude[:]
    elev = fin.root.gla.cols.H_ell[:]
    agc = fin.root.gla.cols.Gain[:]
    txenergy = fin.root.gla.cols.Tx_energy[:]
    energy = fin.root.gla.cols.Energy[:]
    reflect = fin.root.gla.cols.Reflect[:]
    Load_tide = fin.root.gla.cols.Load_tide[:]
    Ocean_tide = fin.root.gla.cols.Ocean_tide[:]

    secs00 = (secs00 - REFSDN)*24*60*60.  # from SDN to secs since 2000
    lon[lon<0] += 360.			  # from -180/180 to 0/360
    elev = elev + Ocean_tide + Load_tide  # to detide

    t.append((orbit, secs00, lat, lon, elev, agc, txenergy, energy, reflect))
    t.flush()


def to_array(fname):
    # from HDF4 to HDF5 files
    fin = sd.SD(fname)  # HDF4 files
    d = fin.datasets()

    #orbit = fin.select('id').get() 
    secs00 = fin.select('time').get() 
    lat = fin.select('lat').get() 
    lon = fin.select('lon').get() 
    elev = fin.select('elev').get() 
    agc = fin.select('gain').get() 
    energy = fin.select('wave_energy').get()
    reflect = fin.select('reflect').get()
    icesvar = fin.select('icesvar').get()
    #pressure = fin.select('pressure').get()
    #sat = fin.select('sat_cor_product').get()

    nrow, ncol = lon.shape[0], 9
    orbit = np.empty(nrow, 'f8')
    revNo = int(os.path.split(fname)[-1].split('_')[2])
    orbit.fill(revNo)

    secs00 += 43200     # move time 12 hours back (starting midnight) 
    lon[lon<0] += 360.  # from -180/180 to 0/360

    fout = tb.openFile(os.path.splitext(fname)[0] + '.h5', 'w')
    atom = tb.Float64Atom()
    shape = (nrow, ncol)
    filters = tb.Filters(complib='blosc', complevel=9)
    c = fout.createCArray('/', 'data', atom=atom, shape=shape, filters=filters)

    c[:,0] = orbit
    c[:,1] = secs00
    c[:,2] = lat
    c[:,3] = lon
    c[:,4] = elev
    c[:,5] = agc
    c[:,6] = energy
    c[:,7] = reflect
    c[:,8] = icesvar
    fout.close()


print 'converting %d files ...' % len(files)
for fname in files:
    #to_table(fname)
    to_array(fname)
    for fid in tb.file._open_files.values():
        fid.close() 
print 'done.'
