#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Interpolate using Splines.

Uses the GP pacakge from 'sklearn' to interpolate spatial data points (2d).

"""

import sys
import numpy as np
import tables as tb
from matplotlib import pyplot as pl

import altimpy as ap

np.random.seed(1)

try:
    fname = sys.argv[1]
except:
    fname = ''

# get fields to interpolate
f = tb.openFile(fname, 'a')
elev = f.root.dh_mean_all[:]
gain = f.root.dg_mean_all[:]
lon = f.root.lon[:]
lat = f.root.lat[:]

# generate 2d mask from 3d array, with coords
mask = ap.get_mask(elev)
lon2d, lat2d = np.meshgrid(lon, lat)

# filter out outliers (reduce variability) prior interpolating. Cut-off 3 std 
# for each time step independently
print 'range before:', np.nanmin(elev), np.nanmax(elev)
elev = ap.filter_std(elev, n=3, per_field=True)
gain = ap.filter_std(gain, n=3, per_field=True)
print 'range after:', np.nanmin(elev), np.nanmax(elev)

# containers for interpolated fields
elev_interp = np.empty_like(elev)
gain_interp = np.empty_like(gain)
elev_interp_err = np.empty_like(elev)
gain_interp_err = np.empty_like(gain)

# interpolate each (time-step) field at a time
krig = ap.Kriging2d()

for k, field in enumerate(elev):
    field, error = krig.interpolate(field, mask, lon2d, lat2d)
    elev_interp[k], elev_interp_err[k] = field, error
    '''
    pl.figure()
    pl.imshow(field, origin='lower', interpolation='nearest')
    pl.figure()
    pl.imshow(error, origin='lower', interpolation='nearest')
    pl.show()
    sys.exit()
    '''

for k, field in enumerate(gain):
    field, error = krig.interpolate(field, mask, lon2d, lat2d)
    gain_interp[k], gain_interp_err[k] = field, error

# save interpolated fields
f.createArray('/', 'dh_mean_all_interp', elev_interp)
f.createArray('/', 'dg_mean_all_interp', gain_interp)
f.createArray('/', 'dh_mean_all_interp_err', elev_interp_err)
f.createArray('/', 'dg_mean_all_interp_err', gain_interp_err)

f.close()
