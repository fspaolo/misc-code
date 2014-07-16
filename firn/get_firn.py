import sys
import numpy as np
import tables as tb
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

import altimpy as ap

# firn densification model
FILE_FDM = 'FM_ANT3K27_1979-2011.nc'

# altimeter database
FILE_ALT = '/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5'

f1 = nc.Dataset(FILE_FDM)
d = f1.variables
h_fdm = d['zs']
t_fdm = d['time'][:]   # years
lon_fdm = d['lon'][:]  # 2d
lat_fdm = d['lat'][:]  # 2d

f2 = tb.openFile(FILE_ALT, 'a')
h_alt = f2.root.dh_mean_corr_short_all[:]
t_alt = ap.num2year(f2.root.time_all[:])
lat_alt = f2.root.lat[:]  # 1d
lon_alt = f2.root.lon[:]  # 1d
lon_alt = ap.lon_180_360(lon_alt, inverse=True)

# get lon/lat of non-null altim ts
h_alt[np.isnan(h_alt)] = 0
h_alt = h_alt.sum(axis=0)            # 3d -> 2d
i_alt, j_alt = np.where(h_alt != 0)  # 2d indices
lonlat_alt = np.column_stack((lon_alt[j_alt], lat_alt[i_alt]))

# find nearest points in the firn model
i_fdm, j_fdm = ap.find_nearest2(lon_fdm, lat_fdm, lonlat_alt)
(k1,k2), = ap.find_nearest(t_fdm, [t_alt[0], t_alt[-1]])

nt = k2 - k1 + 1
ny, nx = h_alt.shape

h_fdm_new = f2.createCArray('/', 'h_firn', shape=(nt,ny,nx), 
                            atom=tb.atom.Float64Atom(dflt=np.nan)) 
t_fdm_new = f2.createCArray('/', 'time_firn', shape=(nt,), 
                            atom=tb.atom.Int32Atom(dflt=0))

# save one ts at a time (memory efficient!)
n = 1
for i1, j1, i2, j2 in zip(i_alt, j_alt, i_fdm, j_fdm):
    h_fdm_new[:, i1, j1] = h_fdm[k1:k2+1, i2, j2]
    print 'time series #', n 
    n += 1

t_fdm_new[:] = ap.year2num(t_fdm[k1:k2+1])  # chose time conversion !!!

f2.flush()
f2.close()
f1.close()
