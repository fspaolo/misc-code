import sys
import numpy as np
import tables as tb
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from numba import jit

import altimpy as ap

FILE_FIRN = '/Users/fpaolo/data/firn/FM_ANT3K27_1979-2011.nc'
FILE_ALTIM = '/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
FILE_OUT = '/Users/fpaolo/data/shelves/firn_ice_shelves.h5'


def as_frame(data, z, y, x):
    """3d Array -> Data Frame."""
    try:
        return pd.Panel(data, items=z, major_axis=y, minor_axis=x
                        ).to_frame(filter_observations=False).T
    except:
        raise
        print 'already a DataFrame'
        return data


def as_array(data):
    """Data Frame -> 3d Array."""
    try:
        return data.T.to_panel().values
    except:
        print 'already an Array'
        return data


# array_to_array one ts at a time so computer doesn't freeze!
@jit
def array_to_array(arr_a, arr_b, i_a, j_a, i_b, j_b):
    n = 1
    for ia, ja, ib, jb in zip(i_a, j_a, i_b, j_b):
        arr_a[:, ia, ja] = arr_b[:, ib, jb]
        print 'time series #', n 
        n += 1
    return arr_a


# read firn
print 'reading firn...'
f1 = nc.Dataset(FILE_FIRN)
d = f1.variables
h_firn = d['zs'] #[:]
t_firn = d['time'][:]   # years
lon_firn = d['lon'][:]  # 2d
lat_firn = d['lat'][:]  # 2d

# read altim
print 'reading altim...'
f2 = tb.openFile(FILE_ALTIM, 'a')
h_altim = f2.root.dh_mean_mixed_const_xcal[:-1,...]  # minus last to agree with firn ts
t_altim = ap.num2year(f2.root.time_xcal[:-1])  # years
lat_altim = f2.root.lat[:]      # 1d
lon_altim = f2.root.lon[:]      # 1d
lon_altim = ap.lon_180_360(lon_altim, inverse=True)

# get lon/lat only of complete (no gaps) altim ts
h_altim = h_altim.sum(axis=0)                    # 3d -> 2d
i_altim, j_altim = np.where(~np.isnan(h_altim))  # 2d indices
lonlat_altim = np.column_stack((lon_altim[j_altim], lat_altim[i_altim]))

# find nearest lon/lat in the firn model
i_firn, j_firn = ap.find_nearest2(lon_firn, lat_firn, lonlat_altim)

# find nearest altim times in the firn times
k_firn, = ap.find_nearest(t_firn, t_altim)

# new firn grid => same altim resolution with original firn time
nt, ny, nx = h_firn.shape[0], h_altim.shape[0], h_altim.shape[1]
h_firn_new = np.full((nt, ny, nx), np.nan, dtype='f8')

# space interpolation (out-of-core)
#h_firn_new[:, i_altim, j_altim] = h_firn[:, i_firn, j_firn]
h_firn_new = array_to_array(h_firn_new, h_firn, i_altim, j_altim, i_firn, j_firn)

del h_firn

# 3-month average, and time interpolation
print 'smoothing firn...'
h_firn_new = as_frame(h_firn_new, t_firn, lat_altim, lon_altim)
h_firn_smooth = pd.rolling_mean(h_firn_new, 45, center=True)
h_firn_smooth = h_firn_smooth.iloc[k_firn,:]

# frame -> array
h_firn_new = as_array(h_firn_new)
h_firn_smooth = as_array(h_firn_smooth)

print 'saving...'
f3 = tb.open_file(FILE_OUT, 'w')
f3.create_array('/', 'firn', h_firn_new)
f3.create_array('/', 'firn_smooth', h_firn_smooth)
f3.create_array('/', 'time', t_firn)
f3.create_array('/', 'time_smooth', t_altim)
f3.create_array('/', 'lon', lon_altim)
f3.create_array('/', 'lat', lat_altim)
print 'done.'

f3.flush()
f3.close()
f2.close()
f1.close()
