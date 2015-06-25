"""
Apply local SL/IB to 2d fields (dh/dt) at grid-cell level.

"""

import numpy as np
import tables as tb
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.basemap import interp
from astropy.convolution import convolve, Gaussian2DKernel

FILE_DHDT = '/Users/fpaolo/data/shelves/h_trendfit.h5.byfirst3_'
FILE_IB = '/Users/fpaolo/data/ibe/DPDT_data.mat'
FILE_SL = '/Users/fpaolo/data/sealevel/aviso/MSL_Map_MERGED_Global_IB_RWT_NoGIA_Adjust.nc' 


# read dh/dt
fin = tb.open_file(FILE_DHDT, 'a')
lon = fin.root.lon[:]
lat = fin.root.lat[:]
poly_lasso_rate = fin.root.poly_lasso_rate[:]  # m/yr
line_lstsq_rate = fin.root.line_lstsq_rate[:]

xx, yy = np.meshgrid(lon, lat)

if 1:  # plot the before
    plt.figure()
    x = poly_lasso_rate.copy()
    plt.imshow(x, origin='lower', interpolation='nearest', vmin=-.1, vmax=.1)

# IB correction (inverse-barometer trend)
#-----------------------------------------------------------------

# read dP/dt
dic = loadmat(FILE_IB, squeeze_me=True, chars_as_strings=True)
lon_ib = dic['LON']
lat_ib = dic['LAT'][::-1]
ib_rate = np.flipud(dic['DPDT'][0][0]) * 1e-2  # mbar/yr (= -cm/yr) -> -m/yr
ib_rate *= -1  # atmospheric pressure -> inverse barometer

# interpolate
ib_rate_ = interp(ib_rate, lon_ib, lat_ib, xx, yy, order=1) # linear

# correct -> SUBTRACT IB
poly_lasso_rate -= ib_rate_
line_lstsq_rate -= ib_rate_

# SL correction (regional sea-level trend)
#-----------------------------------------------------------------

# read SL
if 0:
    # U. Col. 
    ny = 716
    nx = 1440
    x0 = 0
    x1 = 360
    y0 = -89.5
    y1 = 89.5
    null = 999.
    sl_rate = np.loadtxt('cu/sl.txt')
    sl_rate = np.flipud(sl_rate.ravel().reshape(ny, nx))
    sl_rate[sl_rate==null] = np.nan
    sl_rate *= 1e-3  # mm/yr -> m/yr
    lon_sl = np.linspace(x0, x1, nx)
    lat_sl = np.linspace(y0, y1, ny)
else:
    # AVISO
    f = nc.Dataset(FILE_SL)
    sl_rate = f.variables['sea_level_trends'][:] * 1e-3  # masked array, mm/yr -> m/yr
    lon_sl = f.variables['longitude'][:]
    lat_sl = f.variables['latitude'][:]
    f.close()

# interpolate
sl_rate_ = interp(sl_rate, lon_sl, lat_sl, xx, yy, order=1) # lin
sl_rate2 = interp(sl_rate, lon_sl, lat_sl, xx, yy, order=0) # nn
sl_rate_[np.isnan(sl_rate_)] = sl_rate2[np.isnan(sl_rate_)]

if 1: # propagate
    sl_rate_ = pd.DataFrame(sl_rate_, index=lat, columns=lon)
    sl_rate_ = sl_rate_.bfill().values

if 1: # smooth
    gauss_kernel = Gaussian2DKernel(2)  # std of 2 pixel
    sl_rate_ = convolve(sl_rate_, gauss_kernel, boundary='extend', normalize_kernel=True)

# correct -> SUBTRACT SL
poly_lasso_rate -= sl_rate_ 
line_lstsq_rate -= sl_rate_

#---------------------------------------------------------------------

if 1:  # plot the after
    plt.figure()
    plt.imshow(poly_lasso_rate, origin='lower', interpolation='nearest', vmin=-.1, vmax=.1)
    plt.show()

# save data
if 1:
    fin.create_array('/', 'poly_lasso_rate_ib_sl', poly_lasso_rate)
    fin.create_array('/', 'line_lstsq_rate_ib_sl', line_lstsq_rate)
    fin.flush()
    print 'file out ->', FILE_DHDT

fin.close()
