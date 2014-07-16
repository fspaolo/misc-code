"""
Note1: create one single file per post-processed variable, and one associated
XDMF with the same name (see write_xdmf.py).

Note2: for movies of seasonal and interannual h(t) use a different postproc.py

"""
import sys
import numpy as np
import scipy as sp
import pandas as pd
import tables as tb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altimpy as ap
from mpl_toolkits.basemap import interp

import scipy.ndimage as ni

# CHECK AND EDIT THE 'IF 0/1' IN THE CODE
PLOT = False
DIR = '/Users/fpaolo/data/shelves/' 
FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
FILE_OUT = 'h_postproc.h5'
FILE_MSK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'

# for plotting
#LON, LAT = 115.125, -67.125   # Totten, B
#LON, LAT = 260.625, -75.125   # PIG, GL
LON, LAT = 186.375, -79.375  # Ross, with problem at the ends


def step_filt(x, **kw):
    x, m = ap.step_filt(x, **kw)
    '''
    if not np.isnan(m).all():
        plt.plot(m)
        plt.show()
    '''
    return x


def peak_filt(y, x=None, iterative=True):
    if not np.isnan(y).all():
        #plt.plot(y, '--r', linewidth=2)
        y = ap.peak_filt(x, y, iterative=iterative)
        #plt.plot(y, 'b', linewidth=2)
        #plt.show()
    return y


def std_filt(y, x=None, max_std=None):
    """Filter entire vector if the detrended std > max_std."""
    y, std = ap.std_series_filt(x, y, max_std=max_std)
    if std > max_std: print 'ts-std:', std
    return y


def percent_filt(y, min_perc=None):
    y, perc = ap.percent_filt(y, min_perc=min_perc)
    if perc > 0 and perc < min_perc: 
        print 'ts-perc:', perc
    return y


def as_frame(data, z, y, x):
    try:
        return pd.Panel(data, items=z, major_axis=y, minor_axis=x
                        ).to_frame(filter_observations=False).T
    except:
        print 'already a DataFrame'
        return data


def as_array(data):
    try:
        return data.T.to_panel().values
    except:
        print 'already an Array'
        return data


def write_slabs(fid, var_name, data):
    """Save 3d array into several slabs, for XDMF."""
    g = fid.create_group('/', 'data')
    for i, d in enumerate(data):
        fid.create_array(g, var_name +'_%02d' % i, d)


ap.rcparams()

# read data
print('loading data...')
fin = tb.open_file(DIR + FILE_IN)
data = fin.root.dh_mean_mixed_const_xcal[:]
error = fin.root.dh_error_xcal[:]
time = ap.num2year(fin.root.time_xcal[:])
lon = fin.root.lon[:]
lat = fin.root.lat[:]
xx, yy = np.meshgrid(lon, lat)
nz, ny, nx = data.shape

#f1 = tb.open_file('/Users/fpaolo/code/tseries/offset_linfit.h5')
f1 = tb.open_file('/Users/fpaolo/code/tseries/offset_absval.h5')
offset_12 = f1.root.offset_12[:]
offset_23 = f1.root.offset_23[:]
f1.close()

print('done')

if 0: # (for testing only) subset
    print 'subsetting...'
    region = ap.pig
    data, lon, lat = ap.get_subset(region, data, lon, lat)
    xx, yy = np.meshgrid(lon, lat)
    nt, ny, nx = data.shape                # i,j,k = t,y,x
    print 'done'

if 1: # (yes) filter time
    _, data = ap.time_filt(time, data, from_time=1994, to_time=2013)
    time, error = ap.time_filt(time, error, from_time=1994, to_time=2013)

dt = ap.year2date(time)

if 1: # (yes) filter ts with xcal offset > 1.5  # offset of dh_mean or dh_mean_mixed_???
    print 'xcalib filtering...'
    i, j = np.where((offset_12 > 1.5) | (offset_23 > 1.5))
    print 'xcalib filtered:', len(i)
    '''
    for ii,jj in zip(i, j):
        plt.plot(time, data[:,ii,jj])
        plt.show()
    '''
    data[...,i,j] = np.nan
    print 'done'

if 1: # (yes) filter step changes
    print 'step filtering...'
    data = as_frame(data, dt, lat, lon)
    data = data.apply(step_filt, delta=3, window=7, raw=True)
    print 'done'

if 1: # (yes) filter peaks (h > 3 * std from the trend)
    print 'peak filtering...'
    data = as_frame(data, dt, lat, lon)
    data = data.apply(peak_filt, x=time, iterative=True, raw=True)

if 1: # (yes) filter ts with std > max_std 
    print 'std filtering...'
    data = as_frame(data, dt, lat, lon)
    data = data.apply(std_filt, x=time, max_std=2.3, raw=True)
    print 'done'

if 1: # (yes) filter out uncomplete time series
    print 'percent filtering...'
    data = data.apply(percent_filt, min_perc=0.7, raw=True)
    print 'done'

# spherical -> cartesian
xed, yed = ap.cell2node(lon, lat)
xed2d, yed2d = np.meshgrid(xed, yed)
xyz = np.column_stack(ap.sph2xyz(xed2d.ravel(), yed2d.ravel()))

data = as_frame(data, dt, lat, lon)
data = data.apply(ap.referenced, to='mean', raw=True)    # to mean !!!
data = as_array(data)
error[np.isnan(data)] = np.nan

# plot
#---------------------------------------------------------------------

if PLOT: # plot time series
    data = as_array(data)
    j, = ap.find_nearest(lon, LON)
    i, = ap.find_nearest(lat, LAT)
    i, j = i[0], j[0]
    y = data[:,i,j]
    plt.plot(time, y, linewidth=2)
    plt.show()
    print len(y[~np.isnan(y)])
    print 'percentage of data points: %.1f' \
          % ((len(y[~np.isnan(y)]) / float(len(y))) * 100)
    exit()

# save data 
#---------------------------------------------------------------------

if 1:
    print('saving data...')
    fout = tb.open_file(DIR + FILE_OUT, 'a')
    '''
    try:
        fout.create_array('/', 'time', time)
    except:
        pass
    try:
        fout.create_array('/', 'lon', lon)
        fout.create_array('/', 'lat', lat)
    except:
        pass
    try:
        fout.create_array('/', 'xyz_nodes', xyz)
    except:
        pass
    try:
        fout.create_array('/', 'xx', xx)
        fout.create_array('/', 'yy', yy)
    except:
        pass
    fout.create_array('/', 'dh_mean_mixed_const_xcal', data)
    write_slabs(fout, 'dh_mean_mixed_const_xcal', data)
    '''
    fout.create_array('/', 'dh_error_xcal', error)
    fout.flush()
    fout.close()
    print('done')
    print('out -> ' + DIR + FILE_OUT)

fin.close()


