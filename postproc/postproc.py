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

try:
    import seaborn as sns
    sns.set(palette=ap.tab20)
except:
    pass

# ==> CHECK AND EDIT THE 'IF 0/1' IN THE CODE <==
PLOT = False
SAVE = True
FILE_IN = '/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids.h5_mts_byfirst3'
FILE_OUT = '/Users/fpaolo/data/shelves/h_postproc.h5.byfirst3_'
FILE_OFFSET = '/Users/fpaolo/data/shelves/offset_absval.h5.byfirst2'
FILE_MSK = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.h5'

# for plotting
#LON, LAT = 115.125, -67.125   # Totten, B
LON, LAT = 260.625, -75.125   # PIG, GL
#LON, LAT = 186.375, -79.375  # Ross, with problem at the ends
#LON, LAT = 69.375, -72.125   # Amery

# grid cells to remove
lonlat = [
(114.375, -67.125),
#(115.875, -67.12),
 (144.375, -67.62),
 (168.375, -73.37),
 (175.125, -77.625),
 (175.625, -77.875),
 (175.875, -77.62),
 (175.875, -77.625),
 (175.875, -77.875),
 (176.625, -77.87),
 (176.625, -77.875),
 #(177.375, -77.875),
 #(179.625, -77.875),
 #(180.375, -77.875),
 (187.875, -78.375),
 (189.375, -78.625),
 (190.125, -78.625),
 (190.875, -78.625),
 (190.875, -78.875),
 (191.625, -78.625),
 (212.625, -77.37),
 (212.625, -77.375),
 #(214.125, -77.12),
 (259.875, -75.125),
 (260.625, -75.125),
 (264.375, -72.37),
 (264.375, -72.375),
 (270.375, -72.87),
 (270.375, -72.875),
 (277.875, -79.12),
 (28.125, -70.62),
 #(282.375, -77.87),
 (283.125, -73.875),
 #(286.125, -73.37),
 (287.625, -73.375),
 #(288.375, -71.12),
 #(292.875, -72.12),
 (299.625, -67.125),
 (31.875, -69.87),
 (31.875, -69.875),
 (316.857, -78.125),
 #(317.625, -78.125),
 #(318.375, -78.125),
 #(319.125, -78.125),
 #(319.875, -78.125),
 (320.265, -78.125),
 (320.625, -78.125),
 (321.375, -78.12),
 (321.375, -78.125),
 (323.625, -78.375),
 (334.875, -75.12),
 (335.625, -74.875),
 (336.375, -75.125),
 (336.375, -75.375),
 #(336.375, -75.625),
 (337.125, -74.12),
 (337.125, -74.125),
 (337.875, -74.125),
 (337.875, -75.875),
 (345.375, -73.87),
 (346.125, -72.87),
 (346.875, -72.62),
 (346.875, -72.625),
 (359.625, -71.37),
 (38.625, -69.875),
 (67.875, -72.625),
 #(69.375, -71.62),
 (69.375, -72.125),
 #(72.375, -69.87),
 #(99.375, -66.12),
 ]

def step_filt(y, plot=False, **kw):
    if plot:
        y_orig = y.copy()
    y, m = ap.step_filt(y, **kw)
    if plot and np.nansum(y_orig) != np.nansum(y):
        plt.plot(y_orig, linewidth=2)
        plt.plot(y, linewidth=2)
        plt.plot(m, '--', linewidth=2)
        plt.show()
    return y


def peak_filt(y, x=None, iterative=True, n_std=3, max_deg=3, plot=False):
    if not np.isnan(y).all():
        if plot:
            y_orig = y.copy()
        y = ap.peak_filt(x, y, iterative=iterative, n_std=n_std, max_deg=max_deg)
        if plot and np.count_nonzero(np.isnan(y_orig)) != np.count_nonzero(np.isnan(y)):
            plt.plot(y_orig, '--r', linewidth=2)
            plt.plot(y, 'b', linewidth=2)
            plt.show()
    return y


def std_filt(y, x=None, max_std=None, max_deg=3, plot=False):
    """Filter entire vector if the detrended std > max_std."""
    if plot:
        print y.name
        y_orig = y.copy()
    y, std = ap.std_series_filt(x, y, max_std=max_std, max_deg=max_deg)
    if plot and std > max_std: 
        print 'ts-std:', std
        plt.plot(y_orig)
        plt.show()
    return y


def percent_filt(y, min_perc=None, plot=False):
    if plot:
        y_orig = y.copy()
    y, perc = ap.percent_filt(y, min_perc=min_perc)
    if plot and perc > 0 and perc < min_perc: 
        print 'ts-perc:', perc
        plt.plot(y_orig)
        plt.show()
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
fin = tb.open_file(FILE_IN)
data = fin.root.dh_mean_mixed_const_xcal[:]
error = fin.root.dh_error_xcal[:]
time = ap.num2year(fin.root.time_xcal[:])
lon = fin.root.lon[:]
lat = fin.root.lat[:]
xx, yy = np.meshgrid(lon, lat)
nz, ny, nx = data.shape

f1 = tb.open_file(FILE_OFFSET)
offset_12 = f1.root.offset_12[:]
offset_23 = f1.root.offset_23[:]
f1.close()
print('done')

if 0: # (for testing only) subset
    print lon.shape, lat.shape, data.shape
    i, j = ap.where_isnan('Totten', lon, lat)
    data[:,i,j] = np.nan
    '''
    plt.imshow(data[10], origin='lower', interpolation='nearest', 
               extent=(lon.min(), lon.max(), lat.min(), lat.max()), 
               aspect='auto')
    plt.show()
    exit()
    '''

if 1: # (yes) filter time
    _, data = ap.time_filt(time, data, from_time=1994, to_time=2013)
    time, error = ap.time_filt(time, error, from_time=1994, to_time=2013)

dt = ap.year2date(time)

# remove bad grid cells (visual inspection)
if 1:
    ii, jj = ap.find_nearest2(xx, yy, lonlat)
    k = 0
    for i, j in zip(ii, jj):
        print k, lonlat[k]
        '''
        plt.plot(time, data[:,i,j])
        plt.show()
        '''
        data[:,i,j] = np.nan
        k += 1
    print 'removed grid cells:', k

# (NO) filter ts with xcal offset > 1.5  # offset of dh_mean!
if 0: 
    print 'xcalib filtering...'
    i, j = np.where((offset_12 > 2) | (offset_23 > 2))

    for ii, jj in zip(i,j):
        plt.plot(data[...,ii,jj])
        plt.show()

    data[...,i,j] = np.nan
    print 'done.'
    exit()

# (yes) filter step changes
if 1:
    print 'step filtering...'
    data = as_frame(data, dt, lat, lon)
    data = data.apply(step_filt, delta=3, window=15, plot=False, raw=True)

# (yes) filter peaks
if 1:
    print 'peak filtering...'
    data = as_frame(data, dt, lat, lon)
    data = data.apply(peak_filt, x=time, n_std=3, max_deg=3, iterative=True, plot=False, raw=True)

# (NO) filter ts with std > max_std 
if 0:
    print 'std filtering...'
    data = as_frame(data, dt, lat, lon)
    data = data.apply(std_filt, x=time, max_std=2, max_deg=3, plot=False, raw=False)

# (yes) filter out uncomplete time series
if 1:
    print 'percent filtering...'
    data = data.apply(percent_filt, min_perc=0.7, plot=False, raw=True)

# spherical -> cartesian
xed, yed = ap.cell2node(lon, lat)
xed2d, yed2d = np.meshgrid(xed, yed)
xyz = np.column_stack(ap.sph2xyz(xed2d.ravel(), yed2d.ravel()))

data = as_frame(data, dt, lat, lon)
data = data.apply(ap.referenced, to='mean', raw=True)    # to mean !!!
data = as_array(data)
error[np.isnan(data)] = np.nan

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

# save data 
#---------------------------------------------------------------------

if SAVE:
    print('saving data...')
    fout = tb.open_file(FILE_OUT, 'w')
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
    fout.create_array('/', 'dh_error_xcal', error)
    fout.flush()
    fout.close()
    print('out -> ' + FILE_OUT)

fin.close()



