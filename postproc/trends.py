import sys
import numpy as np
import scipy as sp
import pandas as pd
import tables as tb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altimpy as ap

PLOT = False
DIR = '/Users/fpaolo/data/shelves/' 
FNAME = 'all_19920716_20111015_shelf_tide_grids_mts.h5'


def hpfilter(y, lamb=10): 
    return sm.tsa.filters.hpfilter(y, lamb=lamb)[1]


def gfilter(arr):
    return ap.gfilter(arr, 0.7)


def trend(year, dh):
    nt = len(year)
    _, ny, nx = dh.shape
    dhdt = np.zeros((ny, nx), 'f8') * np.nan
    for i in range(ny):
        for j in range(nx):
            ii, = np.where(~np.isnan(dh[:,i,j]))
            if len(ii) < 8:
                continue
            # TREND
            m, c = ap.linear_fit(year, dh[:,i,j], return_coef=True)
            dhdt[i,j] = m
            if m == 0: 
                continue
    return dhdt


ap.rcparams()

print 'loading data...'
f = tb.openFile(DIR + FNAME)
time = ap.num2year(f.root.time_xcal[:])
lon = f.root.lon[:]
lat = f.root.lat[:]
xed = f.root.x_edges[:]
yed = f.root.y_edges[:]
h0 = f.root.dh_mean_xcal[:]
h = f.root.dh_mean_xcal_interp_short_const[:]
#h = f.root.dh_mean_short_const_xcal[:]
h2 = f.root.dh_mean_xcal_short_const[:]
#h2 = f.root.dh_mean_short_const_xcal[:]
g = f.root.dg_mean_xcal[:]
g2 = f.root.dg_mean_xcal[:]
dt = ap.year2date(time)


# add interpolated grid-cells
ind = np.where(~np.isnan(h2))
h[ind] = h2[ind]

# gaussian filter (in space) 
print 'smoothing seasonal fields...'
h = ap.gfilter2d(h, .5)

# array -> panel -> dataframe
data = pd.Panel(h, items=dt, major_axis=lat, minor_axis=lon
                ).to_frame(filter_observations=False).T

# HP filter (in time)
print 'applying HP filter...'
#hpfilt = data.apply(hpfilter, lamb=7)
hpfilt = data

# split data
n = int(len(hpfilt)/4.)
hpfilt1 = hpfilt.T.to_panel().values[0*n:1*n,...]
hpfilt2 = hpfilt.T.to_panel().values[1*n:2*n,...]
hpfilt3 = hpfilt.T.to_panel().values[2*n:3*n,...]
hpfilt4 = hpfilt.T.to_panel().values[3*n:-1,...]

print hpfilt.shape

trends = np.zeros((4, hpfilt1.shape[1], hpfilt1.shape[2]), 'f8') * np.nan
time2 = np.array([time[0], time[n], time[2*n], time[3*n]])
print time2

# fit linear trends
print 'fitting trends...'
trends[0,...] = trend(time[0:n], hpfilt1)
trends[1,...] = trend(time[n:2*n], hpfilt2)
trends[2,...] = trend(time[2*n:3*n], hpfilt3)
trends[3,...] = trend(time[3*n:-1], hpfilt4)


# smooth fields (sigma between 0.6-7)
print 'smoothing fields...'
trends = ap.gfilter2d(trends, 0.7)


# plot time series: (1,161), (2,161), (3,161)
if PLOT:
    plt.imshow(trends[3], origin='lower', interpolation='nearest')
    plt.show()
    sys.exit()


# regrid fields
print 'regridding fields...'
inc = 3
trends, xx, yy = ap.regrid2d(trends, lon, lat, inc_by=inc)

# create 3d coordinates of nodes (x,y,z)
xed = np.linspace(xed.min(), xed.max(), inc * len(lon) + 1)
yed = np.linspace(yed.min(), yed.max(), inc * len(lat) + 1)
xx, yy = np.meshgrid(xed, yed)
lon = xx.ravel()
lat = yy.ravel()
xyz = np.column_stack(ap.sph2xyz(lon, lat))

#---------------------------------------------------------------------
# save data 
#---------------------------------------------------------------------

print 'saving data...'
ff = tb.openFile(DIR + 'elev.h5', 'a')
#ff.createArray('/', 'xyz_nodes_regrid', xyz)
ff.createArray('/', 'time_linear', time2)
ff.createGroup('/', 'linear_dhdt')

for k, field in enumerate(trends):
    ff.createArray('/linear_dhdt', 'elev_%02d' % k, field)
ff.close()
print 'done.'

f.close()

