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

def gradient(y, dt=.25):
    return np.gradient(y.values, dt)

def apply_to_panel(func, panel, *args, **kw):
    for item in panel.items:
        panel[item] = func(panel[item], *args, **kw)

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

# annual averages (forward-backward)
'''
print 'calculating annual averages...'
ann1 = pd.rolling_mean(data, 5, min_periods=3, center=True)
ann2 = pd.rolling_mean(data[::-1], 5, min_periods=3, center=True)[::-1]
annual = ann1.combine_first(ann2).T.to_panel().values
del ann1, ann2
'''

# HP filter (in time)
print 'applying HP filter...'
hpfilt = data.apply(hpfilter, lamb=7)

# fit polynomial
'''
print 'fitting curves...'
spline = ap.spline2d(time, h, window=3, min_pts=50)
#poly = ap.polyfit2d(time, h, deg=25, min_pts=50)  # at least 75% of pts
'''

# calculate derivative
print 'taking derivatives...'
hpfilt_grad = hpfilt.apply(gradient, dt=.25)

# start series at zero
print 'referencing time series...'
hpfilt = hpfilt.apply(ap.referenced, to='first')

# smooth fields (sigma between 0.6-7)
print 'smoothing fields...'
hpfilt = hpfilt.T.to_panel()
hpfilt_grad = hpfilt_grad.T.to_panel()
apply_to_panel(ap.gfilter, hpfilt, .7)
apply_to_panel(ap.gfilter, hpfilt_grad, .7)

# plot time series: (1,161), (2,161), (3,161)
if PLOT:
    '''
    i, j = 1, 161
    '''
    xx, yy = np.meshgrid(lon, lat)
    i, j = ap.find_nearest2(xx, yy, [(116, -67)])
    a, b = xx[i,j], yy[i,j]
    print a, b
    '''
    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(time, h0[:,i,j], linewidth=2)
    ap.intitle('uncorr. elevation', ax=ax1)
    ax2 = plt.subplot(212)
    plt.plot(time, g[:,i,j], linewidth=2, c='k')
    ap.intitle('backscatter', ax=ax2)
    #plt.savefig('totten_d1.png')
    '''

    plt.plot(time, h[:,i,j], linewidth=1, label='seasonal')
    #plt.plot(time, h_cycle, linewidth=2)
    #plt.plot(time, spline[:,i,j], linewidth=2, label='spline')
    plt.plot(time, hpfilt.values[:,i,j], linewidth=2, label='hpfilter')
    plt.plot(time, hpfilt_grad.values[:,i,j], linewidth=1)
    #plt.plot(time, annual[:,i,j], 'k', linewidth=2, label='h: annual')
    plt.show()
    sys.exit()

    plt.xlim(1992, 2012)
    plt.legend().draw_frame(False)
    #plt.ylabel('elevation (m)')
    #ap.intitle('(D) lon, lat = %.1f, %.1f' % (a, b), 4)
    #plt.savefig('totten_d2.png')
    plt.show()
    sys.exit()


# regrid fields
print 'regridding fields...'
inc = 3
hpfilt, xx, yy = ap.regrid2d(hpfilt.values, lon, lat, inc_by=inc)
hpfilt_grad, xx, yy = ap.regrid2d(hpfilt_grad.values, lon, lat, inc_by=inc)

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
ff.createArray('/', 'xyz_nodes_regrid', xyz)
ff.createGroup('/', 'regrid_gfilt_hpfilt_gfilt')
ff.createGroup('/', 'regrid_gfilt_hpfilt_grad_gfilt')

for k, field in enumerate(hpfilt):
    ff.createArray('/regrid_gfilt_hpfilt_gfilt', 'elev_%02d' % k, field)
for k, field in enumerate(hpfilt_grad):
    ff.createArray('/regrid_gfilt_hpfilt_grad_gfilt', 'elev_%02d' % k, field)
ff.close()
print 'done.'

f.close()

