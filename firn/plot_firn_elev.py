import sys
import datetime as dt
import numpy as np
import scipy as sp
import tables as tb
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

import altimpy as ap

FILE1 = 'FM_ANT3K27_1979-2011.nc'
FILE2 = '/Users/fpaolo/data/shelves/ross_19920715_20111015.h5' #larsenc_19920715_20111015.h5'
FILE3 = '/Users/fpaolo/data/coastline/moa_coastfile_ll.h5'
FILE4 = '/Users/fpaolo/data/coastline/antarctica_gl_ll.h5'
#BBOX = (-71.3854, -73.0697, -58.0593, -63.0486)
BBOX = (-126.863, -46.7492, 53.2309, -47.0738)
#BBOX = (-33.4259, -46.9022, 147.058, -46.9198)

f1 = nc.Dataset(FILE1)
d1 = f1.variables
firn = d1['zs']
year1 = d1['time'][:]
lons1 = d1['lon'][:]  # 2d
lats1 = d1['lat'][:]  # 2d
ism = d1['ism'][:]
lsm = d1['lsm'][:]
dt1 = ap.year2date(year1)
#dt1 = year1

f2 = tb.openFile(FILE2)
elev = f2.root.elev[:]
time2 = f2.root.time[:]
lon2 = f2.root.lon[:]  # 1d
lat2 = f2.root.lat[:]  # 1d
lon2 = ap.lon_180_360(lon2, inverse=True)
lons2, lats2 = np.meshgrid(lon2, lat2)
points2 = np.column_stack((lons2.ravel(), lats2.ravel()))
#points2 = (lons2[6,4], lats2[6,4])
dt2 = ap.num2date(time2)
#dt2 = ap.num2year(time2)

'''
f3 = tb.openFile(FILE3)
d3 = f3.root.data[:]
y3, x3 = d3[:,0], d3[:,1]
x3 = ap.lon_180_360(x3, inverse=True)

f4 = tb.openFile(FILE4)
d4 = f4.root.data[:]
y4, x4 = d4[:,0], d4[:,1]
x4 = ap.lon_180_360(x4, inverse=True)
'''

# crop firn grid
#i1, j1 = ap.find_nearest2(lons1, lats1, points2)
k1, = np.where( (dt1 > dt.datetime(1992,1,1)) & (dt1 < dt.datetime(2012,1,1)) )
#i_min, i_max, j_min, j_max = i1.min(), i1.max(), j1.min(), j1.max()
##lons1, lats1 = lons1[i_min:i_max,j_min:j_max], lats1[i_min:i_max,j_min:j_max]
##firn = firn[k1,i_min:i_max,j_min:j_max]
#lons1, lats1 = lons1[130:160,30:65], lats1[130:160,30:65]
#firn = firn[k1,130:160,30:65]
#dt1 = dt1[k1]

#lons1 = lons1[::-1,:]
#lats1 = lats1[::-1,:]

# indices of cropped grid
i1, j1 = ap.find_nearest2(lons1, lats1, points2)
i2, j2 = ap.find_nearest2(lons2, lats2, points2)

print lons1[-1,0], lats1[-1,0], lons1[0,-1], lats1[0,-1]

for i_1, j_1, i_2, j_2 in zip(i1, j1, i2, j2):

    x1, y1 = lons1[i_1,j_1].round(3), lats1[i_1,j_1].round(3)
    x2, y2 = lons2[i_2,j_2].round(3), lats2[i_2,j_2].round(3)
    print 'firn lon/lat:', x1, y1
    print 'elev lon/lat:', x2, y2

    if np.alltrue(np.isnan(elev[:,i_2,j_2])): continue

    fig = plt.figure()
    plt.plot(dt1[k1], firn[k1,i_1,j_1], linewidth=2, label='firn')
    plt.plot(dt2[:], elev[:,i_2,j_2], linewidth=2, label='elev')
    plt.legend().draw_frame(False)
    #plt.xlim(1992, 2012)

    plt.figure()
    m = ap.viz.make_proj_stere(BBOX)
    m = ap.viz.plot_grid_proj(m, lons1, lats1, firn[1000,...])

    '''
    ij = np.where((ism == 1) | (lsm == 1))
    x11, y11 = lons1[ij].ravel(), lats1[ij].ravel()
    xx11, yy11 = m(x11, y11)
    m.plot(xx11, yy11, 'g.')
    '''

    m = ap.viz.plot_grid_proj(m, lons2, lats2, elev[10,...], masked=False, alpha=.2)
    pp = m.drawparallels(np.arange(-90.,-60.,4), labels=[0,0,1,0], color='0.6')
    mm = m.drawmeridians(np.arange(-180,180.,8), labels=[0,1,0,0], color='0.6')

    '''
    xx3, yy3 = m(x3, y3)
    m.plot(xx3, yy3, 'k.')
    xx4, yy4 = m(x4, y4)
    m.plot(xx4, yy4, 'k.')
    '''
    px, py = m(x1, y1)
    m.plot(px, py, 'bo')
    px, py = m(x2, y2)
    m.plot(px, py, 'go')

    fig.autofmt_xdate()
    plt.show()

#f_firn.close()
