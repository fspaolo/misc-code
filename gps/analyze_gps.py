import sys
import numpy as np
import tables as tb
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from glob import glob

import altimpy as ap

#------------------------------------------------------------
# to join all gps files into one hdf5
#------------------------------------------------------------
fnames1 = '/Users/fpaolo/data/gps/forShijieLiu/*.llh'
fnames2 = '/Users/fpaolo/data/gps/spider_web_groundtracks/*.llh'

files = glob(fnames1) + glob(fnames2)

# load all data files
t = lon = lat = h = []
for f in files:
    d = np.loadtxt(f)
    t = np.append(t, d[:,0])      # day
    lon = np.append(lon, d[:,1])  # lon
    lat = np.append(lat, d[:,2])  # lat
    h = np.append(h, d[:,3])      # elev

t = ap.day2year(t, since=(2000, 1, 1))
x, y = ap.ll2xy(lon, lat)
x, y = lon, lat

'''
# create hdf5 file
f = tb.openFile('amery_gps_all.h5', 'w')
f.createArray('/', 'time', t)
f.createArray('/', 'lon', lon)
f.createArray('/', 'lat', lat)
f.createArray('/', 'x', x)
f.createArray('/', 'y', y)
f.createArray('/', 'h', h)
f.close()
'''
#------------------------------------------------------------

fname1 = '/Users/fpaolo/data/gps/amery_gps_all.h5'
fname2 = '/Users/fpaolo/data/shelves/all_19920716_20111015_shelf_tide_grids_mts.h5'

with tb.openFile(fname1) as f1, tb.openFile(fname2) as f2:

    # gps
    '''
    t = ap.year2date(f1.root.time[:])
    lon = f1.root.lon[:]
    lat = f1.root.lat[:]
    h = f1.root.h[:]
    df_gps = pd.DataFrame({'lon': lon, 'lat': lat, 'h': h}, index=t)
    print df_gps.head()
    del t, lon, lat, h
    '''

    # altimetry
    h0 = f2.root.dh_mean_all[:]
    #h1 = f2.root.dh_mean_corr_short_all[:]
    #h2 = f2.root.h_firn[:]
    #g = f2.root.dg_mean_all[:]
    t0 = ap.num2date(f2.root.time_all[:])
    #t2 = ap.num2date(f2.root.time_firn[:])
    lon = f2.root.x_edges[:]
    lat = f2.root.y_edges[:]

    '''
    # maps grid indexes to coordinates
    ii, jj = xrange(len(lat)), xrange(len(lon))
    ij2ll = dict([((i,j),(la,lo)) for i,j,la,lo in zip(ii,jj,lat,lon)])

    df_h0 = pd.Panel(h0, items=t0, major_axis=ii, minor_axis=jj
                     ).to_frame(filter_observations=False).T
    del h0, t0, lon, lat
    print df_h0
    sys.exit()
    '''

#------------------------- FIGURES --------------------------
ap.rcparams()

i1 = np.where( t < 1997)
i2 = np.where((t > 1997) & ( t < 2000))
i3 = np.where((t > 2000) & ( t < 2002))
i4 = np.where((t > 2002) & ( t < 2004))
i5 = np.where(t > 2004)

plt.imshow(h0[10,...], extent=(lon[0], lon[-1], lat[0], lat[-1]), origin='lower', interpolation='nearest')
# plot GPS locations
plt.plot(x[i1], y[i1], 'b.', mec='blue', label='1995')
plt.plot(x[i2], y[i2], 'c.', mec='cyan', label='1999')
plt.plot(x[i3], y[i3], 'g.', mec='green', label='2001')
plt.plot(x[i4], y[i4], 'y.', mec='yellow', label='2003')
plt.plot(x[i5], y[i5], 'r.', mec='red', label='2006')
plt.legend(loc=3).draw_frame(False)
ap.intitle('Amery GPS locations', 2)
plt.xlabel('x (km)')
plt.ylabel('y (km)')
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.savefig('gps_locations_amery.png', bbox_inches='tight')
plt.show()

'''
# plot GPS time series
plt.plot(t[i1], h[i1], 'b.', mec='blue')
plt.plot(t[i2], h[i2], 'c.', mec='cyan')
plt.plot(t[i3], h[i3], 'g.', mec='green')
plt.plot(t[i4], h[i4], 'y.', mec='yellow')
plt.plot(t[i5], h[i5], 'r.', mec='red')
ap.intitle('Amery GPS tseries', 2)
plt.xlabel('time')
plt.ylabel('WGS84 h (m)')
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.savefig('gps_tseries_amery.png', bbox_inches='tight')
plt.show()
'''

'''
“If you know the enemy and know yourself, you need not fear the result of a hundred battles", ―The Art of War. Maybe this can help: http://onlyamodel.com/2011/speed-up-plot-rendering-in-pythonmatplotlib/
'''
