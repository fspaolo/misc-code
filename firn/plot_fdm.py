import sys
import numpy as np
import scipy as sp
import netCDF4 as nc
import matplotlib.pyplot as plt

import altimpy as ap

lons = np.array([ 294.5,  295.5,  296.5,  297.5,  298.5,  299.5])
lats = np.array([-69.4, -69. , -68.6, -68.2, -67.8, -67.4, -67. , -66.6, -66.2, -65.8])
lons[lons>180] -= 360
px, py = np.meshgrid(lons, lats)
points = np.column_stack((px.ravel(), py.ravel()))

'''
plon, plat = -67.0, -81.4
dx, dy = .3, .3
'''
bbox = (-82, -76.2, -26, -79.5)

if len(sys.argv) < 2:
    raise IOError('no input file!')

fname = sys.argv[1]

fin = nc.Dataset(fname)
d = fin.variables

firn = d['zs']
year = d['time'][:]
lon = d['lon'][:]
lat = d['lat'][:]
ism = d['ism']

j, i = ap.find_nearest2(lon, lat, points)

k, = np.where((year > 1992) & (year < 2012))

'''
plt.figure()
m = ap.viz.make_proj_stere((lon[-1,0], lat[-1,0], lon[0,-1], lat[0,-1]), lat_ts=-90)
px, py = m(lon[i,j], lat[i,j])

m.imshow(firn[1000,...], origin='lower', interpolation='nearest')
m.plot(px, py, 'o')
plt.title('lon, lat = %.3f, %.3f' % (plon, plat))
'''

for ii, jj in zip(i,j):
    print lon[jj,ii], lat[jj,ii]
    fig = plt.figure()
    plt.plot(year[k], firn[k,jj,ii])
    #plt.ylim(-0.8, 0.8)
    fig.autofmt_xdate()

    plt.show()

fin.close()
