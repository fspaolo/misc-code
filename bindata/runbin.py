import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.basemap import cm  # GMT
import numpy as np
import tables as tb
from netCDF4 import Dataset
import os, sys

try:
    from matplotlib.colors import LightSource
except ImportError:
    raise ImportError("Need a more recent version of matplotlib!")

import bindata

fname = '/Volumes/LaCie1TB/ANTARCTICA/ENVISAT/2009/all_1978-all_0710_tide_wilkins.h5'
#fname = '/Volumes/LaCie1TB/ANTARCTICA/ENVISAT/2009/all_1985_0411-all_0411_tide.h5' 

# LOAD

#data = np.loadtxt(fname)
h5f = tb.openFile(fname, 'r')
data = h5f.root.data.read()
h5f.close()

x = data[:,0]
y = data[:,1]
z1 = data[:,6]
z2 = data[:,7]
inc1 = data[:,22]
inc2 = data[:,23]
tid1 = data[:,24]
tid2 = data[:,25]

z = (z2 + inc2 - tid2) - (z1 + inc1 - tid1)

# filter

ind, = np.where((-5 <= z) & (z <= 5))
x = x[ind]
y = y[ind]
z = z[ind]

# grid

dx = 0.5  # for contourf !!!
dy = 0.15
#dx = 1.2
#dy = 0.3

left, right = x.min(), x.max()
bottom, top = y.min(), y.max()
xi = np.arange(left, right+dx, dx)
yi = np.arange(bottom, top+dy, dy)
xx, yy = np.meshgrid(xi, yi)

grid, bins = bindata.bindata(x, y, z, xi, yi, ppbin=True)
#grid = griddata(x, y, z, xi, yi)

##### PLOT

fig = plt.figure(figsize=(9,9))
ax = plt.axes([0,0,1,1])

# Antactic Peninsula
left, right, bottom, top = -80, -50, -74, -61
lon0, lat0, lon1, lat1 = -67.5, -67, -56, -68.7

# use major and minor sphere radii from WGS84 ellipsoid
m = Basemap(projection='lcc', lat_1=-72, lat_2=-62, lon_0=0,
            llcrnrlon=lon0, llcrnrlat=lat0, urcrnrlon=lon1, urcrnrlat=lat1,\
            rsphere=(6378137.00, 6356752.3142))

m.drawparallels(np.arange(-80.,81.,1), labels=[0,0,0,0], color='0.4')
m.drawmeridians(np.arange(-180.,181.,3), labels=[0,0,0,0], color='0.4')

TOPO = True
SHORE = True
RES = 400.
cmap = cm.jet
vmin, vmax = -5, 80  # Envisat ice-shelf
#vmin, vmax = -10000, 80000 
inc = 10
cblabel = 'dh/dt (cm/yr)'
title = '[1] Seasat-Envisat (1978-2009)'
figname = 'mask_border.ps'

ticks = np.arange(vmin, vmax+inc, inc)          # cb ticks
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#bounds = np.arange(vmin, vmax+inc/3., inc/3.)  # cb discrete
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#----------------------------------------------------------------
# plot shoreline
#----------------------------------------------------------------

SHORE = True
if SHORE:
    dir = '/Users/fpaolo/data/coastline'
    files = ['antarctica_gl_ll.h5',
             'moa_coastfile_ll.h5',
             'moa_islands_ll.h5']
    for f in files:
        fname = os.path.join(dir, f)
        h5f = tb.openFile(fname)
        shore = h5f.root.data.read()
        h5f.close()
        lon = shore[:,1] 
        lat = shore[:,0] 
        ind, = np.where((left <= lon) & (lon <= right) & \
                        (bottom <= lat) & (lat <= top))
        x, y = m(lon[ind], lat[ind])
        m.scatter(x, y, s=0.2, c='k', marker='o', edgecolors='none')

#----------------------------------------------------------------
# plot topo
#----------------------------------------------------------------

TOPO = True
if TOPO:
    etopo1name='/Users/fpaolo/data/topo/ETOPO1_Ice_g_gmt4.grd'
    etopo1 = Dataset(etopo1name,'r')
    lons = etopo1.variables["lon"][:]
    lats = etopo1.variables["lat"][:]
    i_lon, = np.where((left <= lons) & (lons <= right))
    i_lat, = np.where((bottom <= lats) & (lats <= top))
    topo = etopo1.variables["z"][i_lat,i_lon]
    lons = lons[i_lon]
    lats = lats[i_lat]

    # transform to nx x ny regularly spaced native projection grid
    nx = int((m.xmax-m.xmin)/RES)+1; ny = int((m.ymax-m.ymin)/RES)+1
    topo, x, y = m.transform_scalar(topo, lons, lats, nx, ny, returnxy=True)

    # create light source object.
    ls = LightSource(azdeg = 100, altdeg = 45)

    # convert data to rgb array including shading from light source.
    topom = np.ma.masked_where((topo < 0), topo)
    rgb = ls.shade(topom, cmap)
    rgb = ls.shade(topom, cm.gray_r)
    im = m.imshow(rgb)

    shelf = np.ma.masked_where((topom > 90), topo)
    im = m.imshow(shelf, cmap=cm.gray_r)

#----------------------------------------------------------------


extent = (left-dx/2., right+dx/2., bottom-dy/2., top+dy/2.)
# minimum values for colorbar. filter our nans which are in the grid
zmin = grid[np.where(np.isnan(grid) == False)].min()
zmax = grid[np.where(np.isnan(grid) == False)].max()

#x, y = m(x, y)
#m.plot(x, y, 'k.')
xx, yy = m(xx-dx/2.5, yy-dy/2)
m.pcolor(xx, yy, grid, cmap=cm.jet_r, vmin=zmin, vmax=zmax) #shading='faceted', )
#m.contourf(xx, yy, grid, 1)
#m.imshow(grid, extent=extent, cmap=cm.Spectral_r, interpolation='nearest')
'''
plt.scatter(x, y, c=z, s=8, cmap='Spectral_r', edgecolors='none')
plt.xlim(extent[0], extent[1])
plt.ylim(extent[2], extent[3])
plt.title('ERS-2 (1995) dh=Jun-May [m]')
plt.xlabel('lon')
plt.ylabel('lat')
plt.colorbar()

plt.figure()
plt.imshow(grid, extent=extent, aspect='auto', origin='lower', \
           cmap=cm.Spectral_r, interpolation='nearest')
#plt.plot(xx, yy, 'r.')
plt.title('ERS-2 (1995) dh=Jun-May "binned" [m]')
plt.xlabel('lon')
plt.ylabel('lat')
plt.colorbar()
plt.grid(True)

plt.figure()
bins[bins==0] = np.nan
plt.imshow(bins, extent=extent, aspect='auto', origin='lower', \
           cmap=cm.copper_r, interpolation='nearest')
plt.title('Number of crossovers per bin')
plt.xlabel('lon')
plt.ylabel('lat')
'''
# add a map scale
length = 50  # km 
x1, y1 = 0.88*m.xmax, 0.9*m.ymax  # position
lon1,lat1 = m(x1, y1, inverse=True)
m.drawmapscale(lon1, lat1, lon1, lat1, length, fontsize=12, \
               barstyle='fancy', units='km') #, labelstyle='fancy')
#plt.colorbar()
plt.savefig('bins.png', dpi=150, orientation='portrait')

plt.show()
