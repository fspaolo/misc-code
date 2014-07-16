import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap, interp
import sys

from osgeo import osr, gdal

import libmask as lm

# load MODIS MOA Image Map 
#---------------------------------------------------------------------

fname = 'moa750_r1_hp1.tif'    # GeoTIFF 750m resolution

ds = gdal.Open(fname) 
MOA = ds.ReadAsArray()
gt = ds.GetGeoTransform() 

nx = ds.RasterXSize       # number of pixels in x
ny = ds.RasterYSize       # number of pixels in y 
dx = gt[1]                # pixel size in x
dy = gt[5]                # pixel size in y 
xmin = gt[0]
ymax = gt[3]
ymin = ymax + nx*gt[4] + ny*dy 
xmax = xmin + nx*dx + ny*gt[2] 
x = np.arange(xmin, xmax, dx)    # Polar stereo coords
y = np.arange(ymax, ymin, dy)    # in reverse order !!!
print y[0], y[-1]

fig = plt.figure(figsize=(10,9))

# projection 1 --------------------------------------------------------

'''
# MOA parameters: http://nsidc.org/data/moa/users_guide.html
m1 = Basemap(projection='stere', lat_ts=-71,
             lon_0=0, lat_0=-90,                   # projection center (NOT grid center!)
             urcrnrlon=49.998067125199078, urcrnrlat=-56.496665877185166,
             llcrnrlon=-131.58249796785486, llcrnrlat=-52.306577823917102, 
             rsphere=(6378137.00, 6356752.3142),   # ellipsoid (WGS84 major and minor)
             #rsphere=6371200.,                    # sphere (mean radius)
             #fix_aspect=False,
             #anchor='C',
             )

m1.drawparallels(np.arange(-90.,0.,2), labels=[1,1,0,0], color='0.4')
m1.drawmeridians(np.arange(-180.,180.,10), labels=[0,0,1,1], color='0.4')
'''

# MOA-based mask transformation -----------------------------------------

res = 10
m = lm.Mask('mask_ice_1km_2008_0410c.h5')

x1, y1 = np.meshgrid(m.x_mask[::res], m.y_mask[::res])
MASK = m.m_mask[::res,::res]
m.closemask()

lon, lat = m.mapxy(x1, y1, slat=71, slon=-70, hemi='s', units='km')
x2, y2 = m.mapll(lon, lat, slat=71, slon=0, hemi='s', units='m')
#x3, y3 = m1(lon, lat)    # transform to native coords

# plot ------------------------------------------------------------------

# in native map coordinates !!! 

# control points
#xx, yy = m1([-79, -49.9, -52.836819, -131.582498, 49.998067, -143.207254], 
#            [-68.4, -69.6, -54.475658, -52.306578, -56.496666, -87.642248])
#m1.plot(xx, yy, 'or', markersize=15)

# MOA grid
#m1.imshow(MOA[::2, ::2], cmap=cm.gray_r, origin="upper")
xx1, yy1 = np.meshgrid(x[::5], y[::5])
pl.imshow(MOA[::5, ::5], extent=(xx1[0,0], xx1[0,-1], yy1[-1,0], yy1[0,0]), cmap=cm.gray, origin="upper")
#m1.contourf(xx1, yy1, MOA[::10, ::10], levels=(10, 100, 200), cmap=cm.gray)

# MASK grid
#m1.contour(x3, y3, MASK, levels=(0.5, 1.5), linewidths=2)    # on geographic proj
pl.contour(x2, y2, MASK, levels=(0.5, 1.5), linewidths=2)   # on grid proj

pl.show()
