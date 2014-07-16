import numpy as np
import numpy.ma as ma
from numpy.random import uniform
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

from bindata import bindata, plotbins  # the bindata.py module

# make up some randomly distributed data
npts = 700 
x = uniform(-2,2,npts)
y = uniform(-2,2,npts)
z = x*np.exp(-x**2-y**2)

# define the grid.
xi = np.linspace(-2.1, 2.1, 20)
yi = np.linspace(-2.1, 2.1, 20)

# grid the data.
zi = griddata(x, y, z, xi, yi)
#zi[zi<-0.3] = np.nan

plt.figure()
# contour the gridded data, plotting dots at the randomly spaced data points.
#CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
#CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.Spectral_r)
plotbins(xi, yi, zi)
plt.colorbar()
plt.scatter(x, y, marker='o', c='b', s=3)
plt.xlim(xi.min(), xi.max())
plt.ylim(yi.min(), yi.max())
plt.title('Gridded data (%d points)' % npts)
plt.savefig('gridded-data.png')

#####
# bin the data.
zi = bindata(x, y, z, xi, yi)
#zi[zi<-0.3] = np.nan

plt.figure()
# contour the gridded data, plotting dots at the randomly spaced data points.
#CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
#CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.Spectral_r)
plotbins(xi, yi, zi)
plt.colorbar() # draw colorbar
#####

# plot data points.
plt.scatter(x, y, marker='o', c='b', s=3)
plt.xlim(xi.min(), xi.max())
plt.ylim(yi.min(), yi.max())
plt.title('Binned data (%d points)' % npts)
plt.savefig('binned-data.png')

plt.show()
