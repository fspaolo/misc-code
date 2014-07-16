doc = """ 
  Grid irregularly spaced data on bins containing the median values.
""" 
""" 
 Fernando Paolo <fpaolo@ucsd.edu>
 October 22, 2010
"""

import numpy as np
import tables as tb
import argparse as ap
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.mlab import griddata

# command line arguments
'''
parser = ap.ArgumentParser(description=doc)
group = parser.add_mutually_exclusive_group()
parser.add_argument('file', nargs='+', help='HDF5 file[s] to read')
parser.add_argument('-x', dest='loncol', type=int, default=3, 
		    help='column of utc85 data (0,1..) [default 1]')
parser.add_argument('-y', dest='loncol', type=int, default=3, 
		    help='column of utc85 data (0,1..) [default 1]')
group.add_argument('-m', dest='month', action='store_const', const=True,
                    help='separate by month [default by year]')
group.add_argument('-w', dest='m', nargs=2, type=int,
                    help='separate by month-window (M=1..12)')
parser.add_argument('-d', dest='dir', action='store_const', const=True,
		    default=False, help='create directories [default no]')

args =  parser.parse_args()
files = args.file
col = args.col
month = args.month
window = args.m
dir = args.dir
'''

#fname = '/Users/fpaolo/data/altim/ERS2/fris/1995/x2sys_xy.txt'
#data = np.loadtxt(fname)
#data = np.load('geosat_all.npy')
#np.save('geosat_all.npy', data)

# make up some randomly distributed data
#x = uniform(-2,2,npts)
#y = uniform(-2,2,npts)
#z = x*np.exp(-x**2-y**2)
'''
x = data[:,3]
y = data[:,2]
z = data[:,4]
ind, = np.where((-5 <= z) & (z <= 5))
#ind, = np.where(((-78+360) <= x) & (x <= (-58+360)) & (-73 <= y) & (y <= -50))
x = x[ind]
y = y[ind]
z = z[ind]
'''

npr = np.random
npts = 3000.                            # the total number of data points.
x = npr.normal(size=npts)            # create some normally distributed dependent data in x.
y = npr.normal(size=npts)            # ... do the same for y.
zorig = x**2 - y**2                    # z is a function of the form z = f(x, y).
noise = npr.normal(scale=1.0, size=npts) # add a good amount of noise
z = zorig + noise                    # z = f(x, y) = x**2 + y**2

# define grid.
dx = 0.5
dy = 0.5
xi = np.arange(x.min(), x.max()+dx, dx)
yi = np.arange(y.min(), y.max()+dy, dy)
xx, yy = np.meshgrid(xi, yi)

# grid the data.
#grid = griddata(x,y,z,xi,yi)

ncol = xi.shape[0]
nrow = yi.shape[0]
grid = np.empty((nrow,ncol), 'f8') * np.nan
ni = np.empty((nrow,ncol), 'f8') * np.nan
for row in np.arange(nrow-1):
    for col in np.arange(ncol-1):
        ind, = np.where((xi[col] <= x) & (x < xi[col+1]) & \
                        (yi[row] <= y) & (y < yi[row+1]))
        n = len(ind)
        if n > 0:
            grid[row,col] = np.median(z[ind])
            ni[row,col] = n

###

plt.figure()
# minimum values for colorbar. filter our nans which are in the grid
zmin = grid[np.where(np.isnan(grid) == False)].min()
zmax = grid[np.where(np.isnan(grid) == False)].max()
# contour the gridded data, plotting dots at the randomly spaced data points.
#CS = plt.contour(xi,yi,grid,15,linewidths=0.5,colors='k')
#CS = plt.contourf(grid, 15, cmap=plt.cm.jet)
plt.pcolor(xi, yi, grid, vmin=zmin, vmax=zmax, cmap='Spectral_r')
plt.plot(x, y, 'k.')
plt.colorbar()   # draw colorbar

plt.figure()
zmin = bins[np.where(np.isnan(bins) == False)].min()
zmax = bins[np.where(np.isnan(bins) == False)].max()
bins[bins==0] = np.nan
plt.pcolor(bins, vmin=zmin, vmax=zmax, cmap='copper_r')
plt.colorbar()   # draw colorbar

plt.show()
