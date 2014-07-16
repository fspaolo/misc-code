import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import griddata2

npr = np.random
npts = 3000.                            # the total number of data points.
x = npr.normal(size=npts)            # create some normally distributed dependent data in x.
y = npr.normal(size=npts)            # ... do the same for y.
zorig = x**2 + y**2                      # z is a function of the form z = f(x, y).
noise = npr.normal(scale=1.0, size=npts) # add a good amount of noise
z = zorig + noise                    # z = f(x, y) = x**2 + y**2

dx = 0.5
dy = 0.5
xi = np.arange(x.min(), x.max()+dx, dx)
yi = np.arange(y.min(), y.max()+dy, dy)
xx, yy = np.meshgrid(xi, yi)

# plot some profiles / cross-sections for some visualization.  our
# function is a symmetric, upward opening paraboloid z = x**2 + y**2.
# We expect it to be symmetric about and and y, attain a minimum on
# the origin and display minor Gaussian noise.

plt.ion()   # pyplot interactive mode on

# x vs z cross-section.  notice the noise.
plt.plot(x, z, '.')
plt.title('X vs Z=F(X,Y=constant)')
plt.xlabel('X')
plt.ylabel('Z')

# y vs z cross-section.  notice the noise.
plt.plot(y, z, '.')
plt.title('Y vs Z=F(Y,X=constant)')
plt.xlabel('Y')
plt.ylabel('Z')

# now show the dependent data (x vs y).  we could represent the z data
# as a third axis by either a 3d plot or contour plot, but we need to
# grid it first....
plt.plot(x, y, '.')
plt.title('X vs Y')
plt.xlabel('X')
plt.ylabel('Y')

# enter the gridding.  imagine drawing a symmetrical grid over the
# plot above.  the binsize is the width and height of one of the grid
# cells, or bins in units of x and y.
binsize = 0.5
grid, bins, binloc = griddata2.griddata(x, y, z, binsize=binsize)  # see this routine's docstring


# minimum values for colorbar. filter our nans which are in the grid
zmin    = grid[np.where(np.isnan(grid) == False)].min()
zmax    = grid[np.where(np.isnan(grid) == False)].max()

# plot the results.  first plot is x, y vs z, where z is a filled level plot.
extent = (x.min(), x.max(), y.min(), y.max()) # extent of the plot
tlt.figure()
plt.imshow(grid, extent=extent, cmap=cm.Spectral_r, origin='lower', interpolation='nearest')
#plt.pcolor(xi, yi, grid, vmin=zmin, vmax=zmax, cmap='Spectral_r')
#plt.plot(xx, yy, 'k.')
#plt.plot(x, y, 'k.')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Z = F(X, Y)')
plt.colorbar()

# minimum values for colorbar. filter our nans which are in the grid
zmin    = bins[np.where(np.isnan(bins) == False)].min()
zmax    = bins[np.where(np.isnan(bins) == False)].max()
bins[bins==0] = np.nan

# now show the number of points in each bin.  since the independent data are
# Gaussian distributed, we expect a 2D Gaussian.
plt.figure()
plt.imshow(bins, cmap=cm.copper_r, origin='lower', aspect='auto', interpolation='nearest')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('X, Y vs The No. of Pts Per Bin')
plt.colorbar()

plt.show()
