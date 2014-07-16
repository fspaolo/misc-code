import numpy as np
import pylab as pl
import tables as tb
import matplotlib.mlab as ml 
import bindata as bn
import sys
from scipy.interpolate import griddata, interp2d, Rbf, \
                              bisplrep, bisplev, RectBivariateSpline
from scipy.ndimage import gaussian_filter, median_filter, \
                          laplace, generic_filter, rotate, map_coordinates
from mpl_toolkits.basemap import interp


def get_xyz(fname, xcol=0, ycol=1, zcol=3):
    try:
        # load data from HDF5 file
        f = tb.openFile(fname)
        d = f.root.data.read()
        f.close()
    except:
        d = np.loadtxt(fname)
    return d[:,xcol], d[:,ycol], d[:,zcol]


def make_xyz(npts=200):
    """Make up some randomly distributed data.
    """
    np.random.seed(1234)
    x = np.random.uniform(0, 1, npts)
    y = np.random.uniform(0, 1, npts)
    n = np.random.uniform(-.05, .05, npts) # noise
    #z = x*np.exp(-x**2 - y**2) + n
    z = x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
    return x, y, z


def make_grid(x, y, nx, ny):
    """Define the grid.
    """
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xx, yy = np.meshgrid(xi, yi)
    return xi, yi, xx, yy


def grid2pts(xx, yy, zz):
    """Convert from grid (2D) to array (1D) w/o NaNs.
    """
    x, y, z = xx.ravel(), yy.ravel(), zz.ravel()
    i = ~np.isnan(z)
    return x[i], y[i], z[i]


def plot(x, y, xi, yi, zz, title='gridded data'):
    """Contour the gridded data.
    """
    CS = pl.contour(xi, yi, zz,  15, linewidths=0.5, colors='k')
    CS = pl.contourf(xi, yi, zz, 15, cmap=pl.cm.jet)
    pl.colorbar() 
    # plot data points
    pl.scatter(x, y, marker='o', c='b', s=5)
    # labels
    incx = (x.max() - x.min())/20.
    incy = (y.max() - y.min())/20.
    pl.xlim(x.min()-incx, x.max()+incx)
    pl.ylim(y.min()-incy, y.max()+incy)
    pl.title('%s (%d points)' % (title, x.shape[0]))
    pl.show()


def main():
    npts, nx, ny = 200, 30, 5 
    
    ### make or get data
    if len(sys.argv) == 1:
        x, y, z = make_xyz(npts)
    else:
        x, y, z = get_xyz(sys.argv[1], xcol=0, ycol=1, zcol=7)
    
    # make grid
    #----------------------------------------------------------------
    xi, yi, xx, yy = make_grid(x, y, nx, ny)
    
    ### binning
    #zz = bn.bindata(x, y, z, xi, yi, method='mean')
    #zz = np.ma.getdata(zz)
    #x, y, z = grid2pts(xx, yy, zz)
    #xi, yi, _, _ = make_grid(x, y, 100, 100)

    #zz, yi, xi = np.histogram2d(y, x, bins=[4, 10])

    #pl.hexbin(x, y, gridsize=20)
    
    ### gridding scattered pts
    #----------------------------------------------------------------
    zz = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear') # good!!!
    
    #zz = ml.griddata(x, y, z, xi, yi, interp='nn')                     # good!!!

    #func = interp2d(x, y, z, kind='linear')
    #zz = func(xi, yi)
    
    #rbf = Rbf(x, y, z, epsilon=3)
    #zz = rbf(xx, yy)
    
    #tck = bisplrep(x, y, z)
    #zz = bisplev(xi, yi, tck)
    
    ### filtering
    #----------------------------------------------------------------
    #zz = gaussian_filter(zz, .2, order=0)
    #zz = median_filter(zz, 5)
    #zz = laplace(zz)
    #f = lambda x: np.median(x)
    #zz = generic_filter(zz, f, size=10)
    
    ### regrindding (resample grid)
    #----------------------------------------------------------------
    #_, _, xx, yy = make_grid(x, y, 100, 100)
    zz = interp(zz, xi, yi, xx, yy, order=1)                          # good!!!

    #rbs = RectBivariateSpline(xi, yi, zz, kx=3, ky=3)
    #xi, yi, _, _ = make_grid(x, y, 100, 100)
    #zz = rbs(xi, yi)
    

    #plot(x, y, xi, yi, zz)
    #bn.plotbins(xi, yi, zz)
    pl.imshow(zz, extent=[xi[0], xi[-1], yi[0], yi[-1]], 
              origin='lower', interpolation='nearest')
    #pl.colorbar()
    pl.show()

if __name__ == '__main__':
    main()
