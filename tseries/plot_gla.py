import sys
import numpy as np
import tables as tb
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as img
from mpl_toolkits.basemap import interp

sys.path.append('/Users/fpaolo/code/misc')
import util
import viz

# input
#---------------------------------------------------------------------

MFILE = '/Users/fpaolo/data/masks/scripps_antarctica_masks/scripps_antarctica_mask1km_v1.tif'
fname2 = '/Users/fpaolo/data/tmp/raxla/ice_20031027_20091006_grids4_all_n_mean.h5'
#fname2 = '/Users/fpaolo/data/tmp/raxla/envi_20031027_20091006_grids4_all_mean.h5'

# HDF5
fname1 = sys.argv[1]

f1 = tb.openFile(fname1)
time2 = f1.root.table.cols.time2[:]
#dh1 = f1.root.dh_mean_corr_short[:]
dh1 = f1.root.dh_mean[:]

f2 = tb.openFile(fname2)
dh2 = f2.getNode('/', 'dh_mean')[:]
dh1[np.isnan(dh2)] = np.nan
dh2[np.isnan(dh1)] = np.nan

dh = dh1

x_edges = f1.root.x_edges[:]
y_edges = f1.root.y_edges[:]
lon = f1.root.lon[:]
lat = f1.root.lat[:]

# ross
kk, = np.where(((150 < lon) & (lon < 215)))
'''
# amery
kk, = np.where(((50 < lon) & (lon < 80)))
'''
# ross
jj, = np.where((-76 > lat))
dh = dh[:,jj,:]
lat = lat[jj]
try:
    dh = dh[:,:,kk]
    lon = lon[kk]
except:
    pass

# PARAMETERS -------------------------------------

BIAS = False
ABS_VAL = 0.06  # for trend (ice=0.05, envi=0.06)
width = 1.5  # gaussian width (both=1.5)
GRID_CELL = (1, 5)
#BBOX_REG = (-82, -76.2, -26, -79.5)  # FRIS
BBOX_REG = (-156, -76, 153, -81.6)   # ROSS
#BBOX_REG = (72, -74.2, 68, -67.5)    # AMERY
#BBOX_REG = (-134, -58, 50, -57)       # ANTARCTICA
TITLE = ''
SAVEFIG = 'ross1'
#CMAP = cm.jet_r # cm.gist_rainbow #cm.spectral_r # cm.RdBu_r # cm.jet_r
CMAP = viz.colormap('rgb')
LABEL = 'Elevation change rate (m/yr)'

#-------------------------------------------------

days = np.asarray([dt.datetime.strptime(str(t), '%Y%m%d').toordinal() for t in time2])
year = np.round(days/365., 2) - .33
print year

'''
k, = np.where((year > 2006.1) & (year < 2012.5))
year = year[k]
dh = dh[k,...]
'''

title = 'FRIS 1992-2012\n(ERS1, ERS2, ENVI)'
legend = 'Elevation change rate m/yr'
ticklabels = '0', '4', '8 m/yr'
ticks = -ABS_VAL, 0, ABS_VAL 
colormap = [
    (0, 0.001, 2, 3), # value
    (1, 1, 1, 1), # red
    (0, 0, 1, 1), # green
    (0, 0, 0, 1), # blue
    (0, 1, 1, 1), # alpha
]
colorexp = 1.0
colorlim = ticks[0], ticks[-1]
inches = 6.4, 3.6
extent = (-121.1, -113.8), (32.1, 35.5)
ylim = 150.0
xlim = ylim * inches[0] / inches[1]
axis = -xlim, xlim, -ylim, ylim

#---------------------------------------------------------------------

def linear_fit(x, y, return_coef=False):
    """
    Fit a straight-line by Ordinary Least Squares.

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    x, y = np.asarray(x), np.asarray(y)
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y))) # & (y!=0))) #<<<<<<<<<< check the 0!!!!!
    if len(ind) < 2:
        return (np.nan, np.nan)
    x, y = x[ind], y[ind]
    A = np.ones((len(x), 2))
    A[:,0] = x
    m, c = np.linalg.lstsq(A, y)[0]
    if return_coef:
        return (m, c)
    else:
        x_pol = np.linspace(x.min(), x.max(), 200)
        y_pol = m*x_pol + c
        return (x_pol, y_pol)


def gaussian_filter(grid, width):
    """Gaussian smoothing."""
    k = np.where((np.isnan(grid)) | (grid == 0))
    grid[k] = 0
    grid = img.gaussian_filter(grid, width, order=0, output=None, mode='reflect', cval=0.0)
    grid[k] = np.nan
    return grid


def filter_npts(X, npts=10):
    _, ny, nx = X.shape
    for i in range(ny):
        for j in range(nx):
            ii, = np.where(~np.isnan(X[:,i,j]))
            if len(ii) < npts:
                X[:,i,j] = np.nan
    return X


def filter_abs(X, abs_val=2):
    X[np.abs(X)>abs_val] = np.nan
    return X


def area_grid_cells(X, x_edges, y_edges):
    """
    X is a 2D grid.
    """
    deg2rad = lambda x: (x*np.pi)/180.
    R = 6378    # Earth's radius in m
    c = (np.pi/180) * R**2
    A = np.empty_like(X)
    ny, nx = X.shape
    for i in range(ny):
        for j in range(nx):
            lat1, lat2 = deg2rad(y_edges[i]), deg2rad(y_edges[i+1])
            lon1, lon2 = deg2rad(x_edges[j]), deg2rad(x_edges[j+1])
            A[i,j] = c * np.abs(np.sin(lat1) - np.sin(lat2)) * np.abs(lon1 - lon2)
    '''
    plt.imshow(A, origin='lower')
    plt.colorbar()
    plt.show()
    sys.exit()
    '''
    return A


def area_weighted_mean_ts(X, A):
    """
    X is 3D with one ts at every grid-cell
    A is 2D with area of every grid-cell
    """
    nt, _, _ = X.shape
    ts = np.zeros(nt, 'f8')
    #X[X==0] = np.nan  # <<<<<<<<<<<<<<<<<<<<<??????????????????
    X = np.ma.masked_invalid(X)
    for k in range(nt):
        G = X[k,...]
        W = A.copy()
        W[np.isnan(G)] = 0
        s = W.sum()
        if s != 0:
            W /= s 
        else:
            W[:] = 0 
        ts[k] = np.sum(W*G)  # area-weighted average
    '''
    plt.imshow(G, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.show()
    sys.exit()
    '''
    return ts


def plot_grid(lon, lat, dhdt, cell=None, plot_cell=False, contourf=True, savefig=None):
    fig = plt.figure()
    m = viz.make_proj_stere(BBOX_REG)
    plon, plat = lon[GRID_CELL[1]], lat[GRID_CELL[0]]
    # shift grid half degree
    lon -= (lon[1] - lon[0])/2.
    lat -= (lat[1] - lat[0])/2.
    lon, lat = np.meshgrid(lon, lat)
    xx, yy = m(lon, lat)
    x, y = m(plon, plat)
    print 'point cell (lon lat x y):', plon, plat, x, y
    dhdt = np.ma.masked_invalid(dhdt)
    m.pcolormesh(xx, yy, dhdt, cmap=CMAP, vmin=-ABS_VAL, vmax=ABS_VAL, alpha=1)
    p_ = m.drawparallels(np.arange(-90.,-60, 2), labels=[1,0,0,0], color='0.3')
    m_ = m.drawmeridians(np.arange(-180,180., 15), labels=[0,0,1,0], color='0.3')
    #m.plot(x, y, 'mo', markersize=8)
    if plot_cell:
        lon, lat = util.box(cell)
        x, y = m(lon, lat)
        m.plot(x, y, 'k', linewidth=2)
    if contourf:
        fig2 = plt.figure()
        m.contourf(xx, yy, dhdt, 25, cmap=CMAP)
    ###
    '''
    x2, y2 = np.loadtxt('/Users/fpaolo/data/masks/scripps/scripps_iceshelves_v1_geod.txt', 
                        usecols=(0,1), unpack=True, comments='%')
    x2, y2 = m(x2, y2)
    m.plot(x2, y2, 'o', markersize=1.2)
    '''
    ###
    #plt.colorbar().set_label(LABEL)
    rect = 0.25, 0.11, 0.5, 0.02
    viz.colorbar(fig, CMAP, (-ABS_VAL,ABS_VAL), title='Elevation change rate (m/yr)',
                 rect=rect)
    '''
    w = 50.0 / (axis[1] - axis[0])
    rect = 0.142 - w, 0.08, 2 * w, 0.02
    cmap = CMAP
    viz.colorbar(fig, cmap, colorlim, legend, rect, ticks, ticklabels, 
        size=7, weight='bold', color='k', edgecolor='w')
    leg = fig.add_axes([0, 0, 1, 1])
    leg.set_axis_off()
    viz.text(leg, 0.70, 0.90, title, ha='center', va='top', size=10,
        weight='bold', color='k', edgecolor='w')
    '''
    if savefig is not None:
        plt.savefig(savefig+'_map.png')
    return fig


def plot_ts(x, y, title='', savefig=None):
    x_pol, y_pol = util.poly_fit(x, y, order=3)
    x_lin, y_lin = linear_fit(x, y)
    m, c = linear_fit(x, y, return_coef=True)
    
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot((111))

    x = x.round(2)
    #n = y_lin[0]  # zero at the begining 
    n = y_lin[0] + (y_lin[-1] - y_lin[0])/2.  # zero at the middle 
    plt.plot(x, y-n, linewidth=3)
    #plt.plot(x_pol, y_pol-n)
    plt.plot(x_lin, y_lin-n, 'r')

    viz.add_inner_title(ax, 'dh/dt = %.1f cm/yr' % (m*100), loc=1)
    viz.add_inner_title(ax, title, loc=2)
    plt.ylim(-.3, .3)
    plt.xlim(2003.5, 2010.1)
    #plt.xlabel('years')
    plt.ylabel('Elevation change (m)')
    #fig.autofmt_xdate()
    if savefig is not None:
        pass
        #plt.savefig(savefig+'_ts.png')
        #np.savetxt(savefig+'_ts.txt', np.column_stack((x, y)))
    return fig


def regrid(lon, lat, dhdt, factor=10):
    m, n = len(lon), len(lat)
    lon2 = np.linspace(lon[0], lon[-1], m*factor)
    lat2 = np.linspace(lat[0], lat[-1], n*factor)
    xx, yy = np.meshgrid(lon2, lat2)
    dhdt2 = interp(dhdt, lon, lat, xx, yy, order=1)                          # good!!!
    return lon2, lat2, dhdt2

#---------------------------------------------------------------------

t = year

dh = filter_npts(dh, npts=4)
#dh = filter_abs(dh, abs_val=2)
A = area_grid_cells(dh[0,...], x_edges, y_edges)
ts = area_weighted_mean_ts(dh, A)

### linear dh/dt

nt = len(year)
_, ny, nx = dh.shape
dhdt = np.zeros((ny, nx), 'f8') * np.nan
for i in range(ny):
    for j in range(nx):
        ii, = np.where(~np.isnan(dh[:,i,j]))
        if len(ii) < 10:
            continue
        # DIFFERENCE
        # TREND
        m, c = linear_fit(t, dh[:,i,j], return_coef=True)
        dhdt[i,j] = m
        if m == 0: 
            continue
        #tt, y = spline_interp(t, dh[:,i,j])
        #tt, y = poly_pol(t, dh[:,i,j], order=2)
        #plt.plot(tt, y, 'k')
        #plt.plot(t, dh0[:,i,j], 'g', linewidth=2)
        #plt.plot(t, dh[:,i,j], 'b', linewidth=2)
        #plt.plot(t, (m*t+c), 'b')
        #plt.plot(t, dh0[:,i,j], 'g')
        #plt.show()

# Gaussian smoothing
#dhdt = gaussian_filter(dhdt, width)

# filter values
#dhdt[np.abs(dhdt)>ABS_VAL] = np.nan
#dhdt[np.isnan(dhdt)] = 0

#lon, lat, dhdt = regrid(lon, lat, dhdt, factor=2)

dhdt = gaussian_filter(dhdt, width)

'''
plt.imshow(dhdt, origin='lower', interpolation='nearest')
plt.show()
'''
plot_ts(t, ts, title=TITLE, savefig=SAVEFIG)
plot_grid(lon, lat, dhdt, contourf=False, savefig=SAVEFIG)
'''
lon2, lat2, dhdt2 = regrid(lon, lat, dhdt, factor=2)
plot_grid(lon2, lat2, dhdt2)
'''
plt.show()

for fid in tb.file._open_files.values():
    fid.close()
