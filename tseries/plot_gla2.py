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

# ICESat inter-campaign biases R633 (cm)
# T.Urban
'''
bias = np.asarray([-6.8, -5.2, 0.6, -3.8, -3.4, -0.1, -0.3, -0.2, 
                   -0.9, 1.3, 0.8, -0.5, -1.4, -0.4, 1.6, 5.2, -1.1])
'''
# J.Zwally
bias = np.asarray([-4.01, 5.08, 3.61, -0.31, -6.93, 6.90, 0.86, -0.14, 
                   -3.12, 7.15, 0.63, 3.86, 1.54, -6.85, -8.29, 0, 0])
bias -= bias[0]
bias[0] = 0
bias[np.isnan(bias)] = 0
bias /= 100.


# HDF5
fname1 = sys.argv[1]
fname2 = sys.argv[2]

f1 = tb.openFile(fname1)
time2 = f1.root.table.cols.time2[:]
#dh1 = f1.root.dh_mean[:]
dh1 = f1.root.n_ad[:] + f1.root.n_da[:]

f2 = tb.openFile(fname2)
dh2 = f2.getNode('/', 'dh_mean')[:]
dh1[np.isnan(dh2)] = np.nan
dh2[np.isnan(dh1)] = np.nan

dh1[np.isnan(dh1)] = 0
dhdt = dh1.sum(axis=0)

dh = dh1

'''
fname = sys.argv[1]
f = tb.openFile(fname)
nrows = f.root.table.nrows
satname = f.root.table.cols.satname[:nrows]
time2 = f.root.table.cols.time2[:nrows]
#dh = f.root.dh_mean_corr_short[:nrows]
dh = f.root.dh_mean[:nrows]
'''

x_edges = f1.root.x_edges[:]
y_edges = f1.root.y_edges[:]
lon = f1.root.lon[:]
lat = f1.root.lat[:]

# ross
'''
kk, = np.where(((150 < lon) & (lon < 215)))
# amery
kk, = np.where(((50 < lon) & (lon < 80)))
dh = dh[:,:,kk]
lon = lon[kk]
# ross
jj, = np.where((-76 > lat))
dh = dh[:,jj,:]
lat = lat[jj]
'''

# PARAMETERS -------------------------------------

BIAS = False
ABS_VAL = 0.04  # for trend
width = 1.8  # gaussian width
GRID_CELL = (1, 5)
#BBOX_REG = (-82, -76.2, -26, -79.5)  # FRIS
#BBOX_REG = (-156, -76, 154, -81.4)   # ROSS
#BBOX_REG = (72, -74.2, 68, -67.5)    # AMERY
BBOX_REG = (-134, -58, 50, -57)       # ANTARCTICA
TITLE = ''
SAVEFIG = 'ant3'
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

# remember to take off a piece of the grid (land) !!!!!!!!
'''
dh[:,:,:235] = np.nan  # for FRIS_ALL
dh[:,:,270:] = np.nan  # for FRIS_ALL
dh[:,20:,:] = np.nan  # for FRIS_ALL
dh[:,:11,:] = np.nan  # for FRIS_ALL

dh[:,0:38,:] = np.nan  # for amery_ALL
dh[:,40:,:] = np.nan  # for amery_ALL
dh[:,:,0:58] = np.nan  # for amery_ALL
dh[:,:,59:] = np.nan  # for amery_ALL
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

print 'lon/lat:', lon[GRID_CELL[1]], lat[GRID_CELL[0]]
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
    p_ = m.drawparallels(np.arange(-90.,-60, 6), labels=[0,0,0,1], color='0.8')
    m_ = m.drawmeridians(np.arange(-180,180., 12), labels=[1,0,0,0], color='0.8')
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
    m.plot(x2, y2, 'o', markersize=0.8)
    '''
    ###
    #plt.colorbar().set_label(LABEL)
    #viz.colorbar(fig, CMAP, (-ABS_VAL,ABS_VAL), title='m/yr')
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


def plot_map(lon, lat, grid, bbox, mfile, mres=1, **kw):
    """
    **kw : keyword args
        contourf=[True/False]
        vmin=int
        vmax=int
    """
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    m = viz.make_proj_stere(bbox)
    m = viz.plot_grid_proj(m, lon, lat, grid, shift=True, **kw)
    plt.colorbar(orientation='vertical', shrink=.65)
    x, y, mask = viz.get_gtif_subreg(m, mfile, mres)
    mask = np.ma.masked_values(mask, 4)
    m.imshow(mask, zorder=1, cmap=plt.cm.gray_r)
    #m = plot_boundaries(m, (lon.min(),lon.max(),lat.min()-10,lat.max()+3))
    p_ = m.drawparallels(np.arange(-90.,-60,3), labels=[0,0,0,1], color='0.2')
    m_ = m.drawmeridians(np.arange(-180,180.,5), labels=[1,0,0,0], color='0.2')
    plt.savefig('map.png')
    return fig1 


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
    plt.xlim(2003.5, 2009.6)
    #plt.xlabel('years')
    plt.ylabel('Elevation change (m)')
    fig.autofmt_xdate()
    if savefig is not None:
        plt.savefig(savefig+'_ts.png')
        np.savetxt(savefig+'_ts.txt', np.column_stack((x, y)))
    plt.show()
    return fig


def plot_ts_mean(t, Y):
    y = np.ma.masked_invalid(Y)
    ts = np.mean(np.mean(y, axis=1), axis=1)

    #x_pol, y_pol = spline_interp(t, ts, smooth=0.1)
    x_pol, y_pol = util.poly_fit(t, ts, order=3)
    x_lin, y_lin = linear_fit(t, ts)
    m, c = linear_fit(t, ts, return_coef=True)

    fig = plt.figure()
    ax = fig.add_subplot((111))

    n = y_lin[0]  # shift
    plt.plot(t, ts-n, linewidth=2)
    plt.plot(x_pol, y_pol-n)
    plt.plot(x_lin, y_lin-n)

    viz.add_inner_title(ax, 'linear trend = %.3f cm/yr' % (m*100), loc=2)
    #plt.ylim(-0.1, 0.25)
    plt.xlabel('time')
    plt.ylabel('dh (m)')
    plt.title('Mean TS')
    plt.show()
    return fig


def plot_ts_full(t, Y, (i,j)):
    y = np.ma.masked_invalid(Y)
    mean_t = np.mean(np.mean(y, axis=1), axis=1)
    mean_xy = np.mean(y, axis=0)
    ts = mean_t + mean_xy[i,j] + Y[:,i,j]

    #x_pol, y_pol = spline_interp(t, ts, smooth=0.1)
    x_pol, y_pol = util.poly_fit(t, ts, order=3)
    x_lin, y_lin = linear_fit(t, ts)
    m, c = linear_fit(t, ts, return_coef=True)

    fig = plt.figure()
    ax = fig.add_subplot((111))

    n = y_lin[0]  # shift
    plt.plot(t, ts-n, linewidth=2)
    plt.plot(x_pol, y_pol-n)
    plt.plot(x_lin, y_lin-n)

    viz.add_inner_title(ax, 'linear trend = %.1f cm/yr' % (m*100), loc=2)
    #plt.ylim(ts.min(), ts.max())
    plt.xlabel('time')
    plt.ylabel('dh (m)')
    plt.title('Mean TS')
    plt.show()
    return fig


def plot_grid_and_ts(lon, lat, dhdt, t, dh, (i,j), **kw):
    cell = (lon[j], lon[j+1], lat[i], lat[i+1])
    plot_grid(lon, lat, dhdt, cell, **kw)
    #plot_ts(t, dh[:,i,j])
    ###
    '''
    import netCDF4 as nc
    plon, plat = -67.0, -81.4
    dx, dy = .3, .3
    fname = sys.argv[2]
    fin = nc.Dataset(fname)
    d = fin.variables
    firn = d['zs']
    year = d['time'][:]
    lon = d['lon'][:]
    lat = d['lat'][:]
    i, j = np.where((lon > plon-dx) & (lon < plon+dx) & (lat > plat-dy) & (lat < plat+dy))
    t, = np.where((year > 1992) & (year < 2012))
    print i, j, lon[i,j], lat[i,j]
    plt.plot(year[t], firn[t,i,j], 'r', linewidth=2)
    fin.close()
    '''
    ###
    plt.show()


def regrid(lon, lat, dhdt, factor=10):
    m, n = len(lon), len(lat)
    lon2 = np.linspace(lon[0], lon[-1], m*factor)
    lat2 = np.linspace(lat[0], lat[-1], n*factor)
    xx, yy = np.meshgrid(lon2, lat2)
    dhdt2 = interp(dhdt, lon, lat, xx, yy, order=1)                          # good!!!
    return lon2, lat2, dhdt2

#---------------------------------------------------------------------

t = year

# Gaussian smoothing
#dhdt = gaussian_filter(dhdt, width)

# filter values
#dhdt[np.abs(dhdt)>ABS_VAL] = np.nan
#dhdt[np.isnan(dhdt)] = 0

#lon, lat, dhdt = regrid(lon, lat, dhdt, factor=2)

'''
# plot single time series
for i in range(ny):
    for j in range(nx):
        if np.abs(dhdt[i,j]) < ABS_VAL:
            ii, = np.where(~np.isnan(dh[:,i,j]))
            dh[ii[0],i,j] = np.nan
            plot_grid_and_ts(lon, lat, dhdt, t, dh, (i,j), plot_cell=True, contourf=False)
            plot_ts(t, dh[:,i,j])
            #plot_ts_full(t, dh, (i,j))
            for s in ['ers1', 'ers2', 'envi']:
                k, = np.where(s == satname)
                #dh[dh==0] = np.nan
                plt.plot(t[k], dh[k,i,j], linewidth=2)
        plt.show()
sys.exit()
'''
'''
plt.imshow(dhdt, origin='lower', interpolation='nearest')
plt.show()
'''
#sys.exit()

'''
plot_grid_and_ts(lon, lat, dhdt, t, dh, GRID_CELL, plot_cell=False, contourf=False)
plot_ts_mean(t, dh)
plot_map(lon, lat, dhdt, BBOX_REG, MFILE, mres=1, cmap=cm.jet_r, 
         contourf=False, vmin=-ABS_VAL, vmax=ABS_VAL)
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
