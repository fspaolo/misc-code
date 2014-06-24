#!/urs/bin/env python

import os
import sys
import numpy as np
import tables as tb
import netCDF4 as nc
import scipy.ndimage as img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import interp

sys.path.append('/Users/fpaolo/code/misc')
import util
import viz

# parameters
#---------------------------------------------------------------------

MAX_DH = 3  # for difference (dh)
MIN_NPTS = 6
FROM_YEAR, TO_YEAR = 1991, 2013 

#---------------------------------------------------------------------

fname = sys.argv[1]

def get_dh_data(fname):
    d = {}
    f = tb.openFile(fname)
    nrows = f.root.table_all.nrows
    d['sat_name'] = f.root.table_all.cols.sat_name[:nrows]
    d['year'] = f.root.table_all.cols.year[:nrows]
    d['dh'] = f.root.dh_mean_corr_all[:nrows]
    d['x_edges'] = f.root.x_edges[:]
    d['y_edges'] = f.root.y_edges[:]
    d['lon'] = f.root.lon[:]
    d['lat'] = f.root.lat[:]
    return d


def filter_time(year, dh, from_year=1992, to_year=2012):
    k, = np.where((year > from_year) & (year < to_year))
    year = year[k]
    dh = dh[k,...]
    return [year, dh]


def linear_fit(x, y, return_coef=False):
    """
    Fit a straight-line by Ordinary Least Squares.

    If `return_coef=True` returns the slope (m) and intercept (c).
    """
    ind, = np.where((~np.isnan(x)) & (~np.isnan(y) & (y!=0)))
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


def filter_npts(X, npts=6):
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
    X[X==0] = np.nan
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
    return ts


def get_ts(dh, x_edges, y_edges, MAX_DH, MIN_NPTS):
    dh = filter_abs(dh, abs_val=MAX_DH)
    dh = filter_npts(dh, npts=MIN_NPTS)
    A = area_grid_cells(dh[0,...], x_edges, y_edges)
    ts = area_weighted_mean_ts(dh, A)
    ts[1] = np.nan
    return ts


def get_subreg(dh, inds):
    dh2 = dh.copy()
    dh2[:,:,:inds[0]] = np.nan
    dh2[:,:,inds[1]:] = np.nan
    dh2[:,:inds[2],:] = np.nan
    dh2[:,inds[3]:,:] = np.nan
    return dh2

#---------------------------------------------------------------------

def plot_ts(ax, x, y, num=1):
    x_pol, y_pol = util.poly_fit(x, y, order=3)
    x_lin, y_lin = linear_fit(x, y)
    m, c = linear_fit(x, y, return_coef=True)
    x = x.round(2)
    n = y_lin[0]    # reference to zero
    ax.plot(x, y-n, linewidth=2.5)
    #ax.plot(x_pol, y_pol-n)
    ax.plot(x_lin, y_lin-n, 'r')
    if num != 7:
        viz.add_inner_title(ax, '(%d)' % num, loc=2)
    viz.add_inner_title(ax, 'trend = %.1f cm/yr' % (m*100), loc=3)
    return ax

def create_subplots(fig, m=6, n=1):
    ax = {}
    k = m*n
    for i in range(1, k+1):
        iplot = m*100 + n*10 + i
        ax[i] = fig.add_subplot(iplot, axisbg='#FFF8DC')
        ax[i].xaxis.set_ticks(range(1992, 2014, 2))
        ax[i].yaxis.set_ticks([-0.4, -0.2, 0.0, 0.2, 0.4])
        plt.xlim(1991.5, 2012.5)
        plt.ylim(-.4, .4)
        plt.ylabel('dh (m)')
        if i == 1:
            plt.title('Area-averaged dh (white boxes)')
    return ax

#---------------------------------------------------------------------

d = get_dh_data(fname)
dh = d['dh']
year = d['year']
lon = d['lon']
lat = d['lat']
x_edges = d['x_edges']
y_edges = d['y_edges']

# boxes
#---------------------------------------------------------------------

k = {}    # grid indices
'''
k[1] = (23,35,9,20)    # Ronne FZ west
k[2] = (23,35,1,9)     # Ronne GL west
k[3] = (35,51,13,23)   # Ronne FZ east
k[4] = (35,51,1,9)     # Ronne GL east
k[5] = (52,70,9,12)     # Filchner FZ
k[6] = (52,70,3,9)      # Filchner GL
'''

k[1] = (24,35,9,18)    # Ronne FZ west
k[2] = (24,35,2,9)     # Ronne GL west
k[3] = (35,50,9,23)   # Ronne FZ east
k[4] = (35,50,1,9)     # Ronne GL east
k[5] = (55,69,9,13)     # Filchner FZ
k[6] = (55,69,3,9)      # Filchner GL

fig1 = plt.figure(figsize=(9,18))

ax = create_subplots(fig1, 6, 1)

#---------------------------------------------------------------------

j = 1
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 2
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 3
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 4
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 5
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

j = 6
dh2 = get_subreg(dh, k[j])
ts = get_ts(dh2, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)

#---------------------------------------------------------------------

'''
j = 7
ts = get_ts(dh, x_edges, y_edges, MAX_DH, MIN_NPTS)
plot_ts(ax[j], year, ts, num=j)
viz.add_inner_title(ax[j], 'whole ice shelf', loc=2)
'''

#---------------------------------------------------------------------


fig1.autofmt_xdate()
plt.savefig('ts_fris.png', dpi=150, bbox_inches='tight')
#os.system('cp ts_fris.pdf /Users/fpaolo/posters/scar12/figures/')

plt.show()

for h5f in tb.file._open_files.values():
    h5f.close()
