"""
Module containing functions and classes used by:

xover2grid.py
xover2box.py
x2grid.py

Notes
-----
- all SDs are calculated using N-1

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 14, 2011

import os
import sys
import re
import numpy as np
import scipy as sp
import tables as tb
import datetime as dt
import matplotlib.pyplot as plt

try:
    import altimpy as apy
except:
    pass

# Biases: subtract this correction from the data (dh) 

# Tim Urban [m]
BIAS_U = {'20031027': -0.068, # 2a   
          '20040305': -0.052, # 2b 
          '20040604': 0.006,  # 2c
          '20041021': -0.038, # 3a
          '20050307': -0.034, # 3b
          '20050606': -0.001, # 3c 
          '20051107': -0.003, # 3d
          '20060311': -0.002, # 3e
          '20060610': -0.009, # 3f
          '20061111': 0.013,  # 3g
          '20070329': 0.008,  # 3h
          '20071019': -0.005, # 3i
          '20080305': -0.014, # 3j
          '20081012': -0.004, # 3k
          '20081206': 0.016,  # 2d
          '20090326': 0.052,  # 2e
          '20091006': -0.011} # 2f

# Jay Zwally [m]
BIAS_Z = {'20031027': -0.0401, # 2a   
          '20040305': 0.0508,  # 2b 
          '20040604': 0.0361,  # 2c
          '20041021': -0.0031, # 3a
          '20050307': -0.0693, # 3b
          '20050606': 0.0690,  # 3c 
          '20051107': 0.0086,  # 3d
          '20060311': -0.0014, # 3e
          '20060610': -0.0312, # 3f
          '20061111': 0.0715,  # 3g
          '20070329': 0.0063,  # 3h
          '20071019': 0.0386,  # 3i
          '20080305': 0.0154,  # 3j
          '20081012': -0.0685, # 3k
          '20081206': -0.0829, # 2d
          '20090326': 0,       # 2e
          '20091006': 0}       # 2f

# Adrian Borsa (asc, des) [m]
BIAS_B = {'20031027': (-0.0125, 0.025), # 2a   
          '20040305': (0.051, 0.011),   # 2b 
          '20040604': (0.0142, 0.057),  # 2c
          '20041021': (0.011, 0.063),   # 3a
          '20050307': (0.067, 0),       # 3b
          '20050606': (0.051, 0.071),   # 3c 
          '20051107': (0.076, 0.032),   # 3d
          '20060311': (0.099, -0.019),  # 3e
          '20060610': (-0.041, 0.093),  # 3f
          '20061111': (0.017, -0.048),  # 3g
          '20070329': (-0.048, 0.0120), # 3h
          '20071019': (0.059, 0.042),   # 3i
          '20080305': (0.087, 0.041),   # 3j
          '20081012': (0, 0.0132),      # 3k
          '20081206': (0.057, 0),       # 2d
          '20090326': (0.093, -0.002),  # 2e
          '20091006': (0, 0)}           # 2f


class OutGrids(object):
    def __init__(self, ny, nx):
        self.dh_mean = np.zeros((ny,nx), 'f8') * np.nan
        self.dh_error = np.zeros_like(self.dh_mean) * np.nan
        self.dh_error2 = np.zeros_like(self.dh_mean) * np.nan
        self.dg_mean = np.zeros_like(self.dh_mean) * np.nan
        self.dg_error = np.zeros_like(self.dh_mean) * np.nan
        self.dg_error2 = np.zeros_like(self.dh_mean) * np.nan
        self.n_ad = np.zeros_like(self.dh_mean) * np.nan
        self.n_da = np.zeros_like(self.dh_mean) * np.nan


class OutputContainers(object):
    def __init__(self, fname_out, shape):
        f = tb.openFile(fname_out, 'w')
        nt, ny, nx = shape
        filters = tb.Filters(complib='zlib', complevel=9)
        atom = tb.Atom.from_type('float64', dflt=np.nan)  # dflt is important!
        self.time1 = f.createCArray('/', 'time1', atom, (nt,), '', filters)
        self.time2 = f.createCArray('/', 'time2', atom, (nt,), '', filters)
        self.lon = f.createCArray('/', 'lon', atom, (nx,), '', filters)
        self.lat = f.createCArray('/', 'lat', atom, (ny,), '', filters)
        self.x_edges = f.createCArray('/', 'x_edges', atom, (nx+1,), '', filters)
        self.y_edges = f.createCArray('/', 'y_edges', atom, (ny+1,), '', filters)
        self.dh_mean = f.createCArray('/', 'dh_mean', atom, (nt,ny,nx), '', filters)
        self.dh_error = f.createCArray('/', 'dh_error', atom, (nt,ny,nx), '', filters)
        self.dh_error2 = f.createCArray('/', 'dh_error2', atom, (nt,ny,nx), '', filters)
        self.dg_mean = f.createCArray('/', 'dg_mean', atom, (nt,ny,nx), '', filters)
        self.dg_error = f.createCArray('/', 'dg_error', atom, (nt,ny,nx), '', filters)
        self.dg_error2 = f.createCArray('/', 'dg_error2', atom, (nt,ny,nx), '', filters)
        self.n_ad = f.createCArray('/', 'n_ad', atom, (nt,ny,nx), '', filters)
        self.n_da = f.createCArray('/', 'n_da', atom, (nt,ny,nx), '', filters)
        self.file = f


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


def get_data(fname, tide_code='fortran'):
    """Load data from file (with datamatrix)."""
    f = tb.openFile(fname)
    data = f.root.data[:]         # in-memory -> faster!
    d = {}
    d['lon'] = data[:,0]
    d['lat'] = data[:,1]
    d['h1'] = data[:,6]
    d['h2'] = data[:,7]
    d['g1'] = data[:,8]
    d['g2'] = data[:,9]
    d['fmode1'] = data[:,10]
    d['fmode2'] = data[:,11]
    d['fmask1'] = data[:,16]
    d['fmask2'] = data[:,17]
    d['fbord1'] = data[:,18]
    d['fbord2'] = data[:,19]
    d['ftrk1'] = data[:,22]
    d['ftrk2'] = data[:,23]
    d['tide1'] = data[:,24]       # tide + load
    d['tide2'] = data[:,25]
    if tide_code == 'matlab':
        d['tide1'] += data[:,26]  # add load
        d['tide2'] += data[:,27]
    f.close()
    del data
    return d


def get_info(fname):
    """Get 'satname', 'time1', 'time2' from file name."""
    satname = os.path.basename(fname).split('_')[0]
    time1, time2 = re.findall('\d\d\d\d\d\d+', fname) 
    return satname, np.int32(time1), np.int32(time2)


def filter_data(d, ice_only=True):
    # d is a dictionary containing all the data
    # mode: 0/1/2 = ocean/ice/none (for all RA), fine/medium/coarse (for Envi)
    # border: 0/1 = no/yes
    # tide: -9999 = outside boundary or land
    cond_a = (d['fmode1'] == 0) & (d['fmode2'] == 0)  # ocean-mode
    cond_b = (d['fmode1'] == 1) & (d['fmode2'] == 1)  # ice-mode
    if ice_only:
        cond1 = cond_b
    else:
        cond1 = (cond_a | cond_b)

    cond2 = (d['fmode1'] == 0) & (d['fmode2'] == 0)  # fine-mode
    cond3 = (d['fbord1'] == 0) & (d['fbord2'] == 0)
    cond4 = (d['tide1'] != -9999) & (d['tide2'] != -9999)

    if d['satname'] in ['ers1', 'ers2']:
        ind, = np.where( cond1 & cond3 & cond4 )
    elif d['satname'] in ['envi']:
        ind, = np.where( cond2 & cond3 & cond4 )
    else:
        print "'satname' neither 'ers-1/2', nor 'envi'!"
        ind = []
    if len(ind) < 1:
        d = None                    # no points left
    else:
        for k in d.keys():
            if np.ndim(d[k]) > 0:
                d[k] = d[k][ind]  # filter
                #print 'V filtering', k
            else:
                #print 'X not filtering', k
                pass
    return d


def apply_campaign_bias(d, source=None):
    # substract biases!
    if source == 'urban':
        d['h1'] -= BIAS_U[d['camp1']]
        d['h2'] -= BIAS_U[d['camp2']]
    elif source == 'zwally':
        d['h1'] -= BIAS_Z[d['camp1']]
        d['h2'] -= BIAS_Z[d['camp2']]
    elif source == 'borsa':
        iasc1, = np.where(d['ftrk1'] == 0)
        ides1, = np.where(d['ftrk1'] == 1)
        iasc2, = np.where(d['ftrk2'] == 0)
        ides2, = np.where(d['ftrk2'] == 1)
        d['h1'][iasc1] -= BIAS_B[d['camp1']][0]
        d['h1'][ides1] -= BIAS_B[d['camp1']][1]
        d['h2'][iasc2] -= BIAS_B[d['camp2']][0]
        d['h2'][ides2] -= BIAS_B[d['camp2']][1]
    else:
        pass
    return d


def digitize(lon, lat, x_range, y_range, dx, dy):
    """Digitize lons and lats."""
    x_edges = np.arange(x_range[0], x_range[-1]+dx, dx)
    y_edges = np.arange(y_range[0], y_range[-1]+dy, dy)
    j_bins = np.digitize(lon, bins=x_edges)
    i_bins = np.digitize(lat, bins=y_edges)
    nx, ny = len(x_edges)-1, len(y_edges)-1
    hx, hy = dx/2., dy/2.
    lon_d = (x_edges + hx)[:-1]
    lat_d = (y_edges + hy)[:-1]
    return (lon_d, lat_d, j_bins, i_bins, x_edges, y_edges, nx, ny)


def compute_dh_ad_da(h1, h2, ftrk1, ftrk2, return_index=False):
    '''
    Take arrays `h1` and `h2` and compute `dh_ad` and `dh_da`,
    '''
    dh = h2 - h1                                       # always t2 - t1 !
    i_ad, = np.where((ftrk2 == 0) & (ftrk1 == 1))  # dh_ad
    i_da, = np.where((ftrk2 == 1) & (ftrk1 == 0))  # dh_da
    if return_index:
        return [dh, i_ad, i_da]
    else:
        return [dh[i_ad], dh[i_da]]


def where_ad_da(ftrk1, ftrk2):
    '''
    Find tracks asc/des and des/asc.
    '''
    i_ad, = np.where((ftrk2 == 0) & (ftrk1 == 1))  # dh_ad
    i_da, = np.where((ftrk2 == 1) & (ftrk1 == 0))  # dh_da
    return [i_ad, i_da]


def compute_ordinary_mean(x1, x2, useall=False):
    '''
    Mean of the average values of the arrays `x1` and `x2`: 
    (<x1> + <x2>)/2.

    If `useall=True` returns <x1>|<x2> as the mean value if one 
    of the arrays is empty, otherwise returns NaN.
    '''
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        mean = (x1.mean() + x2.mean())/2.
    elif useall and n1 > 0:
        mean = x1.mean()
    elif useall and n2 > 0:
        mean = x2.mean()
    else:
        mean = np.nan
    return mean 


def compute_weighted_mean(x1, x2, useall=False, median=False):
    '''
    Weighted mean of the average/median values of arrays `x1` and `x2`: 
    (n1 <x1> + n2 <x2>)/(n1 + n2)

    If `useall=True` returns <x1>|<x2> as the mean value if one 
    of the arrays is empty, otherwise returns NaN.
    '''
    x1 = x1[~np.isnan(x1)]  # exclude NaNs
    x2 = x2[~np.isnan(x2)]
    n1, n2 = len(x1), len(x2)
    if median: 
        if n1 > 0: 
            x1_m = np.median(x1)  # robust
        else:
            x1_m = np.nan
        if n2 > 0: 
            x2_m = np.median(x2)
        else:
            x2_m = np.nan
    else: 
        if n1 > 0:
            x1_m = np.mean(x1)    # non robust
        else:
            x1_m = np.nan
        if n2 > 0:
            x2_m = np.mean(x2)
        else:
            x2_m = np.nan
    if n1 > 0 and n2 > 0:
        mean = (n1*x1_m + n2*x2_m)/(n1 + n2)
    elif useall and n1 > 0:
        mean = x1_m
    elif useall and n2 > 0:
        mean = x2_m
    else:
        mean = np.nan
    return mean


def compute_weighted_error(x1, x2, useall=False):
    """
    Weighted mean *standard error* for the stimated <dh> according 
    Davis et. al. (2001, 2004, 2006).

    Notes
    -----
    If `useall=True` returns 'se1' or 'se2' as the mean error if one 
    of the arrays is empty, otherwise returns NaN.

    If only one obs is available, 10% of its absolute value is used as error.

    se = sd / sqrt(N) = sqrt(var/N), N = sum(ni)
    wi = ni / sum(ni)
    se = sqrt(n1**2 * se1**2 + n2**2 * se2**2) / (n1 + n2)
       = sqrt(n1 * sd1**2 + n2 * sd2**2) / (n1 + n2)

    """
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    n1, n2 = len(x1), len(x2)
    sd1, sd2 = x1.std(ddof=1), x2.std(ddof=1)
    if n1 == 1:
        sd1 = 0.1 * np.abs(x1[0])
    if n2 == 1:
        sd2 = 0.1 * np.abs(x2[0])
    # estimate error
    if n1 > 0 and n2 > 0:
        se = np.sqrt(n1 * sd1**2 + n2 * sd2**2) / (n1 + n2)
    elif useall and n1 > 0:
        se = sd1 / np.sqrt(n1)
    elif useall and n2 > 0:
        se = sd2 / np.sqrt(n2)
    else:
        se = np.nan
    return se


def compute_wingham_error(x1, x2, useall=False):
    """
    Error for the stimated <dh> *modified* from Wingham et. al. (2009). 

    Whingham (2009) uses the *variance* of the error-TS formed (as 
    single error for the resulting TS). Here the individual 
    *differences* are used as errors for individual points.

    Notes
    -----
    If `useall=True` returns 'std1' or 'std2' as the error if one 
    of the arrays is empty, otherwise returns NaN.

    If only one obs is available, 10% of its absolute value is used as error.

    This method does not take into account the number of observations.

    """
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    n1, n2 = len(x1), len(x2)
    sd1, sd2 = x1.std(ddof=1), x2.std(ddof=1)
    if n1 == 1:
        sd1 = 0.1 * np.abs(x1[0])
    if n2 == 1:
        sd2 = 0.1 * np.abs(x2[0])
    # estimate error
    if n1 > 0 and n2 > 0:
        err = np.abs(x1.mean() - x2.mean())
    elif useall and n1 > 0:
        err = sd1
    elif useall and n2 > 0:
        err = sd2
    else:
        err = np.nan
    return err


def compute_num_obs(x1, x2, useall=False):
    n1, n2 = len(x1), len(x2)
    if (n1 > 0 and n2 > 0):
        nobs = [n1, n2] 
    elif useall and (n1 > 0 or n2 > 0):
        nobs = [n1, n2] 
    else:
        nobs = [np.nan, np.nan] 
    return nobs


def _std_editing_iterative(x, nsd=3, return_index=False):
    """
    Iterative filtering: all values greater than <nsd>-sigmas
    (standard deviation) till convergence.
    """
    niter = 0
    while True: 
        y = x[~np.isnan(x)]   # ignore NaNs
        if len(y) < 3: break  # min o 3 to calc std
        sd = y.std(ddof=1)    
        i, = np.where(np.abs(x) > nsd*sd)
        if len(i) > 0:
            x[i] = np.nan
            niter += 1
        else:
            break
    if return_index:
        return np.where(~np.isnan(x))[0]
    else:
        return x[~np.isnan(x)]


def _std_editing(x, nsd=3, return_index=False):
    """
    Filter out all values greater than `nsd`-sigma (standar deviation).
    """
    x = x[~np.isnan(x)]  # ignore NaNs
    if len(x) > 2:       # min of 3 to calc std
        sd = x.std(ddof=1)
        ind, = np.where(np.abs(x) <= nsd*sd)
    else:
        ind = range(len(x))
        #ind = np.argmin(np.abs(x)) # return the value closer to zero
    if return_index:
        return ind 
    else:
        return x[ind]


def std_editing(x, nsd=3, return_index=False, iterative=False):
    """
    Filter out all values greater than `nsd`-sigma.
    """
    if iterative:
        return _std_editing_iterative(x, nsd, return_index)
    else:
        return _std_editing(x, nsd, return_index)


def abs_editing(x, absval, return_index=False):
    """
    Filter out all values greater than `absval`.
    """
    i, = np.where(np.abs(x) <= absval)
    if return_index:
        return i 
    else:
        return x[i]


def gaussian_filter(grid, width):
    """Gaussian smoothing."""
    ii = np.where(np.isnan(grid)) # | (grid == 0))
    grid[ii] = 0
    grid = sp.ndimage.gaussian_filter(grid, width, order=0, 
        output=None, mode='reflect', cval=0.0)
    grid[ii] = np.nan
    return grid


def need_to_save_now(pos, fname, files):
    """
    Find when a TS is formed and need to be saved. 

    A TS is formed when (one of the followings):
    1) current file is the last one
    2) current satname is different from next satname
    3) current time1 is different from next time1
    """
    try:
        fname_next = files[pos+1]  # next file
    except:
        return True                # last file
    name = os.path.basename(fname)
    name_next = os.path.basename(fname_next)
    satname, time1 = name.split('_')[:2]
    sat_next, time1_next = name_next.split('_')[:2]
    if satname != sat_next or time1 != time1_next:
        return True
    else:
        return False


# warnning: it is switched --> y, x !!!
def bin_by_mean(lon, lat, z, bins=10, range=None):
    bins = bins[::-1] 
    range = range[::-1]
    wsum, _ = np.histogramdd((lat, lon), weights=z, bins=bins, range=range)
    ppbin, edges = np.histogramdd((lat, lon), bins=bins, range=range) 
    #ppbin[ppbin==0] = np.nan
    #ppbin = np.ma.masked_equal(ppbin, 0)
    return (wsum/ppbin), ppbin, edges[1], edges[0]


def get_sub_domains(l, r, n):
    ds = (r - l)/n
    sdom = {}
    for i, lon in enumerate(np.arange(l, r, ds)):
        sdom[i+1] = (lon, lon+ds)
    return sdom


def get_sect_num(fname):
    num = re.findall('_\d\d_', fname)[0]
    return int(num.strip('_'))


def print_info(x_edges, y_edges, lon, lat, dx, dy, N, n_ad, n_da, source):
    print '-'*55
    print 'grid edges: l/r/b/t = %.2f/%.2f/%.2f/%.2f deg' \
        % (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    print 'grid centers: l/r/b/t = %.2f/%.2f/%.2f/%.2f deg' \
        % (lon[0], lon[-1], lat[0], lat[-1])
    print 'grid spacing: dx x dy = %.2f x %.2f deg' % (dx, dy)
    print 'grid size: nx x ny = %d x %d (x %d grids)' \
        % (len(lon), len(lat), N)
    nad, nda = np.sum(n_ad[~np.isnan(n_ad)]), np.sum(n_da[~np.isnan(n_da)])
    print 'total crossovers used: %d (ad) + %d (da) = %d' % (nad, nda, nad+nda)
    print 'inter-campaign biases:', source
    print '-'*55


def plot_ts(table):
    """
    Plot dh and dAGC time series and the correlation dAGC x dh.
    """
    sys.path.append('/Users/fpaolo/code/misc')
    from util import poly_fit
    # load data from Table
    time2 = table.cols.time2[:] 
    month = table.cols.month[:] 
    dh_mean = table.cols.dh_mean[:] 
    dh_error = table.cols.dh_error[:] 
    dg_mean = table.cols.dg_mean[:] 
    dg_error = table.cols.dg_error[:] 
    dates = [dt.datetime(y, m, 15) for y, m in zip(time2, month)]
    # plot TS
    fig = plt.figure()
    plt.subplot(211)
    plt.errorbar(dates, dh_mean, yerr=dh_error, linewidth=2)
    plt.ylabel('dh (m)')
    plt.subplot(212)
    plt.errorbar(dates, dg_mean, yerr=dg_error, linewidth=2)
    plt.ylabel('dAGC (dB)')
    fig.autofmt_xdate()
    # plot correlation
    dg_fit, dh_fit, _ = poly_fit(dg_mean, dh_mean)
    plt.figure()
    plt.plot(dg_mean, dh_mean, 'o')
    plt.plot(dg_fit, dh_fit, linewidth=2.5)
    plt.xlabel('dAGC (dB)')
    plt.ylabel('dh (m)')
    corr = np.corrcoef(dg_mean, dh_mean)[0,1]
    print 'correlation = %.2f' % corr


def plot_grids(x, y, g1, g2, g3, g4):
    """
    Plot the 4 calculated grids: dh, dAGC, n_ad, n_da.

    Notes
    -----
    `pcolor` cannot have NaN, use `masked array` instead.
    `pcolor` does not preserve the lon,lat aspect ratio like `imshow`.
    """
    try:
        apy.rcparams()
    except:
        pass
    #sys.path.append('/Users/fpaolo/code/misc')
    #import viz
    #cmap = viz.colormap('rgb')
    g1 = np.ma.masked_invalid(g1)
    g2 = np.ma.masked_invalid(g2)
    g3 = np.ma.masked_invalid(g3)
    g4 = np.ma.masked_invalid(g4)
    x = np.ma.masked_invalid(x)
    y = np.ma.masked_invalid(y)
    xx, yy = np.meshgrid(x, y)
    fig = plt.figure()
    plt.subplot(211)
    plt.pcolor(xx, yy, g1)
    plt.colorbar()
    plt.subplot(212)
    plt.pcolor(xx, yy, g2)
    plt.colorbar()
    #viz.colorbar(fig, cmap, (-2,2))
    fig = plt.figure()
    plt.subplot(211)
    plt.pcolor(xx, yy, g3, cmap=plt.cm.copper_r)
    plt.colorbar()
    plt.subplot(212)
    plt.pcolor(xx, yy, g4, cmap=plt.cm.copper_r)
    plt.colorbar()


