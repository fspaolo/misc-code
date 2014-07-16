"""
Make regional area-average time series

"""
import sys
import numpy as np
import pandas as pd
import tables as tb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altimpy as ap
from scipy.signal import detrend
from patsy import dmatrix
from sklearn.linear_model import LassoCV

PLOT = True
FILE_IN = 'h_raw4.h5'
DIR = '/Users/fpaolo/data/shelves/' 
#FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
UNITS = 'm/yr'

#plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 11
'''
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.2
plt.rcParams['grid.color'] = '0.5'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 0
#plt.rcParams['ytick.major.size'] = 0
plt.rcParams['xtick.labelsize'] = 12 # 18
#plt.rcParams['ytick.labelsize'] = 12 # 18
#plt.rcParams['xtick.major.pad'] = 6
#plt.rcParams['ytick.major.pad'] = 6
'''



# regression (or residual) sum of squares 
rss = lambda y_true, y_pred, ax=0: np.nansum((y_true - y_pred)**2, axis=ax)
# mean squared error
mse = lambda y_true, y_pred, ax=0: np.nanmean((y_true - y_pred)**2, axis=ax)
# root mean squared error
rmse = lambda y_true, y_pred, ax=0: np.sqrt(mse(y_true, y_pred, ax=ax))


def kfold(X, K, randomise=True):
    """Generates K training-validation pairs from the indices in X.
    
    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.
    
    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    ind = xrange(len(X))
    if randomise: 
        from random import shuffle
        ind = list(ind)
        shuffle(ind)
    for k in xrange(K):
    	i_training = [i for i in ind if i % K != k]
    	i_testing = [i for i in ind if i % K == k]
    	yield i_training, i_testing

def polyfit_kfold(x, y, k=10, deg=2, randomise=False):
    mse_ = [] 
    for i_train, i_test in kfold(x, k, randomise=randomise):
        p = np.polyfit(x[i_train], y[i_train], deg)
        y_pred = np.polyval(p, x[i_test])
        y_true = y[i_test]
        mse_ = np.append(mse_, mse(y_true, y_pred))
    return np.mean(mse_)

def polyfit_select(x, y, cv=10, max_deg=3, randomise=False):
    poly_order = np.arange(1,max_deg+1)
    poly_mse = np.zeros(len(poly_order), 'f8')
    for i, n in enumerate(poly_order):
        poly_mse[i] = polyfit_kfold(x, y, k=cv, deg=n, randomise=randomise)
    return poly_order[poly_mse.argmin()], poly_mse

def polyfit_cv(x, y, cv=10, max_deg=3, return_coef=False, randomise=False):
    n, mse = polyfit_select(x, y, cv=cv, max_deg=max_deg, randomise=randomise)
    p = np.polyfit(x, y, n)
    y_pred = np.polyval(p, x)
    out = y_pred
    if return_coef:
        out = [p, n, mse]
    return out




def gradient(y, dt=0.25):
    return np.gradient(y.values, dt)


shelves = [
    ap.queenmaud,
    ap.fris,
    ap.larsen,
    ap.belling,
    ap.amundsen,
    ap.ris,
    ap.tottmosc,
    ap.westshac,
    ap.ais,
    ]

names = [
    'Queen Maud',
    'Filchner-Ronne',
    'Eastern AP',
    'Bellingshaus.',
    'Amundsen',
    'Ross',
    'Totten+Mosc.',
    'West+Shack.',
    'All Antarct.',
    ]

print 'loading data...'
fin = tb.openFile(DIR + FILE_IN)
try:
    time = ap.num2year(fin.root.time[:])
except:
    time = fin.root.time[:]
lon = fin.root.lon[:]
lat = fin.root.lat[:]
d = fin.root.dh_mean_mixed_const_xcal[:]
#d = fin.root.dg_mean_xcal[:]
#e = fin.root.dh_error_xcal[:]
#d = fin.root.n_ad_xcal[:]
nz, ny, nx = d.shape
dt = ap.year2date(time)

if 0:
    reg = (207, 216, -78, -76)
    d, lon, lat = ap.get_subset(reg, d, lon, lat)
    plt.imshow(d[10], extent=(lon.min(), lon.max(), lat.min(), lat.max()), 
               origin='lower', interpolation='nearest', aspect='auto')
    plt.grid(True)
    plt.show()
    sys.exit()

if 1:
    time, d = ap.time_filt(time, d, from_time=1994, to_time=2013)
    #time, e = time_filt(time, e, from_time=1992, to_time=20013)

# bin-area-average time series
df = pd.DataFrame(index=time)
df2 = pd.DataFrame(index=time)
for k, s in zip(names, shelves):
    shelf, x, y = ap.get_subset(s, d, lon, lat)
    #error, x, y = ap.get_subset(s, e, lon, lat)
    A = ap.get_area_cells(shelf[10], x, y)
    ts, _ = ap.area_weighted_mean(shelf, A)
    #ts2, _ = ap.area_weighted_mean(error, A)
    df[k] = ts
    #df2[k] = ts2
    if 0:
        print k
        plt.imshow(shelf[10], extent=(x.min(), x.max(), y.min(), y.max()), 
                   origin='lower', interpolation='nearest', aspect='auto')
        plt.show()

df.fillna(0, inplace=True)  # leave this!!!

if 0:
    df = df.apply(ap.hp_filt, lamb=7)

if 0:
    df = df.apply(detrend)

if 0:
    df = df.apply(gradient, dt=0.25)

if 1:
    df = df.apply(ap.referenced, to='mean')

ncols = 1
nrows = int(np.ceil(len(df.columns) / float(ncols)))

# plot
fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(6.5,14))
fig.patch.set_facecolor('white')

# calculate linear trend
zeros = np.zeros_like(time)
n = 0
for j in range(ncols):
    for i in range(nrows):

        #------------------------- plot -----------------------------

        print n, k
        k = df.columns[n]
        s = df[k]
        if 0:
            np.savetxt('time.txt', time)
            np.savetxt(k[:3]+'.txt', s)
        m, c = ap.linear_fit_robust(time, s.values, return_coef=True)
        x, y = ap.linear_fit_robust(time, s.values, return_coef=False)
        pol = np.polyfit(time, s.values, 2)
        yy = np.polyval(pol, time)
        #axs[i,j].fill_between(time, s.values+2*r, s.values-2*r, facecolor='0.5',
        #                      edgecolor='w', alpha=0.2)
        #axs[i,j].plot(time, zeros, ':', c='0.5', linewidth=0.5)
        axs[i].plot(time, y, c='0.2', linewidth=0.75, zorder=2)
        axs[i].plot(time, s.values, 's', c='0.5', markersize=4, zorder=1)
        if 0:
            axs[i].plot(time, yy, c='b', linewidth=1.5, zorder=2)
        if 1:
            s = np.asarray(s.values)
            axs[i].plot(time, ap.lasso_cv(time, s, max_deg=3), c='b', linewidth=1.75, zorder=4)
            axs[i].plot(time, polyfit_cv(time, s, cv=10, max_deg=3), c='r', linewidth=1, zorder=5)

        #----------------------- settings ---------------------------

        if i == 0:
            ap.intitle('%s %.2f %s' % (k, m, UNITS), ax=axs[i],  loc=8,
                       pad=-1, borderalpha=0.7)
        else:
            ap.intitle('%s %.2f' % (k, m), ax=axs[i], loc=8,
                       pad=-1, borderalpha=0.7)

        if i != nrows-1:
            ap.adjust_spines(axs[i], ['left'], pad=15)
        else:
            ap.adjust_spines(axs[i], ['left', 'bottom'], pad=15)
            axs[i].set_xticks([1994, 1997, 2000, 2003, 2006, 2009, 2012])
        mn, mx = ap.get_limits(s)
        axs[i].set_yticks([mn, 0, mx])
        axs[i].set_ylim(mn, mx)
        axs[i].set_xlim(1994, 2012)
        axs[i].tick_params(axis='both', direction='out', length=6, width=1,
                        labelsize=12)
        axs[i].set_clip_on(False)
        n += 1

fig.subplots_adjust(left=0.17, right=0.92, bottom=0.01, top=0.95, wspace=0.25, hspace=0.28)
axs[4].set_ylabel('Elevation change (m)', fontsize='large')
fig.autofmt_xdate()
#plt.savefig('g_old_ann.png', dpi=150)
#plt.savefig('backscatter_ts2.png', dpi=150)
plt.show()

#---------------------------------------------------------------------
# save data 
#---------------------------------------------------------------------

if 0:
    print 'saving data...'
    fout = tb.openFile(DIR + FILE_OUT, 'a')
    try:
        fout.createArray('/', 'time', time)
    except:
        pass
    try:
        fout.createArray('/', 'xyz_nodes', xyz)
    except:
        pass
    try:
        fout.createArray('/', 'xx', xx)
        fout.createArray('/', 'yy', yy)
    except:
        pass
    #write_slabs(fout, 'dh_mean_xcal_short_const', data)
    #write_slabs(fout, 'dh_mean_short_const_xcal', data)
    #write_slabs(fout, 'dh_mean_xcal', data)
    #write_slabs(fout, 'dh_error_xcal', data)
    write_slabs(fout, 'nobs_xcal', data)
    fout.flush()
    fout.close()
    fin.close()

    print 'out ->', DIR + FILE_OUT


