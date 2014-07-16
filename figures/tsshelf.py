"""
Make full-ice-shelf area-average time series

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
FILE_DATA = 'h_postproc.h5'
FILE_RATES = 'h_integrate.csv'
FILE_AREA = 'area_grid_cells.h5'
DIR = '/Users/fpaolo/data/shelves/' 
#FILE_DATA = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_oce'
UNITS = 'cm/yr'

plt.rcParams['font.size'] = 11

shelves = [
    ap.fimbulw,
    ap.fimbule,
    ap.lazarev,
    ap.amery,
    ap.west,
    ap.shackleton,
    ap.totten,
    ap.moscow,
    ap.rosse,
    ap.rossw,
    ap.sulzberger,
    ap.getz,
    ap.dotson,
    ap.crosson,
    ap.thwaites,
    ap.pig,
    ap.abbot,
    ap.stange,
    ap.bach,
    ap.wilkins,
    ap.georges,
    ap.georgen,
    ap.larsenb,
    ap.larsenc,
    ap.larsend,
    ap.ronne,
    ap.filchner,
    ap.brunt,
    ap.riiser,
    ap.ais,
    ]

names = [
    'Fimbul W',
    'Fimbul E',
    'Lazarev',
    'Amery',
    'West',
    'Shackleton',
    'Totten',
    'Moscow',
    'Ross E',
    'Ross W',
    'Sulzberger',
    'Getz',
    'Dotson',
    'Crosson',
    'Thwaites',
    'Pine Island',
    'Abbot',
    'Stange',
    'Bach',
    'Wilkins',
    'George S',
    'George N',
    'Larsen B',
    'Larsen C',
    'Larsen D',
    'Ronne',
    'Filchner',
    'Brunt',
    'Riiser',
    'All Antarctica',
    ]

def gradient(y, dt=0.25):
    return np.gradient(y.values, dt)


def rate(x, y):
    """Average rate of change."""
    return (y[-1] - y[0]) / (x[-1] - x[0])


def rate_err(x, yerr, independent=True):
    """Uncertainty for rate of change."""
    if independent:
        err = (np.sqrt(2) * yerr) / (x[-1] - x[0])  # independent errors
    else:
        err = (2 * yerr) / (x[-1] - x[0])           # dependent errors
    return err


print 'loading data...'
fd = tb.openFile(DIR + FILE_DATA)
time = fd.root.time[:]
lon = fd.root.lon[:]
lat = fd.root.lat[:]
d = fd.root.dh_mean_mixed_const_xcal[:] * 1e2
e = fd.root.dh_error_xcal[:] * 1e2
nz, ny, nx = d.shape
dt = ap.year2date(time)

rates_ = pd.read_csv(DIR + FILE_RATES, index_col=0)
rates = rates_['dhdt_poly(cm/yr)']
rates_err = rates_['dhdt_poly_err(cm/yr)']

fa = tb.open_file(DIR + FILE_AREA)
area = fa.root.area[:]
fa.close()

if 0:
    reg = ap.west
    d, lon, lat = ap.get_subset(reg, d, lon, lat)
    e, lon, lat = ap.get_subset(reg, e, lon, lat)
    plt.figure()
    plt.imshow(d[10], extent=(lon.min(), lon.max(), lat.min(), lat.max()), 
               origin='lower', interpolation='nearest', aspect='auto')
    plt.grid(True)
    plt.show()
    sys.exit()

# area-average time series
df = pd.DataFrame(index=time)
df2 = pd.DataFrame(index=time)

for k, s in zip(names, shelves):
    shelf, x, y = ap.get_subset(s, d, lon, lat)
    error, x, y = ap.get_subset(s, e, lon, lat)
    A, _, _ = ap.get_subset(s, area, lon, lat)
    #A = ap.get_area_cells(shelf[10], x, y)
    ts, _ = ap.area_weighted_mean(shelf, A)
    ts2 = ap.area_weighted_mean_err(error, A)
    df[k] = ts
    df2[k] = ts2
    if 0:
        print k
        plt.imshow(shelf[10], extent=(x.min(), x.max(), y.min(), y.max()), 
                   origin='lower', interpolation='nearest', aspect='auto')
        plt.show()

#df.fillna(0, inplace=True)  # leave this!!!

if 0:
    df = df.apply(ap.hp_filt, lamb=7)

if 0:
    df = df.apply(detrend)

if 0:
    df = df.apply(gradient, dt=0.25)

if 1:
    df = df.apply(ap.referenced, to='mean')

ncols = 3
nrows = int(np.ceil(len(df.columns) / float(ncols)))

# plot
fig, axs = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(16.5,14))
fig.patch.set_facecolor('white')

# calculate trends
zeros = np.zeros_like(time)
n = 0
for j in range(ncols):
    for i in range(nrows):

        #------------------------- plot -----------------------------

        print n, k
        k = df.columns[n]
        s = df[k]
        err = df2[k]
        m, c = ap.linear_fit(time, s.values, return_coef=True)
        x, y = ap.linear_fit(time, s.values, return_coef=False)
        axs[i,j].plot(time, y, c='0.2', linewidth=0.75, zorder=1)
        axs[i,j].plot(time, s.values, 's', c='0.5', markersize=4, clip_on=False, zorder=3)
        #axs[i,j].errorbar(time, s.values, yerr=3*err, fmt='s', c='0.5', markersize=4, zorder=1)
        if 1:
            # poly lasso 
            poly = ap.lasso_cv(time, s.values, cv=10, max_deg=3)
            poly_err = ap.gse(poly, s)
            poly_rate = (rate(time, poly)).round(1)
            poly_rate_err = (rate_err(time, poly_err, independent=False) * 3).round(1)
            axs[i,j].plot(time, poly, c='b', linewidth=1.5, zorder=2)
            #axs[i,j].errorbar(time, poly, yerr=poly_err, c='b', linewidth=1.75, zorder=4)
        if 0:
            # poly lstsq
            axs[i,j].plot(time, ap.lstsq_cv(time, s.values, cv=10, max_deg=3,
                          randomise=True), c='r', linewidth=1.5)

        #----------------------- settings ---------------------------

        if i == 0:
            ap.intitle('%s %.1f$\pm$%.1f %s' % (k, rates[k], rates_err[k], UNITS),
                       ax=axs[i,j],  loc=8, pad=-1, borderalpha=0.8)
            '''
            ap.intitle('%s %.1f$\pm$%.1f %s' % (k, poly_rate, poly_rate_err, UNITS),
                       ax=axs[i,j],  loc=8, pad=-1, borderalpha=0.8)
            '''
        else:
            ap.intitle('%s %.1f$\pm$%.1f' % (k, rates[k], rates_err[k]),
                       ax=axs[i,j], loc=8, pad=-1, borderalpha=0.8)
            '''
            ap.intitle('%s %.1f$\pm$%.1f' % (k, poly_rate, poly_rate_err),
                       ax=axs[i,j], loc=8, pad=-1, borderalpha=0.8)
            '''

        if i != nrows-1:
            ap.adjust_spines(axs[i,j], ['left'], pad=15)
        else:
            ap.adjust_spines(axs[i,j], ['left', 'bottom'], pad=15)
            axs[i,j].set_xticks([1994, 1997, 2000, 2003, 2006, 2009, 2012])
        mn, mx = ap.get_limits(s)
        mn, mx = int(mn), int(mx) #<<<<<<<<<<<<<<<<<<<<<<<< for cm
        axs[i,j].set_yticks([mn, 0, mx])
        axs[i,j].set_ylim(mn, mx)
        axs[i,j].set_xlim(1994, 2012)
        axs[i,j].tick_params(axis='both', direction='out', length=6, width=1,
                        labelsize=12)
        n += 1

fig.subplots_adjust(left=0.07, right=0.95, bottom=0.01, top=0.95, wspace=0.25, hspace=0.28)
axs[5,0].set_ylabel('Elevation change (cm)', fontsize='large', labelpad=13)
fig.autofmt_xdate()
plt.savefig('Sup1_ts_shelves_v3.png', dpi=150, bbox_inches='tight')
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
    fd.close()

    print 'out ->', DIR + FILE_OUT


