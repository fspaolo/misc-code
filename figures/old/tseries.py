import sys
import numpy as np
import pandas as pd
import tables as tb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import altimpy as ap
from scipy.signal import detrend

PLOT = True
DIR = '/Users/fpaolo/data/shelves/' 
#FILE_IN = 'all_19920716_20111015_shelf_tide_grids_mts.h5.ice_old'
FILE_IN = 'h_raw3.h5'

#plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 11
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 0.2
plt.rcParams['grid.color'] = '0.5'
#plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['xtick.major.size'] = 0
#plt.rcParams['ytick.major.size'] = 0
plt.rcParams['xtick.labelsize'] = 12 # 18
#plt.rcParams['ytick.labelsize'] = 12 # 18
#plt.rcParams['xtick.major.pad'] = 6
#plt.rcParams['ytick.major.pad'] = 6


shelves = [
        #ap.fimbule,
        #ap.fris,
        #ap.antpen,
        #ap.amundsen,
        ap.ross,
        #ap.amery,
        #ap.ais,
        ap.wais,
        #ap.riiser,
        #ap.larsenc,
        #ap.larsend,
        #ap.ronne,
        #ap.stange,
        #ap.abbot,
    ]
names = [
        #'Fimbul',
        #'Filchner-Ronne',
        #'Antarctic Peninsula',
        #'Amundsen',
        'Ross',
        #'Amery',
        #'AIS',
        'WAIS',
        #'RIISER',
        #'LARSEN C',
        #'LARSEN D',
        #'RONNE',
        #'STANGE',
        #'ABBOT',
    ]
# EAIS -> see below in the code

nrows = 3
ncols = 1

# load data
print 'loading data...'
fin = tb.openFile(DIR + FILE_IN)
try:
    time = ap.num2year(fin.root.time_xcal[:])
except:
    time = fin.root.time[:]
lon = fin.root.lon[:]
lat = fin.root.lat[:]
d = fin.root.dh_mean_mixed_const_xcal[:]
ny, nx, nz = d.shape
dt = ap.year2date(time)

time, d = ap.time_filt(time, d, from_time=1994, to_time=2013)

# area-average time series
df = pd.DataFrame(index=time)
for k, s in zip(names, shelves):
    shelf, x, y = ap.get_subreg(s, d, lon, lat)
    A = ap.get_area_cells(shelf[10], x, y)
    ts, _ = ap.area_weighted_mean(shelf, A)
    df[k] = ts
    plt.imshow(shelf[10], extent=(x.min(), x.max(), y.min(), y.max()), 
               origin='lower', interpolation='nearest', aspect='auto')
    plt.show()

# Add EAST1 and EAST2
shelf1, x1, y = ap.get_subreg(ap.eais1, d, lon, lat)
shelf2, x2, y = ap.get_subreg(ap.eais2, d, lon, lat)
shelf = np.concatenate((shelf2, shelf1), axis=2)
x = np.concatenate((x2, x1), axis=0)
A = ap.get_area_cells(shelf[10], x, y)
ts, ac = ap.area_weighted_mean(shelf, A)
df['EAIS'] = ts
plt.imshow(shelf[10], extent=(x.min(), x.max(), y.min(), y.max()), 
           origin='lower', interpolation='nearest', aspect='auto')
plt.show()


#df = df.apply(ap.hp_filt, lamb=7)
#df = df.apply(detrend)
df = df.apply(ap.referenced, to='mean')

np.savetxt('h_eais_raw.txt', np.column_stack((time, df['EAIS'])), fmt='%.6f')
np.savetxt('h_wais_raw.txt', np.column_stack((time, df['WAIS'])), fmt='%.6f')
exit()

# plot
fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=False, figsize=(7,8))
fig.patch.set_facecolor('white')

zeros = np.empty_like(time) * 0
n = 0
for i in range(nrows):
    k = df.columns[i]
    s = df[k]
    m, c = ap.linear_fit(time, s.values, return_coef=True)
    x, y = ap.linear_fit(time, s.values)
    #axs[i].plot(time, zeros, ':', c='0.5', linewidth=0.5)
    axs[i].plot(x, y, c='0.5', linewidth=0.7)
    axs[i].plot(time, s.values, 'b', linewidth=2)
    ap.intitle(k, ax=axs[i], loc=8)
    ap.intitle('%.2f m/yr' % m, ax=axs[i], loc=4)
    #ap.intitle('%.2f dB/yr' % m, ax=axs[i], loc=5)
    if i != nrows-1:
        ap.adjust_spines(axs[i], ['left'])
    else:
        ap.adjust_spines(axs[i], ['left', 'bottom'])
    axs[i].set_yticks([s.min().round(1), s.max().round(1)])
    #axs[i].set_ylim(s.min().round(1), s.max().round(1))
    #axs[i].set_yticks([-.4, .6])
    #axs[i].set_ylim(-.4, .6)
    n += 1
    if n == len(df): break

midle = len(axs) / 2
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.01, top=0.95, wspace=0.3, hspace=0.25)
axs[0].set_xticks([1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012])
#axs[midle].set_ylabel('Backscatter change (dB)', fontsize='large')
axs[midle].set_ylabel('Elevation change (m)', fontsize='large')
plt.xlim(1992, 2012.2)
fig.autofmt_xdate()
#plt.savefig('elevation_uncor_det_lin_.png', dpi=150)
#plt.savefig('backscatter_ts2.png', dpi=150)
plt.show()

df.plot(legend=True, linewidth=3)
plt.xlim(1992, 2012.5)
plt.ylabel('Elevation change (m)')
#plt.ylabel('Backscatter change (dB)')
plt.show()

#---------------------------------------------------------------------
# save data 
#---------------------------------------------------------------------

if not PLOT:
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
    write_slabs(fout, 'nobs_xcal', data)
    fout.flush()
    fout.close()
    fin.close()

    print 'out ->', DIR + FILE_OUT


