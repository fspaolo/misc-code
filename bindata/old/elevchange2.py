'''
 0 xlon
 1 xlat

 2 orbit_1
 3 orbit_2

 4 utc85_1
 5 utc85_2

 6 elev_1
 7 elev_2

 8 agc_1
 9 agc_2

 10 fmode_1
 11 fmode_2 : tracking mode
     0 = ocean | fine
     1 = ice | medium
     2 = none | coarse

 12 fret_1
 13 fret_2 : waveform was retracked
     0 = no
     1 = yes

 14 fprob_1
 15 fprob_2 : problem retracking    # don't use this for Geosat/GM
     0 = no (passed all the tests)
     1 = yes (didn't pass at least one test)

 16 fmask_1
 17 fmask_2 : MOA mask
     0 = land
     1 = water 
     2 = ice-shelf

 18 fbord_1
 19 fbord_2 : is border
     0 = no
     1 = yes
 
 20 ftrack_1
 21 ftrack_2 : separate tracks
     0 = ascending
     1 = descending
     2 = invalid
 
 22 inc_1
 23 inc_2 : inter-mission bias increment
 
 24 tide_1
 25 tide_2 : tide correction
 
 26 load_1
 27 load_2 : load tide correction
'''

import os
import sys
import re
import numpy as np
import scipy as sp
import tables as tb
import datetime as dt
import matplotlib.pyplot as plt


#---------------------------------------------------------------------
# edit
#---------------------------------------------------------------------

#left, right, bottom, top = -85, -20, -82, -74  # fris
#left, right, bottom, top = -61.2, -50, -77, -76.75  # bin1
#left, right, bottom, top = -74.54, -73.54, -79.15, -78.85  # bin2
left, right, bottom, top = -61.7, -60.4, -78.2, -77.8  # bin3
#left, right, bottom, top = -63.7, -62.4, -78.2, -77.8  # bin4
#left, right, bottom, top = -63.7, -62.4, -79.2, -78.8  # bin5
dx = right - left
dy = top - bottom
absval = 10
single_ts = True
tide = True

x_range = (left, right)
y_range = (bottom, top)

#---------------------------------------------------------------------

class TimeSeries(tb.IsDescription):
    sat = tb.StringCol(10, pos=1)
    t_ref = tb.StringCol(10, pos=2)
    year = tb.Int32Col(pos=3)
    month = tb.Int32Col(pos=4)
    dh = tb.Float64Col(pos=5)
    se_dh = tb.Float64Col(pos=6)
    dg = tb.Float64Col(pos=7)
    se_dg = tb.Float64Col(pos=8)
    n_ad = tb.Int32Col(pos=9)
    n_da = tb.Int32Col(pos=10)


def dh_ad_da(h1, h2, ftrack1, ftrack2, return_index=False):
    dh = h2 - h1                                       # always t2 - t1
    i_ad, = np.where((ftrack2 == 0) & (ftrack1 == 1))  # dh_ad
    i_da, = np.where((ftrack2 == 1) & (ftrack1 == 0))  # dh_da
    if return_index:
        return dh, i_ad, i_da
    else:
        return dh[i_ad], dh[i_da]


def dh_mean_stan_dard(x1, x2, useall=False):
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
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


def dh_mean_weighted(x1, x2, useall=False):
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        mean = (n1*x1.mean() + n2*x2.mean())/(n1 + n2)
    elif useall and n1 > 0:
        mean = x1.mean()
    elif useall and n2 > 0:
        mean = x2.mean()
    else:
        mean = np.nan
    return mean


def se_mean_weighted(x1, x2, useall=False):
    # C.Davis error
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        error = np.sqrt(n1*x1.var() + n2*x2.var())/(n1 + n2)
    elif useall and n1 > 0:
        error = x1.std()/n1 
    elif useall and n2 > 0:
        error = x2.std()/n2
    else:
        error = np.nan
    return error


def se_mean_weighted2(x1, x2, useall=False):
    # D.Wingham error
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        error = np.abs(x1.mean() - x2.mean())
    elif useall and n1 > 0:
        error = x1.std()/n1
    elif useall and n2 > 0:
        error = x2.std()/n2
    else:
        error = np.nan
    return error


def sd_iterative_editing(x, nsd=3, return_index=False):
    niter = 0
    while True: 
        sd = x[~np.isnan(x)].std()  # ignore NaNs
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


def abs_editing(x, absval, return_index=False):
    i, = np.where(np.abs(x) <= absval)
    if return_index:
        return i 
    else:
        return x[i]


def apply_tides(d):
    d['h1'] = d['h1'] - d['tide1'] + d['load1']
    d['h2'] = d['h2'] - d['tide2'] + d['load2']
    return d


def get_time(fname):
    t1, t2 = re.findall('\d\d\d\d\d\d+', fname) 
    y2, m2 = np.int32(t2[:4]), np.int32(t2[4:6])
    return y2, m2


def get_sat_and_ref(fname):
    return os.path.basename(fname).split('_')[:2]


def get_fname_out(files):
    """Min and max time for output file name."""
    times = [re.findall('\d\d\d\d\d\d+', fname) for fname in files]
    t_1 = [t1 for t1, t2 in times]
    t_2 = [t2 for t1, t2 in times]
    t_min, t_max =  min(t_1), max(t_2)
    #times = np.array(times, 'i4').reshape(len(times), 2)
    #t_min, t_max = str(times.min()), str(times.max())
    path, name = os.path.split(files[0])  # path from any file
    sat = name.split('_')[0]
    name = '_'.join([sat, t_min, t_max, 't.h5'])
    return os.path.join(path, name)


def save_tseries(pos, fname, files):
    try:
        fname_next = files[pos+1]  # next file
    except:
        return True                # last file
    name = os.path.basename(fname)
    name_next = os.path.basename(fname_next)
    sat, reftime = name.split('_')[:2]
    sat_next, reftime_next = name_next.split('_')[:2]
    if sat != sat_next or reftime != reftime_next:
        return True
    else:
        return False

#---------------------------------------------------------------------

files = sys.argv[1:]
files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s))

isfirst = True
for pos, fname in enumerate(files):

    f = tb.openFile(fname)
    data = f.root.data

    # 1. FILTER DATA FIRST

    fmode1 = data[:,10]
    fmode2 = data[:,11]
    fbord1 = data[:,18]
    fbord2 = data[:,19]

    #condition = ((fbord1 == 0) & (fbord2 == 0))
    #condition = ((fmode1 == fmode2) & (fbord1 == 0) & (fbord2 == 0)) 
    condition = ((fmode1 == 1) & (fmode2 == 1) & (fbord1 == 0) & (fbord2 == 0)) # ice
    #condition = ((fmode1 == 0) & (fmode2 == 0) & (fbord1 == 0) & (fbord2 == 0)) # fine
    ind, = np.where(condition)

    if len(ind) < 1:
        f.close()
        continue

    d = {}
    d['lon'] = data[:,0]
    d['lat'] = data[:,1]
    d['h1'] = data[:,6]
    d['h2'] = data[:,7]
    d['g1'] = data[:,8]
    d['g2'] = data[:,9]
    d['ftrack1'] = data[:,20]
    d['ftrack2'] = data[:,21]
    if tide:
        d['tide1'] = data[:,24]
        d['tide2'] = data[:,25]
        d['load1'] = data[:,26]
        d['load2'] = data[:,27]

    d['lon'] = d['lon'][ind]
    d['lat'] = d['lat'][ind]
    d['h1'] = d['h1'][ind]
    d['h2'] = d['h2'][ind]
    d['g1'] = d['g1'][ind]
    d['g2'] = d['g2'][ind]
    d['ftrack1'] = d['ftrack1'][ind]
    d['ftrack2'] = d['ftrack2'][ind]
    if tide:
        d['tide1'] = d['tide1'][ind]
        d['tide2'] = d['tide2'][ind]
        d['load1'] = d['load1'][ind]
        d['load2'] = d['load2'][ind]

    # 2. APPLY CORRECTIONS 

    if tide:
        d = apply_tides(d)

    del data, fmode1, fmode2, fbord1, fbord2
    if tide:
        del d['tide1'], d['tide2'], d['load1'], d['load2']

    #-----------------------------------------------------------------
    print 'binning ...'

    # binning
    x_edges = np.arange(x_range[0], x_range[-1]+dx, dx)
    y_edges = np.arange(y_range[0], y_range[-1]+dy, dy)
    j_bins = np.digitize(d['lon'], bins=x_edges)
    i_bins = np.digitize(d['lat'], bins=y_edges)
    nx, ny = len(x_edges)-1, len(y_edges)-1

    # output grids 
    dh_mean = np.empty((ny,nx), 'f8')
    se_mean = dh_mean.copy()
    dg_mean = dh_mean.copy()
    sg_mean = dh_mean.copy()
    n_ad = np.empty((ny,nx), 'i4')
    n_da = n_ad.copy() 

    # calculations per bin
    for i in xrange(ny):
        for j in xrange(nx):
            ind, = np.where((j_bins == j+1) & (i_bins == i+1))
            #---------------------------------------------------------
            # separate --> asc/des, des/asc 
            dh_ad, dh_da = dh_ad_da(d['h1'][ind], d['h2'][ind], 
                           d['ftrack1'][ind], d['ftrack2'][ind])

            # filter
            dh_ad = sd_iterative_editing(dh_ad, nsd=3)
            dh_da = sd_iterative_editing(dh_da, nsd=3)
            dh_ad = abs_editing(dh_ad, absval=absval)
            dh_da = abs_editing(dh_da, absval=absval)

            # mean values
            #dh_mean[i,j] = dh_mean_standard(dh_ad, dh_da, useall=False) 
            dh_mean[i,j] = dh_mean_weighted(dh_ad, dh_da, useall=False) 
            se_mean[i,j] = se_mean_weighted(dh_ad, dh_da, useall=False) 
            ##se_mean[i,j] = se_mean_weighted2(dh_ad, dh_da, useall=False) 
            n_ad[i,j] = len(dh_ad)
            n_da[i,j] = len(dh_da)
            print len(dh_ad), len(dh_da)

            dg_ad, dg_da = dh_ad_da(d['g1'][ind], d['g2'][ind], 
                           d['ftrack1'][ind], d['ftrack2'][ind])
            dg_ad = sd_iterative_editing(dg_ad, nsd=3)
            dg_da = sd_iterative_editing(dg_da, nsd=3)

            dg_mean[i,j] = dh_mean_weighted(dg_ad, dg_da, useall=False) 
            sg_mean[i,j] = se_mean_weighted(dg_ad, dg_da, useall=False) 
            #---------------------------------------------------------

    if len(dh_mean) < 1:
        f.close()
        continue

    # get time for each average dh
    sat, t_ref = get_sat_and_ref(fname)
    year2, month2 = get_time(fname)  

    #----------------------------------------------------------------------

    if isfirst:
        fname_out = get_fname_out(files) 
        title = 'FRIS Time Series'
        filters = tb.Filters(complib='blosc', complevel=9)
        db = tb.openFile(fname_out, 'w')
        g = db.createGroup('/', 'tseries')
        t = db.createTable(g, 'ts', TimeSeries, title, filters)
        isfirst = False

    t.row['sat'] = sat
    t.row['t_ref'] = t_ref
    t.row['year'] = year2
    t.row['month'] = month2
    t.row['dh'] = dh_mean[0,0]
    t.row['se_dh'] = se_mean[0,0]
    t.row['dg'] = dg_mean[0,0]
    t.row['se_dg'] = sg_mean[0,0]
    t.row['n_ad'] = n_ad[0,0]
    t.row['n_da'] = n_da[0,0]
    t.row.append()
    #data = np.rec.array(tuple([sat, t_ref, year2, month2, dh_mean[0,0], 
    #    se_mean[0,0], dg_mean[0,0], sg_mean[0,0], n_ad[0,0], n_da[0,0]]), 
    #    dtype=t.dtype)
    #t.append(data)
    t.flush()

    #------------------------------------------------------------------
    f.close()

year = t.cols.year[:] 
month = t.cols.month[:] 
dh = t.cols.dh[:] 
se_dh = t.cols.se_dh[:] 
dates = [dt.datetime(y, m, 15) for y, m in zip(year, month)]
plt.errorbar(dates, dh, yerr=se_dh, linewidth=2)

print 'done.'

db.flush()
db.close()
plt.show()
print 'file out -->', fname_out
