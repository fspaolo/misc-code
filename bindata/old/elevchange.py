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
import numpy as np
import scipy as sp
import tables as tb
import matplotlib.pyplot as plt

files = sys.argv[1:]

def dh_ad_da(h1, h2, ftrack1, ftrack2, inds=False):
    dh = h2 - h1                             # always t2 - t1
    i_ad, = np.where((ftrack2 == 0) & (ftrack1 == 1))  # dh_ad
    i_da, = np.where((ftrack2 == 1) & (ftrack1 == 0))  # dh_da
    if inds:
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
    x1 = np.ma.masked_invalid(x1)
    x2 = np.ma.masked_invalid(x2)
    n1, n2 = len(x1), len(x2)
    if n1 > 0 and n2 > 0:
        error = np.sqrt(n1*x1.var() + n2*x2.var())/(n1 + n2) # <<<<< verify this!!!
    elif useall and n1 > 0:
        error = x1.std()/n1                                  # <<<<< verify this!!!
    elif useall and n2 > 0:
        error = x2.std()/n2
    else:
        error = np.nan
    return error


def sd_iterative_editing(x, nsd=3, inds=False):
    niter = 0
    while True: 
        sd = x[~np.isnan(x)].std()  # ignore NaNs
        i, = np.where(np.abs(x) > nsd*sd)
        if len(i) > 0:
            x[i] = np.nan
            niter += 1
        else:
            #print 'iterations:', niter
            break
    if inds:
        return np.where(~np.isnan(x))[0]
    else:
        return x[~np.isnan(x)]


def abs_editing(x, absval, inds=False):
    i, = np.where(np.abs(x) <= absval)
    if inds:
        return i 
    else:
        return x[i]


# warnning: y, x !!!
def bin_by_mean(lon, lat, z, bins=10, range=None):
    bins = bins[::-1] 
    range = range[::-1]
    wsum, _ = np.histogramdd((lat, lon), weights=z, bins=bins, range=range)
    ppbin, edges = np.histogramdd((lat, lon), bins=bins, range=range) 
    #ppbin[ppbin==0] = np.nan
    #ppbin = np.ma.masked_equal(ppbin, 0)
    return (wsum/ppbin), ppbin, edges[1], edges[0]

#----------------------------------------------------------------

for fname in files:

    f = tb.openFile(fname)
    data = f.root.data

    # 1. FILTER DATA FIRST
    # - retracking --> done!
    # - problem retracking --> done!
    # - equal modes
    # - exclude border

    fmode1 = data[:,10]
    fmode2 = data[:,11]
    fbord1 = data[:,18]
    fbord2 = data[:,19]

    ind, = np.where( \
            #    (fbord1 == 0) & (fbord2 == 0) )
            (fmode1 == fmode2) & (fbord1 == 0) & (fbord2 == 0) )

    if len(ind) < 2:
        continue

    lon = data[ind,0]
    lat = data[ind,1]
    h1 = data[ind,6]
    h2 = data[ind,7]
    g1 = data[ind,8]
    g2 = data[ind,9]
    ftrack1 = data[ind,20]
    ftrack2 = data[ind,21]
    tide1 = data[ind,24]
    tide2 = data[ind,25]
    load1 = data[ind,26]
    load2 = data[ind,27]

    # 2. APPLY CORRECTIONS 
    # - tide
    # - load tide

    h1 = h1 - tide1 + load1
    h2 = h2 - tide2 + load2

    #-----------------------------------------------------------------
    # edit
    #-----------------------------------------------------------------

    ext = '_1.0x0.3.h5'
    #x_range = (-100, -95)
    #y_range = (-72.5, -71)
    x_range = (lon.min(), lon.max())
    y_range = (lat.min(), lat.max())
    dx = 1.0
    dy = 1/3.
    ABSVAL = 5 

    #-----------------------------------------------------------------

    # binning
    x_edges = np.arange(x_range[0], x_range[-1]+dx, dx)
    y_edges = np.arange(y_range[0], y_range[-1]+dy, dy)
    j_bins = np.digitize(lon, bins=x_edges)
    i_bins = np.digitize(lat, bins=y_edges)
    nx, ny = len(x_edges)-1, len(y_edges)-1

    print 'REGION X:', x_range[0], x_edges[0], x_range[-1], x_edges[-1] 
    print 'REGION Y:', y_range[0], y_edges[0], y_range[-1], y_edges[-1]

    # output grids 
    dh_mean = np.empty((ny,nx), 'f8')
    se_mean = np.empty((ny,nx), 'f8')
    dg_mean = np.empty((ny,nx), 'f8')
    sg_mean = np.empty((ny,nx), 'f8')
    n_ad = np.empty((ny,nx), 'i4')
    n_da = np.empty((ny,nx), 'i4')

    # calculations per bin
    for i in xrange(ny):
        for j in xrange(nx):
            #ind, = np.where((x_edges[j] <= lon) & (lon < x_edges[j+1]) & \
            #                 (y_edges[i] <= lat) & (lat < y_edges[i+1]))
            ind, = np.where((j_bins == j+1) & (i_bins == i+1))
            #=========================================================
            dh_ad, dh_da = dh_ad_da(h1[ind], h2[ind], ftrack1[ind], ftrack2[ind])
            dh_ad = sd_iterative_editing(dh_ad, nsd=3)
            dh_da = sd_iterative_editing(dh_da, nsd=3)
            dh_ad = abs_editing(dh_ad, absval=ABSVAL)
            dh_da = abs_editing(dh_da, absval=ABSVAL)

            #dh_mean[i,j] = dh_mean_standard(dh_ad, dh_da, useall=False) 
            dh_mean[i,j] = dh_mean_weighted(dh_ad, dh_da, useall=True) 
            se_mean[i,j] = se_mean_weighted(dh_ad, dh_da, useall=True) 
            n_ad[i,j] = len(dh_ad)
            n_da[i,j] = len(dh_da)

            dg_ad, dg_da = dh_ad_da(g1[ind], g2[ind], ftrack1[ind], ftrack2[ind])
            dg_ad = sd_iterative_editing(dg_ad, nsd=3)
            dg_da = sd_iterative_editing(dg_da, nsd=3)

            dg_mean[i,j] = dh_mean_weighted(dg_ad, dg_da, useall=False) 
            sg_mean[i,j] = se_mean_weighted(dg_ad, dg_da, useall=False) 
            #=========================================================

    '''
    dh_mean2, _dh_mean2, x_edges, y_edges = \
        bin_by_mean(lon, lat, x, (nx,ny), (x_range,y_range))
    '''

    #------------------------------------------------------------

    # 3. store results in 2D dh_means

    #filters = tb.Filters(complib='blosc', complevel=9)
    #atom = tb.Atom.from_dtype(data.dtype)
    #shape = (dg_mean.shape)
    fout = tb.openFile(os.path.splitext(fname)[0] + ext, 'w')
    g = fout.createGroup('/', 'data')
    a1 = fout.createArray(g, 'lon', x_edges)
    a2 = fout.createArray(g, 'lat', y_edges)
    a3 = fout.createArray(g, 'dh_mean', dh_mean)
    a4 = fout.createArray(g, 'n_ad', n_ad)
    a5 = fout.createArray(g, 'n_da', n_da)
    #dout = fout.createCArray(fout.root,'data', atom=atom, shape=shape,
    #                         filters=filters)
    #dout[:,:-2] = data[:] 
    #dout[:,-2:] = flags[:] 
    fout.close()

    f.close()

    #dh.append(dh_mean)
    #se.append(se_mean)

#----------------------------------------------------------------
'''
t = np.arange(len(dh))
t2 = np.linspace(t[0], t[-1], 100)
coeff = np.polyfit(t, dh, 8)
yfit = np.polyval(coeff, t2)

plt.errorbar(t, dh, yerr=se, linewidth=1)
plt.plot(t2, yfit, linewidth=2)
'''
#print np.alltrue(np.nan_to_num(dh_mean) == np.nan_to_num(dh_mean2))
extent = [x_range[0], x_range[-1], y_range[0], y_range[-1]]
plt.figure(1)
plt.subplot(211)
fig1 = plt.imshow(n_ad, extent=extent, origin='lower', interpolation='nearest')
plt.colorbar()
plt.subplot(212)
fig2 = plt.imshow(n_da, extent=extent, origin='lower', interpolation='nearest')
plt.colorbar()
plt.figure(2)
fig2 = plt.imshow(dh_mean, extent=extent, origin='lower', interpolation='nearest')
plt.colorbar()
plt.show()
