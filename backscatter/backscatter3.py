#!/usr/bin/env python
"""
FOR INDEPENDENT NON-CALIBRATED TIME SERIES !!!

Applies the backscatter correction to a set of time series.

Implements the correction using changes in AGC, following Zwally, et al. (2005)
and Yi, et al. (2011). See docstrings in function backscatter_corr() for 
details in the algorithm.

Notes
-----
ERS-1: 1992/07-1996/04
ERS-2: 1995/07-2003/04
Envi:  2002/10-2011/11

Example
-------
$ python backscatter3.py /data/alt/ra/ers1/hdf/antarctica/xovers/ers1_*_mts.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# February 7, 2012

import sys
import numpy as np
import tables as tb
import altimpy as ap
import matplotlib.pyplot as plt
from funcs import plot_rs, plot_ts, plot_map

#-------------------------------------------------------------------------
# global variables
#-------------------------------------------------------------------------

PLOT_TS = False    # correlation, R(t), and sensitivity, S(t), included
PLOT_MAP = False
SAVE_TO_FILE = True
DIFF = False       # False/True = absolute values (mixed)/differenced series (short)

# correct each satellite independently
T_NAME = 'time'
H_NAME = 'dh_mean'
G_NAME = 'dg_mean'
R_NAME = 'corrcoef_mixed_const'
S_NAME = 'corrgrad_mixed_const'
SAVE_AS_NAME = 'dh_mean_mixed_const'

#BBOX = (-82, -76.2, -26, -79.5)  # FRIS
BBOX = (-156, -76, 154, -81.2)   # ROSS
#BBOX = (72, -74.2, 68, -67.5)   # AMERY

# for plotting
MASK_FILE = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.tif'

#-------------------------------------------------------------------------
# map to database -> struct
#-------------------------------------------------------------------------

class GetData(object):
    def __init__(self, fname, mode='r'):
        fin = tb.openFile(fname, mode)
        self.file = fin
        self.satname = fin.getNode('/satname')
        self.time = fin.getNode('/time')
        self.lon = fin.getNode('/lon')
        self.lat = fin.getNode('/lat')
        self.x_edges = fin.getNode('/x_edges')
        self.y_edges = fin.getNode('/y_edges')
        self.dh_mean = fin.getNode('/dh_mean')
        self.dh_error = fin.getNode('/dh_error')
        self.dh_error2 = fin.getNode('/dh_error2')
        self.dg_mean = fin.getNode('/dg_mean')
        self.dg_error = fin.getNode('/dg_error')
        self.dg_error2 = fin.getNode('/dg_error2')
        self.n_ad = fin.getNode('/n_ad')
        self.n_da = fin.getNode('/n_da')
        try:
            self.time_xcal = fin.getNode('/time_xcal')
            self.dg_mean_xcal = fin.getNode('/dg_mean_xcal')
            self.dh_mean_xcal = fin.getNode('/dh_mean_xcal')
            self.dg_mean_xcal_interp = fin.getNode('/dg_mean_xcal_interp')
            self.dh_mean_xcal_interp = fin.getNode('/dh_mean_xcal_interp')
        except:
            pass

#-------------------------------------------------------------------------

def main():

    fname_in = sys.argv[1]

    din = GetData(fname_in, 'a')
    time = ap.num2date(getattr(din, T_NAME)[:])
    lon = din.lon[:]
    lat = din.lat[:]
    sat = din.satname[:]
    nrows = len(time)
    nt, ny, nx =  getattr(din, H_NAME).shape    # i,j,k = t,y,x

    RR = np.empty((nt,ny,nx), 'f8') * np.nan
    SS = np.empty((nt,ny,nx), 'f8') * np.nan

    print 'processing time series:'
    isfirst = True

    # iterate over every grid cell (all times): i,j = y,x
    #-----------------------------------------------------------------

    for i in xrange(ny):
        for j in xrange(nx):
            print 'time series of grid-cell:', i, j

            dh = getattr(din, H_NAME)[:nrows,i,j]
            dg = getattr(din, G_NAME)[:nrows,i,j]
            dh_cor = np.zeros_like(dh)

            if np.alltrue(np.isnan(dh)): continue

            #---------------------------------------------------------

            # pull and correct a chunk of the array at a time
            for s in np.unique(sat):
                k = np.where(sat == s)
                dh_cor[k], R, S = ap.backscatter_corr(dh[k], dg[k], 
                                  diff=DIFF, robust=True)
                RR[k,i,j] = R
                SS[k,i,j] = S

            #---------------------------------------------------------

            if PLOT_TS:
                dh_cor = ap.referenced(dh_cor, to='first')
                dh = ap.referenced(dh, to='first')
                dg = ap.referenced(dg, to='first')
                fig = plot_rs(time, RR[:,i,j], SS[:,i,j])
                for s in np.unique(sat):
                    k = np.where(sat == s)
                    r, s = np.mean(RR[k,i,j]), np.mean(SS[k,i,j])
                    try:
                        fig = plot_ts(time[k], lon[j], lat[i], dh_cor[k], 
                                      dh[k], dg[k], r, s, diff=DIFF)
                    except:
                        print 'something wrong with ploting!'
                        print 'dh:', dh
                        print 'dg:', dg
                if fig is None: continue
                plt.show()

            # save one TS per grid cell at a time
            #---------------------------------------------------------

            if not SAVE_TO_FILE: continue

            if isfirst:
                # open or create output file
                isfirst = False
                atom = tb.Atom.from_type('float64', dflt=np.nan)
                filters = tb.Filters(complib='zlib', complevel=9)
                try:
                    c1 = din.file.createCArray('/', SAVE_AS_NAME, atom, 
                                               (nt,ny,nx), '', filters)
                except:
                    c1 = din.file.getNode('/', SAVE_AS_NAME)
                c2 = din.file.createCArray('/', R_NAME, atom, 
                                           (nt,ny,nx), '', filters)
                c3 = din.file.createCArray('/', S_NAME, atom, 
                                           (nt,ny,nx), '', filters)
            c1[:,i,j] = dh_cor

    if SAVE_TO_FILE:
        c2[:] = RR
        c3[:] = SS

    if PLOT_MAP:
        RR = RR[0]  # change accordingly
        SS = SS[0]
        plot_map(lon, lat, np.abs(RR), BBOX, MASK_FILE, mres=1, vmin=0, vmax=1)
        plt.title('Correlation Coefficient, R')
        plt.savefig('map_r.png')
        plot_map(lon, lat, SS, BBOX, MASK_FILE, mres=1, vmin=-0.2, vmax=0.7)
        plt.title('Correlation Gradient, S')
        plt.savefig('map_s.png')
        plt.show()

    din.file.close()

    if SAVE_TO_FILE:
        print 'out file -->', fname_in


if __name__ == '__main__':
    main()
