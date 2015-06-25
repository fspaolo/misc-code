#!/usr/bin/env python
"""
FOR JOINT CROSS-CALIBRATED TIME SERIES !!!

Applies the backscatter correction to a set of time series.

Implements the correction using changes in AGC, following Zwally, et al. (2005)
and Yi, et al. (2011). See docstrings in function backscatter_corr() for 
details in the algorithm.

Usage
-----
$ python backscatter2.py ~/data/fris/xover/seasonal/all_*_mts.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# February 7, 2012

import sys
import numpy as np
import tables as tb
import altimpy as ap
import matplotlib.pyplot as plt
from funcs import plot_rs, plot_ts, plot_map

# ERS-1: 1992/07-1996/04
# ERS-2: 1995/07-2003/04
# Envi:  2002/10-2011/11

#-------------------------------------------------------------------------
# global variables
#-------------------------------------------------------------------------

PLOT_TS = True    # R(t) and S(t) included
PLOT_MAP = False
SAVE_TO_FILE = True
DIFF = False       # False=dH/dG (abs vals: MIX), True=DdH/DdG (diff: SHORT)
TVAR = False      # for time variable correlation: R(t), S(t)
NPTS = 13         # sliding-window, number of pts for correlation at each time
TINT = False      # time intervals for sattelite-independent correlations
INTERVALS = [(1992, 1996), (1996, 2003), (2003, 2012)]  # t intervals [t1,t2) 

# correct continuous cross-calibrated TS
T_NAME = 'time_xcal'
H_NAME = 'dh_mean_xcal'
G_NAME = 'dg_mean_xcal'
R_NAME = 'corrcoef_xcal_mixed_const'
S_NAME = 'corrgrad_xcal_mixed_const'
SAVE_AS_NAME = 'dh_mean_xcal_mixed_const'

#BBOX = (-82, -76.2, -26, -79.5)  # FRIS
BBOX = (-156, -76, 154, -81.2)   # ROSS
#BBOX = (72, -74.2, 68, -67.5)   # AMERY

MFILE = '/Users/fpaolo/data/masks/scripps/scripps_antarctica_mask1km_v1.tif'

# grid-cells with negative correlation (FRIS)
I = [ 3,  3,  4,  4,  5,  7,  7,  7,  7,  7,  7,  8,  8,  9,  9, 10, 
     13, 13, 17, 19, 20, 21, 21, 21, 21, 22, 22, 22, 23, 23, 23]
J = [61, 63, 62, 65, 54, 40, 70, 73, 74, 75, 76, 40, 69, 39, 56, 19, 
     20, 21, 46, 35, 36, 36, 37, 77, 78, 76, 78, 79, 39, 76, 77]

#-------------------------------------------------------------------------
# pointer to data -> struct
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
        self.time_xcal = fin.getNode('/time_xcal')
        self.dg_mean_xcal = fin.getNode('/dg_mean_xcal')
        self.dh_mean_xcal = fin.getNode('/dh_mean_xcal')

#-------------------------------------------------------------------------

def main():

    fname_in = sys.argv[1]

    din = GetData(fname_in, 'a')
    time = ap.num2date(getattr(din, T_NAME)[:])
    lon = din.lon[:]
    lat = din.lat[:]
    nrows = len(time)
    nt, ny, nx =  getattr(din, H_NAME).shape    # i,j,k = t,y,x

    RR = np.empty((nt,ny,nx), 'f8') * np.nan
    SS = np.empty((nt,ny,nx), 'f8') * np.nan

    if TINT:
        intervals = [ap.year2date(tt) for tt in INTERVALS]

    if TINT:
        print 'using time-interval correlation'
    elif TVAR:
        print 'using time-variable correlation'
    else:
        print 'using constant correlation'
    print 'processing time series:'
    isfirst = True

    # iterate over every grid cell (all times): i,j = y,x
    #-----------------------------------------------------------------

    for i in xrange(ny):
        for j in xrange(nx):
            print 'time series of grid-cell:', i, j

            dh = getattr(din, H_NAME)[:nrows,i,j]
            dg = getattr(din, G_NAME)[:nrows,i,j]

            if np.alltrue(np.isnan(dh)): continue

            #---------------------------------------------------------

            if TINT:
                # satellite-dependent R and S
                dh_cor, RR[:,i,j], SS[:,i,j] = \
                    ap.backscatter_corr3(dh, dg, time, intervals, 
                                         diff=DIFF, robust=True)
            elif TVAR:
                # time-varying R and S
                dh_cor, RR[:,i,j], SS[:,i,j] = \
                    ap.backscatter_corr2(dh, dg, diff=DIFF, 
                                         robust=True, npts=NPTS)
            else:
                # constant R and S
                dh_cor, RR[:,i,j], SS[:,i,j] = \
                    ap.backscatter_corr(dh, dg, diff=DIFF, robust=True)

            #---------------------------------------------------------

            # plot figures
            if PLOT_TS:
                dh_cor = ap.referenced(dh_cor, to='first')
                dh = ap.referenced(dh, to='first')
                dg = ap.referenced(dg, to='first')
                k, = np.where(~np.isnan(RR[:,i,j]))
                r = np.mean(RR[k,i,j])
                s = np.mean(SS[k,i,j])
                fig = plot_rs(time, RR[:,i,j], SS[:,i,j])
                fig = plot_ts(time, lon[j], lat[i], dh_cor, dh, dg, 
                              r, s, diff=DIFF)
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
                c1 = din.file.createCArray('/', SAVE_AS_NAME, atom, 
                                           (nt,ny,nx), '', filters)
                c2 = din.file.createCArray('/', R_NAME, atom, 
                                           (nt,ny,nx), '', filters)
                c3 = din.file.createCArray('/', S_NAME, atom, 
                                           (nt,ny,nx), '', filters)
            c1[:,i,j] = dh_cor

    if SAVE_TO_FILE:
        c2[:] = RR
        c3[:] = SS

    if PLOT_MAP:
        if TVAR:
            # 3D -> 2D
            RR = np.mean(RR[~np.isnan(RR)], axis=0)
            SS = np.mean(SS[~np.isnan(SS)], axis=0)
        plot_map(lon, lat, np.abs(RR), BBOX, MFILE, mres=1, vmin=0, vmax=1)
        plt.title('Correlation Coefficient, R')
        plt.savefig('map_r.png')
        plot_map(lon, lat, SS, BBOX, MFILE, mres=1, vmin=-0.2, vmax=0.7)
        plt.title('Correlation Gradient, S')
        plt.savefig('map_s.png')
        plt.show()

    din.file.close()

    if SAVE_TO_FILE:
        print 'out file -->', fname_in


if __name__ == '__main__':
    main()
