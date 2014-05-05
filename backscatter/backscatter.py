#!/usr/bin/env python
"""
Applies the backscatter correction to a set of time series.

Implements the correction using changes in AGC, following 
Zwally, et al. (2005) and Yi, et al. (2011):
        
    H_corr = H - S*G - H0

where H = dh(t0,t), G = dAGC(t0,t) and S = dH/dG.

Example
-------
$ python backscatter.py ~/data/fris/xover/seasonal/all_*_mts.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# February 7, 2012

import argparse as ap

from funcs import *

#-------------------------------------------------------------------------
# global variables
#-------------------------------------------------------------------------

PLOT_TS = True    # R(t) and S(t) included
PLOT_MAP = False
SAVE_TO_FILE = False
NODE_NAME = ''
INDEPENDENT_TS = False
TERM = 'short'    # mix=dH/dG (abs vals), short=DdH/DdG (differences)
TVAR = False      # for time variable correlation: R(t), S(t)
NPTS = 13         # sliding-window, number of pts for correlation at each time
TINT = True      # time intervals for sattelite-independent correlations
INTERVALS = [(1992, 1996), (1996, 2003), (2003, 2012)]  # t intervals [t1,t2) 

# ERS-1: 1992/07-1996/04
# ERS-2: 1995/07-2003/04
# Envi:  2002/10-2011/11

if INDEPENDENT_TS:
    # correct each satellite-independent TS
    T_NAME = 'time'
    H_NAME = 'dh_mean'
    G_NAME = 'dg_mean'
    R_NAME = 'corrcoef'
    S_NAME = 'corrgrad'
    SAVE_AS_NAME = 'dh_mean_corr'
    ALL_SATELLITES = False
else:
    # correct continuous cross-calibrated TS
    T_NAME = 'time_all'
    H_NAME = 'dh_mean_all'
    G_NAME = 'dg_mean_all'
    R_NAME = 'corrcoef_short_t13_all'
    S_NAME = 'corrgrad_short_t13_all'
    SAVE_AS_NAME = 'dh_mean_corr_short_t13_all'
    ALL_SATELLITES = True

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

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('fin', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: same as input file]')
args = parser.parse_args()


def main(args):

    fname_in = args.fin[0]
    fname_out = get_fname_out(fname_in, fname_out=args.fname_out, sufix='corr.h5')

    ### input data

    din = GetData(fname_in, 'a')
    time = num2dt(getattr(din, T_NAME))
    lon = din.lon
    lat = din.lat
    nrows = len(time)
    nt, ny, nx =  getattr(din, H_NAME).shape    # i,j,k = t,y,x

    if TINT:
        intervals = [year2dt(tt) for tt in INTERVALS]

    if TINT or TVAR:
        RR = np.empty((nt,ny,nx), 'f8') * np.nan
        SS = np.empty((nt,ny,nx), 'f8') * np.nan
    else:
        RR = np.empty((ny,nx), 'f8') * np.nan
        SS = np.empty((ny,nx), 'f8') * np.nan

    print 'using time interval correlation:', TINT
    print 'using time variable correlation:', TVAR
    print 'processing time series ...'
    isfirst = True

    # iterate over every grid cell (all times): i,j = y,x
    #-----------------------------------------------------------------
    for i in xrange(ny):
        for j in xrange(nx):
            i = 3
            j = 163

            ts_ij = din.dh_mean[:,i,j]
            if np.alltrue(np.isnan(ts_ij)): continue

            dh = getattr(din, H_NAME)[:nrows,i,j]
            dg = getattr(din, G_NAME)[:nrows,i,j]

            #---------------------------------------------------------

            if TINT:
                # satellite-independent R and S
                dh_cor, R, S = \
                    backscatter_corr3(dh, dg, time, intervals, term=TERM, robust=True)
                RR[:,i,j] = R
                SS[:,i,j] = S
            elif TVAR:
                # time-varying R and S
                dh_cor, R, S = \
                    backscatter_corr2(dh, dg, term=TERM, robust=True, npts=NPTS)
                RR[:,i,j] = R
                SS[:,i,j] = S
            else:
                # constant R and S
                dh_cor, R, S = \
                    backscatter_corr(dh, dg, term=TERM, robust=True)
                RR[i,j] = R
                SS[i,j] = S

            #---------------------------------------------------------

            # plot figures
            if PLOT_TS: # and not np.alltrue(np.isnan(dh[1:])) \
                    #and not np.alltrue(dh == 0):
                print 'grid-cell:', i, j
                '''
                time[1] = np.nan
                dh[1] = np.nan
                dg[1] = np.nan
                dh_cor[1] = np.nan
                '''
                if TINT or TVAR:
                    k, = np.where(~np.isnan(RR[:,i,j]))
                    r = np.mean(RR[k,i,j])
                    s = np.mean(SS[k,i,j])
                    fig = plot_rs(time, RR[:,i,j], SS[:,i,j])
                    if fig is None: continue
                    #plt.savefig('rs.pdf', dpi=150, bbox_inches='tight')
                    #plt.show()
                    #continue
                else:
                    r = RR[i,j]
                    s = SS[i,j]
                dh_cor = reference_to_first(dh_cor)
                dh = reference_to_first(dh)
                dg = reference_to_first(dg)
                fig = plot_tseries(time, lon[j], lat[i], dh_cor, dh,
                    dg, r, s, term=TERM)
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
                c = din.file.createCArray('/', SAVE_AS_NAME, atom, (nt,ny,nx), '', filters)
                if TVAR:
                    c2 = din.file.createCArray('/', R_NAME, atom, (nt,ny,nx), '', filters)
                    c3 = din.file.createCArray('/', S_NAME, atom, (nt,ny,nx), '', filters)
                else:
                    c2 = din.file.createCArray('/', R_NAME, atom, (ny,nx), '', filters)
                    c3 = din.file.createCArray('/', S_NAME, atom, (ny,nx), '', filters)

            c[:,i,j] = dh_cor

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
    main(args)
