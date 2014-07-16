#!/usr/bin/env python
"""
Applies the backscatter correction to a set of time series.

Implements the correction using changes in AGC, following Zwally, et al. (2005)
and Yi, et al. (2011):
        
    H_corr = H - S*G - H0

where H is dh(t0,t), G is dAGC(t0,t) and S is dG/dH.

Example
-------
$ python backscatter.py ~/data/fris/xover/seasonal/envi_*_mean.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# February 7, 2012

import argparse as ap

from funcs_gla import *

# global variables
#-------------------------------------------------------------------------

PLOT = False
SAVE_TO_FILE = True
NODE_NAME = ''
INDEPENDENT_TS = True
TERM = 'short'    # mix=dH/dG, short=DdH/DdG

if INDEPENDENT_TS:
    # correct independent TS
    H_NAME = 'dh_mean'
    G_NAME = 'dg_mean'
    R_NAME = 'corrcoef_short'
    S_NAME = 'corrgrad_short'
    TABLE = 'table'
    TIME2 = 'time2'
    SAVE_AS_NAME = 'dh_mean_corr_short'
    ALL_SATELLITES = False
else:
    # correct cross-calibrated TS
    H_NAME = 'dh_mean_all'
    G_NAME = 'dg_mean_all'
    R_NAME = 'corrcoef_all'
    S_NAME = 'corrgrad_all'
    TABLE = 'table_all'
    TIME2 = 'time2_all'
    SAVE_AS_NAME = 'dh_mean_all_corr'
    ALL_SATELLITES = True

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

    # input data --> dictionary and file
    #---------------------------------------------------------------------

    d, fin = get_data_from_file(fname_in, NODE_NAME, 'a', all_sat=ALL_SATELLITES)

    #---------------------------------------------------------------------

    N, ny, nx =  d[H_NAME].shape    # i,j,k = t,y,x
    RR = np.empty((ny,nx), 'f8') * np.nan
    SS = np.empty((ny,nx), 'f8') * np.nan

    print 'processing time series ...'
    isfirst = True

    # iterate over every grid cell (all times): i,j = y,x
    #-----------------------------------------------------------------

    for i in xrange(ny):
        for j in xrange(nx):

            #i, j = 1, 42 
            nrows = d[TABLE].nrows
            time2 = d[TIME2]
            lon = d['lon']
            lat = d['lat']
            dh_mean = d[H_NAME][:nrows,i,j]
            dg_mean = d[G_NAME][:nrows,i,j]

            dh_mean_corr, R, S = \
                backscatter_corr(dh_mean, dg_mean, term=TERM, robust=True)

            if R is None:
                R = np.nan
                S = np.nan

            if np.ndim(R) == 1:
                RR[i,j] = R
                SS[i,j] = S
            else:
                RR[i,j] = np.mean(R)
                SS[i,j] = np.mean(S)

            # plot figures
            if PLOT and not np.alltrue(np.isnan(dh_mean[1:])) \
                    and not np.alltrue(dh_mean == 0):
                print 'grid-cell:', i, j
                time2 = i2dt(time2[1:])
                dh_mean_corr, dh_mean, dg_mean = dh_mean_corr[1:], dh_mean[1:], dg_mean[1:]
                dh_mean_corr = reference_to_first(dh_mean_corr)
                dh_mean = reference_to_first(dh_mean)
                dg_mean = reference_to_first(dg_mean)
                plot_tseries(time2, lon[j], lat[i], dh_mean_corr, dh_mean, dg_mean, R, S, term=TERM)
                plt.show()

            # save one TS per grid cell at a time
            #---------------------------------------------------------

            if not SAVE_TO_FILE: continue

            if isfirst:
                # open or create output file
                isfirst = False
                atom = tb.Atom.from_dtype(dh_mean_corr.dtype)
                filters = tb.Filters(complib='zlib', complevel=9)
                try:
                    g = fin.getNode('/', NODE_NAME)
                    c = fin.createCArray(g, SAVE_AS_NAME, atom, 
                        (N,ny,nx), '', filters)
                    c2 = fin.createCArray(g, R_NAME, atom, 
                        (ny,nx), '', filters)
                    c3 = fin.createCArray(g, S_NAME, atom, 
                        (ny,nx), '', filters)
                except:
                    c = fin.getNode('/%s' % NODE_NAME, SAVE_AS_NAME)

            c[:,i,j] = dh_mean_corr

    if SAVE_TO_FILE:
        c2[:] = RR[:]
        c3[:] = SS[:]

    plt.figure()
    plt.imshow(RR, origin='lower', interpolation='nearest')
    plt.colorbar(orientation='horizontal', shrink=0.6)
    plt.title('Corr Coef')
    plt.figure()
    plt.imshow(SS, origin='lower', interpolation='nearest')
    plt.colorbar(orientation='horizontal', shrink=0.6)
    plt.title('Corr Grad')
    plt.show()

    fin.close()

    print 'out file -->', fname_in


if __name__ == '__main__':
    main(args)
