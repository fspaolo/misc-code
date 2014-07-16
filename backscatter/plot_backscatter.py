#!/usr/bin/env python
"""
Plots the backscatter correction to a set of time series.

Example
-------
$ python plot_backscatter.py ~/data/fris/xover/seasonal/envi_*_mean.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# July 30, 2012

import argparse as ap

from funcs import *
import viz

# global variables
#-------------------------------------------------------------------------

PLOT_TS = True
PLOT_MAPS = False
INDEPENDENT_TS = False
TERM = 'short'    # mix/short
NODE_NAME = ''

BBOX_REG = (-82, -76.2, -26, -79.5)  # FRIS
#BBOX_REG = (-156, -76, 154, -81.2)   # ROSS
#BBOX_REG = (68, -74.5, 70, -67.5)   # AMERY

MOA_FILE = '/Users/fpaolo/data/MOA/moa750_r1_hp1.tif'
MASK_FILE = '/Users/fpaolo/data/masks/scripps_antarctica_masks/scripps_antarctica_mask1km_v1.tif'
MOA_RES = 10
MASK_RES = 1

if INDEPENDENT_TS:
    # correct independent TS
    H_NAME = 'dh_mean'
    G_NAME = 'dg_mean'
    R_NAME = 'corrcoef'
    S_NAME = 'corrgrad'
    TABLE = 'table'
    YEAR = 'year'
    ALL_SATELLITES = False
else:
    # correct cross-calibrated TS
    H_NAME = 'dh_mean_all'
    G_NAME = 'dg_mean_all'
    R_NAME = 'corrcoef_all'
    S_NAME = 'corrgrad_all'
    TABLE = 'table_all'
    YEAR = 'year_all'
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
    #-----------------------------------------------------------------

    d, fin = get_data_from_file(fname_in, NODE_NAME, 'a', all_sat=ALL_SATELLITES)

    N, ny, nx =  d[H_NAME].shape    # i,j,k = t,y,x
    RR = np.empty((ny,nx), 'f8') * np.nan
    SS = np.empty((ny,nx), 'f8') * np.nan

    print 'processing time series ...'

    # iterate over every grid cell (all times): i,j = y,x
    #-----------------------------------------------------------------

    for i in xrange(ny):
        for j in xrange(nx):

            i, j = 8, 42 
            nrows = d[TABLE].nrows
            year = d[YEAR]
            lon = d['lon']
            lat = d['lat']
            dh_mean = d[H_NAME][:nrows,i,j]
            dg_mean = d[G_NAME][:nrows,i,j]

            dh_mean_corr, R, S = \
                backscatter_corr2(dh_mean, dg_mean, term=TERM, robust=False, npts=9)

            if np.ndim(R) == 0:
                RR[i,j] = R
                SS[i,j] = S
            else:
                RR[i,j] = np.mean(R)
                SS[i,j] = np.mean(S)

            # plot time series
            #---------------------------------------------------------

            if PLOT_TS and not np.alltrue(np.isnan(dh_mean[1:])) \
                    and not np.alltrue(dh_mean == 0):
                print 'grid-cell: [%d,%d]' % (i, j), 'lon/lat: %.2f/%.2f' % (lon[j], lat[i])
                year = year[1:]
                dh_mean_corr, dh_mean, dg_mean = dh_mean_corr[1:], dh_mean[1:], dg_mean[1:]
                dh_mean_corr = reference_to_first(dh_mean_corr)
                dh_mean = reference_to_first(dh_mean)
                dg_mean = reference_to_first(dg_mean)
                plot_tseries(year, lon[j], lat[i], dh_mean_corr, dh_mean, dg_mean, R, S, term=TERM)
                plt.figure()
                plt.subplot(211)
                plt.plot(R)
                plt.subplot(212)
                plt.plot(S)
                plt.show()
                sys.exit()

    # plot R and S maps
    #-----------------------------------------------------------------

    if PLOT_MAPS:
        fig1 = plot_map(lon, lat, RR, BBOX_REG, MASK_FILE, MASK_RES)
        plt.title('Correlation Coefficient, R')
        fig2 = plot_map(lon, lat, SS, BBOX_REG, MASK_FILE, MASK_RES)
        plt.title('Correlation Gradient, S')
        plt.show()

    fin.close()


if __name__ == '__main__':
    main(args)
