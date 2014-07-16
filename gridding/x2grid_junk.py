#!/usr/bin/env python
"""
Take crossover files as input and construct grids of average `dh`, `dAGC`, 
`errors` and `n_obs` at every grid cell: 

Notes
-----
- Reads HDF5 files with a 2d array named 'data' (a data matrix).
- Process one sat *and* one subgrid at a time and merge the files later on!
- Dimensions are: i,j,k = t,y,x
- All the SDs are calculated using N-1 (ddof=1)
- It applies tide and load corrections
- Do not use multiple modes! Use ERS-1/2 -> ice-mode, Envi -> fine-mode.
- For mode selection edit 'filter_data()'
- Warning: without 'USEALL' several grid cells end up empty (in small/difficult regions).
- To avoid Unix limitation on number of cmd args, pass instead of file names a
  string as "/path/to/data/file_*_name.ext"

Example
-------
$ python x2grid.py -r 0 360 -82 -62 -d .75 .25 -n 10 \
                   "/data/alt/ra/ers1/hdf/antarctica/xovers/ers1_*_tide.h5"

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap
import altimpy as alt
from glob import glob
from scipy.ndimage import gaussian_filter
from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = False
SAVE_TO_FILE = True

ABS_VAL = 15          # (m), accept data if |dh| < ABS_VAL
NUM_STD = 3           # sigma for editing: accept if |dh| < NUM_STD * std
ICE_ONLY = False      # use ice-only mode, otherwise uses both ice + ocean
ITERATIVE = False     # iterative sigma editing
MEDIAN = True         # use median (instead of mean) for weighted avrg: [w1*M(x1) + w2*M(x2)]/(w1+w2)
USEALL = True         # use one of the average/median <ad> or <da> if the other is missing
TIDE_CODE = 'fortran' # for tide computed using 'fortran' or 'matlab' code
GAUSS_SMOOTH = False  # for plotting only!
GAUSS_WIDTH = 0.2

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='crossover files to read (HDF5 2D array)')
parser.add_argument('-r', dest='region', nargs=4, type=float, 
    metavar=('l', 'r', 'b', 't'), default=(0, 360, -82, -62),
    help='coordinates of full-grid domain: left right bottom top')
parser.add_argument('-d', dest='delta', nargs=2, type=float, 
    metavar=('dx', 'dy'), default=(1.0, 0.3),
    help='size of grid cells: dx dy (deg), default 1.0 x 0.3')
parser.add_argument('-n', dest='nsectors', default=1, type=int,
    help='total number of sectors (sub-grids), default 1')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name, default /input/path/fname_grid.h5')
parser.add_argument('-s', dest='suffix', default='_grid.h5',
    help="suffix of output file, default '_grid.h5'")
args = parser.parse_args()

def main(args):

    # input args
    #---------------------------------------------------------------------
    if len(args.files) > 1:
        files = args.files
    else:
        files = glob(args.files[0])  # using 'glob'

    fname_out = args.fname_out
    region = args.region
    x_range = region[:2]
    y_range = region[2:]
    dx = args.delta[0]
    dy = args.delta[1]
    nsect = args.nsectors
    suffix = args.suffix

    if nsect > 1:
        subdom = get_sub_domains(x_range[0], x_range[1], nsect)

    if len(files) > 1:
        files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s))

    print 'processing %d files...' % len(files)

    for fname in files:

        if nsect > 1:
            sectnum = get_sect_num(fname) 
            x_range = subdom[sectnum]

        d = get_data(fname)
        d['satname'], d['time1'], d['time2'] = get_info(fname)

        # pre-processing
        #-----------------------------------------------------------------

        d['lon'] = alt.lon_180_360(d['lon'], region=region)

        # filter data -> select modes and "good" data points
        d = filter_data(d, ice_only=ICE_ONLY)

        # go to next file
        if d is None:   
            print 'No data left after filtering!\nFile:', fname
            continue

        # apply tide and load corrections
        """
        The sign of the correction is the same (subtract), but the phase of the
        load tide is ~180 degrees off the ocean tide. E.g., if the ocean tide at
        (t,x,y) is +1.0 m, the load tide is probably -0.03 m (more or less), so
        the correction equation would be:
        
        tide_free = measured - (+1.0) - (-0.03) = measured - (+0.97)
        """
        d['h1'] -= d['tide1']
        d['h2'] -= d['tide2']

        # digitize lons and lats (create the grid)
        lon, lat, j_bins, i_bins, x_edges, y_edges, nx, ny = \
            digitize(d['lon'], d['lat'], x_range, y_range, dx, dy)

        h1, h2 = d['h1'], d['h2']
        g1, g2 = d['g1'], d['g2']
        ftrk1, ftrk2 = d['ftrk1'], d['ftrk2']

        # Output grids 
        g = OutGrids(ny, nx)

        # calculations per grid cell
        #-----------------------------------------------------------------

        for i in xrange(ny):
            for j in xrange(nx):
                '''
                # data corresponding to the 'i,j' grid cell
                i_cell, = np.where((x_edges[j] <= d['lon']) & \
                                   (d['lon'] < x_edges[j+1]) & \
                                   (y_edges[i] <= d['lat']) & \
                                   (d['lat'] < y_edges[i+1]))
                '''
                # indices of data corresponding to the 'i,j' grid cell
                i_cell, = np.where((j_bins == j+1) & (i_bins == i+1))

                if len(i_cell) == 0: continue

                # xovers per cell 
                dh = h2[i_cell] - h1[i_cell]    # always t2 - t1 !
                dg = g2[i_cell] - g1[i_cell]

                # separate in asc/des-des/asc 
                i_ad, i_da = where_ad_da(ftrk1[i_cell], ftrk2[i_cell])
                dh_ad = dh[i_ad]
                dh_da = dh[i_da]
                dg_ad = dg[i_ad]
                dg_da = dg[i_da]

                # filter absolute values
                i_ad = abs_editing(dh_ad, absval=ABS_VAL, return_index=True)
                i_da = abs_editing(dh_da, absval=ABS_VAL, return_index=True)
                dh_ad = dh_ad[i_ad]
                dh_da = dh_da[i_da]
                dg_ad = dg_ad[i_ad]
                dg_da = dg_da[i_da]

                # filter standard deviation
                i_ad = std_editing(dh_ad, nsd=NUM_STD, iterative=ITERATIVE, return_index=True)
                i_da = std_editing(dh_da, nsd=NUM_STD, iterative=ITERATIVE, return_index=True)
                if len(i_ad) == 0 and len(i_da) == 0:
                    pass
                else:
                    dh_ad = dh_ad[i_ad]
                    dh_da = dh_da[i_da]
                    dg_ad = dg_ad[i_ad]
                    dg_da = dg_da[i_da]

                # mean values
                g.dh_mean[i,j] = compute_weighted_mean(dh_ad, dh_da, useall=USEALL, median=MEDIAN) 
                g.dh_error[i,j] = compute_weighted_error(dh_ad, dh_da, useall=USEALL) 
                g.dh_error2[i,j] = compute_wingham_error(dh_ad, dh_da, useall=USEALL) 
                g.dg_mean[i,j] = compute_weighted_mean(dg_ad, dg_da, useall=USEALL, median=MEDIAN) 
                g.dg_error[i,j] = compute_weighted_error(dg_ad, dg_da, useall=USEALL) 
                g.dg_error2[i,j] = compute_wingham_error(dg_ad, dg_da, useall=USEALL) 
                g.n_ad[i,j], g.n_da[i,j] = compute_num_obs(dh_ad, dh_da, useall=USEALL)

                # gaussian smooth
                if PLOT and GAUSS_SMOOTH and not SAVE_TO_FILE:
                    g.dh_mean = gaussian_filter(g.dh_mean, GAUSS_WIDTH)
                    g.dg_mean = gaussian_filter(g.dg_mean, GAUSS_WIDTH)

            # save the grids
            #-----------------------------------------------------------------

            if SAVE_TO_FILE:
                # save one set of grids per iteration (i.e., per file)
                fname_out = fname.replace('.h5', suffix)
                out = OutputContainers(fname_out, (1,ny,nx))
                out.lon[:] = lon
                out.lat[:] = lat
                out.x_edges[:] = x_edges
                out.y_edges[:] = y_edges
                out.time1[:] = d['time1']
                out.time2[:] = d['time2']
                out.dh_mean[:] = g.dh_mean
                out.dh_error[:] = g.dh_error
                out.dh_error2[:] = g.dh_error2
                out.dg_mean[:] = g.dg_mean
                out.dg_error[:] = g.dg_error
                out.dg_error2[:] = g.dg_error2
                out.n_ad[:] = g.n_ad
                out.n_da[:] = g.n_da
                out.file.flush()
                out.file.close()

        try:
            print_info(x_edges, y_edges, lon, lat, dx, dy, 1, g.n_ad, g.n_da, source='None')
        except:
            pass
        
        if PLOT:
            try:
                plt.plot(d['lon'], d['lat'], '.')
                plot_grids(x_edges, y_edges, g.dh_mean, g.dh_error, g.n_ad, g.n_da)
                plt.show()
            except:
                print 'no data to plot!'
        
        if SAVE_TO_FILE:
            print 'file out -->', fname_out, '\n'


if __name__ == '__main__':
    main(args)
