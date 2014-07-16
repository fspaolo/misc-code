#!/usr/bin/env python
"""
Take crossover files as input and construct grids of average `dh`, `dAGC`, 
`errors` and `n_obs` at every grid cell: 

1) single_crossover_file --> single_grid
2) multiple_crossover_files --> multiple_grids

Notes
-----
- Reads HDF5 files with a 2d array named 'data' (a data matrix).
- Process one sat *and* one subgrid at a time and merge the files later on.
- Indices: i,j,k = t,y,x
- All the SDs are calculated using N-1 (ddof=1)
- It applies tide and load corrections
- Do not use multiple modes! Use ERS-1/2 -> ice-mode, Envi -> fine-mode.
- For mode selection edit 'filter_data()'
- Warning: without 'USEALL' several grid cells end up empty (in small/difficult regions).
- To avoid Unix limitation on number of cmd args, pass instead of file names a
  string as "/path/to/data/file_*_name.ext"

Example
-------
$ python x2grid.py -r 0 360 -82 -62 -d .25 .75 -n 10 \
                   "/data/alt/ra/ers1/hdf/antarctica/xovers/ers1_*_tide.h5"

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap
from glob import glob
from mpl_toolkits.basemap import interp
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter

from funcs_tmp import *

# global variables
#-------------------------------------------------------------------------

PLOT = False
SAVE_TO_FILE = True

NODE_NAME = ''
STRUCT = None         # data structure: None/'gla', use gla for Envi/ICESat comparison
ABS_VAL = 15          # (m), accept data if |dh| < ABS_VAL
NUM_STD = 3           # sigma for editing: accept if |dh| < NUM_STD * std
ITERATIVE = False     # iterative sigma editing
MEDIAN = True         # use median (instead of mean) for weighted avrg: [w1*M(x1) + w2*M(x2)]/(w1+w2)
USEALL = True         # use one of the average/median <ad> or <da> if the other is missing
MAX_SIZE_DATA = 512   # (MB), to load data in memory
TIDE_CODE = 'fortran' # for tide computed using fortran or matlab code
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
    help='output file name, default /input/path/sat_tmin_tmax_grid.h5')
parser.add_argument('-p', dest='prefix', default=None,
    help='prefix of output file, default same as input')
parser.add_argument('-s', dest='suffix', default='grid.h5',
    help="suffix of output file, default 'grid.h5'")
args = parser.parse_args()

def main(args):

    # input args
    #---------------------------------------------------------------------
    if len(args.files) > 1:
        files = args.files
    else:
        # use 'glob' instead
        files = glob(args.files[0])

    fname_out = args.fname_out
    region = args.region
    x_range = region[:2]
    y_range = region[2:]
    dx = args.delta[0]
    dy = args.delta[1]
    nsect = args.nsectors
    prefix = args.prefix
    suffix = args.suffix
    N = len(files)

    if nsect > 1:
        subdom = get_sub_domains(x_range[0], x_range[1], nsect)

    if len(files) > 1:
        files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s))

    print 'processing %d files...' % len(files)

    isfirst = True
    for pos, fname in enumerate(files):

        if nsect > 1:
            sectnum = get_sect_num(fname) 
            x_range = subdom[sectnum]

        In = Input(fname)
        In.get_time_from_fname()   # get sat/time info for every grid
        d = In.get_data_from_file(MAX_SIZE_DATA, TIDE_CODE, struct=STRUCT)

        # pre-processing
        #-----------------------------------------------------------------

        # convert -180/+180 <-> 0/360 if needed
        d['lon'] = lon_180_to_360(d['lon'], region)

        # filter data -> select modes and "good" data points
        d = filter_data(d)

        # go to next file. A "void" will be left in the 3D array if no data!
        if d is None:   
            print 'No data left after filtering!\nFile:', fname
            In.fin.close()
            continue

        # apply corrections 
        d = apply_tide_corr(d, TIDE_CODE)

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
                #g.dh_mean[i,j] = compute_ordinary_mean(dh_ad, dh_da, useall=USEALL) 
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

        if not SAVE_TO_FILE: 
            In.fin.close()
            continue

        if isfirst:
            Out = Output(files, fname_out, g.dh_mean, (N,ny,nx), NODE_NAME, prefix, suffix)
            Out.get_fname_out()
            Out.create_output_containers()

            # save info
            n = 0
            Out.lon[:] = lon
            Out.lat[:] = lat
            Out.x_edges[:] = x_edges
            Out.y_edges[:] = y_edges

            isfirst = False

        # save one set of grids per iteration (i.e., per file)
        # --> test row.append() instead!
        #Out.table.append([(d['satname'], d['time1'], d['time2'])]) 
        #Out.table.flush()
        Out.time1[n] = d['time1']
        Out.time2[n] = d['time2']
        Out.dh_mean[n,...] = g.dh_mean
        Out.dh_error[n,...] = g.dh_error
        Out.dh_error2[n,...] = g.dh_error2
        Out.dg_mean[n,...] = g.dg_mean
        Out.dg_error[n,...] = g.dg_error
        Out.dg_error2[n,...] = g.dg_error2
        Out.n_ad[n,...] = g.n_ad
        Out.n_da[n,...] = g.n_da
        n += 1

        In.fin.close()

    try:
        print_info(x_edges, y_edges, lon, lat, dx, dy, n, Out.n_ad, Out.n_da, source='None')
    except:
        try:
            print_info(x_edges, y_edges, lon, lat, dx, dy, 1, g.n_ad, g.n_da, source='None')
        except:
            pass

    if SAVE_TO_FILE:
        Out.fout.flush()
        Out.fout.close()

    if PLOT:
        try:
            plt.plot(d['lon'], d['lat'], '.')
            plot_grids(x_edges, y_edges, g.dh_mean, g.dh_error, g.n_ad, g.n_da)
            plt.show()
        except:
            print 'no data to plot!'

    if SAVE_TO_FILE:
        print 'file out -->', Out.fname_out


if __name__ == '__main__':
    main(args)
