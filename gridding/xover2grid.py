#!/usr/bin/env python
"""
Take crossover files as input and construct grids of `average dh, dAGC, 
errors, #obs` at every grid cell: 

single_crossover_file    --> single_grid
multiple_crossover_files --> multiple_grids

Notes
-----
* Process one satellite at a time and merge the files later on.
* indices: i,j,k = t,x,y

Example
-------
$ python xover2grid.py ~/data/fris/xover/seasonal/ers1_199*_tide.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

ABSVAL = 10  # (m) to edit abs(dh)
MAX_SIZE_DATA = 512  # (MB) to load data in memory

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='crossover files to read (HDF5 2D array)')
parser.add_argument('-r', dest='region', nargs=4, type=float, 
    metavar=('l', 'r', 'b', 't'), default=(-100, -20, -82, -75),
    help='coordinates of grid domain: left right bottom top')
parser.add_argument('-d', dest='delta', nargs=2, type=float, 
    metavar=('dx', 'dy'), default=(1.2, 0.4),
    help='size of grid cells: dx dy (deg) [default: 1.2 x 0.4]')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/sat_tmin_tmax_dh_grids.h5]')

args = parser.parse_args()

def main(args):

    # input args
    #---------------------------------------------------------------------

    files = args.files
    x_range = (args.region[0], args.region[1])
    y_range = (args.region[2], args.region[3])
    dx = args.delta[0]
    dy = args.delta[1]

    files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s))

    fname_out = args.fname_out
    fname_out = get_fname_out(files, fname_out=fname_out, sufix='dh_grids.h5')

    # generate *one* set of grids *per file*
    #---------------------------------------------------------------------

    isfirst = True
    for pos, fname in enumerate(files):

        file_in = tb.openFile(fname)
        check_if_can_be_loaded(file_in.root.data, MAX_SIZE_DATA)
        data = file_in.root.data.read()      # in-memory --> faster!

        # get sat/time info for every grid
        sat_name, ref_time, year, month = get_time_from_fname(fname)  

        # (1) FILTER DATA FIRST

        d = {}
        d['fmode1'] = data[:,10]
        d['fmode2'] = data[:,11]
        d['fbord1'] = data[:,18]
        d['fbord2'] = data[:,19]

        '''
        condition = ((d['fmode1'] == d['fmode2']) & \
                     (d['fbord1'] == 0) & (d['fbord2'] == 0)) 
        '''
        if sat_name == 'ers1' or sat_name == 'ers2':
            condition = ((d['fmode1'] == 1) & (d['fmode2'] == 1) & \
                         (d['fbord1'] == 0) & (d['fbord2'] == 0))  # ice mode
        elif sat_name == 'envisat' or sat_name == 'geosat' or sat_name == 'gfo':
            condition = ((d['fmode1'] == 0) & (d['fmode2'] == 0) & \
                         (d['fbord1'] == 0) & (d['fbord2'] == 0))  # fine mode/ocean mode

        ind, = np.where(condition)

        if len(ind) < 1:    # go to next file
            file_in.close()
            continue

        data = data[ind,:]

        d['lon'] = data[:,0]
        d['lat'] = data[:,1]
        d['h1'] = data[:,6]
        d['h2'] = data[:,7]
        d['g1'] = data[:,8]
        d['g2'] = data[:,9]
        d['ftrack1'] = data[:,20]
        d['ftrack2'] = data[:,21]
        d['tide1'] = data[:,24]
        d['tide2'] = data[:,25]
        d['load1'] = data[:,26]
        d['load2'] = data[:,27]

        # (2) APPLY CORRECTIONS 

        d = apply_tide_and_load_corr(d)

        del data, d['fmode1'], d['fmode2'], d['fbord1'], d['fbord2']
        del d['tide1'], d['tide2'], d['load1'], d['load2']

        # digitize lons and lats
        #-----------------------------------------------------------------

        x_edges = np.arange(x_range[0], x_range[-1]+dx, dx)
        y_edges = np.arange(y_range[0], y_range[-1]+dy, dy)
        j_bins = np.digitize(d['lon'], bins=x_edges)
        k_bins = np.digitize(d['lat'], bins=y_edges)
        nx, ny = len(x_edges)-1, len(y_edges)-1
        hx, hy = dx/2., dy/2.
        lon = (x_edges + hx)[:-1]
        lat = (y_edges + hy)[:-1]

        # output grids 
        #-----------------------------------------------------------------

        dh_mean = np.empty((ny,nx), 'f8') * np.nan
        dh_error = np.empty_like(dh_mean) * np.nan
        dg_mean = np.empty_like(dh_mean) * np.nan
        dg_error = np.empty_like(dh_mean) * np.nan
        n_ad = np.empty((ny,nx), 'i4')
        n_da = np.empty_like(n_ad)

        # calculations per grid cell
        #-----------------------------------------------------------------

        for k in xrange(ny):
            for j in xrange(nx):
                '''
                ind, = np.where((x_edges[j] <= d['lon']) & \
                                (d['lon'] < x_edges[j+1]) & \
                                (y_edges[k] <= d['lat']) & \
                                (d['lat'] < y_edges[k+1]))
                '''
                # single grid cell
                ind, = np.where((j_bins == j+1) & (k_bins == k+1))

                ### dh time series

                # separate --> asc/des, des/asc 
                dh_ad, dh_da = compute_dh_ad_da(d['h1'][ind], d['h2'][ind], 
                               d['ftrack1'][ind], d['ftrack2'][ind])

                # filter
                #dh_ad = std_iterative_editing(dh_ad, nsd=3)
                #dh_da = std_iterative_editing(dh_da, nsd=3)
                dh_ad = abs_value_editing(dh_ad, absval=ABSVAL)
                dh_da = abs_value_editing(dh_da, absval=ABSVAL)

                # mean values
                #dh_mean[k,j] = compute_ordinary_mean(dh_ad, dh_da, useall=False) 
                dh_mean[k,j] = compute_weighted_mean(dh_ad, dh_da, useall=False) 
                dh_error[k,j] = compute_weighted_mean_error(dh_ad, dh_da, useall=False) 
                #dh_error[k,j] = compute_wingham_error(dh_ad, dh_da, useall=False) 
                n_ad[k,j] = len(dh_ad)
                n_da[k,j] = len(dh_da)

                ### dAGC time series

                dg_ad, dg_da = compute_dh_ad_da(d['g1'][ind], d['g2'][ind], 
                               d['ftrack1'][ind], d['ftrack2'][ind])

                #dg_ad = std_iterative_editing(dg_ad, nsd=3)
                #dg_da = std_iterative_editing(dg_da, nsd=3)

                dg_mean[k,j] = compute_weighted_mean(dg_ad, dg_da, useall=False) 
                dg_error[k,j] = compute_weighted_mean_error(dg_ad, dg_da, useall=False) 

        # save the grids
        #-----------------------------------------------------------------

        if isfirst:
            # open or create output file
            isfirst = False
            atom = tb.Atom.from_dtype(dh_mean.dtype)
            nk, nj = dh_mean.shape    # ny,nx
            dout, file_out = create_output_containers(fname_out, atom, (nj,nk))

            # save info
            dout['x_edges'][:] = x_edges
            dout['y_edges'][:] = y_edges
            dout['lon'][:] = lon
            dout['lat'][:] = lat

        # append one set of grids per file
        dout['table'].append([(sat_name, ref_time, year, month)]) ### test row.append() instead!
        dout['dh_mean'].append(dh_mean.reshape(1, nj, nk))
        dout['dh_error'].append(dh_error.reshape(1, nj, nk))
        dout['dg_mean'].append(dg_mean.reshape(1, nj, nk))
        dout['dg_error'].append(dg_error.reshape(1, nj, nk))
        dout['n_ad'].append(n_ad.reshape(1, nj, nk))
        dout['n_da'].append(n_da.reshape(1, nj, nk))
        dout['table'].flush()

        file_in.close()

    file_out.flush()
    file_out.close()

    print_info(x_edges, y_edges, lon, lat, dx, dy)

    plot_grids(x_edges, y_edges, dh_mean, dg_mean, n_ad, n_da)
    plt.show()

    print 'file out -->', fname_out


if __name__ == '__main__':
    main(args)
