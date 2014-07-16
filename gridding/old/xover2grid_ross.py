#!/usr/bin/env python
"""
Take crossover files as input and construct grids of average `dh`, `dAGC`, 
`errors`, `#obs` at every grid cell: 

1) single_crossover_file    --> single_grid
2) multiple_crossover_files --> multiple_grids

Notes
-----
* Read HDF5 files with a 2D array called `data`
* Process one satellite at a time and merge the files later on.
* Indices: i,j,k = t,y,x

Example
-------
$ python xover2grid.py ~/data/fris/xover/seasonal/ers1_199*_tide.h5

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# December 15, 2011

import argparse as ap
from mpl_toolkits.basemap import interp
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter

from funcs import *

# global variables
#-------------------------------------------------------------------------

PLOT = True
SAVE_TO_FILE = False
NUM_STD = 4  # number of std to filter out data
ABS_VAL = 10  # (m), accept data if |dh| < ABS_VAL 
MAX_SIZE_DATA = 512  # (MB), to load data in memory
NODE_NAME = 'ross'
SUFFIX = 'ross_grids.h5'
TIDE_CODE = 'fortran' # fortran/matlab

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
    help='output file name [default: /input/path/sat_tmin_tmax_grids.h5]')

args = parser.parse_args()

def main(args):

    # input args
    #---------------------------------------------------------------------

    files = args.files
    x_range = (args.region[0], args.region[1])
    y_range = (args.region[2], args.region[3])
    dx = args.delta[0]
    dy = args.delta[1]
    N = len(files)

    files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s))

    fname_out = args.fname_out
    fname_out = get_fname_out(files, fname_out=fname_out, sufix=SUFFIX)

    # generate *one* set of grids *per file*
    #---------------------------------------------------------------------

    print 'processing files ...'

    isfirst = True
    for pos, fname in enumerate(files):

        file_in = tb.openFile(fname)
        check_if_can_be_loaded(file_in.root.data, MAX_SIZE_DATA)
        data = file_in.root.data.read()      # in-memory --> faster!

        # get sat/time info for every grid
        sat_name, ref_time, date, year, month = get_time_from_fname(fname)  

        # (1) FILTER DATA FIRST

        d = {}
        d['fmode1'] = data[:,10]
        d['fmode2'] = data[:,11]
        d['fbord1'] = data[:,18]
        d['fbord2'] = data[:,19]
        d['tide1'] = data[:,24]
        d['tide2'] = data[:,25]

        '''
        condition = ((d['fmode1'] == d['fmode2']) & \
                     (d['fbord1'] == 0) & (d['fbord2'] == 0)) 
        '''
        # ice mode
        if sat_name == 'ers1' or sat_name == 'ers2':
            condition = ((d['fmode1'] == 1) & (d['fmode2'] == 1) & \
                         (d['fbord1'] == 0) & (d['fbord2'] == 0) & \
                         (d['tide1'] != -9999) & (d['tide2'] != -9999))  
        # fine mode/ocean mode
        elif sat_name == 'envi' or sat_name == 'geosat' or \
             sat_name == 'gfo':
            condition = ((d['fmode1'] == 0) & (d['fmode2'] == 0) & \
                         (d['fbord1'] == 0) & (d['fbord2'] == 0) & \
                         (d['tide1'] != -9999) & (d['tide2'] != -9999))  

        ind, = np.where(condition)

        if len(ind) < 1:    # go to next file. A "void" will be left in the 3D array!
            file_in.close()
            continue

        data = data[ind,:]

        # -180/+180 -> 0/360
        _lon = data[:,0]
        _lon[_lon<0] += 360
        d['lon'] = _lon 

        #d['lon'] = data[:,0]
        d['lat'] = data[:,1]
        d['h1'] = data[:,6]
        d['h2'] = data[:,7]
        d['g1'] = data[:,8]
        d['g2'] = data[:,9]
        d['ftrack1'] = data[:,20]
        d['ftrack2'] = data[:,21]
        d['tide1'] = data[:,24]
        d['tide2'] = data[:,25]
        if TIDE_CODE == 'matlab':
            d['load1'] = data[:,26]  # not needed if using OTPS Fortran:
            d['load2'] = data[:,27]  # load corr is already in `tide` corr

        # (2) APPLY CORRECTIONS 

        if TIDE_CODE == 'matlab':
            d = apply_tide_and_load_corr(d)
            del d['tide1'], d['tide2'], d['load1'], d['load2']
        else:
            d = apply_tide_corr(d)
            del d['tide1'], d['tide2']

        del data, d['fmode1'], d['fmode2'], d['fbord1'], d['fbord2']

        # digitize lons and lats
        #-----------------------------------------------------------------

        x_edges = np.arange(x_range[0], x_range[-1]+dx, dx)
        y_edges = np.arange(y_range[0], y_range[-1]+dy, dy)
        j_bins = np.digitize(d['lon'], bins=x_edges)
        i_bins = np.digitize(d['lat'], bins=y_edges)
        nx, ny = len(x_edges)-1, len(y_edges)-1
        hx, hy = dx/2., dy/2.
        lon = (x_edges + hx)[:-1]
        lat = (y_edges + hy)[:-1]

        # output grids 
        #-----------------------------------------------------------------

        dh_mean = np.zeros((ny,nx), 'f8') * np.nan
        dh_error = np.zeros_like(dh_mean) * np.nan
        dg_mean = np.zeros_like(dh_mean) * np.nan
        dg_error = np.zeros_like(dh_mean) * np.nan
        n_ad = np.zeros((ny,nx), 'i4')
        n_da = np.zeros_like(n_ad)

        # calculations per grid cell
        #-----------------------------------------------------------------

        for i in xrange(ny):
            for j in xrange(nx):
                '''
                ind, = np.where((x_edges[j] <= d['lon']) & \
                                (d['lon'] < x_edges[j+1]) & \
                                (y_edges[i] <= d['lat']) & \
                                (d['lat'] < y_edges[i+1]))
                '''
                # single grid cell
                ind, = np.where((j_bins == j+1) & (i_bins == i+1))

                ### dh time series

                # separate --> asc/des, des/asc 
                dh_ad, dh_da = compute_dh_ad_da(d['h1'][ind], d['h2'][ind], 
                               d['ftrack1'][ind], d['ftrack2'][ind])

                # filter
                dh_ad = abs_editing(dh_ad, absval=ABS_VAL)
                dh_da = abs_editing(dh_da, absval=ABS_VAL)
                dh_ad = std_editing(dh_ad, nsd=NUM_STD, iterative=False)
                dh_da = std_editing(dh_da, nsd=NUM_STD, iterative=False)

                # mean values
                #dh_mean[i,j] = compute_ordinary_mean(dh_ad, dh_da, useall=False) 
                #dh_error[i,j] = compute_wingham_error(dh_ad, dh_da, useall=False) 
                dh_mean[i,j] = compute_weighted_mean(dh_ad, dh_da, useall=False, median=False) 
                dh_error[i,j] = compute_weighted_mean_error(dh_ad, dh_da, useall=False) 
                n_ad[i,j] = len(dh_ad)
                n_da[i,j] = len(dh_da)

                ### dAGC time series

                dg_ad, dg_da = compute_dh_ad_da(d['g1'][ind], d['g2'][ind], 
                               d['ftrack1'][ind], d['ftrack2'][ind])

                dg_ad = std_editing(dg_ad, nsd=NUM_STD, iterative=False)
                dg_da = std_editing(dg_da, nsd=NUM_STD, iterative=False)

                dg_mean[i,j] = compute_weighted_mean(dg_ad, dg_da, useall=False) 
                dg_error[i,j] = compute_weighted_mean_error(dg_ad, dg_da, useall=False) 

                # smooth 2D field
                #=========================================================
                '''
                ii = np.where(np.isnan(dh_mean))
                dh_mean[ii] = 0 # cannot have NaNs or be a masked array for `order=3`
                dh_mean = gaussian_filter(dh_mean, 0.5, order=0, output=None, mode='reflect', cval=0.0)
                dh_mean[ii] = np.nan
                '''
                #=========================================================

        # save the grids
        #-----------------------------------------------------------------

        if not SAVE_TO_FILE: continue

        if isfirst:
            # open or create output file
            isfirst = False
            n = 0
            dout, file_out = create_output_containers(
                             fname_out, dh_mean, (N,ny,nx), 
                             NODE_NAME, (1,ny,nx)) # <-- chunkshape

            # save info
            dout['lon'][:] = lon
            dout['lat'][:] = lat
            dout['x_edges'][:] = x_edges
            dout['y_edges'][:] = y_edges

        # save one set of grids per iteration (i.e., per file)
        # --> test row.append() instead!
        dout['table'].append([(sat_name, ref_time, date, year, month)]) 
        dout['table'].flush()
        dout['dh_mean'][n,...] = dh_mean
        dout['dh_error'][n,...] = dh_error
        dout['dg_mean'][n,...] = dg_mean
        dout['dg_error'][n,...] = dg_error
        dout['n_ad'][n,...] = n_ad
        dout['n_da'][n,...] = n_da
        n += 1

        file_in.close()

    if SAVE_TO_FILE:
        file_out.flush()
        file_out.close()

    try:
        print_info(x_edges, y_edges, lon, lat, dx, dy, n)
    except:
        print_info(x_edges, y_edges, lon, lat, dx, dy, 1)

    if PLOT:
        # uncomment options bolow
        xx, yy = np.meshgrid(lon, lat)     # output grid
        ind = np.where(np.isnan(dh_mean))
        dh_mean[ind] = 0 # cannot have NaNs or be a masked array for `order=3`

        #dh_mean = interp(dh_mean, lon, lat, xx, yy, order=1)                          # good!!!

        #rbs = RectBivariateSpline(lat, lon, dh_mean, kx=3, ky=3)
        #dh_mean = rbs(lat, lon)

        #x = xx.ravel()
        #y = yy.ravel()
        #z = dh_mean.ravel()
        #dh_mean = griddata((x, y), z, (lon[None,:], lat[:,None]), method='cubic') 

        dh_mean = gaussian_filter(dh_mean, 0.5, order=0, output=None, mode='reflect', cval=0.0)

        dh_mean[ind] = np.nan

        plot_grids(x_edges, y_edges, dh_mean, dg_mean, n_ad, n_da)
        plt.show()

    print 'file out -->', fname_out


if __name__ == '__main__':
    main(args)
