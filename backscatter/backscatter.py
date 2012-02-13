"""
Applys the backscatter correction to a set of time series.

Implements the correction using the AGC following Zwally, et al. (2005).

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# February 7, 2012

import argparse as ap

from funcs import *

# global variables
#-------------------------------------------------------------------------

SAVE_TO_FILE = True
NODE_NAME = 'fris'
PLOT = False

#-------------------------------------------------------------------------

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file_in', nargs=1, 
    help='HDF5 file with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: same as input file]')

args = parser.parse_args()

def main(args):

    fname_in = args.file_in[0]
    fname_out = get_fname_out(fname_in, fname_out=args.fname_out, sufix='corr.h5')

    # input data --> dictionary and file
    #---------------------------------------------------------------------

    d, file_in = get_data(fname_in, NODE_NAME, 'a')

    #---------------------------------------------------------------------

    sat_missions = np.unique(d['sat_name'])
    nt, nx, ny =  d['dh_mean'].shape    # i,j,k = t,x,y

    print 'processing time series ...'

    isfirst = True
    # iterate over every sat mission
    for sat in sat_missions:

        # iterate over every grid cell (all times): j,k = x,y
        #-----------------------------------------------------------------

        for k in xrange(ny):
            for j in xrange(nx):

                dh_mean, dg_mean = d['dh_mean'][:,j,k], d['dg_mean'][:,j,k]
                dh_mean[0] = np.nan
                dg_mean[0] = np.nan
                dh_mean_corr, R, dHdG = \
                        backscatter_corr(dh_mean, dg_mean, robust=True)
                dh_mean[0] = 0.
                dg_mean[0] = 0.
                dh_mean_corr[0] = 0.

                # plot figures
                if PLOT: plot_figures(dh_mean_corr, dh_mean, dg_mean, R, dHdG)

                # save one TS per grid cell at a time
                #---------------------------------------------------------

                if not SAVE_TO_FILE: continue

                if isfirst:
                    # open or create output file
                    isfirst = False
                    atom = tb.Atom.from_dtype(dh_mean_corr.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    try:
                        g = file_in.getNode('/', NODE_NAME)
                        c = file_in.createCArray(g, 'dh_mean_corr', atom, (nt,nx,ny), '', filters)
                    except:
                        c = file_in.getNode('/%s' % NODE_NAME, 'dh_mean_corr')

                c[:,j,k] = dh_mean_corr

    file_in.close()

    print 'out file -->', fname_in


if __name__ == '__main__':
    main(args)
