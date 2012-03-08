#!/usr/bin/env python
"""
Perform cross-calibration of time series (given a 3D array).

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 8, 2012

import sys
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
    help='HDF5 files with grids to read (3D arrays)')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: same as input file]')

args = parser.parse_args()


def main(args):

    fname_in = args.file_in[0]
    fname_out = get_fname_out(fname_in, fname_out=args.fname_out, 
                              sufix='grids_calib.h5')

    # input data --> dictionary and file
    #---------------------------------------------------------------------

    d, file_in = get_data(fname_in, NODE_NAME, 'a')

    sat_missions = np.unique(d['sat_name'])
    N, ny, nx = d['dh_mean'].shape    # i,j,k = t,y,x

    print 'calibrating time series ...'

    # iterate over every grid cell (all times): j,k = y,x
    #-----------------------------------------------------------------

    isfirst = True

    for i in xrange(ny):
        for j in xrange(nx):
            print 'TEST'
            df = create_df_with_sats(d['table'], d['dh_mean_corr'], j, i,
                                     is_seasonal=True, ref_month=2)
            print df
            sys.exit()



if __name__ == '__main__':
    main(args)
