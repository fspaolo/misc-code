#!/usr/bin/env python
"""
"""
# Fernando Paolo <fpaolo@ucsd.edu>
# March 7, 2012

import os
import sys
import re
import numpy as np
import tables as tb
import pandas as pn
import argparse as arg

from funcs import *

# global variables
#-------------------------------------------------------------------------

SAVE_TO_FILE = True
NODE_NAME = ''

#-------------------------------------------------------------------------

# parse command line arguments
parser = arg.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='HDF5 files with 3d grids to read')
parser.add_argument('-o', dest='fname_out', default=None,
    help='output file name [default: /input/path/all_t1_t2_grids_mts.h5]')

args = parser.parse_args()


def get_output_dimensions(fnames, node_name):
    N= 0
    for fname in fnames:
        f = tb.openFile(fname, 'r')
        if node_name:
            data = f.getNode('/', node_name)
        else:
            data = f.root
        N += data.table.nrows
        ny, = data.lat.shape
        nx, = data.lon.shape
        f.close()
    return (N, ny, nx)


def get_fname_out2(files, fname_out=None, prefix=None, suffix=None):
    """
    Construct the output file name with the min and max times 
    from the input files.
    """
    path, name = os.path.split(files[0])  # path from any file
    if fname_out is None:
        times = [re.findall('\d\d\d\d\d\d+', fname) for fname in files]
        t_1 = [t1 for t1, t2 in times]
        t_2 = [t2 for t1, t2 in times]
        t_min, t_max =  min(t_1), max(t_2)
        if prefix is None:
            prefix = name.split('_')[0]  # sat name
        if suffix is None:
            suffix = name.split('_')[3:]
            suffix = '_'.join(suffix)
        name = '_'.join([prefix, t_min, t_max, suffix])
    else:
        name = fname_out
    return os.path.join(path, name)


def main(args):

    files = args.files
    fname_out = args.fname_out

    files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s)) # important!!!

    n1, n2 = 0, 0

    isfirst = True
    for fname in files:

        # input data
        #-------------------------------------------------------------

        d, fin = get_data_from_file(fname, NODE_NAME)

        # output data
        #-------------------------------------------------------------

        if not SAVE_TO_FILE: 
            fin.close()
            continue

        if isfirst:
            # open or create output file
            fname_out = get_fname_out2(files, fname_out=fname_out, 
                prefix='all', suffix=None)
            N, ny, nx = get_output_dimensions(files, NODE_NAME)
            arr = np.empty(1, 'f8')
            dout, fout = create_output_containers(fname_out, arr, 
                (N,ny,nx), NODE_NAME, None)

            # save info
            dout['lon'][:] = d['lon'][:]
            dout['lat'][:] = d['lat'][:]
            dout['x_edges'][:] = d['x_edges'][:]
            dout['y_edges'][:] = d['y_edges'][:]

            isfirst = False

        nrows = d['table'].nrows
        n2 += nrows 
        print 'total: %d, ' % N, 'indices: [%d:%d], ' % (n1, n2), 'nrows: %d' % nrows

        dout['table'].append(d['table'][:nrows])
        dout['table'].flush()
        dout['dh_mean'][n1:n2,...] = d['dh_mean'][:nrows]
        try:
            dout['dh_mean_corr'][n1:n2,...] = d['dh_mean_corr'][:nrows]
        except:
            pass
        dout['dh_error'][n1:n2,...] = d['dh_error'][:nrows]
        dout['dh_error2'][n1:n2,...] = d['dh_error2'][:nrows]
        dout['dg_mean'][n1:n2,...] = d['dg_mean'][:nrows]
        dout['dg_error'][n1:n2,...] = d['dg_error'][:nrows]
        dout['dg_error2'][n1:n2,...] = d['dg_error2'][:nrows]
        dout['n_ad'][n1:n2,...] = d['n_ad'][:nrows]
        dout['n_da'][n1:n2,...] = d['n_da'][:nrows]

        n1 = n2
        fin.close()

    fout.flush()
    fout.close()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
