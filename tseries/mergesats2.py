#!/usr/bin/env python
"""
Example
-------
python mergesats2.py /data/alt/ra/ers1/hdf/antarctica/xovers/*_mts.h5 /data/alt/ra/ers2/hdf/antarctica/xovers/*_mts.h5 /data/alt/ra/envi/hdf/antarctica/xovers/*_mts.h5

Notes
-----
- Output file is saved in the first path of the argument list.
- Check in the code the variables to merge and add new ones, also check:
    - GetData()
    - OutputContainer()

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

SAVE_TO_FILE = True

# parse command line arguments
parser = arg.ArgumentParser()
parser.add_argument('files', nargs='+', 
    help='HDF5 files with grids to read (3D arrays)')
args = parser.parse_args()


def get_output_dimensions(fnames):
    N = 0
    for fname in fnames:
        f = tb.openFile(fname, 'r')
        dh_mean = f.getNode('/dh_mean')
        nt, ny, nx = dh_mean.shape
        N += nt
        f.close()
    return (N, ny, nx)


def get_fname_out2(files, prefix=None, suffix=None):
    """
    Construct the output file name with the min and max times 
    from the input files.
    """
    path, name = os.path.split(files[0])  # path from any file
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
    return os.path.join(path, name)


def main(args):

    files = args.files
    files.sort(key=lambda s: re.findall('\d\d\d\d\d\d+', s)) # important!!!

    n1, n2 = 0, 0

    isfirst = True
    for fname in files:

        # input data
        din = GetData(fname)

        # output data
        if not SAVE_TO_FILE: 
            din.file.close()
            continue

        if isfirst:
            # open or create output file
            fname_out = get_fname_out2(files, prefix='all', suffix=None)
            N, ny, nx = get_output_dimensions(files)
            dout = OutputContainers(fname_out, (N,ny,nx), (N/3,ny,nx))

            # save info
            dout.lon[:] = din.lon
            dout.lat[:] = din.lat
            dout.x_edges[:] = din.x_edges
            dout.y_edges[:] = din.y_edges

            isfirst = False

        nt = din.time.shape[0]
        n2 += nt
        print 'total time steps: {0}, chunk indices: [{1}:{2}], chunk nt: {3}'.format(N, n1, n2, nt)

        dout.satname[n1:n2] = din.satname
        dout.time[n1:n2] = din.time
        dout.dh_mean[n1:n2,...] = din.dh_mean[:nt]
        dout.dh_error[n1:n2,...] = din.dh_error[:nt]
        dout.dh_error2[n1:n2,...] = din.dh_error2[:nt]
        dout.dg_mean[n1:n2,...] = din.dg_mean[:nt]
        dout.dg_error[n1:n2,...] = din.dg_error[:nt]
        dout.dg_error2[n1:n2,...] = din.dg_error2[:nt]
        dout.n_ad[n1:n2,...] = din.n_ad[:nt]
        dout.n_da[n1:n2,...] = din.n_da[:nt]
        try:
            dout.dh_mean_mixed_const[n1:n2,...] = din.dh_mean_mixed_const[:nt]
        except:
            pass
        try:
            dout.dh_mean_short_const[n1:n2,...] = din.dh_mean_short_const[:nt]
        except:
            pass

        n1 = n2
        din.file.close()

    dout.file.flush()
    dout.file.close()

    print 'out file -->', fname_out


if __name__ == '__main__':
    main(args)
