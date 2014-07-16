#!/usr/bin/env python

import sys
import numpy as np
import tables as tb
import argparse as ap
import matplotlib.pyplot as plt

from masklib import *

# in km, a third flag with data buffer. Use `None` to disable.
BUF2 = 50   

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 (2D) file(s) to read')
parser.add_argument('-f', dest='maskfile', default=\
    '/data/alt/masks/scripps/scripps_antarctica_mask1km_v1.h5', 
    help='path to mask file [/data/alt/masks/scripps/'
    'scripps_antarctica_mask1km_v1.h5]')
parser.add_argument('-b', dest='buffer', default=3, type=int,
    help='cut-off distance in km from any border, default 3 km')  
parser.add_argument('-x', dest='loncol', default=3, type=int,
    help='column of longitude in the file: 0,1,.., default 3')  
parser.add_argument('-y', dest='latcol', default=2, type=int,
    help='column of latitude in the file: 0,1,.., default 2')  
args = parser.parse_args()


def main(args):
    files = args.files
    maskfile = args.maskfile
    buf = args.buffer
    loncol = args.loncol
    latcol = args.latcol

    print 'processing %d files ...' % len(files)
    ### mask
    if BUF2 is not None:
        xm, ym, mask = get_mask(maskfile, paddzeros=BUF2+2)
    else:
        xm, ym, mask = get_mask(maskfile, paddzeros=0)
    xm, ym = np.rint(xm), np.rint(ym)

    for fname in files:
        ### input
        f = tb.openFile(fname)
        data = f.root.data
        lon = data[:,loncol]
        lat = data[:,latcol]

        flg1, flg2 = apply_mask(lon, lat, xm, ym, mask, buf=buf, slon=0)
        if BUF2 is not None:
            _, flg3 = apply_mask(lon, lat, xm, ym, mask, buf=BUF2, slon=0)

        ### output 
        fnameout = os.path.splitext(fname)[0] + '_mask.h5'
        if BUF2 is not None:
            save_arr_as_mat(fnameout, [data[:], flg1, flg2, flg3])
        else:
            save_arr_as_mat(fnameout, [data[:], flg1, flg2])

        close_files()


if '__main__' == __name__:
    main(args)
