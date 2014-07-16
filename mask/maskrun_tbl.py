#!/usr/bin/env python

import sys
import numpy as np
import tables as tb
import argparse as ap
import matplotlib.pyplot as plt

from masklib import *

TABLE_NAME = 'mask'
BUF2 = 50   # in km

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 (Tables) file[s] to read')
parser.add_argument('-f', dest='maskfile', default=\
    '/data/alt/masks/scripps_antarctica_masks/scripps_antarctica_mask1km_v1.h5', 
    help='path to mask file [/data/alt/masks/scripps_antarctica_masks/'
    'scripps_antarctica_mask1km_v1.h5]')  
parser.add_argument('-b', dest='buffer', default=3, type=int,
    help='cut-off distance in km from any border [default: 3 km]')  
args = parser.parse_args()


def main(args):
    files = args.files
    maskfile = args.maskfile
    buf = args.buffer

    print 'processing %d files ...' % len(files)
    xm, ym, mask = get_mask(maskfile, paddzeros=BUF2+2)
    xm, ym = np.rint(xm), np.rint(ym)
    for fname in files:
        # input
        f = tb.openFile(fname, 'a')
        try:
            tbl = f.root.idr
        except:
            tbl = f.root.gla
        lon = tbl.cols.lon[:]
        lat = tbl.cols.lat[:]

        flg1, flg2 = apply_mask(lon, lat, xm, ym, mask, buf=buf, slon=0)
        _, flg3 = apply_mask(lon, lat, xm, ym, mask, buf=BUF2, slon=0)

        save_arr_as_tbl(fname, TABLE_NAME, {'fmask': flg1, 'fbord': flg2, 'fbuf': flg3})

        close_files()


if '__main__' == __name__:
    main(args)
