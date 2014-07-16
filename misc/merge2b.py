#!/usr/bin/env python
"""
Merge several HDF5 or ASCII files.

Merge all files that have a common (given) pattern in the name.
The patterns may be numbers and/or characters. Example: 'YYYYMMDD', 
where YYYY is year, MM is month and DD is day.

"""
# Fernando <fpaolo@ucsd.edu>
# November 2, 2012 

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 2D file[s] to merge')
parser.add_argument('-p', dest='pattern', default="_\d\d\d\d\d\d\d\d", 
    help="pattern to match in the file names, default '_\d\d\d\d\d\d\d\d'")
parser.add_argument('-o', dest='prefix', default='all_', 
    help='prefix of output file name, default all_')
parser.add_argument('-s', dest='suffix', default='', 
    help='suffix of output file name, default none')
args = parser.parse_args()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 

def get_files_to_merge(files, pattern):
    tomerge = {}
    patterns = np.unique(re.findall(pattern, ' '.join(files)))
    for s in patterns:
        tomerge[s] = [f for f in files if s in f]
    return tomerge

def get_fname_out(stem, fnamein, pref='', suf=''):
    path = os.path.split(fnamein)[0]
    return os.path.join(path, ''.join([pref, stem, suf, '.h5']))

def get_ndim_out(files):
    nrows = 0
    for fname in files:
        f = tb.openFile(fname, 'r')
        data = f.getNode('/data')
        nrow, ncols = data.shape
        nrows += nrow
        f.close()
    return (nrows, ncols)

def merge_files(fnameout, shape, files):
    print 'merging:\n', files
    print 'into:\n', fnameout, '...'
    fout = tb.openFile(fnameout, 'w')
    atom = tb.Atom.from_type('float64')
    filters = tb.Filters(complib='zlib', complevel=9)
    dout = fout.createCArray('/', 'data', atom=atom, 
        shape=shape, filters=filters)
    i1, i2 = 0, 0
    for fnamein in files:
        fin = tb.openFile(fnamein, 'r')
        data = fin.getNode('/data')
        i2 += data.shape[0]
        dout[i1:i2] = data[:]
        i1 = i2
    close_files()
    print 'done.'


if len(args.files) > 1:
    files = args.files
else:
    # to avoid Unix limitation on number of cmd args: use `glob`
    # and pass a _str_ with `dir + file pattern` instead of files
    from glob import glob
    files = glob(args.files[0])   

PATTERN = str(args.pattern)
pref = args.prefix
suf = args.suffix
#path, _ = os.path.split(files[0])          # path of first file

print 'pattern to match:', PATTERN
print 'merging files: %d ...' % len(files)

tomerge = get_files_to_merge(files, PATTERN)
for patt, fnames in tomerge.items():
    fnameout = get_fname_out(patt, fnames[0], pref, suf)
    shape = get_ndim_out(fnames)
    merge_files(fnameout, shape, fnames)

close_files()

print 'done.'
