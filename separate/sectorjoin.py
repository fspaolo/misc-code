#!/usr/bin/env python
"""
Join several HDF5 files following geographic sectors.

Join all files that have a common (given) pattern in the name. The patterns
may be numbers and/or characters. Example: 'YYYYMMDD', where YYYY is year, MM
is month and DD is day.

Notes
-----
- 'left' is inclusive and 'right' is not.
- To avoid Unix limitation on number of args use 'glob' and pass a _str_ with
  'dir + file pattern` instead of files

"""
# Fernando <fpaolo@ucsd.edu>
# November 2, 2012 

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap

from funcs import lon_180_to_360, define_sectors

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 2D file[s] to merge')
parser.add_argument('-p', dest='pattern', default="_\d\d\d\d\d\d\d\d", 
    help="pattern to match in the file names, default '_\d\d\d\d\d\d\d\d'")
parser.add_argument('-r', dest='range', nargs=2, type=float, 
    metavar=('x1', 'x2'), help='longitude range (in deg): left right')
parser.add_argument('-d', dest='step', type=float, default=90,
    metavar=('DX'), help='sector size (in deg), default 90')
parser.add_argument('-x', dest='loncol', default=3, type=int,
    help='column of longitude in the file (0,1,..), default 3')
parser.add_argument('-o', dest='prefix', default='all_', 
    help='prefix of output file name, default all_')
parser.add_argument('-s', dest='suffix', default='', 
    help='suffix of output file name, default none')
parser.add_argument('-n', dest='count', action='store_const', const=True, \
    default=False, help='count number of tasks and exit, default no')
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

def get_fname_out(stem_out, fname_in, pref='', suf=''):
    path = os.path.split(fname_in)[0]
    return os.path.join(path, ''.join([pref, stem_out, suf, '.h5']))

def get_shape_from_files(files):
    nrows = 0
    for fname in files:
        f = tb.openFile(fname, 'r')
        data = f.getNode('/data')
        nrow, ncols = data.shape
        nrows += nrow
        f.close()
    return (nrows, ncols)

def merge_files(fname_out, files, sectors, ncols=9, loncol=0):
    print 'merging:\n', files
    print 'into:\n', fname_out, '...'
    fout = tb.openFile(fname_out, 'w')
    atom = tb.Atom.from_type('float64')
    filters = tb.Filters(complib='zlib', complevel=9)
    dout = fout.createEArray('/', 'data', atom=atom, 
        shape=(0, ncols), filters=filters)
    for fname_in in files:
        fin = tb.openFile(fname_in, 'r')
        data = fin.getNode('/data')
        ###
        lon = lon_180_to_360(data[:,loncol])
        ii = int(re.findall('\d\d', fname_in)[-1]) # sector num
        x1, x2 = sectors[ii-1]
        ind, = np.where((x1 <= lon) & (lon < x2))  # only data in the sector
        if len(ind) > 0:
            dout.append(data[ind,:])
        else:
            print '{0} has no data points in sector {1]'.format(fname_in, ii)
        ###
    close_files()
    print 'done.'

def merge_all(tomerge, sectors, loncol=10, pref='', suf=''):
    for patt, fnames in tomerge.items():
        fname_out = get_fname_out(patt, fnames[0], pref, suf)
        nrows, ncols = get_shape_from_files(fnames)
        merge_files(fname_out, fnames, sectors, ncols, loncol)

#---------------------------------------------------------------------

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
count = args.count
loncol = args.loncol
x1, x2 = args.range
dx = args.step

print 'pattern to match:', PATTERN
print 'merging files: %d ...' % len(files)

sectors = define_sectors(x1, x2, dx=dx, buf=0)
tomerge = get_files_to_merge(files, PATTERN)
if count: 
    print 'number of tasks:', len(tomerge.items())
    print 'sectors:', sectors
    sys.exit()
merge_all(tomerge, sectors, loncol=loncol, pref=pref, suf=suf)

close_files()
