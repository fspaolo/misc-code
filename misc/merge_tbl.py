#!/usr/bin/env python
"""
Merge several HDF5 or ASCII files.

Merge all files that have a common (given) pattern in the name.
The patterns may be numbers and/or characters. Example: 'YYYYMM', 
where YYYY is year and MM is month.

"""
# Fernando <fpaolo@ucsd.edu>
# August 25, 2012 

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5/ASCII file[s] to merge')
parser.add_argument('-p', dest='pattern', default="_\d\d\d\d", 
                    help="pattern to match in the file names " 
                    "[default: '_\d\d\d\d\d\d']")
parser.add_argument('-o', dest='prefix', default='all_', 
                    help='prefix of output file name [default: all_]')
parser.add_argument('-s', dest='suffix', default='', 
                    help='suffix of output file name [default: none]')
parser.add_argument('-c', dest='struct', default='idr', 
                    help='data (table) structure: idr or gla [default: idr]')
args = parser.parse_args()


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


if len(args.files) > 1:
    files = args.files
else:
    # to avoid Unix limitation on number of cmd args: use `glob`
    from glob import glob
    # pass a str with `dir + file pattern` instead of files
    files = glob(args.files[0])   

PATTERN = str(args.pattern)
prefix = args.prefix
suffix = args.suffix
struct = args.struct
ext = '.h5'

print 'pattern to match:', PATTERN

if files:
    path, _ = os.path.split(files[0])          # path of first file
    print 'merging files: %d ...' % len(files)
else:
    print 'no input files!'
    sys.exit()

while files:

    ### 1) search first file that matches the pattern

    for j, ff in enumerate(files):    
        match = re.search(PATTERN, ff)
        if match:
            isfirst = True
            match = match.group()
            # check if there is character in common (duplicate)
            if prefix[-1] == match[0] and match[0] in ['_', '-']:
                pattern = match[1:]                    
            else:
                pattern = match[:]                    
            # avoid duplicating the extension in the file name
            pattern, _ = os.path.splitext(pattern)
            fname = ''.join([prefix, pattern, suffix, ext])
            outfile = os.path.join(path, fname)
            break
        else:
            files[j] = None          # mark no matched files
            continue

    ### 2) match all files to the first found 

    for i, f in enumerate(files):    
        if match and f and (match in f):
            # tables to save
            #---------------------------------------------------------
            fin = tb.openFile(f)
            if struct == 'idr':
                table1_out = fin.root.idr
            elif struct == 'gla':
                table1_out = fin.root.gla
            else:
                raise IOError('-c must be idr/gla')
            table2_out = fin.root.mask
            table3_out = fin.root.trk

            if isfirst:
                filters = tb.Filters(complib='zlib', complevel=9)
                fout = tb.openFile(outfile, 'w')
                t1 = fout.createTable('/', table1_out.name, table1_out.description, '', filters)
                t2 = fout.createTable('/', table2_out.name, table2_out.description, '', filters)
                t3 = fout.createTable('/', table3_out.name, table3_out.description, '', filters)
                isfirst = False
            t1.append(table1_out[:])
            t2.append(table2_out[:])
            t3.append(table3_out[:])
            t1.flush()
            t2.flush()
            t3.flush()
            #---------------------------------------------------------
            files[i] = None      # mark merged files

    files = filter(None, files)  # filter out already processed files (marked `None`)
    if not isfirst:
        print 'file created:', outfile
    else:
        print 'no matching files!'
    close_files()

print 'done.'
