#!/usr/bin/env python
"""
Merge several HDF5 or ASCII files.

Merge all files that have a common (given) pattern in the name.
The patterns may be numbers and/or characters. Example: 'YYYYMM', 
where YYYY is year and MM is month.

Fernando <fpaolo@ucsd.edu>
November 4, 2010
"""

import numpy as np
import tables as tb
import argparse as ap
import re
import sys
import os

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('file', nargs='+', help='HDF5/ASCII file[s] to merge')
parser.add_argument('-p', dest='pattern', default="_\d\d\d\d", 
                    help="pattern to match in the file names " 
                    "[default: '_\d\d\d\d']")
parser.add_argument('-o', dest='prefix', default='all_', 
                    help='prefix of output file name [default: all_]')
parser.add_argument('-a', dest='ascii', default=False, action='store_const',
                    const=True, help='reads and writes ASCII files ' 
                    '[default HDF5]')

args = parser.parse_args()

if len(args.file) > 1:
    files = args.file
else:
    # to avoid Unix limitation on number of cmd args: use `glob`
    from glob import glob
    # pass a str with `dir + file pattern` instead of files
    files = glob(args.file[0])   

PATTERN = str(args.pattern)


print 'pattern to match:', PATTERN

if args.ascii:
    ext = '.txt'
    print 'reading and writing ASCII files'
else:
    ext = '.h5'
    print 'reading and writing HDF5 files'

if files:
    path, _ = os.path.split(files[0])          # path of first file
    print 'merging files: %d ...' % len(files)
else:
    print 'no input files!'
    sys.exit()

while files:

    ### search first file that matches the pattern

    for j, ff in enumerate(files):    
        match = re.search(PATTERN, ff)
        if match:
            isfirst = True
            match = match.group()
            # check if there is character in common (duplicate)
            if args.prefix[-1] == match[0] and match[0] in ['_', '-']:
                suffix = match[1:]                    
            else:
                suffix = match[:]                    
            # avoid duplicating the extension in the file name
            suffix, ext2 = os.path.splitext(suffix)
            fname = ''.join([args.prefix, suffix, ext])
            outfile = os.path.join(path, fname)
            break
        else:
            files[j] = None          # mark no matched files
            continue

    ### match all files to the first found 

    for i, f in enumerate(files):    
        if match and f and match in f:
            if args.ascii:
                data = np.loadtxt(f)
                # still need to complete for ASCII !!!!!
                print 'still need to complete for ASCII !!!!!'
            else:
                fin = tb.openFile(f)
                data = fin.root.data
                #data = fin.root.data.read()

                if isfirst:
                    fout = tb.openFile(outfile, 'w')
                    shape = (0, data.shape[1])
                    atom = tb.Atom.from_dtype(data.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    dout = fout.createEArray(fout.root, 'data', atom=atom, 
                                             shape=shape, filters=filters)
                    isfirst = False

                dout.append(data[:])
                fin.close()

            files[i] = None            # mark merged files
            
    files = filter(None, files)    # filter out marked files
    if not args.ascii:
        try:                               # if 'fout' was open
            if fout.isopen:
                fout.close()
                print 'file created:', outfile
        except:
            print 'no matching files!'
            pass

print 'done.'
