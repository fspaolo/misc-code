"""
Filter selected flags.

Fernando Paolo <fpaolo@ucsd.edu>
August 25, 2012
"""

import os
import sys
import numpy as np
import tables as tb
import argparse as ap

ftrkcol = 12  # track flag column

parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 Table file(s) to read')
args = parser.parse_args()


def save_arr(fname, data):
    """
    Open/Create specific output file and save specific variables.
    """
    if '.txt' in fname:
        np.savetxt(fname, data, fmt='%f')
    elif '.h5' in fname:
        fout = tb.openFile(fname, 'w')
        shape = data.shape
        atom = tb.Atom.from_dtype(data.dtype)
        filters = tb.Filters(complib='zlib', complevel=9)
        dout = fout.createCArray('/','data', atom=atom, 
                                 shape=shape, filters=filters)
        dout[:] = data
        fout.close()
    else:
        print 'no output file extension!'
        pass


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


files = args.files
print 'filtering files:', len(files), '...'

nasc = 0
ndes = 0
for f in files:
    ### input
    fin = tb.openFile(f)
    data = fin.getNode('/data')
    ftrk = data[:,ftrkcol] 
    nrows = ftrk.shape[0]

    ### proc
    if nrows > 0:
        i_a, = np.where(ftrk == 0)
        i_d, = np.where(ftrk == 1)

        ### output
        if (i_a.shape[0] > 0) and (i_d.shape[0] > 0):  # both are needed
            fname_a = os.path.splitext(f)[0] + '_a.h5' 
            fname_d = os.path.splitext(f)[0] + '_d.h5' 
            save_arr(fname_a, data[i_a,:])
            save_arr(fname_d, data[i_d,:])
            nasc += i_a.shape[0]
            ndes += i_d.shape[0]

print 'done!'
print 'ascending points:', nasc
print 'descending points:', ndes
close_files()
