#!/usr/bin/env python
"""
Merge several HDF5 or ASCII files.

Merge all files that have a common (given) pattern in the name.
The patterns may be numbers and/or characters. Example: 'YYYYMMDD', 
where YYYY is year, MM is month and DD is day.

"""
# Fernando Paolo <fpaolo@ucsd.edu>
# Jan 17, 2013

import os
import sys
import re
import numpy as np
import tables as tb
import argparse as ap
from mpi4py import MPI
from Queue import Queue

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='+', help='HDF5 2D file[s] to merge')
parser.add_argument('-p', dest='pattern', default="_\d\d\d\d\d\d\d\d", 
    help='pattern to match in the file names, default "_\d\d\d\d\d\d\d\d"')
parser.add_argument('-o', dest='prefix', default='all_', 
    help='prefix of output file name, default all_')
parser.add_argument('-s', dest='suffix', default='', 
    help='suffix of output file name, default none')
parser.add_argument('-c', dest='ncols', required=True, nargs=1,
    help='number of columns in the (2D) files, default none')
parser.add_argument('-n', dest='count', action='store_const', const=True, \
    default=False, help='count number of tasks and exit, default no')
args = parser.parse_args()

# If needed, uses `glob` to avoid Unix limitation on number of cmd args.
# To use it, instead of _file names_ pass a _str_ with "dir + file pattern".
if len(args.files) > 1:
    FILES = args.files
else:
    from glob import glob
    FILES = glob(args.files[0])   

PATTERN = str(args.pattern)
PREFIX = args.prefix
SUFFIX = args.suffix
NCOLS = args.ncols
COUNT = args.count

WORKTAG = 1
DIETAG = 2

class Work(object):
    """Get subsets of files to merge."""
    def __init__(self, pattern, files):
        tomerge = Queue()
        #########################################################
        print 'pattern:', pattern
        patterns = np.unique(re.findall(pattern, ' '.join(files)))
        print 'patterns', patterns
        #########################################################
        for s in patterns:
            tomerge.put( (s, [f for f in files if s in f]) ) # ('patt', [files])
        self.work = tomerge

    def get_next(self):
        if self.work.empty():
            return None
        return self.work.get()
 

def do_work(work, ncols, pref='', suf=''):
    """Merge the subset of files in work."""
    patt, fnames = work
    fnameout = get_fname_out(patt, fnames[0], pref, suf)
    merge_files(fnameout, ncols, fnames)


def get_fname_out(stem, fnamein, pref='', suf=''):
    path = os.path.split(fnamein)[0]
    return os.path.join(path, ''.join([pref, stem, suf, '.h5']))


def merge_files(fname, ncols, files):
    print 'merging:\n', files
    print 'into:\n', fname, '...'
    fout = tb.openFile(fname, 'w')
    atom = tb.Atom.from_type('float64')
    filters = tb.Filters(complib='zlib', complevel=9)
    dout = fout.createEArray('/', 'data', atom=atom, 
        shape=(0, ncols), filters=filters)
    for fnamein in files:
        fin = tb.openFile(fnamein, 'r')
        data = fin.getNode('/data')
        dout.append(data[:])
    close_files()
    print 'done.'


def close_files():
    for fid in tb.file._open_files.values():
        fid.close() 


def process_result(result):
    pass

# MPI functions

def master(comm):
    num_procs = comm.Get_size()
    status = MPI.Status()
    
    # generate work queue
    wq = Work(PATTERN, FILES)
    if COUNT: print 'ntasks:', wq.work.qsize(); sys.exit()  # run serial

    # Seed the slaves, send one unit of work to each slave (rank)
    for rank in xrange(1, num_procs):
        work = wq.get_next()
        comm.send(work, dest=rank, tag=WORKTAG)
    
    # Loop over getting new work requests until there is no more work to be done
    while True:
        work = wq.get_next()
        if not work: break
    
        # Receive results from a slave
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        #process_result(result)

        # Send the slave a new work unit
        comm.send(work, dest=status.Get_source(), tag=WORKTAG)
    
    # No more work to be done, receive all outstanding results from slaves
    for rank in xrange(1, num_procs): 
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        #process_result(result)

    # Tell all the slaves to exit by sending an empty message with DIETAG
    for rank in xrange(1, num_procs):
        comm.send(0, dest=rank, tag=DIETAG)


def slave(comm):
    my_rank = comm.Get_rank()
    status = MPI.Status()

    while True:
        # Receive a message from the master
        work = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        # Check the tag of the received message.
        if status.Get_tag() == DIETAG: break 

        # Do the work
        result = do_work(work, NCOLS, PREFIX, SUFFIX)

        # Send the result back
        comm.send(result, dest=0, tag=0)
        

def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    my_name = MPI.Get_processor_name()
    #comm.Barrier()
    #start = MPI.Wtime()
    
    if my_rank == 0:
        print 'pattern to match:', PATTERN
        print 'total files:', len(FILES)
        master(comm)
    else:
        slave(comm)
    close_files()

    #comm.Barrier()
    #end = MPI.Wtime()
    #print 'time:', end - start


if __name__ == '__main__':
    main()

