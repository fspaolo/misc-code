#!/usr/bin/env python
"""
NOTE
----
1) edit ASC, DES, DATE and REG patterns according input files.
2) run mpi_ntasks.py <same args> <same files> to set up the number of nodes.

EXAMPLE
-------
mpiexec -v -machinefile $PBS_NODEFILE python mpi_x2sys.py \
    '/home/fpaolo/code/x2sys/x2sys.py -s shelf -r' ~/data/envi/*_?

September 1, 2012
Fernando Paolo <fpaolo@ucsd.edu>
"""

import os
import sys
import re
import numpy as np
from itertools import combinations as comb
from mpi4py import MPI

### [edit] patterns in file name

ASC = '_a'
DES = '_d'
DATE = '\d\d\d\d\d\d\d\d'
REG = '_\d\d_'  # `None` for no regions

def simple_partitioning(length, num_procs):
    sublengths = [length/num_procs]*num_procs
    for i in range(length % num_procs):    # treatment of remainder
        sublengths[i] += 1
    return sublengths

def get_subproblem_input_args(input_args, my_rank, num_procs):
    sub_ns = simple_partitioning(len(input_args), num_procs)
    my_offset = sum(sub_ns[:my_rank])
    my_input_args = input_args[my_offset:my_offset+sub_ns[my_rank]]
    return my_input_args

def program_to_run(string):
    if '.py' in string:
        run = 'python '
    else:
        run = '' # './'
    return run

def sep_files_by_reg(fnames, pattern):
    if pattern is None:
        return fnames
    byreg = []
    patterns = np.unique(re.findall(pattern, ' '.join(fnames)))
    for s in patterns:
        byreg.append([f for f in fnames if s in f])
    return byreg  # list of lists

def get_all_comb(files_byreg):
    tolist = lambda iterobj: [elem for elem in iterobj]
    if isinstance(files_byreg[0], list):
        pairs = []
        for fnames in files_byreg:  # one list at a time
            pairs.extend(tolist(comb(fnames, 2)))
    else:
        pairs = tolist(comb(files_byreg, 2))
    return pairs

date = lambda s: re.findall(DATE, s)

def remove_duplicates(pairs):
    for i, (f1, f2) in enumerate(pairs):
        if (ASC in f1 and ASC in f2) or \
           (DES in f1 and DES in f2) or \
           date(f1) == date(f2):
            pairs[i] = 'x' 
    return [ff for ff in pairs if ff != 'x'] 


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_procs = comm.Get_size()

prog_and_args = sys.argv[1]
files_in = sys.argv[2:]

files_in = sep_files_by_reg(files_in, REG) 
pairs = get_all_comb(files_in)
pairs = remove_duplicates(pairs)
run = program_to_run(prog_and_args)
my_pairs = get_subproblem_input_args(pairs, my_rank, num_procs)

print 'pairs of files rank #%d (of %d):\n' % (my_rank, num_procs), my_pairs
for f1, f2 in my_pairs:
    os.system('%s%s %s %s' % (run, prog_and_args, f1, f2))
    #print '%s%s %s %s' % (run, prog_and_args, f1, f2)
