#!/usr/bin/env python

import os
import sys
import numpy as np
from mpi4py import MPI


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


comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
num_procs = comm.Get_size()

prog_and_args = sys.argv[1]
files_in = sys.argv[2:]

run = program_to_run(prog_and_args)
my_files = get_subproblem_input_args(files_in, my_rank, num_procs)
os.system('%s%s %s' % (run, prog_and_args, ' '.join(my_files)))

print '%s%s %s' % (run, prog_and_args, ' '.join(my_files))
