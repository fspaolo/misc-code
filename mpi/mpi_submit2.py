#!/usr/bin/env python
"""
For basic applications, MPI is as easy to use as any other 
message-passing system. The sample code below contains the complete 
communications skeleton for a dynamically load balanced master/slave 
application. Following the code is a description of the few functions 
necessary to write typical parallel applications.

important parameters
--------------------

status = MPI.Status()               # where all info is stored

# Receive results from a slave
result = comm.recv(                 # message buffer
         source=MPI.ANY_SOURCE,     # receive from any sender (-1)
         tag=MPI.ANY_TAG,           # any type of message (-1)
         status=status)             # info about the received msg (class)

# Send the slave a new work unit
comm.send(work,                     # message buffer
         dest=status.Get_source(),  # to whom we just received from
         tag=WORKTAG)               # user chosen message tag

"""
# Fernandoo Paolo <fpaolo@ucsd.edu>
# Jan 15, 2013

import os
import sys
import numpy as np
from mpi4py import MPI
from Queue import Queue

WORKTAG = 1
DIETAG = 2

class Work(object):
    def __init__(self, prog, files):
        # importat: sort by file size in decreasing order!
        files.sort(key=lambda f: os.stat(f).st_size, reverse=True)
        q = Queue()
        for f in files:
            q.put(' '.join([prog, f]))
        self.work = q

    def get_next(self):
        if self.work.empty():
            return None
        return self.work.get()
 

def do_work(work):
    if '.py' in work:
        os.system('python ' + work)
    else:
        os.system(work)  # for './'
    return


def process_result(result):
    pass


def master(comm):
    num_procs = comm.Get_size()
    status = MPI.Status()
    
    # generate work queue
    wq = Work(sys.argv[1], sys.argv[2:])

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
        result = do_work(work)

        # Send the result back
        comm.send(result, dest=0, tag=0)
        

def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    my_name = MPI.Get_processor_name()
    #comm.Barrier()
    #start = MPI.Wtime()
    
    if my_rank == 0:
        master(comm)
    else:
        slave(comm)
    
    #comm.Barrier()
    #end = MPI.Wtime()
    #print 'time:', end - start


if __name__ == '__main__':
    main()
