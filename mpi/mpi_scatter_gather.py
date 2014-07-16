'''
Email 1
-------
@Aron/@Konstantin: your suggestion is very valid, however take into 
account that such approach will use pickle under the hood, then it is 
not the most efficient way to communicate array data. The code below 
is a far better, it should get near-C speed. 

Email 2
-------
Then I would first try to scatter() all the tasks at once at the 
beginning, and finally gather() results at the end. For doing 
scatter(), you have to make at processor 0 a list of lists of tasks, 
the 'outer' list with comm.size 'inner' sublists (of similar length). 
This way, even the master can do useful computation. 

The above strategy are just a few of lines. If you notice that the 
unbalance is to high (some processes finish too early compared to 
others), you can go from a more elaborated scheme using point-to-point 
(Send/Recv/Probe).i
'''

import sys
import numpy as np 
from mpi4py import MPI 

def pprint(comm, buffer):
    rank = comm.rank
    size = comm.size
    for r in xrange(size):
        if rank == r:
            print "[%d] %r" % (rank, buffer) 
        comm.Barrier()

def func(input):
    return input**2.


comm = MPI.COMM_WORLD 
rank = comm.Get_rank() 
size = comm.Get_size() 

if rank==0: 
    # process 0 is the root, it has the data to scatter 
    sendbuf = np.arange(size*size, dtype=np.float64).reshape(size, size)
    print '\ndata created at processor 0:'
    print '[0]', sendbuf
else: 
    # processes other than root has None or empty container
    sendbuf = None 
    #sendbuf = np.zeros(size*size, dtype=np.float64)

# all processes receive data 
recvbuf = np.zeros(size, dtype=np.float64)  # emptpy container on each processor
comm.Scatter(sendbuf, recvbuf, root=0)
#comm.Bcast(sendbuf, root=0)

if rank==0: print '\ndata distributed to processors:'
pprint(comm, recvbuf)

# perform operation (in parallel)
recvbuf = func(recvbuf)

# gather (on processor 0) distributed subresults
comm.Gather(recvbuf, sendbuf, root=0)
#comm.Allgather(recvbuf, sendbuf)
#comm.Recv(recvbuf, source=0)
#comm.Reduce(recvbuf, sendbuf, op=sum, root=0)


if rank==0: print '\ndata processed in parallel:'
pprint(comm, recvbuf)
if rank==0: print '\ndata gathered back to processor 0:'
pprint(comm, sendbuf)
