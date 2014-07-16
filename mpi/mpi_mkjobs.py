import os
import sys
import re
import numpy as np
from itertools import combinations as comb
from glob import glob

ASC = '_a'
DES = '_d'
DATE = '\d\d\d\d\d\d\d\d'

sats = {#'geo': '/home/fpaolo/data/geo/*_{0}_?', 
        #'gfo': '/home/fpaolo/data/gfo/*_{0}_?', 
        #'ers1': '/home/fpaolo/data/ers1/*_{0}_?', 
        #'ers2': '/home/fpaolo/data/ers2/*_{0}_?',
        #'envi': '/home/fpaolo/data/envi/*_{0}_?', 
        #'envi_rxl': '/home/fpaolo/data/envi/raxla/*_{0}_?',
        'ice_rxl': '/home/fpaolo/data/ice/raxla/*_{0}_?_filt'}

sect = ['01', '02', '03', '04']

if len(sys.argv) == 2:    # specifies wall time
    wtime = sys.argv[1]
elif len(sys.argv) == 3:  # specifies nodes and time
    nodes = sys.argv[1]
    wtime = sys.argv[2]
else:
    nodes = 50
    wtime = 25            # default time and nodes (1 task/proc)

comb2 = lambda f: [ff for ff in comb(f, 2)]
date = lambda s: re.findall(DATE, s)

def remove_duplicates(pairs):
    for i, (f1, f2) in enumerate(pairs):
        if (ASC in f1 and ASC in f2) or \
           (DES in f1 and DES in f2) or \
           date(f1) == date(f2):
            pairs[i] = 'x' 
    return [ff for ff in pairs if ff != 'x'] 

def get_job_args(sats, sect):
    args = {}
    for sat, files in sats.items():
        for reg in sect:
            jname = 'x2s_{0}_{1}'.format(sat, reg)
            args[jname] = files.format(reg)
    return args

def create_jobs(job, args):
    for jname, files in args.items():
        fnames = glob(files)
        fcombs = comb2(fnames)
        fpairs = remove_duplicates(fcombs)
        #nodes = int(len(fpairs)/8.)
        f = open(jname+'.sh', 'w')
        f.write(job.format(jname, nodes, wtime, files))
        f.write('\n# total input files: {0}'.format(len(fnames)))
        f.write('\n# total combinations: {0}'.format(len(fcombs)))
        f.write('\n# pairs w/o duplicates: {0}\n'.format(len(fpairs)))
        f.close()
        print jname+'.sh'

def create_batch(args):
    f = open('submit.sh', 'w')
    for jname in np.sort(args.keys()):
        f.write('qsub {0}.sh &\n'.format(jname))
    f.close()
    print 'submit.sh'

job = """
#!/bin/bash

#PBS -A fpaolo-lab
#PBS -q batch 
#PBS -N {0}
#PBS -o {0}.out
#PBS -e {0}.err 
#PBS -l nodes={1}:ppn=8
#PBS -l walltime=0:{2}:00
#PBS -M fpaolo@ucsd.edu 
#PBS -m abe
#PBS -V

echo 'PBS_JOBID:' $PBS_JOBID
echo 'PBS_O_WORKDIR:' $PBS_O_WORKDIR
echo 'PBS_O_QUEUE:' $PBS_O_QUEUE
echo 'PBS_NODEFILE:' $PBS_NODEFILE

# Change into original workdir
cd $PBS_O_WORKDIR

mpiexec -v -machinefile $PBS_NODEFILE python mpi_x2sys.py '/home/fpaolo/code/x2sys/x2sys.py -r' {3}
"""

args = get_job_args(sats, sect)
create_jobs(job, args)
create_batch(args)
