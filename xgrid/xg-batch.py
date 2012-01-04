#!/usr/bin/env python
"""
Generate and submit batch files for Apple's Xgrid computing. 

This script is a wrapper around the `xgrid` command line tool. It 
facilitates the construction and submission of batch files for 
*multiple* input files. 

Examples
--------
To see the available options::

$ python xg-batch.py -h

To generate and submit a batch file with jobs that call a program
`prog` with command line argument `arg` on several files::

$ python xg-batch.py -s -j jobname -c "/path/to/prog -a arg" /path/to/files/*.ext 

See Also
--------
Xgrid: http://www.apple.com/science/hardware/gridcomputing.html
Tutorials: http://macresearch.org/the_xgrid_tutorials
Wiki/Docs/FAQ/etc: http://tengrid.com

"""
# Fernando <fpaolo@ucsd.edu>
# March 6, 2011

import os
import sys
import re
import argparse as ap
import plistlib as pl
from glob import glob

# parse command line arguments
parser = ap.ArgumentParser()
parser.add_argument('files', nargs='*', help='file[s] to process '
    '[ex: /path/to/files/*.ext]')
parser.add_argument('-j', dest='jobname', default='job', help='a symbolic '
    'name of the job [default: job]')
parser.add_argument('-o', dest='batchfile', default=None, help='name of the '
    'output batch file [default: <jobname>.xml]')
parser.add_argument('-c', dest='execute', required=True, help="command "
    "(with args) to execute [ex: -c '/path/to/prog -a arg']")
parser.add_argument('-d', dest='depends', nargs=1, #metavar=("'id1 id2 ...'"), 
    default=None, help="do not schedule this job until the following "
    "job (id's) have finished successfully [ex: -d 'id1 id2 ...' or "
    "-d dir_with_id_files] ")
parser.add_argument('-e', dest='email', default='', help='email to send '
    'notification of job state')
parser.add_argument('-0', dest='multitask', default=False, action='store_const', 
    const=True, help='create one task per input files (multi-task) '
    '[default]')
parser.add_argument('-1', dest='singletask', default=False, action='store_const', 
    const=True, help='create a single task with all input files '
    '[default: multi-task]')
parser.add_argument('-2', dest='combinedtask', default=False, action='store_const', 
    const=True, help='create one task every two combined input files ' 
    '[default: multi-task]')
parser.add_argument('-s', dest='submit', default=False, action='store_const', 
    const=True, help='submit batch file after creation')

args = parser.parse_args()
files = args.files
execute = args.execute.split()
batchfile = args.batchfile
jobname = args.jobname
depends = args.depends
email = args.email
submit = args.submit
task = {}
task['single'] = args.singletask
task['multi'] = args.multitask
task['combined'] = args.combinedtask

if any(task.values()):
    pass
else:
    task['multi'] = True          # default action

if task['combined']:
    from itertools import combinations
    files = combinations(files, 2)

if not batchfile: 
    batchfile = jobname + '.xml'  # job name plus extension

# define a basic structure for an Xgrid batch file.
PLIST = {
    'jobSpecification': {
        'applicationIdentifier': 'com.apple.xgrid.cli',
        'name': jobname,
        'notificationEmail': email,
        'schedulerParameters': {'dependsOnJobs': []},
        'inputFiles': {},
        'taskSpecifications': {},
        }
    }

 
def create_tasks(*args):
    """Create individual tasks given command and arguments/files.
    """
    cmd, arguments = execute[0], execute[1:]
    tasks = {}
    # `-1` all files -> one task (sequential)
    if task['single'] and files:             
        print 'all files in one task (single)'
        tasks.setdefault(`0`, {'command': cmd, 
                               'arguments': arguments + files})
    # `-0` each file -> one task (parallel)
    elif task['multi'] and files:                            
        print 'each file in one task (multi)'
        for i, f in enumerate(files):  
            tasks.setdefault(`i`, {'command': cmd, 
                                   'arguments': arguments + [f]})
    # `-2` every two files -> one task (parallel)
    elif task['combined'] and files:                            
        print 'two files in one task (combined)'
        for i, (f1, f2) in enumerate(files):  
            tasks.setdefault(`i`, {'command': cmd, 
                                   'arguments': arguments + [f1, f2]})
    # no files
    else:                                       
        tasks.setdefault(`0`, {'command': cmd, 
                               'arguments': arguments})
    return tasks


def input_files(*args):
    """Check if input files contain relative path (local files). 
       
    If local files, encode64 their content to be sent (over the 
    network) to the agent machines.
    """ 
    inputFiles = {}
    for f in files:
       if not os.path.isabs(f):
           file = open(f).read()
           inputFiles.setdefault(f, {'fileData': pl.Data(file), 
                                     'isExecutable': 'YES'})
    return inputFiles


def submit_batch(*args):
    """Submit batch file (job) to the controller. 

    Check first if the environmental variables for controller `name`
    and `password` are defined, otherwise raise an error.
    """
    controller = os.environ.get('XGRID_CONTROLLER_HOSTNAME')
    password = os.environ.get('XGRID_CONTROLLER_PASSWORD')
    if (not controller) or (not password):
        print 'error submitting: no controller name/password specified!'
        print 'please set the following environmental variables:'
        print 'XGRID_CONTROLLER_HOSTNAME=hostname'
        print 'XGRID_CONTROLLER_PASSWORD=password'
        sys.exit()
    print 'job ID file -> %s.id' % jobname 
    print 'job submitted:'
    os.system('xgrid -job batch %s > %s.id' % (batchfile, jobname))
    os.system('cat %s.id' % jobname)


def get_ids(*args):
    """Get job IDs. 

    Get a list of ID numbers passed on the command line or by
    scanning the content of ID files in a *given* directory.
    """
    ids = []
    if depends is not None:    # str with numbers or directory
        if os.path.isdir(depends[0]):
            files = glob('%s/*.id' % depends[0])  # search for `*.id` files
            for fname in files:
                idNo = re.findall('\d+', open(fname).read())
                ids.append(idNo[0])
        else:
            ids = depends[0].split()
    return ids


def main():
    PLIST['jobSpecification']['taskSpecifications'] \
        = create_tasks(execute, task)
    PLIST['jobSpecification']['inputFiles'] = input_files(files) 
    PLIST['jobSpecification']['schedulerParameters']['dependsOnJobs'] \
        = get_ids(depends) 

    pl.writePlist(PLIST, batchfile)

    print 'batch file ->', batchfile 

    if submit: 
        submit_batch(batchfile, jobname)


if __name__ == '__main__':
    main()
