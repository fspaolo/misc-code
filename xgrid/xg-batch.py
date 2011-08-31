#!/usr/bin/env python
"""
Generate and submit batch files for Xgrid computing. 

This script is a wrapper around the `xgrid` command line tool. It 
facilitates the construction and submission of batch files for 
*multiple* input files. 

Examples
--------

To see the available options::

$ python batch.py -h

To generate and submit a batch file with jobs that call a program
`prog` with command line argument `arg` on several files::

$ python batch.py -s -j jobname -c "/path/to/prog -a arg" /path/to/files/*.ext 

See Also
--------

Xgrid: ``http://www.apple.com/server/macosx/technology/xgrid.html``

"""
# Fernando <fpaolo@ucsd.edu>
# March 6, 2011

import argparse as ap
import plistlib as pl
from glob import glob
import re
import os
import sys

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
parser.add_argument('-1', dest='onetask', default=False, action='store_const', 
    const=True, help='create one single task with all files '
    '[default: multi-task]')
parser.add_argument('-s', dest='submit', default=False, action='store_const', 
    const=True, help='submit batch file after creation')

args = parser.parse_args()
if not args.batchfile: 
    args.batchfile = args.jobname + '.xml'  # job name plus extension


# define a basic structure for an Xgrid batch file.
PLIST = {
    'jobSpecification': {
        'applicationIdentifier': 'com.apple.xgrid.cli',
        'name': args.jobname,
        'notificationEmail': args.email,
        'schedulerParameters': {'dependsOnJobs': []},
        'inputFiles': {},
        'taskSpecifications': {},
        }
    }

 
def create_tasks(args):
    """Create individual tasks given command and arguments/files.
    """
    execute = args.execute.split()
    cmd, arguments = execute[0], execute[1:]
    tasks = {}
    if args.files and args.onetask:             
        # all files -> one task
        tasks.setdefault(`0`, {'command': cmd, 
                               'arguments': arguments + args.files})
    elif args.files:                            
        # each file -> one task
        for i, f in enumerate(args.files):  
            tasks.setdefault(`i`, {'command': cmd, 
                                   'arguments': arguments + [f]})
    else:                                       
        # no files
        tasks.setdefault(`0`, {'command': cmd, 
                               'arguments': arguments})
    return tasks


def input_files(args):
    """Check if input files contain relative path (local files). 
       
    If local files, encode64 their content to be sent (over the 
    network) to the agent machines.
    """ 
    inputFiles = {}
    for f in args.files:
       if not os.path.isabs(f):
           file = open(f).read()
           inputFiles.setdefault(f, {'fileData': pl.Data(file), 
                                     'isExecutable': 'YES'})
    return inputFiles


def submit_batch(args):
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
    print 'job ID file -> %s.id' % args.jobname 
    print 'job submitted:'
    os.system('xgrid -job batch %s > %s.id' % (args.batchfile, args.jobname))
    os.system('cat %s.id' % args.jobname)


def get_ids(args):
    """Get job IDs. 

    Get a list of ID numbers passed on the command line or by
    scanning the content of ID files in a *given* directory.
    """
    ids = []
    if args.depends is not None:
        depends = args.depends[0]              # str with numbers or directory
        if os.path.isdir(depends):
            files = glob('%s/*.id' % depends)  # search for *.id files
            for fname in files:
                idNo = re.findall('\d+', open(fname).read())
                ids.append(idNo[0])
        else:
            ids = depends.split()
    return ids


def main():
    PLIST['jobSpecification']['taskSpecifications'] = create_tasks(args)
    PLIST['jobSpecification']['inputFiles'] = input_files(args) 
    PLIST['jobSpecification']['schedulerParameters']['dependsOnJobs'] = get_ids(args) 

    pl.writePlist(PLIST, args.batchfile)

    print 'batch file ->', args.batchfile 

    if args.submit: 
        submit_batch(args)


if __name__ == '__main__':
    main()
